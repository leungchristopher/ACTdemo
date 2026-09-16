"""Run ACT in MuJoCo and save per-episode metrics, actions, and videos."""

import json
from pathlib import Path

import numpy as np

from .dependencies import require_training_dependencies
from .env import CAMERAS, CONTROL_DT


def evaluation_action_steps(cfg, override=None):
    """Validate a runtime queue length without changing the saved policy config."""
    if cfg.temporal_ensemble_coeff is not None:
        raise ValueError("Evaluation requires disabled temporal ensembling")
    if override is None:
        if cfg.n_action_steps != 16:
            raise ValueError("Expected n_action_steps=16; use an explicit evaluation override")
        return 16
    if not 1 <= override <= cfg.chunk_size:
        raise ValueError(f"n_action_steps must be between 1 and chunk_size ({cfg.chunk_size})")
    return override


def evaluate_checkpoint(
    checkpoint,
    output,
    episodes=10,
    seed=1000,
    max_steps=1500,
    device=None,
    video=False,
    viewer=False,
    video_episodes=None,
    n_action_steps=None,
):
    """Evaluate saved ACT weights with an optional execution-length override."""
    require_training_dependencies()
    import time
    from contextlib import nullcontext

    import imageio.v2 as imageio
    import torch
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    from .dataset import scene_hash
    from .env import FluteEnv

    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    if video_episodes is not None and video_episodes < 1:
        raise ValueError("video_episodes must be positive")
    checkpoint, output = Path(checkpoint), Path(output)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Expected a local pretrained_model directory: {checkpoint}")
    if output.exists():
        raise FileExistsError(output)
    cfg = ACTConfig.from_pretrained(checkpoint)
    saved_n_action_steps = cfg.n_action_steps
    cfg.n_action_steps = evaluation_action_steps(cfg, n_action_steps)
    if device:
        cfg.device = device
    policy = ACTPolicy.from_pretrained(checkpoint, config=cfg).to(cfg.device).eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "normalizer_processor": {"device": cfg.device},
        },
        postprocessor_overrides={"unnormalizer_processor": {"device": cfg.device}},
    )
    image_features = cfg.image_features
    for camera in CAMERAS:
        if f"observation.images.{camera}" not in image_features:
            raise ValueError(f"Checkpoint lacks {camera} camera")
    if cfg.robot_state_feature.shape != (6,) or cfg.action_feature.shape != (6,):
        raise ValueError("Expected six state and action dimensions")
    output.mkdir(parents=True)
    results = []
    aborted = False
    for episode in range(episodes):
        env = FluteEnv(seed=seed + episode)
        policy.reset()
        pre.reset()
        post.reset()
        record_video = video and (video_episodes is None or episode < video_episodes)
        writer = (
            imageio.get_writer(output / f"episode_{episode:03d}.mp4", fps=25)
            if record_video
            else None
        )
        if viewer:
            from mujoco import viewer as mj_viewer

            context = mj_viewer.launch_passive(env.model, env.data)
        else:
            context = nullcontext(None)
        actions, states = [], [env.data.qpos.copy()]
        try:
            with context as window, torch.inference_mode():
                for step in range(max_steps):
                    start = time.monotonic()
                    batch = {
                        "observation.state": torch.from_numpy(env.data.qpos[:6].astype(np.float32))
                    }
                    rgb_views = {}
                    for camera in CAMERAS:
                        key = f"observation.images.{camera}"
                        _, height, width = image_features[key].shape
                        rgb = env.render(camera, width, height)
                        rgb_views[camera] = rgb
                        batch[key] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                    action = post(policy.select_action(pre(batch))).squeeze(0).cpu().numpy()
                    if action.shape != (6,) or not np.isfinite(action).all():
                        raise RuntimeError("Policy produced an invalid action")
                    if writer and step % 2 == 0:
                        writer.append_data(rgb_views["overview"])
                    env.step(action)
                    actions.append(action)
                    states.append(env.data.qpos.copy())
                    if window:
                        if not window.is_running():
                            aborted = True
                            break
                        window.sync()
                        time.sleep(max(0, CONTROL_DT - (time.monotonic() - start)))
                    # Require two stable seconds, including withdrawal, before stopping.
                    metrics = env.metrics()
                    if metrics["success"] and metrics["stable_seconds"] >= 2:
                        break
            metrics = env.metrics()
            np.savez_compressed(output / f"episode_{episode:03d}.npz", actions=actions, qpos=states)
            result = dict(seed=seed + episode, steps=len(actions), aborted=aborted, **metrics)
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            if writer:
                writer.close()
            env.close()
        report = dict(
            checkpoint=str(checkpoint.resolve()),
            scene_sha256=scene_hash(),
            n_action_steps=cfg.n_action_steps,
            saved_n_action_steps=saved_n_action_steps,
            device=str(cfg.device),
            results=results,
            success_rate=sum(r["success"] for r in results) / len(results),
        )
        (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        if aborted:
            break
    return report
