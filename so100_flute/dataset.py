"""ACT-style HDF5 episodes, exact simulator states, and action replay."""

import hashlib
import json
import platform
import time
from pathlib import Path

import h5py
import mujoco
import numpy as np

from .env import CAMERAS, CONTROL_DT, JOINT_NAMES, FluteEnv
from .expert import expert_actions
from .scene import ROOT, build_scene

PHASES = (
    "settle",
    "approach",
    "grasp",
    "lift",
    "transfer",
    "flip",
    "align",
    "lower",
    "place",
    "release",
    "unhook",
    "withdraw",
    "retreat",
    "verify",
)
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def scene_hash():
    xml = build_scene().replace(str(ROOT), "${PACKAGE}")
    digest = hashlib.sha256(xml.encode())
    for path in sorted((ROOT / "assets/so100/assets").glob("*")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sim_state(env):
    state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
    mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
    return state


def record_episode(
    path,
    seed=0,
    images=True,
    width=320,
    height=240,
    speed=1.0,
    controller=None,
    initialize=None,
    metadata=None,
):
    """Write a candidate to a temporary file; publish it only if successful."""
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be positive and finite")
    if images and not (0 < width <= 960 and 0 < height <= 720):
        raise ValueError("image size must be within 1..960 by 1..720")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".hdf5.partial")
    if temporary.exists():
        raise FileExistsError(f"Remove stale partial recording explicitly: {temporary}")
    env = FluteEnv(seed=seed)
    started = time.monotonic()
    try:
        if initialize is not None:
            initialize(env)
        with h5py.File(temporary, "x") as file:
            file.attrs.update(
                sim=True,
                schema_version="1.0",
                seed=seed,
                control_dt=CONTROL_DT,
                mujoco_version=mujoco.__version__,
                numpy_version=np.__version__,
                python_version=platform.python_version(),
                speed=speed,
                joint_names=json.dumps(JOINT_NAMES),
                phase_names=json.dumps(PHASES),
                scene_sha256=scene_hash(),
                state_spec=int(STATE_SPEC),
                task="Take a rim-down champagne flute from the shelf and place it upright on the table.",
                action_semantics="absolute joint position targets, radians; observation[t] precedes action[t]",
                source_xy=env.source,
                target_xy=env.target,
                quaternion_order="wxyz",
            )
            if metadata is not None:
                file.attrs["collection_metadata"] = json.dumps(metadata)
            image_sets = {}
            if images:
                for camera in CAMERAS:
                    image_sets[camera] = file.create_dataset(
                        f"observations/images/{camera}",
                        (0, height, width, 3),
                        maxshape=(None, height, width, 3),
                        dtype="u1",
                        chunks=(1, height, width, 3),
                        compression="gzip",
                        compression_opts=1,
                    )
            observations, actions, phases, states, metrics = [], [], [], [sim_state(env)], []
            for phase, action in (controller or expert_actions)(env, speed=speed):
                i = len(actions)
                observations.append(env.observe())
                for camera, dataset in image_sets.items():
                    dataset.resize(i + 1, axis=0)
                    dataset[i] = env.render(camera, width, height)
                actions.append(action.copy())
                phases.append(PHASES.index(phase))
                env.step(action)
                states.append(sim_state(env))
                metrics.append(env.metrics())
            for name in observations[0]:
                file.create_dataset(
                    f"observations/{name}",
                    data=np.array([o[name] for o in observations]),
                    compression="gzip",
                    shuffle=True,
                )
            file.create_dataset("action", data=np.array(actions), compression="gzip", shuffle=True)
            file.create_dataset("phase", data=np.array(phases, dtype=np.uint8))
            file.create_dataset("timestamp", data=np.arange(len(actions)) * CONTROL_DT)
            file.create_dataset(
                "sim/state", data=np.array(states), compression="gzip", shuffle=True
            )
            file.create_dataset(
                "sim/qpos",
                data=np.array(
                    [np.r_[o["qpos"], o["object_pose"]] for o in observations]
                    + [env.data.qpos.copy()]
                ),
                compression="gzip",
                shuffle=True,
            )
            file.create_dataset(
                "success", data=np.array([m["success"] for m in metrics], dtype=bool)
            )
            final = env.metrics()
            file.attrs["success"] = final["success"]
            file.attrs["final_metrics"] = json.dumps(final)
            file.attrs["num_steps"] = len(actions)
        summary = dict(
            seed=seed,
            speed=speed,
            steps=len(actions),
            metrics=final,
            wall_seconds=round(time.monotonic() - started, 3),
        )
        if final["success"]:
            temporary.rename(path)
            summary["file"] = path.name
        else:
            temporary.unlink()
        return summary
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        env.close()


def generate(output, episodes=20, seed=0, images=True, width=320, height=240, max_attempts=None):
    output = Path(output)
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if max_attempts is not None and max_attempts < episodes:
        raise ValueError("max_attempts must be at least episodes")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    build_scene(output / "scene.xml")
    accepted, attempts = [], []
    manifest = dict(
        schema_version="1.0",
        task="inverted_shelf_to_upright_table",
        episodes=accepted,
        attempts=attempts,
        cameras=list(CAMERAS) if images else [],
        image_shape=[height, width, 3] if images else None,
        control_hz=1 / CONTROL_DT,
        scene_sha256=scene_hash(),
        mujoco_version=mujoco.__version__,
    )
    for attempt in range(max_attempts or episodes * 3):
        current_seed = seed + attempt
        speed = float(np.random.default_rng(current_seed + 10000).uniform(0.9, 1.1))
        result = record_episode(
            output / f"episode_{len(accepted):06d}.hdf5", current_seed, images, width, height, speed
        )
        attempts.append(result)
        if result["metrics"]["success"]:
            accepted.append(result)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"seed {current_seed}: {'accepted' if result['metrics']['success'] else 'rejected'}; {len(accepted)}/{episodes}",
            flush=True,
        )
        if len(accepted) == episodes:
            return manifest
    raise RuntimeError(
        f"Only {len(accepted)}/{episodes} successful episodes after {len(attempts)} attempts; see manifest.json"
    )


def replay(path, video=None, camera="overview", verify=True, realtime=False):
    """Re-simulate recorded actions from the full initial integration state."""
    from contextlib import nullcontext

    import imageio.v2 as imageio

    with h5py.File(path, "r") as file:
        if file.attrs["scene_sha256"] != scene_hash():
            raise ValueError("Episode scene/assets do not match this version of the task")
        if file.attrs["mujoco_version"] != mujoco.__version__:
            raise ValueError("Replay verification requires the recorded MuJoCo version")
        env = FluteEnv(seed=int(file.attrs["seed"]))
        env.target[:] = file.attrs["target_xy"]
        env.model.site("target").pos[:2] = env.target
        mujoco.mj_setState(env.model, env.data, file["sim/state"][0], STATE_SPEC)
        mujoco.mj_forward(env.model, env.data)
        writer = (
            imageio.get_writer(str(video), fps=25, codec="libx264", quality=8) if video else None
        )
        error = 0.0
        try:
            if realtime:
                from mujoco import viewer as mj_viewer

                viewer_context = mj_viewer.launch_passive(env.model, env.data)
            else:
                viewer_context = nullcontext(None)
            with viewer_context as viewer:
                for i, action in enumerate(file["action"]):
                    started = time.monotonic()
                    if writer and i % 2 == 0:
                        writer.append_data(env.render(camera, 640, 480))
                    env.step(action)
                    if verify:
                        error = max(
                            error, float(np.max(np.abs(sim_state(env) - file["sim/state"][i + 1])))
                        )
                    if viewer:
                        if not viewer.is_running():
                            break
                        viewer.sync()
                        time.sleep(max(0, CONTROL_DT - (time.monotonic() - started)))
            metrics = env.metrics()
        finally:
            if writer:
                writer.close()
            env.close()
        completed = i + 1 == len(file["action"])
        if verify and error > 1e-7:
            raise AssertionError(f"Action replay diverged: max integration-state error {error:.3g}")
        if verify and completed and not metrics["success"]:
            raise AssertionError(f"Replay did not complete the task: {metrics}")
        return dict(max_state_error=error, steps=i + 1, completed=completed, **metrics)
