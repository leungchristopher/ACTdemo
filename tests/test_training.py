"""Training checks are optional when only simulator dependencies are installed."""

import json
from unittest.mock import patch

import h5py
import numpy as np
import pytest

pytest.importorskip("lerobot")
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from so100_flute.conversion import convert_dataset
from so100_flute.env import CAMERAS, JOINT_NAMES
from so100_flute.evaluation import evaluation_action_steps
from so100_flute.periodic_training import CheckpointEvaluator
from so100_flute.train_command import DEFAULT_CONFIG, training_command


def test_requested_training_recipe(tmp_path):
    cfg = TrainPipelineConfig.from_pretrained(DEFAULT_CONFIG)
    cfg.output_dir = tmp_path / "recipe_check"
    cfg.validate()
    assert cfg.batch_size == 64
    assert cfg.steps == 200_000
    assert cfg.policy.n_decoder_layers == 7
    assert cfg.policy.chunk_size == 100
    assert cfg.policy.n_action_steps == 16
    assert cfg.policy.temporal_ensemble_coeff is None
    assert cfg.optimizer.type == "adamw"
    assert cfg.optimizer.lr == cfg.policy.optimizer_lr_backbone == 2e-5
    assert cfg.scheduler is None
    assert not cfg.dataset.image_transforms.enable
    assert not cfg.policy.push_to_hub
    assert not cfg.wandb.enable


def small_policy():
    # Keep all seven decoder layers, but reduce width for a fast gradient check.
    cfg = ACTConfig(
        n_decoder_layers=7,
        n_action_steps=16,
        chunk_size=100,
        optimizer_lr=2e-5,
        optimizer_lr_backbone=2e-5,
        device="cpu",
        push_to_hub=False,
        dim_model=64,
        dim_feedforward=128,
        n_heads=4,
        n_encoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.overview": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 32, 32)
            ),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    )
    return ACTPolicy(cfg)


def test_seven_decoder_layers_train_and_queue_executes_sixteen_actions():
    torch.set_num_threads(2)
    policy = small_policy()
    optimizer = policy.config.get_optimizer_preset().build(policy.get_optim_params())
    assert isinstance(optimizer, torch.optim.AdamW)
    assert [group["lr"] for group in optimizer.param_groups] == [2e-5, 2e-5]
    batch = {
        "observation.state": torch.randn(2, 6),
        "observation.images.overview": torch.rand(2, 3, 32, 32),
        "action": torch.randn(2, 100, 6),
        "action_is_pad": torch.zeros(2, 100, dtype=torch.bool),
    }
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert len(policy.model.decoder.layers) == 7
    for layer in policy.model.decoder.layers:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters())
    optimizer.step()
    assert [group["lr"] for group in optimizer.param_groups] == [2e-5, 2e-5]
    policy.reset()
    chunk = torch.arange(100.0).view(1, 100, 1).expand(1, 100, 6)
    with patch.object(policy, "predict_action_chunk", return_value=chunk) as predict:
        actions = [policy.select_action({}) for _ in range(17)]
    assert predict.call_count == 2
    assert [a[0, 0].item() for a in actions] == list(range(16)) + [0]


@pytest.mark.parametrize("horizon", [16, 32, 48])
def test_evaluation_override_changes_only_execution_queue(horizon):
    policy = small_policy()
    assert evaluation_action_steps(policy.config) == 16
    length = evaluation_action_steps(policy.config, horizon)
    assert policy.config.n_action_steps == 16
    policy.config.n_action_steps = length
    policy.reset()
    chunk = torch.arange(100.0).view(1, 100, 1).expand(1, 100, 6)
    with patch.object(policy, "predict_action_chunk", return_value=chunk) as predict:
        actions = [policy.select_action({}) for _ in range(horizon + 1)]
    assert predict.call_count == 2
    assert [a[0, 0].item() for a in actions] == list(range(horizon)) + [0]
    assert policy.config.chunk_size == 100
    for invalid in [0, 101]:
        with pytest.raises(ValueError, match="chunk_size"):
            evaluation_action_steps(policy.config, invalid)


def make_source(root):
    root.mkdir()
    entries = []
    for episode, n in enumerate([4, 5]):
        name = f"episode_{episode:06d}.hdf5"
        with h5py.File(root / name, "w") as f:
            f.attrs.update(
                success=True,
                seed=episode,
                control_dt=0.02,
                joint_names=json.dumps(JOINT_NAMES),
                task="Move the flute upright onto the table",
            )
            f.create_dataset("timestamp", data=np.arange(n) * 0.02)
            f.create_dataset("action", data=np.arange(n * 6).reshape(n, 6) + episode * 100)
            f.create_dataset("observations/qpos", data=np.arange(n * 6).reshape(n, 6) * 0.01)
            for c in CAMERAS:
                image = np.zeros((n, 32, 32, 3), dtype=np.uint8)
                image[..., 0] = 25 + episode
                image[..., 1] = np.arange(n)[:, None, None]
                f.create_dataset(f"observations/images/{c}", data=image)
        entries.append({"file": name})
    (root / "manifest.json").write_text(json.dumps({"episodes": entries, "scene_sha256": "test"}))


def test_conversion_preserves_alignment_rgb_and_episode_padding(tmp_path):
    source, output = tmp_path / "hdf5", tmp_path / "lerobot"
    make_source(source)
    report = convert_dataset(source, output)
    assert report["complete"] and report["total_frames"] == 9
    dataset = LeRobotDataset(
        "local/so100_flute",
        root=output,
        delta_timestamps={"action": [i * 0.02 for i in range(100)]},
        video_backend="pyav",
    )
    item = dataset[3]
    np.testing.assert_array_equal(item["action"][0].numpy(), np.arange(18, 24))
    assert item["action_is_pad"].tolist() == [False] + [True] * 99
    np.testing.assert_array_equal(
        item["observation.state"].numpy(), (np.arange(18, 24) * 0.01).astype("float32")
    )
    rgb = item["observation.images.overview"].numpy().transpose(1, 2, 0)
    np.testing.assert_array_equal(
        np.round(rgb * 255).astype("uint8"), np.full((32, 32, 3), [25, 3, 0], dtype="uint8")
    )
    command = training_command(dataset=output, steps=2, batch_size=2, num_workers=0)
    assert "--steps=2" in command
    assert "--batch_size=2" in command
    assert "--persistent_workers=false" in command
    assert "--scheduler" not in " ".join(command)
    assert "so100_flute.periodic_training" in command
    assert "--flute-eval-every=2000" in command
    assert "--save_freq=2000" in command
    assert "--flute-cache-actions=true" in command
    assert "--dataset.repo_id=local/so100_flute" in training_command(
        dataset=output, dataset_repo_id="local/so100_flute"
    )
    assert "lerobot.scripts.lerobot_train" in training_command(
        dataset=output, eval_every=0, cache_actions=False
    )
    with pytest.raises(FileExistsError):
        convert_dataset(source, output)


def test_periodic_evaluation_boundaries_resume_and_best(tmp_path, monkeypatch):
    calls = []

    def run(command, check):
        assert check
        assert "--video" in command
        assert command[command.index("--video-episodes") + 1] == "1"
        calls.append(command)
        output = tmp_path / command[command.index("--output") + 1]
        output.mkdir()
        (output / "episode_000.mp4").write_bytes(b"test video")
        success = len(calls) > 1
        (output / "metrics.json").write_text(
            json.dumps(
                dict(
                    success_rate=float(success),
                    results=[dict(seed=s, aborted=False, success=success) for s in [1000, 1001]],
                )
            )
        )

    monkeypatch.setattr("so100_flute.periodic_training.subprocess.run", run)
    evaluator = CheckpointEvaluator(every=10, episodes=2)
    for step in [9, 10, 10, 20, 30]:
        checkpoint = tmp_path / "checkpoints" / f"{step:06d}"
        (checkpoint / "training_state").mkdir(parents=True, exist_ok=True)
        (checkpoint / "training_state/training_step.json").write_text(json.dumps(dict(step=step)))
        evaluator(checkpoint)
    assert len(calls) == 3
    summary = json.loads((tmp_path / "eval/summary.json").read_text())
    assert [r["step"] for r in summary["checkpoints"]] == [10, 20, 30]
    assert summary["best"]["step"] == 20
    assert summary["best"]["video"].endswith("episode_000.mp4")
    with pytest.raises(ValueError, match="settings differ"):
        CheckpointEvaluator(every=10, episodes=3)(checkpoint)
    # Changing cadence preserves history and does not redo a completed boundary.
    CheckpointEvaluator(every=2, episodes=2)(checkpoint)
    summary = json.loads((tmp_path / "eval/summary.json").read_text())
    assert summary["settings"]["every"] == 2
    assert [r["step"] for r in summary["checkpoints"]] == [10, 20, 30]
    assert summary["best"]["step"] == 20
    assert len(calls) == 3
    checkpoint = tmp_path / "checkpoints/000032"
    (checkpoint / "training_state").mkdir(parents=True)
    (checkpoint / "training_state/training_step.json").write_text('{"step":32}')
    CheckpointEvaluator(every=2, episodes=2)(checkpoint)
    assert len(calls) == 4
    summary = json.loads((tmp_path / "eval/summary.json").read_text())
    assert [r["step"] for r in summary["checkpoints"]] == [10, 20, 30, 32]


def test_failed_periodic_evaluation_retries_without_overwriting(tmp_path, monkeypatch):
    import subprocess

    checkpoint = tmp_path / "checkpoints/000010"
    (checkpoint / "training_state").mkdir(parents=True)
    (checkpoint / "training_state/training_step.json").write_text('{"step":10}')
    outputs = []

    def run(command, check):
        output = tmp_path / command[command.index("--output") + 1]
        output.mkdir()
        outputs.append(output)
        if len(outputs) == 1:
            raise subprocess.CalledProcessError(1, command)
        (output / "episode_000.mp4").write_bytes(b"test video")
        (output / "metrics.json").write_text(
            json.dumps(
                dict(success_rate=0.0, results=[dict(seed=1000, aborted=False, success=False)])
            )
        )

    monkeypatch.setattr("so100_flute.periodic_training.subprocess.run", run)
    evaluator = CheckpointEvaluator(every=10, episodes=1)
    with pytest.raises(subprocess.CalledProcessError):
        evaluator(checkpoint)
    assert not (tmp_path / "eval/summary.json").exists()
    evaluator(checkpoint)
    assert outputs[0].name == "attempt_000"
    assert outputs[1].name == "attempt_001"
    assert (tmp_path / "eval/summary.json").exists()


@pytest.mark.parametrize("episodes", [None, [1]])
@pytest.mark.parametrize("return_uint8", [False, True])
def test_action_cache_preserves_every_sample_and_filtered_episode_boundaries(
    tmp_path, episodes, return_uint8
):
    import pickle

    from so100_flute.fast_dataset import cache_action_targets

    source, output = tmp_path / "hdf5", tmp_path / "lerobot"
    make_source(source)
    convert_dataset(source, output)
    dataset = LeRobotDataset(
        "local/so100_flute",
        root=output,
        episodes=episodes,
        delta_timestamps={"action": [i * 0.02 for i in range(100)]},
        video_backend="pyav",
        return_uint8=return_uint8,
    )
    expected = [dataset[i] for i in range(len(dataset))]
    cache_action_targets(dataset)
    # Spawn workers serialize the dataset. Check that the cache survives that.
    dataset = pickle.loads(pickle.dumps(dataset))
    for i, original in enumerate(expected):
        actual = dataset[i]
        assert actual.keys() == original.keys()
        for key in original:
            if isinstance(original[key], torch.Tensor):
                torch.testing.assert_close(actual[key], original[key], rtol=0, atol=0)
            else:
                assert actual[key] == original[key]
    # Downstream mutation of a chunk must not alter subsequent reads.
    dataset[0]["action"].fill_(9999)
    torch.testing.assert_close(dataset[0]["action"], expected[0]["action"], rtol=0, atol=0)
