import json
import sys
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs/act_flute.json"


def training_command(
    config=DEFAULT_CONFIG,
    *,
    resume=None,
    eval_every=2000,
    eval_episodes=10,
    eval_seed=1000,
    eval_max_steps=1500,
    cache_actions=True,
    **overrides,
):

    config_path = Path(resume or config).resolve()
    cfg = json.loads(config_path.read_text())
    root = Path(overrides.get("dataset") or cfg["dataset"]["root"])
    report = json.loads((root / "conversion.json").read_text())
    if not report.get("complete"):
        raise ValueError("Dataset conversion is incomplete")
    if report["repo_id"] != (overrides.get("dataset_repo_id") or cfg["dataset"]["repo_id"]):
        raise ValueError("Dataset repo_id must match the training config")
    if eval_every < 0 or eval_episodes < 1 or eval_max_steps < 1:
        raise ValueError(
            "Evaluation interval must be nonnegative; episodes and max steps must be positive"
        )
    module = (
        "so100_flute.periodic_training"
        if eval_every or cache_actions
        else "lerobot.scripts.lerobot_train"
    )
    command = [sys.executable, "-m", module, f"--config_path={config_path}"]
    if eval_every or cache_actions:
        command += [
            f"--flute-cache-actions={str(cache_actions).lower()}",
            f"--flute-eval-every={eval_every}",
        ]
    if eval_every:
        command += [
            f"--flute-eval-episodes={eval_episodes}",
            f"--flute-eval-seed={eval_seed}",
            f"--flute-eval-max-steps={eval_max_steps}",
            f"--save_freq={eval_every}",
            "--save_checkpoint=true",
        ]
    if resume:
        command.append("--resume=true")
    mappings = {
        "device": "policy.device",
        "steps": "steps",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "output": "output_dir",
        "dataset": "dataset.root",
        "log_freq": "log_freq",
        "dataset_repo_id": "dataset.repo_id",
    }
    for key, cli_name in mappings.items():
        value = overrides.get(key)
        if value is not None:
            command.append(f"--{cli_name}={value}")
    if overrides.get("num_workers") == 0:
        command.append("--persistent_workers=false")
    return command
