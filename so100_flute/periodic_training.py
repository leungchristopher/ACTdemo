"""
Training + checkpoint evaluations.
Caches action targets and checkpoint evaluation is in a child process, so that thep olicy/optimiser/RNG state are preserved.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class CheckpointEvaluator:
    def __init__(self, every=2000, episodes=10, seed=1000, max_steps=1500):
        if every < 1 or episodes < 1 or max_steps < 1:
            raise ValueError("Evaluation interval, episodes and max steps must be positive")
        self.settings = dict(
            every=every, episodes=episodes, seed=seed, max_steps=max_steps, video_episodes=1
        )

    def __call__(self, checkpoint_dir):
        checkpoint = Path(checkpoint_dir).resolve()
        step = json.loads((checkpoint / "training_state/training_step.json").read_text())["step"]
        if step == 0 or step % self.settings["every"]:
            return
        root = checkpoint.parent.parent / "eval"
        root.mkdir(exist_ok=True)
        summary_path = root / "summary.json"
        summary = (
            json.loads(summary_path.read_text())
            if summary_path.exists()
            else dict(settings=self.settings, checkpoints=[], best=None)
        )
        # Cadence can change on resume; episode protocol must stay comparable.
        previous_protocol = {k: v for k, v in summary["settings"].items() if k != "every"}
        current_protocol = {k: v for k, v in self.settings.items() if k != "every"}
        if previous_protocol != current_protocol:
            raise ValueError(
                "Evaluation settings differ from eval/summary.json; use the original settings on resume"
            )
        if summary["settings"] != self.settings:
            summary["settings"] = self.settings
            write_json(summary_path, summary)
        if any(row["step"] == step for row in summary["checkpoints"]):
            return
        # Preserve partial results after interruption; retries get a new folder.
        step_root = root / f"step_{step:06d}"
        step_root.mkdir(exist_ok=True)
        attempt = 0
        while (step_root / f"attempt_{attempt:03d}").exists():
            attempt += 1
        output = step_root / f"attempt_{attempt:03d}"
        command = [
            sys.executable,
            "-m",
            "so100_flute.training",
            "evaluate",
            str(checkpoint / "pretrained_model"),
            "--output",
            str(output),
            "--episodes",
            str(self.settings["episodes"]),
            "--seed",
            str(self.settings["seed"]),
            "--max-steps",
            str(self.settings["max_steps"]),
            "--video",
            "--video-episodes",
            "1",
        ]
        logging.info("MuJoCo evaluation at step %s: %s episodes", step, self.settings["episodes"])
        # A failed evaluation stops after a durable checkpoint. Resume retries it.
        subprocess.run(command, check=True)
        report = json.loads((output / "metrics.json").read_text())
        results = report["results"]
        if (
            len(results) != self.settings["episodes"]
            or any(r["aborted"] for r in results)
            or [r["seed"] for r in results]
            != list(range(self.settings["seed"], self.settings["seed"] + self.settings["episodes"]))
        ):
            raise RuntimeError(f"Incomplete evaluation: {output}")
        video = output / "episode_000.mp4"
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError(f"Evaluation video is missing or empty: {video}")
        row = dict(
            step=step,
            checkpoint=str(checkpoint / "pretrained_model"),
            metrics=str(output / "metrics.json"),
            video=str(video),
            success_rate=report["success_rate"],
        )
        summary["checkpoints"].append(row)
        summary["checkpoints"].sort(key=lambda row: row["step"])
        # Keep the earlier checkpoint when success rates tie.
        summary["best"] = max(summary["checkpoints"], key=lambda row: row["success_rate"])
        write_json(summary_path, summary)
        logging.info(
            "MuJoCo step %s success %.1f%%; best step %s",
            step,
            100 * row["success_rate"],
            summary["best"]["step"],
        )


def main():
    from .dependencies import require_training_dependencies

    require_training_dependencies()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--flute-eval-every", type=int, default=2000)
    parser.add_argument("--flute-eval-episodes", type=int, default=10)
    parser.add_argument("--flute-eval-seed", type=int, default=1000)
    parser.add_argument("--flute-eval-max-steps", type=int, default=1500)
    parser.add_argument("--flute-cache-actions", choices=["true", "false"], default="true")
    args, remaining = parser.parse_known_args()
    evaluator = (
        CheckpointEvaluator(
            args.flute_eval_every,
            args.flute_eval_episodes,
            args.flute_eval_seed,
            args.flute_eval_max_steps,
        )
        if args.flute_eval_every
        else None
    )
    # Retry evaluation of a resumed boundary checkpoint before further training.
    resume_parser = argparse.ArgumentParser(add_help=False)
    resume_parser.add_argument("--resume", default="false")
    resume_parser.add_argument("--config_path", type=Path)
    resume_args, _ = resume_parser.parse_known_args(remaining)
    if evaluator and resume_args.resume.lower() == "true":
        evaluator(resume_args.config_path.resolve().parent.parent)
    sys.argv = [sys.argv[0], *remaining]
    import lerobot.scripts.lerobot_train as trainer

    original = trainer.update_last_checkpoint
    original_datasets = trainer.make_train_eval_datasets

    def make_cached_datasets(cfg):
        from .fast_dataset import cache_action_targets

        return tuple(cache_action_targets(ds) for ds in original_datasets(cfg))

    def update_and_evaluate(checkpoint_dir):
        result = original(checkpoint_dir)
        if evaluator:
            evaluator(checkpoint_dir)
        return result

    trainer.update_last_checkpoint = update_and_evaluate
    if args.flute_cache_actions == "true":
        trainer.make_train_eval_datasets = make_cached_datasets
    try:
        trainer.main()
    finally:
        trainer.update_last_checkpoint = original
        trainer.make_train_eval_datasets = original_datasets


if __name__ == "__main__":
    main()
