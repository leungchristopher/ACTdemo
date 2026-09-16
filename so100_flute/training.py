import argparse
import json
import shlex
import subprocess
from pathlib import Path

from .conversion import convert_dataset
from .dependencies import require_training_dependencies
from .evaluation import evaluate_checkpoint
from .train_command import DEFAULT_CONFIG, training_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="Convert HDF5 experts to a local LeRobot dataset")
    convert.add_argument("--source", type=Path, default=Path("data/expert_v1"))
    convert.add_argument("--output", type=Path, default=Path("data/lerobot_flute"))
    convert.add_argument("--repo-id", default="local/so100_flute")
    train = commands.add_parser("train", help="Train with the checked-in recipe using LeRobot")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--resume", type=Path, help="Checkpoint pretrained_model/train_config.json")
    train.add_argument("--dataset", type=Path)
    train.add_argument(
        "--dataset-repo-id", help="Dataset identity override, e.g. for corrective fine-tuning"
    )
    train.add_argument("--output", type=Path)
    train.add_argument("--device", choices=["cpu", "mps", "cuda"])
    train.add_argument("--steps", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--log-freq", type=int)
    train.add_argument(
        "--cache-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache target actions to avoid redundant future-frame image decoding",
    )
    train.add_argument(
        "--eval-every",
        type=int,
        default=2000,
        help="Save and evaluate every N updates; 0 disables evaluation",
    )
    train.add_argument("--eval-episodes", type=int, default=10)
    train.add_argument("--eval-seed", type=int, default=1000, help="First fixed validation seed")
    train.add_argument("--eval-max-steps", type=int, default=1500)
    train.add_argument("--dry-run", action="store_true")
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate a trained ACT in the flute environment"
    )
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--output", type=Path, default=Path("outputs/eval_act"))
    evaluate.add_argument("--episodes", type=int, default=10)
    evaluate.add_argument("--seed", type=int, default=1000)
    evaluate.add_argument("--max-steps", type=int, default=1500)
    evaluate.add_argument("--device", choices=["cpu", "mps", "cuda"])
    evaluate.add_argument(
        "--n-action-steps", type=int, help="Evaluation-only actions executed per prediction"
    )
    evaluate.add_argument("--video", action="store_true")
    evaluate.add_argument(
        "--video-episodes", type=int, help="Record only the first N episodes with --video"
    )
    evaluate.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    require_training_dependencies()
    if args.command == "convert":
        print(json.dumps(convert_dataset(args.source, args.output, args.repo_id), indent=2))
    elif args.command == "train":
        kwargs = vars(args).copy()
        kwargs.pop("command")
        dry_run = kwargs.pop("dry_run")
        command = training_command(**kwargs)
        print(shlex.join(command), flush=True)
        if not dry_run:
            subprocess.run(command, check=True)
    else:
        kwargs = vars(args).copy()
        kwargs.pop("command")
        evaluate_checkpoint(**kwargs)


if __name__ == "__main__":
    main()
