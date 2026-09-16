"""Command-line entry points for scene creation, demonstration capture and replay."""

import argparse
import json
from pathlib import Path

from .dataset import generate, replay
from .env import FluteEnv
from .scene import build_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scene = commands.add_parser("scene", help="Export a portable MJCF scene")
    scene.add_argument("--output", type=Path, default=Path("scenes/flute.xml"))
    collect = commands.add_parser(
        "generate", help="Collect successful ACT-style HDF5 demonstrations"
    )
    collect.add_argument("--output", type=Path, default=Path("data/expert_v1"))
    collect.add_argument("--episodes", type=int, default=20)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--state-only", action="store_true")
    collect.add_argument("--width", type=int, default=320)
    collect.add_argument("--height", type=int, default=240)
    collect.add_argument("--max-attempts", type=int)
    playback = commands.add_parser(
        "replay", help="Re-simulate recorded actions and check the result"
    )
    playback.add_argument("episode", type=Path)
    playback.add_argument("--video", type=Path)
    playback.add_argument("--camera", choices=["overview", "wrist", "side"], default="overview")
    playback.add_argument("--viewer", action="store_true", help="Use mjpython on macOS")
    validate = commands.add_parser("validate", help="Verify every episode by replaying its actions")
    validate.add_argument("dataset", type=Path)
    commands.add_parser("view", help="Open the initialized task; use mjpython on macOS")
    args = parser.parse_args()
    if args.command == "scene":
        build_scene(args.output)
        print(args.output)
    elif args.command == "generate":
        result = generate(
            args.output,
            args.episodes,
            args.seed,
            not args.state_only,
            args.width,
            args.height,
            args.max_attempts,
        )
        print(f"Saved {len(result['episodes'])} successful episodes to {args.output}")
    elif args.command == "replay":
        if args.video:
            args.video.parent.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                replay(args.episode, args.video, args.camera, realtime=args.viewer), indent=2
            )
        )
    elif args.command == "validate":
        manifest = json.loads((args.dataset / "manifest.json").read_text())
        if not manifest["episodes"]:
            raise ValueError("Dataset has no accepted episodes")
        results = []
        for entry in manifest["episodes"]:
            result = replay(args.dataset / entry["file"])
            results.append(dict(file=entry["file"], **result))
            print(
                f"{entry['file']}: success; max state error {result['max_state_error']:.3g}",
                flush=True,
            )
        (args.dataset / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
    elif args.command == "view":
        from mujoco import viewer

        env = FluteEnv(randomize=False)
        try:
            viewer.launch(env.model, env.data)
        finally:
            env.close()


if __name__ == "__main__":
    main()
