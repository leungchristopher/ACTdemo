"""Collect successful corrective demonstrations on separate training seeds."""

import argparse
import json
import os
import shutil
from pathlib import Path

import mujoco
import numpy as np

from .dataset import record_episode, scene_hash
from .expert import expert_actions
from .kinematics import IK
from .scene import FLUTE_HEIGHT, GRASP_Z, SHELF_Z, build_scene


def corrective_expert_actions(env, speed=1.0):
    """Demonstrate approach correction and prompt closing, then finish the task.

    The expert is used only for collecting labels. ACT receives no privileged
    source/target coordinates or scripted assistance during policy evaluation.
    """
    from .env import CONTROL_DT
    from .scene import FLUTE_HEIGHT, GRASP_Z, SHELF_Z

    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be positive and finite")
    target = np.r_[env.source, SHELF_Z + FLUTE_HEIGHT - GRASP_Z]
    goal = IK(env.model).solve(target, [0, 0, -1], env.data.qpos[:5])
    # Feedback targets maintain progress instead of easing toward a timed stop.
    for _ in range(round(6 / speed / CONTROL_DT)):
        error = goal - env.data.qpos[:5]
        yield (
            "approach",
            np.r_[
                env.data.qpos[:5] + np.clip(error, -speed * CONTROL_DT, speed * CONTROL_DT), 0.55
            ],
        )
        if np.max(np.abs(goal - env.data.qpos[:5])) < 0.002:
            break
    else:
        raise RuntimeError("Corrective approach did not converge")
    start = float(env.data.qpos[5])
    close_steps = max(2, round(0.6 / speed / CONTROL_DT))
    for i in range(1, close_steps + 1):
        yield "grasp", np.r_[goal, start + (-0.174 - start) * i / close_steps]
    for _ in range(max(2, round(0.3 / speed / CONTROL_DT))):
        yield "grasp", np.r_[goal, -0.174]
    # Reuse the baseline lift/flip/place sequence after establishing the grasp.
    for phase, action in expert_actions(env, speed):
        if phase not in ("settle", "approach", "grasp"):
            yield phase, action


def initialize_near_stem(env):
    """Start in the measured failure region, without using validation rollouts."""
    rng = np.random.default_rng(env.seed + 30000)
    offset = np.array(
        [rng.uniform(-0.012, 0.012), rng.uniform(0.008, 0.025), rng.uniform(-0.004, 0.004)]
    )
    target = np.r_[env.source, SHELF_Z + FLUTE_HEIGHT - GRASP_Z] + offset
    q = IK(env.model).solve(target, [0, 0, -1], env.data.qpos[:5])
    q[4] += rng.uniform(-0.06, 0.06)
    env.data.qpos[:6] = np.r_[
        np.clip(q, env.model.jnt_range[:5, 0], env.model.jnt_range[:5, 1]), rng.uniform(0.55, 0.58)
    ]
    env.data.qvel[:] = 0
    env.data.ctrl[:] = env.data.qpos[:6]
    mujoco.mj_forward(env.model, env.data)


def collect_corrections(output, source="data/expert_v1", episodes=30, seed=200, max_attempts=None):
    output, source = Path(output), Path(source)
    if episodes < 1 or seed < 0:
        raise ValueError("episodes must be positive and seed nonnegative")
    max_attempts = episodes * 3 if max_attempts is None else max_attempts
    if max_attempts < episodes:
        raise ValueError("max_attempts must be at least episodes")
    if set(range(seed, seed + max_attempts)) & (set(range(1000, 1010)) | set(range(2000, 2100))):
        raise ValueError("Correction seeds overlap reserved validation/final-test seeds")
    if output.exists():
        raise FileExistsError(output)
    original = json.loads((source / "manifest.json").read_text())
    if original["scene_sha256"] != scene_hash():
        raise ValueError("Source demonstrations use a different scene")
    shape = original["image_shape"]
    output.mkdir(parents=True)
    manifest = {
        **original,
        "episodes": [],
        "attempts": [],
        "complete": False,
        "correction_recipe": dict(
            controller="corrective_expert_actions",
            normal_start_fraction="one third",
            near_stem_fraction="two thirds",
            requested_new_episodes=episodes,
            seed=seed,
            source=str(source.resolve()),
        ),
    }
    build_scene(output / "scene.xml")

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Immutable HDF5 files are linked to avoid duplicating the original 1.3 GB.
    for entry in original["episodes"]:
        origin = (source / entry["file"]).resolve()
        if not origin.is_relative_to(source.resolve()):
            raise ValueError("Source episode path escapes dataset directory")
        filename = f"episode_{len(manifest['episodes']):06d}.hdf5"
        try:
            os.link(origin, output / filename)
        except OSError:
            shutil.copy2(origin, output / filename)
        manifest["episodes"].append({**entry, "file": filename, "kind": "original"})
    save()
    accepted = 0
    for attempt in range(max_attempts):
        current_seed = seed + attempt
        near_stem = accepted % 3 != 0
        mode = "near_stem" if near_stem else "normal_start"
        filename = f"episode_{len(manifest['episodes']):06d}.hdf5"
        result = record_episode(
            output / filename,
            seed=current_seed,
            images=True,
            width=shape[1],
            height=shape[0],
            speed=1.0,
            controller=corrective_expert_actions,
            initialize=initialize_near_stem if near_stem else None,
            metadata=dict(kind="correction", start=mode, teacher="corrective_expert_actions"),
        )
        result["kind"] = "correction"
        result["start"] = mode
        manifest["attempts"].append(result)
        if result["metrics"]["success"]:
            manifest["episodes"].append(result)
            accepted += 1
        manifest["complete"] = accepted == episodes
        save()
        print(
            f"Correction {accepted}/{episodes}: seed={current_seed}, {mode}, success={result['metrics']['success']}",
            flush=True,
        )
        if accepted == episodes:
            return manifest
    raise RuntimeError(
        f"Only {accepted}/{episodes} corrections succeeded; see {output}/manifest.json"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/expert_guided_v2"))
    parser.add_argument("--source", type=Path, default=Path("data/expert_v1"))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=200)
    args = parser.parse_args()
    collect_corrections(**vars(args))


if __name__ == "__main__":
    main()
