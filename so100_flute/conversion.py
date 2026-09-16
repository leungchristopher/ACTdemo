"""Validate HDF5 demonstrations and convert them to a local LeRobot dataset."""

import json
from pathlib import Path

import h5py
import numpy as np

from .dependencies import require_training_dependencies
from .env import CAMERAS, CONTROL_DT, JOINT_NAMES


def inspect_sources(source):
    """Fail before writing if an episode is unsuccessful or lacks aligned RGB."""
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("complete") is False:
        raise ValueError("Demonstration collection is incomplete")
    entries = manifest["episodes"]
    if not entries:
        raise ValueError("No expert episodes in manifest")
    specs = []
    common_shape = None
    for entry in entries:
        path = (source / entry["file"]).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError(f"Episode escapes dataset directory: {path}")
        with h5py.File(path, "r") as f:
            n = len(f["action"])
            if not bool(f.attrs["success"]) or n == 0:
                raise ValueError(f"Unsuccessful or empty episode: {path}")
            if not np.isclose(f.attrs["control_dt"], CONTROL_DT):
                raise ValueError(f"Expected 50 Hz actions in {path}")
            if json.loads(f.attrs["joint_names"]) != list(JOINT_NAMES):
                raise ValueError(f"Unexpected joint order in {path}")
            for key in ["action", "observations/qpos"]:
                if f[key].shape != (n, 6) or not np.isfinite(f[key][:]).all():
                    raise ValueError(f"Invalid {key} in {path}")
            np.testing.assert_allclose(f["timestamp"][:], np.arange(n) * CONTROL_DT, atol=1e-8)
            for camera in CAMERAS:
                key = f"observations/images/{camera}"
                if key not in f:
                    raise ValueError(
                        f"{path} lacks {camera} images; generate RGB demonstrations first"
                    )
                ds = f[key]
                shape = ds.shape[1:]
                if len(ds) != n or len(shape) != 3 or shape[-1] != 3 or ds.dtype != np.uint8:
                    raise ValueError(f"Invalid RGB shape/type in {path}:{key}")
                if common_shape is None:
                    common_shape = shape
                if shape != common_shape:
                    raise ValueError("All episodes/cameras must have matching image dimensions")
            specs.append(dict(path=path, frames=n, seed=int(f.attrs["seed"])))
    return manifest, specs, common_shape


def convert_dataset(
    source="data/expert_v1", output="data/lerobot_flute", repo_id="local/so100_flute"
):
    """Create a local LeRobot v3 image dataset with exactly aligned actions.

    Images stay lossless. We do not resize, crop, augment, resample or upload.
    A completion marker prevents a partial conversion from being used by train.
    """
    require_training_dependencies()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Conversion output must not exist: {output}")
    manifest, specs, shape = inspect_sources(source)
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(JOINT_NAMES)},
        "action": {"dtype": "float32", "shape": (6,), "names": list(JOINT_NAMES)},
        **{
            f"observation.images.{c}": {
                "dtype": "image",
                "shape": shape,
                "names": ["height", "width", "channels"],
            }
            for c in CAMERAS
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        fps=round(1 / CONTROL_DT),
        robot_type="so100_mujoco",
        features=features,
        use_videos=False,
        image_writer_threads=4,
        video_backend="pyav",
    )
    records = []
    try:
        for episode_index, spec in enumerate(specs):
            with h5py.File(spec["path"], "r") as f:
                for t in range(spec["frames"]):
                    dataset.add_frame(
                        {
                            "observation.state": f["observations/qpos"][t].astype(np.float32),
                            "action": f["action"][t].astype(np.float32),
                            **{
                                f"observation.images.{c}": f[f"observations/images/{c}"][t]
                                for c in CAMERAS
                            },
                            "task": str(f.attrs["task"]),
                        }
                    )
                dataset.save_episode()
            records.append(
                dict(
                    episode_index=episode_index,
                    source_file=spec["path"].name,
                    seed=spec["seed"],
                    frames=spec["frames"],
                )
            )
            print(f"Converted {episode_index + 1}/{len(specs)}: {spec['path'].name}", flush=True)
    finally:
        dataset.finalize()
    report = dict(
        complete=True,
        lerobot_version="0.6.1",
        repo_id=repo_id,
        source_scene_sha256=manifest["scene_sha256"],
        fps=round(1 / CONTROL_DT),
        image_shape=list(shape),
        cameras=list(CAMERAS),
        joint_names=list(JOINT_NAMES),
        episodes=records,
        total_frames=sum(s["frames"] for s in specs),
        action_units="radians",
        action_semantics="absolute position targets; same-frame observation precedes action",
    )
    (output / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
