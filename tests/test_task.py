import h5py
import mujoco
import numpy as np
import pytest

from so100_flute.dataset import PHASES, record_episode, replay
from so100_flute.env import FluteEnv
from so100_flute.expert import expert_actions
from so100_flute.scene import FLUTE_HEIGHT, SHELF_Z, build_scene


def test_scene_export_and_initial_support(tmp_path):
    path = tmp_path / "task.xml"
    build_scene(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nu == 6 and model.nq == 13
    assert model.neq == 0  # Grasp is contact-based, with no weld constraints.
    env = FluteEnv(randomize=False)
    for _ in range(100):
        env.step(env.data.ctrl.copy())
    assert env.data.body("flute").xmat.reshape(3, 3)[2, 2] < -0.999
    assert abs(env.data.qpos[8] - (SHELF_Z + FLUTE_HEIGHT)) < 0.001
    assert not env.metrics()["success"]
    env.close()


@pytest.mark.parametrize("seed", [0, 3, 8, 19, 35, 38, 43, 49])
def test_expert_completes_and_remains_stable(seed):
    env = FluteEnv(seed=seed)
    initial_xy = env.source.copy()
    speed = float(np.random.default_rng(seed + 10000).uniform(0.9, 1.1))
    flipped_in_air = False
    for phase, action in expert_actions(env, speed):
        assert np.all(action >= env.model.actuator_ctrlrange[:, 0] - 1e-6)
        assert np.all(action <= env.model.actuator_ctrlrange[:, 1] + 1e-6)
        env.step(action)
        if phase == "align":
            flipped_in_air |= (
                env.data.body("flute").xmat.reshape(3, 3)[2, 2] > 0.98 and env.data.qpos[8] > 0.1
            )
    assert flipped_in_air
    assert np.linalg.norm(env.data.qpos[6:8] - initial_xy) > 0.20
    assert env.metrics()["success"], env.metrics()
    for _ in range(100):
        env.step(env.data.ctrl.copy())
        assert env.metrics()["success"], env.metrics()
    env.close()


def test_recording_alignment_and_action_replay(tmp_path):
    path = tmp_path / "episode.hdf5"
    result = record_episode(path, seed=2, images=False, speed=1.05)
    assert result["metrics"]["success"]
    with h5py.File(path, "r") as f:
        n = len(f["action"])
        assert f["observations/qpos"].shape == (n, 6)
        assert f["sim/qpos"].shape == (n + 1, 13)
        np.testing.assert_array_equal(f["observations/qpos"], f["sim/qpos"][:-1, :6])
        assert set(f["phase"][:]) == set(range(len(PHASES)))
        assert f["success"][-1]
    verified = replay(path)
    assert verified["success"]
    assert verified["max_state_error"] < 1e-7
    with pytest.raises(FileExistsError):
        record_episode(path, images=False)


def test_action_validation():
    env = FluteEnv()
    with pytest.raises(ValueError):
        env.step([0] * 5)
    with pytest.raises(ValueError):
        env.step([np.nan] * 6)
    env.close()


@pytest.mark.parametrize("seed,near_stem", [(200, False), (201, True), (202, True)])
def test_corrective_demonstrations_complete_and_replay(seed, near_stem, tmp_path):
    from so100_flute.corrections import corrective_expert_actions, initialize_near_stem

    path = tmp_path / "correction.hdf5"
    result = record_episode(
        path,
        seed=seed,
        images=False,
        controller=corrective_expert_actions,
        initialize=initialize_near_stem if near_stem else None,
        metadata={"kind": "correction"},
    )
    assert result["metrics"]["success"]
    with h5py.File(path, "r") as f:
        assert f["phase"][0] == PHASES.index("approach")
        assert f["observations/qpos"].shape == f["action"].shape
        assert "collection_metadata" in f.attrs
    replayed = replay(path)
    assert replayed["success"]
    assert replayed["max_state_error"] == 0
