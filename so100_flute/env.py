"""SO-100 task dynamics, observations, and success criteria.

Units are metres, seconds, and radians. Each control step advances 20 ms.
MuJoCo qpos layout: six robot joints, flute xyz, flute quaternion (wxyz).
MuJoCo qvel layout: six robot velocities, flute linear and angular velocities.
Object state is available for recording and scoring, but ACT uses only RGB and
robot joint positions; see evaluation.py for its observation construction.
"""

import mujoco
import numpy as np

from .scene import FLUTE_HEIGHT, GRASP_Z, SHELF_Z, SOURCE, TARGET, build_scene

JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
CAMERAS = ("overview", "wrist")
CONTROL_DT = 0.02


class FluteEnv:
    """Position-controlled arm with contact-based grasping and deterministic resets."""

    def __init__(self, seed=0, randomize=True):
        self.model = mujoco.MjModel.from_xml_string(build_scene())
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.reset(seed, randomize)

    def reset(self, seed=0, randomize=True):
        from .kinematics import IK

        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.seed = seed
        self.source = SOURCE + (rng.uniform(-0.004, 0.004, 2) if randomize else 0)
        self.target = TARGET + (rng.uniform(-0.006, 0.006, 2) if randomize else 0)
        self.model.site("target").pos[:2] = self.target
        self.data.qpos[6:9] = [*self.source, SHELF_Z + FLUTE_HEIGHT + 0.0002]
        self.data.qpos[9:13] = [0, 1, 0, 0]
        start = np.array([self.source[0], self.source[1] + 0.05, SHELF_Z + FLUTE_HEIGHT - GRASP_Z])
        q = IK(self.model).solve(start, [0, 0, -1])
        self.data.qpos[:6] = np.r_[q, 0.55]
        self.data.ctrl[:] = self.data.qpos[:6]
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.lifted = False
        self.flipped = False
        self.stable_steps = 0
        return self.observe()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("action must be six finite joint position targets in radians")
        self.data.ctrl[:] = np.clip(
            action, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        )
        for _ in range(round(CONTROL_DT / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        pos = self.data.body("flute").xpos
        axis = self.data.body("flute").xmat.reshape(3, 3)[:, 2]
        if axis[2] < -0.95 and pos[2] - FLUTE_HEIGHT > SHELF_Z + 0.025:
            self.lifted = True
        if self.lifted and axis[2] > 0.98 and pos[2] > 0.10:
            self.flipped = True
        self.stable_steps = self.stable_steps + 1 if self.metrics()["placement_valid"] else 0
        return self.observe()

    def observe(self):
        """Return recording fields; this includes privileged simulator state."""
        return {
            "qpos": self.data.qpos[:6].copy(),
            "qvel": self.data.qvel[:6].copy(),
            "object_pose": self.data.qpos[6:13].copy(),
            "object_velocity": self.data.qvel[6:12].copy(),
            "grasp_position": self.data.site("grasp").xpos.copy(),
        }

    def metrics(self):
        """Score lift, flip, and stable upright placement after gripper withdrawal."""
        pos = self.data.body("flute").xpos
        upright = float(self.data.body("flute").xmat.reshape(3, 3)[2, 2])
        xy_error = float(np.linalg.norm(pos[:2] - self.target))
        speed = float(np.linalg.norm(self.data.qvel[6:9]))
        angular_speed = float(np.linalg.norm(self.data.qvel[9:12]))
        table_contact = False
        robot_contact = False
        for c in self.data.contact:
            names = [self.model.geom(int(g)).name for g in (c.geom1, c.geom2)]
            if any(n.startswith("flute_") for n in names):
                table_contact |= "table" in names
                robot_contact |= any(
                    self.model.geom_bodyid[g] in range(1, 8) for g in (c.geom1, c.geom2)
                )
        placement_valid = (
            xy_error < 0.025
            and upright > np.cos(np.deg2rad(8))
            and abs(pos[2]) < 0.006
            and speed < 0.01
            and angular_speed < 0.1
            and table_contact
            and not robot_contact
            and self.data.qpos[5] > 0.3
        )
        success = (
            placement_valid
            and self.lifted
            and self.flipped
            and self.stable_steps >= round(1.0 / CONTROL_DT)
        )
        return dict(
            success=bool(success),
            placement_valid=bool(placement_valid),
            lifted=self.lifted,
            flipped=self.flipped,
            stable_seconds=self.stable_steps * CONTROL_DT,
            xy_error=xy_error,
            upright_cosine=upright,
            base_height=float(pos[2]),
            linear_speed=speed,
            angular_speed=angular_speed,
            table_contact=bool(table_contact),
            robot_contact=bool(robot_contact),
        )

    def render(self, camera="overview", width=640, height=480):
        if self.renderer is None or self.renderer.width != width or self.renderer.height != height:
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
