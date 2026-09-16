"""Solve five arm joints for a grasp position and axis; jaw control is separate."""

import mujoco
import numpy as np
from scipy.optimize import least_squares


class IK:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site("grasp").id
        self.bounds = (model.jnt_range[:5, 0] + 1e-5, model.jnt_range[:5, 1] - 1e-5)

    def solve(self, position, axis, seed=None):
        position, axis = np.asarray(position), np.asarray(axis)
        if seed is None:
            seed = np.array([0.0, -2.0, 1.5, 0.5, 1.57])
        seed = np.clip(np.asarray(seed)[:5], *self.bounds)

        def residual(q):
            self.data.qpos[:5] = q
            mujoco.mj_kinematics(self.model, self.data)
            return np.r_[
                self.data.site_xpos[self.site] - position,
                0.1 * (self.data.site_xmat[self.site].reshape(3, 3)[:, 2] - axis),
            ]

        result = least_squares(
            residual, seed, bounds=self.bounds, ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=150
        )
        error = residual(result.x)
        if np.linalg.norm(error[:3]) > 0.002 or np.linalg.norm(error[3:]) > 0.003:
            raise RuntimeError(f"Unreachable waypoint: {position}, axis={axis}; residual={error}")
        return result.x
