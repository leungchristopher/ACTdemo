"""Generate the baseline demonstration with IK and timed joint waypoints."""

import numpy as np

from .kinematics import IK


def smoothstep(t):
    return t * t * t * (10 + t * (-15 + 6 * t))


def expert_actions(env, speed=1.0):
    """Yield (phase, action). Actions alone drive all robot and object motion."""
    from .env import CONTROL_DT
    from .scene import FLUTE_HEIGHT, GRASP_Z, SHELF_Z

    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be positive and finite")
    ik = IK(env.model)
    q = env.data.qpos[:6].copy()
    down, up = np.array([0, 0, -1]), np.array([0, 0, 1])
    source = np.r_[env.source, SHELF_Z + FLUTE_HEIGHT - GRASP_Z]
    target = np.r_[env.target, GRASP_Z + 0.001]

    def move(phase, position=None, axis=None, jaw=None, duration=1.0, joints=None):
        nonlocal q
        end = q.copy()
        if position is not None:
            end[:5] = ik.solve(position, axis, q)
        if joints is not None:
            end[:5] = joints
        if jaw is not None:
            end[5] = jaw
        start = q.copy()
        n = max(2, round(duration / speed / CONTROL_DT))
        for t in range(1, n + 1):
            yield phase, start + (end - start) * smoothstep(t / n)
        q = end

    yield from move("settle", duration=0.5)
    yield from move("approach", source, down, duration=2)
    yield from move("grasp", jaw=-0.174, duration=1.2)
    yield from move("lift", source + [0, 0, 0.065], down, duration=1.5)
    # Move laterally clear of the shelf before turning the wrist through pi.
    clear = np.r_[env.target, 0.30]
    yield from move("transfer", clear, down, duration=2)
    flipped = q[:5].copy()
    flipped[4] -= np.pi
    yield from move("flip", joints=flipped, duration=2.5)
    yield from move("align", clear, up, duration=0.5)
    yield from move("lower", target + [0, 0, 0.025], up, duration=2.5)
    yield from move("place", target, up, duration=1.2)
    yield from move("release", jaw=0.55, duration=1)
    # Retreat horizontally along the fingers, then raise the empty hand.
    # The SO-100 has one fixed finger: shift it away from the stem first.
    radial = env.target - np.array([0, -0.0452])
    lateral = np.array([-radial[1], radial[0]]) / np.linalg.norm(radial)
    retreat = target.copy()
    retreat[:2] += 0.015 * lateral
    yield from move("unhook", retreat, up, duration=0.8)
    retreat[:2] -= 0.045 * radial / np.linalg.norm(radial)
    yield from move("withdraw", retreat, up, duration=1.5)
    retreat[2] += 0.09
    yield from move("retreat", retreat, up, duration=1.5)
    yield from move("verify", duration=2)
