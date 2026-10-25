"""Symmetric quadruped joint-topology coordinates and expansion."""

from __future__ import annotations

import numpy as np


JOINT_GROUP_NAMES = (
    "front_hip_roll", "front_hip_pitch", "front_knee",
    "rear_hip_roll", "rear_hip_pitch", "rear_knee",
)
JOINT_NAMES = tuple(f"{leg}_{role}" for leg in ("fl", "fr", "rl", "rr")
                    for role in ("hip_roll", "hip_pitch", "knee"))
ACTIVE_NAMES = tuple(f"{name}_active" for name in JOINT_GROUP_NAMES)
LOCK_ANGLE_NAMES = tuple(f"{name}_lock_angle" for name in JOINT_GROUP_NAMES)


def split_design(physical):
    """Split a physical [geometry7, active6, angle6] design."""
    values = np.asarray(physical, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 19 or not np.isfinite(values).all():
        raise ValueError("Topology design must have 19 finite coordinates")
    return values[..., :7], values[..., 7:13], values[..., 13:19]


def expand_symmetric_topology(active6, lock_angles6):
    """Expand paired groups to leg-major joints, mirroring right roll angles."""
    active = np.asarray(active6, dtype=np.float64)
    angles = np.asarray(lock_angles6, dtype=np.float64)
    if active.shape != angles.shape or active.ndim < 1 or active.shape[-1] != 6:
        raise ValueError("Active masks and lock angles require matching final dimension six")
    if not np.isfinite(active).all() or not np.isfinite(angles).all() or not np.isin(active, (0., 1.)).all():
        raise ValueError("Joint masks must be binary and all coordinates finite")
    if np.any(angles * active != 0.):
        raise ValueError("Active joint lock angles must be zero")
    expanded_active = np.concatenate((active[..., :3], active[..., :3], active[..., 3:], active[..., 3:]), axis=-1)
    right_front = angles[..., :3] * np.array([-1., 1., 1.])
    right_rear = angles[..., 3:] * np.array([-1., 1., 1.])
    expanded_angles = np.concatenate((angles[..., :3], right_front, angles[..., 3:], right_rear), axis=-1)
    return expanded_active, expanded_angles
