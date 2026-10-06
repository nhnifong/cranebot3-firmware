"""What "centred over the basket" means, and the move back to it, in the gripper body frame.

The body frame is the room's axes turned with the gripper (rotate_about_vertical by +spin),
with the origin at the gripper and z up. A point the camera sees becomes a body vector from
the gripper to it; the basket is centred when that vector is the one straight down from
the jaws, at the height the operator chose to hang over it.
"""

import numpy as np

from nf_robot.ml.visual_servoing.geometry import (
    CAMERA_POS_BODY, JAW_POS_BODY, camera_to_body, rotate_about_vertical,
)


def target_body(point_cam):
    """A camera-frame point as the body vector from the gripper to it. Takes (3,) or (N, 3)."""
    return camera_to_body(point_cam) + CAMERA_POS_BODY


def centered_body(drop_range_m):
    """The body vector to the basket when centred over it: straight below the jaws, with the
    rangefinder reading drop_range_m. Takes a scalar or (N,)."""
    drop = np.asarray(drop_range_m, dtype=np.float64) - CAMERA_POS_BODY[2] + JAW_POS_BODY[2]
    out = np.zeros(drop.shape + (3,))
    out[..., :2] = JAW_POS_BODY[:2]
    out[..., 2] = JAW_POS_BODY[2] - drop
    return out


def return_body(point_cam, drop_range_m, spin=0.0, centered_spin=None):
    """The gripper move, in the body frame at `spin`, that puts the basket at point_cam back
    where it was when centred. centered_spin is the wrist then, if it has turned since; the
    move is the same room-frame translation either way."""
    if centered_spin is None or centered_spin == spin:
        return target_body(point_cam) - centered_body(drop_range_m)
    now = rotate_about_vertical(target_body(point_cam), -float(spin))
    then = rotate_about_vertical(centered_body(drop_range_m), -float(centered_spin))
    return rotate_about_vertical(now - then, float(spin))


def lateral_return_body(point_cam):
    """The sideways part of return_body, which needs no drop height: centred means straight
    below the jaws whatever the height. Takes (3,) or (N, 3), gives (2,) or (N, 2)."""
    return target_body(point_cam)[..., :2] - JAW_POS_BODY[:2]


def body_to_room(vec_body, spin):
    return rotate_about_vertical(vec_body, -float(spin))
