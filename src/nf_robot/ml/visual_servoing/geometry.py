#!/usr/bin/env python

"""The gripper camera's mount: room to camera for labelling, and back for the robot."""

import numpy as np
from scipy.spatial.transform import Rotation

import nf_robot.common.definitions as definitions

# Camera mount moved from the CAD y-up gripper frame into the z-up body frame (Rx(90)).
_YUP_TO_ZUP = Rotation.from_euler("x", 90, degrees=True)
# The PCB normal, as CAD has it: the sensor board leans this far back from straight down.
_PCB_TILT_DEG = 90.0 - float(np.degrees(definitions.gripper_camera[0][0]))
# The optical axis sits 5.67 degrees forward of the PCB normal, measured from naavox/red-
# dot.
LENS_VS_PCB_DEG = 5.671
# Negative is back, away from the nose.
CAMERA_TILT_DEG = -(_PCB_TILT_DEG - LENS_VS_PCB_DEG)
CAMERA_ROT_BODY = Rotation.from_euler("x", 180 + CAMERA_TILT_DEG, degrees=True)

CAMERA_POS_BODY = _YUP_TO_ZUP.apply(definitions.gripper_camera[1])
# Where the jaws close in the body frame: 2.7cm behind the lens, confirmed by the red-dot
# fit.
JAW_POS_BODY = np.zeros(3)


def lens_to_jaw_body():
    return JAW_POS_BODY - CAMERA_POS_BODY


def rotate_about_vertical(vec, radians):
    """A 3-vector turned by `radians` about the vertical axis; room -> gripper is +spin."""
    return Rotation.from_euler("z", float(radians)).apply(np.asarray(vec, dtype=np.float64))


def delta_in_camera(delta_room, spin):
    """A room-frame delta from the gripper to a point, in the camera's optical frame,
    assuming the gripper hangs level."""
    in_body = rotate_about_vertical(np.asarray(delta_room, dtype=np.float64), spin)
    return CAMERA_ROT_BODY.inv().apply(in_body - CAMERA_POS_BODY)


def point_in_camera(point_room, gripper_pos, spin):
    """A room point in the gripper camera's optical frame, assuming the gripper hangs level."""
    return delta_in_camera(
        np.asarray(point_room, dtype=np.float64) - np.asarray(gripper_pos, dtype=np.float64),
        spin)


def camera_to_body(point_cam):
    return CAMERA_ROT_BODY.apply(np.asarray(point_cam, dtype=np.float64))


def camera_to_room(point_cam, spin):
    return rotate_about_vertical(camera_to_body(point_cam), -float(spin))


def point_in_room(point_cam, gripper_pos, spin):
    """The full inverse of point_in_camera: a camera-frame point back in room coordinates."""
    in_body = camera_to_body(point_cam) + CAMERA_POS_BODY
    return np.asarray(gripper_pos, dtype=np.float64) + rotate_about_vertical(in_body, -float(spin))
