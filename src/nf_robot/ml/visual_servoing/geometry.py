#!/usr/bin/env python

"""The gripper camera's mount, in both directions.

Labelling projects a room point into the camera. The robot, running the model, has to
undo that: the heads report a point in the camera's optical frame and the gantry needs a
room-frame direction to move in. Both directions live here so the transform the robot
flies on cannot drift away from the one the labels were made with - a sign error in
either is invisible in a loss curve and obvious only when the gripper flies away from
the sock.

Nothing here imports torch or lerobot, so the observer can use it on the control path.
"""

import numpy as np
from scipy.spatial.transform import Rotation

import nf_robot.common.definitions as definitions

# The camera mount, moved from the CAD y-up gripper frame (grommet at +y, nose at -z)
# into the z-up body frame the rest of the system uses (pole up +z, nose at +y). Rx(90)
# is the same seam arp_gripper_client.measure_gantry_minus_card crosses.
_YUP_TO_ZUP = Rotation.from_euler("x", 90, degrees=True)
# The PCB normal, as CAD has it: the sensor board leans this far back from straight down.
_PCB_TILT_DEG = 90.0 - float(np.degrees(definitions.gripper_camera[0][0]))
# How far the optical axis sits forward of that normal. The lens does not project square to
# the board it is soldered to, so the board's angle is not the camera's angle, and CAD
# knows only the board.
#
# Measured from naavox/red-dot: a 15mm lid left sitting exactly between the fingertips
# while the gripper climbs straight up from 0.11m to 1.0m, so the dot marks the jaw axis at
# every range. Fitting 666 frames of it puts the optical axis 3.389 degrees back from
# straight down where the board is 9.06, a difference of 5.67 degrees, to a residual of
# 1.8px over the whole climb.
#
# The anchor camera calibration turned up a discrepancy of about 6 degrees between its
# measured and CAD tilts independently. Same part, same size, so this is one property of
# the camera module rather than two coincidences - and it is worth applying to any other
# camera whose pose comes from CAD rather than from a calibration.
LENS_VS_PCB_DEG = 5.671
# Negative is back, away from the nose. Both earlier versions of this file used the board's
# angle for the lens's: the original -9.06 leaned the right way and far too far, and a flip
# to +9.06 leaned the wrong way while landing within 2px of the truth at grasping range -
# which is why it looked right on mined frames and fell apart on a descent.
CAMERA_TILT_DEG = -(_PCB_TILT_DEG - LENS_VS_PCB_DEG)
CAMERA_ROT_BODY = Rotation.from_euler("x", 180 + CAMERA_TILT_DEG, degrees=True)

CAMERA_POS_BODY = _YUP_TO_ZUP.apply(definitions.gripper_camera[1])
# Where the jaws close, in the same body frame. The red-dot fit recovered the lens-to-jaw
# offset as -27.0mm along the nose axis without being told anything about the mount, and
# CAMERA_POS_BODY[1] is +27.0mm: the jaws sit at the gripper body origin and the lens is
# 2.7cm in front of them. The CAD translation is right, and it is the piece every part of
# this system was missing - labels hung the target below the *lens* and the servo loop
# nulled the offset from the *lens*, so both agreed to aim 2.7cm past the fingers.
JAW_POS_BODY = np.zeros(3)


def lens_to_jaw_body():
    """The body-frame vector from the lens to the jaws, which is what aiming has to add."""
    return JAW_POS_BODY - CAMERA_POS_BODY


def rotate_about_vertical(vec, radians):
    """A 3-vector turned by `radians` about the vertical axis, its z component untouched.

    The same sense as the rotate_vector the gripper client and the lerobot robot both
    carry: get_spin is a clockwise bearing, so room -> gripper is a rotation by +spin and
    gripper -> room is by -spin.
    """
    return Rotation.from_euler("z", float(radians)).apply(np.asarray(vec, dtype=np.float64))


def delta_in_camera(delta_room, spin):
    """A room-frame vector from the gripper to a point, in the camera's optical frame.

    Step one is the rotated contact vector that lerobot.label_contact_actions already
    builds: the room-frame vector from the gripper to the target, turned into the gripper
    frame by rotating its horizontal part by `spin`.

    Step two is the fixed camera mount, taken from definitions.gripper_camera rather
    than idealised: the lens sits 2.7cm toward the nose and 6mm up from the body origin,
    and looks 9.06 degrees back from straight down. Both matter at grasping range, where
    the object is only a few centimetres away and 2.7cm is a large part of the frame.

    Takes the delta rather than the two positions because an integrating labeller never
    has a position to give - only how far the gripper has moved since its anchor frame.

    Still ignored: any swing of the gripper away from vertical.
    """
    in_body = rotate_about_vertical(np.asarray(delta_room, dtype=np.float64), spin)
    return CAMERA_ROT_BODY.inv().apply(in_body - CAMERA_POS_BODY)


def point_in_camera(point_room, gripper_pos, spin):
    """A room point in the gripper camera's optical frame, assuming the gripper hangs level."""
    return delta_in_camera(
        np.asarray(point_room, dtype=np.float64) - np.asarray(gripper_pos, dtype=np.float64),
        spin)


def camera_to_body(point_cam):
    """The vector from the lens to a point it sees, in the gripper's z-up body axes."""
    return CAMERA_ROT_BODY.apply(np.asarray(point_cam, dtype=np.float64))


def camera_to_room(point_cam, spin):
    """The vector from the lens to a point it sees, in room axes.

    Its horizontal part is the centering error a servoing loop closes, and it is measured
    from the lens rather than from the gripper body origin on purpose: the labels call the
    target "straight down from the camera by the rangefinder reading", so that is where a
    centered target sits. Charging the 2.7cm nose offset to the wrong point would leave a
    fixed bias in every grasp.
    """
    return rotate_about_vertical(camera_to_body(point_cam), -float(spin))


def point_in_room(point_cam, gripper_pos, spin):
    """The full inverse of point_in_camera: a camera-frame point back in room coordinates."""
    in_body = camera_to_body(point_cam) + CAMERA_POS_BODY
    return np.asarray(gripper_pos, dtype=np.float64) + rotate_about_vertical(in_body, -float(spin))
