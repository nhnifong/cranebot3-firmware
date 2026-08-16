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
# Comes out as Rx(180 - 9.06 deg): looking down, tilted back away from the nose.
CAMERA_ROT_BODY = _YUP_TO_ZUP * Rotation.from_rotvec(definitions.gripper_camera[0])
# Comes out as (0, +0.027, +0.006): 2.7cm toward the nose, 6mm up from the body origin.
CAMERA_POS_BODY = _YUP_TO_ZUP.apply(definitions.gripper_camera[1])


def rotate_about_vertical(vec, radians):
    """A 3-vector turned by `radians` about the vertical axis, its z component untouched.

    The same sense as the rotate_vector the gripper client and the lerobot robot both
    carry: get_spin is a clockwise bearing, so room -> gripper is a rotation by +spin and
    gripper -> room is by -spin.
    """
    return Rotation.from_euler("z", float(radians)).apply(np.asarray(vec, dtype=np.float64))


def point_in_camera(point_room, gripper_pos, spin):
    """A room point in the gripper camera's optical frame, assuming the gripper hangs level.

    Step one is the rotated contact vector that lerobot_label_contact_actions already
    builds: the room-frame vector from the gripper to the target, turned into the gripper
    frame by rotating its horizontal part by `spin`.

    Step two is the fixed camera mount, taken from definitions.gripper_camera rather
    than idealised: the lens sits 2.7cm toward the nose and 6mm up from the body origin,
    and looks 9.06 degrees back from straight down. Both matter at grasping range, where
    the object is only a few centimetres away and 2.7cm is a large part of the frame.

    Still ignored: any swing of the gripper away from vertical.
    """
    delta = np.asarray(point_room, dtype=np.float64) - np.asarray(gripper_pos, dtype=np.float64)
    in_body = rotate_about_vertical(delta, spin)
    return CAMERA_ROT_BODY.inv().apply(in_body - CAMERA_POS_BODY)


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
