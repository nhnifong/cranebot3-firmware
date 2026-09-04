import numpy as np
from math import pi, sqrt
from typing import NamedTuple

from nf_robot.generated.nf import common

# data obtained manually from onshape
# poses are specified as tuples of (rvec, tvec) # ROTATION IS FIRST
# distances are in meters
# rotation vectors are Rodrigues

# rotation and translation vectors of the gripper camera (the frame of reference used in marker detection) in the gripper reference frame
# The camera is not pointed straight down: it sits ~2.7cm toward the front (-z, the nose side) of the
# CAD origin (~4cm from the hang centerline at the grommet) and is tilted 9.06 deg back toward center.
# That tilt is a rotation about the gripper x axis, the same axis as the pi/2 that maps the optical
# axis to straight-down, so it adds to that term. Tilting the optical axis from -y toward +z (center)
# is a negative x rotation, hence pi/2 minus the tilt.
gripper_camera = (np.array([pi/2 - np.radians(9.06),0,0], dtype=float), np.array([0,0.006,-0.027], dtype=float))

# rotation and translation vectors of the gripper IMU in the gripper reference frame
# the BNO085 is mounted in the gripper with it's y axis up, x axis out of the grippers right ear, and X axis pointing out of the back of it's head.
# to translate it into the reference frame of the gripper
hpi = sqrt(2*pi**2)/2 # half hypoteneuse of a right triangle with legs=pi
gripper_imu = (np.array([0., -hpi, -hpi], dtype=float), np.array([0.022, 0.03, 0.029], dtype=float))
# gripper_imu = (np.array([pi/2, 0, 0], dtype=float), np.array([0, 0, 0], dtype=float))

# position of the gripper grommet point in the reference frame of the gripper. rotation is irrelevant
gripper_grommet = (np.array([0,0,0], dtype=float), np.array([0,0.115,0.013], dtype=float))

# position of the gripper center of gravity in the gripper reference frame. rotation is irrelevant
gripper_cog = (np.array([0,0,0], dtype=float), np.array([0,0.055,0.011], dtype=float))

# Effective length of pole based on measured swing with 'polecal'
pole_length_abs500 = 0.4526
pole_length_carbon400 = 0.4389
pole_length_carbon270 = pole_length_carbon400 - 0.13

# distance between gantry origin (where lines meet) and gripper origin
pole_offset_abs500 = 0.5334
pole_offset_carbon400 = 0.5757
pole_offset_carbon270 = pole_offset_carbon400 - 0.13

# z offset of the gripper laser rangefinder from the origin of the gantry when the winch is zeroed.
laser_offset = 0.14 # meters

# position of the anchor grommet point in the reference frame of the anchor. rotation is irrelevant
anchor_grommet = (np.array([0,0,0], dtype=float), np.array([0.018,-0.033,-0.035], dtype=float))

# position in the anchor model where the two walls and top surface meet. rotation is irrelevant
anchor_wall_corner = (np.array([0,0,0], dtype=float), np.array([0.005978, 0.089425, 0.042], dtype=float))

# rotation and translation vectors of the 'gantry' april tag in the gantry reference frame.
# the point where the lines meet is the gantry's origin.
# gantry_april = (np.array([0,pi/2,0], dtype=float), np.array([0.055,0,0.105], dtype=float))
gantry_april = (np.array([pi/2,0,0], dtype=float), np.array([0, -0.065, -0.055], dtype=float))
# flat marker
gantry_flat_april = (np.array([pi/2,0,0], dtype=float), np.array([0, -0.007, -0.0652], dtype=float))

# position in front of a basket marker where objects should be dropped
basket_offset = (np.array([0,0,0], dtype=float), np.array([0,0,0.20], dtype=float))
basket_offset_inv = (np.array([0,0,0], dtype=float), np.array([0,0,-0.202], dtype=float))

assumed_full_line_length = 7.5 # meters

# damiao spool
damiao_empty_spool_diameter = 72.0
damiao_full_spool_diameter_fishing_line = 73.1
damiao_full_spool_diameter_power_line = 86.1

# The state each anchor spool is built to, as (line on a full spool in m, the diameter the spool
# reaches with that much on it in mm). SpiralCalculator ramps the diameter from
# damiao_empty_spool_diameter up to full_diameter across full_length, so both numbers have to
# describe the same spool: get full_length wrong and the reported diameter is right for the wrong
# point on the spiral, which biases length-per-revolution at every angle.
#
# anchor_arp_eval.py winds 15 m low / 7.5 m high normally, 20 m / 12 m with --long. The low
# (indirect) line takes more because it routes out around the eyelet. A power line, when fitted,
# is always on the high spool; it is much thicker than fishing line, so its pile grows far faster
# and it is the entry that really moves between the two windings.
#
#   (winding, spool, line type) -> (full_length_m, full_diameter_mm)
damiao_spool_geometry = {
    # Short winding keeps the values every already-calibrated robot was built against. The low
    # spool carries 15 m but has always been modelled at 7.5; correcting that would shift the
    # indirect line on every short-wound robot, so it is left alone here.
    ('short', 'high', 'fishing'): (assumed_full_line_length, damiao_full_spool_diameter_fishing_line),
    ('short', 'high', 'power'):   (assumed_full_line_length, damiao_full_spool_diameter_power_line),
    ('short', 'low',  'fishing'): (assumed_full_line_length, damiao_full_spool_diameter_fishing_line),
    # Long winding, measured: 20 m of fishing line piles to 75 mm, 12 m of power line to 100 mm.
    # The 12 m fishing case was not measured directly, but the two fishing measurements agree on
    # 0.15 mm of diameter per meter wound (1.1/7.5 and 3.0/20), which puts 12 m at 73.8 mm.
    ('long',  'high', 'fishing'): (12.0, 73.8),
    ('long',  'high', 'power'):   (12.0, 100.0),
    ('long',  'low',  'fishing'): (20.0, 75.0),
}

# arp anchor model
# your "right" when looking at it's face. right is direct line
arp_anchor_right_eyelet = (np.array([0,0,0], dtype=float), np.array([0.031,-0.068,-0.022], dtype=float))
arp_anchor_left_eyelet = (np.array([0,0,0], dtype=float), np.array([-0.031,-0.068,-0.022], dtype=float))
# compose_poses([(np.array([0,pi,0], dtype=float), np.array([0.007, -0.033, 0.071], dtype=float)), (np.array([(90-22)/180*pi,0,0], dtype=float), np.array([0,0,0], dtype=float))])
arp_anchor_camera = (np.array([0.0, 2.60449835, -1.75675632]), np.array([ 0.007, -0.033,  0.071]))

rpi_cam_3_wide_fov = np.array([102, 67])


class PoleGeometry(NamedTuple):
    """Everything about the robot that the gripper's pole changes.

    The gantry origin is where the support lines meet, so swapping the pole moves the
    gripper, the marker, or both relative to that point. Which pole a robot has is in
    config.gripper.pole_type; look the geometry up with pole_geometry().
    """

    # (m) effective pendulum length, gantry pivot to gripper center of mass. Sets the
    # swing frequency the gripper's onboard fit and the host's cancellation both run at,
    # so the wrong one lands the correction at the wrong phase. See host/swing.py.
    swing_length: float
    # (m) straight down from the gantry model origin to the gripper model origin
    gantry_to_gripper: float
    # where the gantry marker sits in the gantry frame, since the two poles carry
    # different markers mounted at different points
    gantry_april: tuple


POLE_GEOMETRY = {
    # A config with no pole recorded predates the field, so it belongs to a robot built
    # with the ABS pole; config_loader backfills it to ABS500 on load.
    common.PoleType.UNSPECIFIED: PoleGeometry(pole_length_abs500, pole_offset_abs500, gantry_april),
    common.PoleType.ABS500: PoleGeometry(pole_length_abs500, pole_offset_abs500, gantry_april),
    common.PoleType.CARBON400: PoleGeometry(pole_length_carbon400, pole_offset_carbon400, gantry_flat_april),
    common.PoleType.CARBON270: PoleGeometry(pole_length_carbon270, pole_offset_carbon270, gantry_flat_april),
}


def pole_geometry(config):
    """The PoleGeometry this robot's configured pole gives it."""
    return POLE_GEOMETRY[config.gripper.pole_type]