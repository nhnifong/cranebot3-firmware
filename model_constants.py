import numpy as np
from cv_common import invert_pose, compose_poses
from scipy.spatial.transform import Rotation as R
from math import pi, sqrt

# data obtained manually from onshape
# poses are specified as tuples of (rvec, tvec) # ROTATION IS FIRST
# distances are in meters
# rotation vectors are Rodrigues

# rotation and translation vectors of the gripper camera (the frame of reference used in aruco detection) in the gripper reference frame
gripper_camera = (np.array([pi/2,0,0], dtype=float), np.array([0,0.004,-0.026], dtype=float))

# rotation and translation vectors of the gripper IMU in the gripper reference frame
hpi = sqrt(2*pi**2)/2 # half hypoteneuse of a right triangle with legs=pi
gripper_imu = (np.array([0., -hpi, -hpi], dtype=float), np.array([0.022, 0.03, 0.029], dtype=float))

# position of the gripper grommet point in the reference frame of the gripper. rotation is irrelevant
gripper_grommet = (np.array([0,0,0], dtype=float), np.array([0,0.115,0.013], dtype=float))

# position of the gripper center of gravity in the gripper reference frame. rotation is irrelevant
gripper_cog = (np.array([0,0,0], dtype=float), np.array([0,0.055,0.011], dtype=float))

# rotation and translation of the anchor camera (the frame of reference used in aruco detection) in the reference frame of the anchor
tilt = 28 # camera look tilt downward from horizontal in degrees.
anchor_camera = compose_poses([
	(np.array([0,pi,0], dtype=float), np.array([0.054, -0.038, 0.017], dtype=float)), # the camera is mounted upside down
    (np.array([(90-tilt)/180*pi,0,0], dtype=float), np.array([0,0,0], dtype=float)),
])

# position of the anchor grommet point in the reference frame of the anchor. rotation is irrelevant
anchor_grommet = (np.array([0,0,0], dtype=float), np.array([0.018,-0.033,-0.035], dtype=float))

# rotation and translation vectors of the 'gantry_front' aruco marker in the gantry reference frame.
gantry_aruco_front = (np.array([0,pi/2,0], dtype=float), np.array([0.055,0,0.105], dtype=float))

# position of the gantry keyring point in the gantry reference frame
gantry_keyring = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# precompute some inverts
gantry_aruco_front_inv = invert_pose(gantry_aruco_front)
anchor_cam_inv = invert_pose(anchor_camera)
gripper_imu_inv = invert_pose(gripper_imu)