import numpy as np
from cv_common import invert_pose, compose_poses
from scipy.spatial.transform import Rotation as R
from math import pi

# data obtained manually from onshape
# poses are specified as tuples of (rvec, tvec) # ROTATION IS FIRST
# distances are in meters
# rotation vectors are Rodrigues

# rotation and translation vectors of the gripper aruco markers in the gripper reference frame.
# the front of the device is the side with the camera mount.
# if you are looking at the front face, the right face is on your right.
gripper_aruco_front = (np.array([0,pi,0], dtype=float), np.array([0,0.061,-0.041], dtype=float))
gripper_aruco_back = (np.array([0,0,0], dtype=float), np.array([0,0.061,0.071], dtype=float))
gripper_aruco_right = (np.array([0,-0.5*pi,0], dtype=float), np.array([-0.051,0.066,0.013], dtype=float))
gripper_aruco_left = (np.array([0,0.5*pi,0], dtype=float), np.array([0.051,0.066,0.013], dtype=float))

# rotation and translation vectors of the gripper camera (the frame of reference used in aruco detection) in the gripper reference frame
gripper_camera = (np.array([0,0,0], dtype=float), np.array([0,0.004,0.026], dtype=float))

# rotation and translation vectors of the gripper IMU in the gripper reference frame
gripper_imu = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# position of the gripper grommet point in the reference frame of the gripper. rotation is irrelevant
gripper_grommet = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# position of the gripper center of gravity in the gripper reference frame. rotation is irrelevant
gripper_cog = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# rotation and translation of the anchor camera (the frame of reference used in aruco detection) in the reference frame of the anchor
anchor_camera = compose_poses([
	(np.array([0,0,0], dtype=float), np.array([0.054, -0.038, 0.010], dtype=float)),
    (np.array([0,1,0], dtype=float)*pi, np.array([0,0,0], dtype=float)), # the camera is mounted upside down
    (np.array([1,0,0], dtype=float)*(55/180*pi), np.array([0,0,0], dtype=float)), # and tilted 125 deg away from the anchor model's z axis
])

# position of the anchor grommet point in the reference frame of the anchor. rotation is irrelevant
anchor_grommet = (np.array([0,0,0], dtype=float), np.array([0.013,-0.025,-0.043], dtype=float))

# rotation and translation vectors of the 'gantry_front' aruco marker in the gantry reference frame.
gantry_aruco_front = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# rotation and translation vectors of the 'gantry_back' aruco marker in the gantry reference frame.
gantry_aruco_back = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# position of the gantry keyring point in the gantry reference frame
gantry_keyring = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))



# precompute some inverts
gantry_aruco_front_inv = invert_pose(gantry_aruco_front)
gantry_aruco_back_inv = invert_pose(gantry_aruco_back)
anchor_cam_inv = invert_pose(anchor_camera)