import numpy as np
from cv_common import invert_pose

# data obtained manually from onshape
# poses are specified as tuples of (rvec, tvec)

# rotation and translation vectors of the 'gripper_front' aruco marker in the gripper reference frame.
gripper_aruco_front = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# rotation and translation vectors of the 'gripper_back' aruco marker in the gripper reference frame.
gripper_aruco_back = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# rotation and translation vectors of the gripper camera in the gripper reference frame
gripper_camera = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# rotation and translation vectors of the gripper IMU in the gripper reference frame
gripper_imu = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# position of the gripper grommet point in the reference frame of the gripper. rotation is irrelevant
gripper_grommet = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# position of the gripper center of gravity in the gripper reference frame. rotation is irrelevant
gripper_cog = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# rotation and translation of the anchor camera in the reference frame of the anchor
anchor_camera = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# position of the anchor grommet point in the reference frame of the anchor. rotation is irrelevant
anchor_grommet = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# rotation and translation vectors of the 'gantry_front' aruco marker in the gantry reference frame.
gantry_aruco_front = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# rotation and translation vectors of the 'gantry_back' aruco marker in the gantry reference frame.
gantry_aruco_back = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

# position of the gantry keyring point in the gantry reference frame
gantry_keyring = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))



# precompute some inverts
gripper_aruco_front_inv = invert_pose(gripper_aruco_front)
gripper_aruco_back_inv = invert_pose(gripper_aruco_back)
gantry_aruco_front_inv = invert_pose(gantry_aruco_front)
gantry_aruco_back_inv = invert_pose(gantry_aruco_back)
anchor_cam_inv = invert_pose(anchor_camera)