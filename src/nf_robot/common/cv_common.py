import os
import cv2
from pupil_apriltags import Detector
import numpy as np
import time
from nf_robot.common.config_loader import *
import nf_robot.common.definitions as model_constants
from scipy.spatial.transform import Rotation
from nf_robot.generated.nf import config as nf_config
import functools

# The marker IDs will correspond to the index in this list.
MARKER_NAMES = [
    'origin',
    'gantry',
    'gamepad',
    'hamper',
    'trash',
    'cal_assist_1',
    'cal_assist_2',
    'cal_assist_3',
    'gamepad_back',
    'hamper_back',
    'trash_back',
    'toys',
    'toys_back',
] # next tag id 13

# AprilTag images are typically downloaded, not generated in code.
# We are using the tag36h11 tag family.
# The images for new tags can be downloaded at
# https://github.com/AprilRobotics/apriltag-imgs/tree/master/tag36h11

DEFAULT_MARKER_SIZE = 0.0945 # The default side length of markers in meters
# Define the physical size of any markers that are not the default size.
SPECIAL_SIZES = {
    'origin': 0.1680, # size in meters
    'cal_assist_1': 0.1640, # shouldn't these have printed with the same dimensions as the origin card?
    'cal_assist_2': 0.1640,
    'cal_assist_3': 0.1640,
    'gantry':       0.0915
}

# These are the 3D corner points of a generic marker of size 1x1 meter.
# We will scale this based on the actual marker size.
BASE_MARKER_POINTS = np.array([
    [-0.5, -0.5, 0],
    [ 0.5, -0.5, 0],
    [ 0.5,  0.5, 0],
    [-0.5,  0.5, 0]
], dtype=np.float32)

# Pre-calculate marker points for known sizes.
DEFAULT_OBJ_POINTS = BASE_MARKER_POINTS * DEFAULT_MARKER_SIZE
SPECIAL_OBJ_POINTS = {
    name: BASE_MARKER_POINTS * size 
    for name, size in SPECIAL_SIZES.items()
}

SF_INPUT_SHAPE = (960, 540)      # Size of the raw frame coming from gripper camera
SF_TARGET_SHAPE = (384, 384)     # Size of the final neural net input (Square)
SF_SCALE_FACTOR = 1.4  # Zoom factor (values less than 1 zoom in)

saved_matrices = {}

def gripper_stabilized_cal(camera_cal: nf_config.CameraCalibration):
    mtx = np.array(camera_cal.intrinsic_matrix).reshape((3,3))
    distortion = np.array(camera_cal.distortion_coeff)
    calibration_shape = (camera_cal.resolution.width, camera_cal.resolution.height) # (1920, 1080)
    # Derive other constants needed for stabilize_frame
    sf_image_ratio = SF_INPUT_SHAPE[0] / calibration_shape[0] # Ratio to scale intrinsics (approx 1/3)
    starting_K = mtx.copy() # if we being using the wide angle camera for the gripper, then this would no long be a copy of the anchor cam cal
    starting_K[0, 0] *= sf_image_ratio  # Scale fx
    starting_K[1, 1] *= sf_image_ratio  # Scale fy
    starting_K[0, 2] *= sf_image_ratio  # Scale cx
    starting_K[1, 2] *= sf_image_ratio  # Scale cy

    # Define Virtual Camera Intrinsics for stabilized gripper frame
    # The optical center (cx, cy) is set to half of the target shape (384/2),
    # not the input shape. This forces the center of the projection to be the center of the square.
    K_new = np.array([
        [starting_K[0, 0] / SF_SCALE_FACTOR, 0,                                  SF_TARGET_SHAPE[0] / 2.0], # cx = 192
        [0,                                  starting_K[1, 1] / SF_SCALE_FACTOR, SF_TARGET_SHAPE[1] / 2.0], # cy = 192
        [0,                                  0,                                  1                    ]
    ])
    return starting_K, K_new

# Stringman tags are from the 'tag36h11' family.
# increase quad_decimate to improve speed at the cost of distance
detector = Detector(families="tag36h11", quad_decimate=1.0)

def _locate_markers(im, K, D):
    # AprilTag detection works on grayscale images.
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    
    if not detections:
        return []

    results = []
    for detection in detections:
        marker_id = detection.tag_id
        corners = detection.corners

        try:
            name = MARKER_NAMES[marker_id]
        except IndexError:
            # Saw a tag that's not part of the defined system
            print(f'Unknown AprilTag spotted with id {marker_id}')
            continue
        
        # Look up the scaled object points specific to this tag
        obj_points = SPECIAL_OBJ_POINTS.get(name, DEFAULT_OBJ_POINTS)
        
        # Use solvePnP to get the rotation and translation vectors (rvec, tvec)
        # This gives the pose of the marker relative to the camera.
        # The coordinate system has the origin at the camera center. The z-axis points from the camera center out the camera lens.
        # The x-axis is to the right in the image taken by the camera, and y is down. The tag's coordinate frame is centered at the center of the tag.
        # From the viewer's perspective, the x-axis is to the right, y-axis down, and z-axis is out of the tag.
        _, r, t = cv2.solvePnP(obj_points, corners, K, D, False, cv2.SOLVEPNP_IPPE_SQUARE)
        
        results.append({
            'n': name,
            'p': (r.reshape((3,)), t.reshape((3,))), # pose tuple. numpy arrays. numpy supposedly has fast pickle hooks
        })
    return results

def locate_markers(im, camera_cal: nf_config.CameraCalibration):
    """
    Detects AprilTags in an image and estimates their pose.
    
    Args:
        im: The input image
        camera_cal: 

    Returns:
        A list of dictionaries, each containing the name, rotation vector (r),
        and translation vector (t) of a detected marker.

    TODO: use a cropped search window that uses slices like im[y:y+h, x:x+w] based on where tags were seen on
    previous frames. search the whole image only if the tag was not seen on the previous frame.
    """

    # Use passed object for camera calibration. keeps function pure for multiprocessing
    mtx = np.array(camera_cal.intrinsic_matrix).reshape((3,3))
    distortion = np.array(camera_cal.distortion_coeff)
    return _locate_markers(im, mtx, distortion)

def locate_markers_gripper(im, camera_cal: nf_config.CameraCalibration):
    """
    Locate markers in the stabilized gripper frame
    """
    if 'K_new' not in saved_matrices:
        saved_matrices['starting_K'], saved_matrices['K_new'] = gripper_stabilized_cal(camera_cal)
    return _locate_markers(im, saved_matrices['K_new'], np.array(camera_cal.distortion_coeff))

def project_pixels_to_floor(normalized_pixels, pose, camera_cal: nf_config.CameraCalibration):
    """
    batch project normalized [0,1] pixel coordinates from a camera's point of view to the floor
    make sure you use the camera pose, not just the anchor pose!
    anchor 3 z rot 2.356194490192345
    """
    # Use passed object for camera calibration.
    K = np.array(camera_cal.intrinsic_matrix).reshape((3,3))
    D = np.array(camera_cal.distortion_coeff)
    image_shape = (camera_cal.resolution.width, camera_cal.resolution.height) # (1920, 1080)

    # Undistort Points
    pts = np.array(normalized_pixels, dtype=np.float64) * image_shape
    uv = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D).reshape(-1, 2).T

    # Rotate Rays to World Frame
    rays = cv2.Rodrigues(np.array(pose[0]))[0] @ np.vstack((uv, np.ones(uv.shape[1])))

    # Calculate Intersections with floor
    tvec = np.array(pose[1], dtype=np.float64).reshape(3, 1)
    with np.errstate(divide='ignore'): # Handle potential div/0
        s = -tvec[2] / rays[2]

    # Filter Valid Points and Return
    mask = (s > 0) & (np.abs(rays[2]) > 1e-6)
    return (tvec + s[mask] * rays[:, mask])[:2].T

def project_floor_to_pixels(floor_points, pose, camera_cal: nf_config.CameraCalibration):
    """
    Project world coordinates on the floor (z=0) back to normalized pixel coordinates.
    """
    # Use passed object for camera calibration.
    K = np.array(camera_cal.intrinsic_matrix).reshape((3,3))
    D = np.array(camera_cal.distortion_coeff)
    image_shape = (camera_cal.resolution.width, camera_cal.resolution.height) # (1920, 1080)

    floor_points = np.array(floor_points, dtype=np.float64)
    
    # Create 3D world points by appending z=0
    zeros = np.zeros((floor_points.shape[0], 1))
    object_points = np.hstack((floor_points, zeros))

    # Extract Camera-to-World rotation and translation
    rvec_c2w = np.array(pose[0], dtype=np.float64)
    tvec_c2w = np.array(pose[1], dtype=np.float64).reshape(3, 1)
    
    R_c2w, _ = cv2.Rodrigues(rvec_c2w)

    # Calculate World-to-Camera transformation for cv2.projectPoints
    R_w2c = R_c2w.T
    tvec_w2c = -R_w2c @ tvec_c2w
    
    # Convert rotation matrix back to rvec for projectPoints
    rvec_w2c, _ = cv2.Rodrigues(R_w2c)

    # Project 3D points to 2D pixel coordinates
    # projectPoints returns shape (N, 1, 2), so we reshape to (N, 2)
    image_points, _ = cv2.projectPoints(object_points, rvec_w2c, tvec_w2c, K, D)
    image_points = image_points.reshape(-1, 2)

    # Normalize coordinates to [0, 1] range
    # We divide by the image width and height provided in image_shape
    normalized_pixels = image_points / image_shape

    return normalized_pixels

def get_rotation_to_center_ray(K, u, v, image_shape):
    """
    Calculates the rotation matrix required to rotate the camera such that
    the ray passing through pixel (u, v) becomes the optical axis (0, 0, 1).
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Back-project pixel to normalized vector in Camera Frame
    # UV coordinates (0..1) to Pixel coordinates
    px = u * image_shape[0]
    py = (1.0 - v) * image_shape[1] # Flip V to match OpenCV Y-down
    
    vec_x = (px - cx) / fx
    vec_y = (py - cy) / fy
    vec_z = 1.0
    
    # Create the Target Vector and Source Vector (Optical Axis)
    target_vec = np.array([vec_x, vec_y, vec_z])
    target_vec = target_vec / np.linalg.norm(target_vec) # Normalize
    
    source_vec = np.array([0, 0, 1]) # We want target_vec to end up here
    
    # Calculate Rotation (Axis-Angle)
    # Rotation axis is perpendicular to both
    rot_axis = np.cross(target_vec, source_vec)
    axis_len = np.linalg.norm(rot_axis)
    
    if axis_len < 1e-6:
        # Vectors are already aligned
        return np.eye(3)
        
    rot_axis = rot_axis / axis_len
    
    # Angle is arccos(dot product)
    angle = np.arccos(np.dot(target_vec, source_vec))
    
    # Create Matrix using Rodrigues
    r_vec = rot_axis * angle
    R_fix, _ = cv2.Rodrigues(r_vec)
    return R_fix

def stabilize_frame(frame, quat, camera_cal: nf_config.CameraCalibration, room_spin=0, range_dist=None, axis_uv_linear=(-0.3182, 0.9845), axis_uv_x_linear=None):
    """
    Warp a video frame to a stationary, centered perspective.
    
    Args:
        frame: Input image
        quat: BNO085 quaternion
        camera_cal: camera calibration of 1920x1080 unstabilized "normal" image
        room_spin: Z-axis offset for room alignment in radians
        range_dist: Distance from camera to floor (meters).
        axis_uv_linear: (slope, intercept) for Y-axis target.
        axis_uv_x_linear: Optional (slope, intercept) for X-axis target. Defaults to Center (0.5).
    """

    if 'K_new' not in saved_matrices:
        saved_matrices['starting_K'], saved_matrices['K_new'] = gripper_stabilized_cal(camera_cal)

    h, w = frame.shape[:2]

    # Physics Rotation (World -> Camera)
    R_room_spin = Rotation.from_euler('z', room_spin).as_matrix()
    r_imu = Rotation.from_quat(quat)
    R_world_to_imu = r_imu.as_matrix().T
    
    R_imu_to_cam = np.array([
        [-1, 0,  0], 
        [0,  0, -1], 
        [0, -1,  0]
    ])
    
    R_world_to_cam = R_imu_to_cam @ R_world_to_imu
    # R_relative un-rotates the camera to align with World Frame
    R_relative = R_room_spin @ R_world_to_cam.T
    
    # Axis Centering (Rotation Fix) ---
    R_fix = np.eye(3)
    
    if range_dist is not None:
        # Y-Axis Logic
        slope_y, intercept_y = axis_uv_linear
        target_v = slope_y * range_dist + intercept_y
        
        # X-Axis Logic (Default to 0.5/Center if not provided)
        if axis_uv_x_linear is not None:
            slope_x, intercept_x = axis_uv_x_linear
            target_u = slope_x * range_dist + intercept_x
        else:
            target_u = 0.5
            
        # Calculate the corrective rotation
        # This rotation maps the Target Ray to the Optical Axis (Z)
        R_fix = get_rotation_to_center_ray(saved_matrices['starting_K'], target_u, target_v, (w, h))

    # Final Homography ---
    # Chain: K_new @ R_relative @ R_fix @ K_inv
    # We apply R_fix FIRST (closest to K_inv) to align the target vector to Z-axis in Camera Frame.
    # Then R_relative aligns that Z-axis (now containing the target) to World Down.
    H = saved_matrices['K_new'] @ R_relative @ R_fix @ np.linalg.inv(saved_matrices['starting_K'])

    # Vertical Flip Matrix
    flip_vertical = np.array([
        [1,  0,  0],
        [0, -1,  SF_TARGET_SHAPE[1]], 
        [0,  0,  1]
    ])
    
    H_final = flip_vertical @ H

    return cv2.warpPerspective(frame, H_final, SF_TARGET_SHAPE, borderMode=cv2.BORDER_REPLICATE, borderValue=(0, 0, 0))