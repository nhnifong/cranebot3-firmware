import os
import cv2
from pupil_apriltags import Detector
import numpy as np
import time
from functools import lru_cache
from config_loader import *
import model_constants
from scipy.spatial.transform import Rotation

# --- Configuration ---
cfg = load_config()
mtx = np.array(cfg.camera_cal.intrinsic_matrix).reshape((3,3))
print(mtx)
distortion = np.array(cfg.camera_cal.distortion_coeff)
sf_calibration_shape = (1920, 1080) 

# The marker IDs will correspond to the index in this list.
marker_names = [
    'origin',
    'gantry',
    'gamepad',
    'hamper',
    'trash',
    'cal_assist_1',
    'cal_assist_2',
    'cal_assist_3',
]

# AprilTag images are typically downloaded, not generated in code.
# You can find printable PNGs for all standard families here:
# https://github.com/AprilRobotics/apriltag-imgs

# Define the physical size of any markers that are not the default size.
special_sizes = {
    'origin': 0.1680, # size in meters
    'cal_assist_1': 0.1640, # shouldn't these have printed with the same dimensions as the origin card?
    'cal_assist_2': 0.1640,
    'cal_assist_3': 0.1640,
    'gantry':       0.0915
}
default_marker_size = 0.08948 # The default side length of markers in meters

# The 'tag36h11' family is a good general-purpose choice.
# Other options include 'tag16h5', 'tag25h9', 'tagCircle21h7', etc.
# increase quad_decimate to improve speed at the cost of distance
detector = Detector(families="tag36h11", quad_decimate=1.0)

def locate_markers(im, mxx=mtx):
    """
    Detects AprilTags in an image and estimates their pose.
    
    Args:
        im: The input image (must be grayscale).

    Returns:
        A list of dictionaries, each containing the name, rotation vector (r),
        and translation vector (t) of a detected marker.
    """
    # AprilTag detection works on grayscale images.
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    
    results = []
    if detections:
        # These are the 3D corner points of a generic marker of size 1x1 meter.
        # We will scale this based on the actual marker size.
        marker_points = np.array([
            [-0.5, -0.5, 0], # Index 0: Bottom-Left
            [ 0.5, -0.5, 0], # Index 1: Bottom-Right
            [ 0.5,  0.5, 0], # Index 2: Top-Right
            [-0.5,  0.5, 0]  # Index 3: Top-Left
        ], dtype=np.float32)

        # marker_points = np.array([[-0.5, 0.5, 0],
        #                   [0.5, 0.5, 0],
        #                   [0.5, -0.5, 0],
        #                   [-0.5, -0.5, 0]], dtype=np.float32)

        for detection in detections:
            marker_id = detection.tag_id
            corners = detection.corners

            try:
                name = marker_names[marker_id]
            except IndexError:
                # Saw a tag that's not part of the defined system
                print(f'Unknown AprilTag spotted with id {marker_id}')
                continue

            # Scale the 3D marker points based on the specific size for this tag
            if name in special_sizes:
                size = special_sizes[name]
            else:
                size = default_marker_size
            
            mp = marker_points * size
            
            # Use solvePnP to get the rotation and translation vectors (rvec, tvec)
            # This gives the pose of the marker relative to the camera.
            # The coordinate system has the origin at the camera center. The z-axis points from the camera center out the camera lens.
            # The x-axis is to the right in the image taken by the camera, and y is down. The tag's coordinate frame is centered at the center of the tag.
            # From the viewer's perspective, the x-axis is to the right, y-axis down, and z-axis is out of the tag.
            _, r, t = cv2.solvePnP(mp, corners, mxx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
            
            # Append the result in a JSON-serializable format
            results.append({
                'n': name,
                'r': r.tolist(),
                't': t.tolist()
            })
    return results

def average_pose(poses):
    """
    Averages a list of pose detection results to provide a more accurate pose.
    """
    rotation_matrices = []
    translation_vectors = []
    for rvec, tvec in poses:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        rotation_matrices.append(rotation_matrix)
        translation_vectors.append(tvec.reshape(3, 1))

    average_translation_vector = np.mean(np.array(translation_vectors), axis=0)

    sum_of_rotation_matrices = np.sum(rotation_matrices, axis=0)
    average_intermediate_matrix = sum_of_rotation_matrices / len(poses)

    U, S, V = np.linalg.svd(average_intermediate_matrix)
    average_rotation_matrix = np.dot(U, V)

    if np.linalg.det(average_rotation_matrix) < 0:
        average_rotation_matrix = -average_rotation_matrix

    average_rotation_vector, _ = cv2.Rodrigues(average_rotation_matrix)
    return average_rotation_vector.reshape((3,)), average_translation_vector.flatten()

def invert_pose(pose):
    """
    Inverts the frame of reference of a pose.
    (e.g., from marker-relative-to-camera to camera-relative-to-marker)
    """
    rvec, tvec = pose
    R_cam_to_marker, _ = cv2.Rodrigues(rvec)
    R_marker_to_cam = R_cam_to_marker.T
    tvec_marker_to_cam = -np.dot(R_marker_to_cam, tvec)
    rvec_marker_to_cam, _ = cv2.Rodrigues(R_marker_to_cam)
    return rvec_marker_to_cam.reshape((3,)), tvec_marker_to_cam

def compose_poses(poses):
    """Composes a chain of relative poses into a single global pose."""
    if not poses:
        return None

    rvec_global, tvec_global = poses[0]
    R_global, _ = cv2.Rodrigues(rvec_global)

    for rvec_relative, tvec_relative in poses[1:]:
        R_relative, _ = cv2.Rodrigues(rvec_relative.reshape((3,)))
        tvec_global = np.dot(R_global, tvec_relative.reshape((3,))) + tvec_global
        R_global = np.dot(R_global, R_relative)

    rvec_global, _ = cv2.Rodrigues(R_global)
    return rvec_global.reshape((3,)), tvec_global

def homogenize_types(poses):
    """Ensures all pose elements are float numpy arrays."""
    return [
        (np.array(r, dtype=float), np.array(t, dtype=float))
        for r, t in poses
    ]

import numpy as np
import cv2

def create_lookat_pose(cam_pos, target_pos):
    """
    Creates a Camera-to-World pose (rvec, tvec) looking at target_pos.
    Convention:
    - tvec: Camera Position in World.
    - rvec: Rotation from Camera Frame (X-Right, Y-Down, Z-Forward) to World.
    """
    cam_pos = np.array(cam_pos, dtype=float)
    target_pos = np.array(target_pos, dtype=float)
    
    z_axis = target_pos - cam_pos
    z_axis = z_axis / np.linalg.norm(z_axis)
    forward = z_axis
    
    right = np.cross(forward, np.array([0,0,1]))
    right = right / np.linalg.norm(right)

    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    R_c2w = np.column_stack((right, down, forward))
    rvec, _ = cv2.Rodrigues(R_c2w)
    
    return (rvec, cam_pos)

def project_pixels_to_floor(normalized_pixels, pose, K=mtx, D=distortion):
    """
    batch project normalized [0,1] pixel coordinates from a camera's point of view to the floor
    make sure you use the camera pose, not just the anchor pose!
    """
    # Undistort Points
    pts = np.array(normalized_pixels, dtype=np.float64) * sf_calibration_shape
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

def project_floor_to_pixels(floor_points, pose, K=mtx, D=distortion, image_shape=sf_calibration_shape):
    """
    Project world coordinates on the floor (z=0) back to normalized pixel coordinates.
    """
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

# Configuration for stabilize_frame
sf_input_shape = (960, 540)      # Size of the raw frame coming from camera
sf_target_shape = (384, 384)     # Size of the final neural net input (Square)

# Scale Logic
sf_image_ratio = sf_input_shape[0] / sf_calibration_shape[0] # Ratio to scale intrinsics (approx 1/3)
sf_scale_factor = 1.4  # Zoom factor (values less than 1 zoom in)

# Scale the Original Intrinsics to match the input video resolution
starting_K = mtx.copy()
starting_K[0, 0] *= sf_image_ratio  # Scale fx
starting_K[1, 1] *= sf_image_ratio  # Scale fy
starting_K[0, 2] *= sf_image_ratio  # Scale cx
starting_K[1, 2] *= sf_image_ratio  # Scale cy

# Define Virtual Camera Intrinsics (K_new)
# The optical center (cx, cy) is set to half of the target shape (384/2),
# not the input shape. This forces the center of the projection to be the center of the square.
K_new = np.array([
    [starting_K[0, 0] / sf_scale_factor, 0,                                  sf_target_shape[0] / 2.0], # cx = 192
    [0,                                  starting_K[1, 1] / sf_scale_factor, sf_target_shape[1] / 2.0], # cy = 192
    [0,                                  0,                                  1                    ]
])

def stabilize_frame(frame, quat, room_spin=0, K=starting_K):
    """
    Warp a gripper video frame to a stationary perspective based on rotation.
    """
    R_room_spin = Rotation.from_euler('z', room_spin).as_matrix()
    r_imu = Rotation.from_quat(quat)
    R_world_to_imu = r_imu.as_matrix().T
    
    # Static transforms (IMU->Cam)
    R_imu_to_cam = np.array([
        [-1, 0,  0],
        [0,  0, -1],
        [0, -1,  0]
    ])
    
    R_world_to_cam = R_imu_to_cam @ R_world_to_imu
    R_relative = R_room_spin @ R_world_to_cam.T
    
    # Base Homography
    H = K_new @ R_relative @ np.linalg.inv(K)

    # Vertical Flip Matrix
    flip_vertical = np.array([
        [1,  0,  0],
        [0, -1,  sf_target_shape[1]], 
        [0,  0,  1]
    ])
    
    H_final = flip_vertical @ H

    # Warp Perspective
    return cv2.warpPerspective(frame, H_final, sf_target_shape, borderMode=cv2.BORDER_REPLICATE, borderValue=(0, 0, 0))

def locate_markers_gripper(im):
    return locate_markers(im, mxx=K_new)

# --- Precompute some inverted poses ---
gantry_april_inv = invert_pose(model_constants.gantry_april)
anchor_cam_inv = invert_pose(model_constants.anchor_camera)
gripper_imu_inv = invert_pose(model_constants.gripper_imu)

def get_rotation_to_center_ray(K, u, v, image_shape):
    """
    Calculates the rotation matrix required to rotate the camera such that
    the ray passing through pixel (u, v) becomes the optical axis (0, 0, 1).
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # 1. Back-project pixel to normalized vector in Camera Frame
    # UV coordinates (0..1) to Pixel coordinates
    px = u * image_shape[0]
    py = (1.0 - v) * image_shape[1] # Flip V to match OpenCV Y-down
    
    vec_x = (px - cx) / fx
    vec_y = (py - cy) / fy
    vec_z = 1.0
    
    # 2. Create the Target Vector and Source Vector (Optical Axis)
    target_vec = np.array([vec_x, vec_y, vec_z])
    target_vec = target_vec / np.linalg.norm(target_vec) # Normalize
    
    source_vec = np.array([0, 0, 1]) # We want target_vec to end up here
    
    # 3. Calculate Rotation (Axis-Angle)
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

def stabilize_frame_offset(frame, quat, room_spin=0, range_dist=None, axis_uv_linear=(-0.3182, 0.9845), axis_uv_x_linear=None, K=starting_K):
    """
    Warp a video frame to a stationary, centered perspective.
    
    Args:
        frame: Input image
        quat: BNO085 quaternion
        room_spin: Z-axis offset for room alignment
        range_dist: Distance from camera to floor (meters).
        axis_uv_linear: (slope, intercept) for Y-axis target.
        axis_uv_x_linear: Optional (slope, intercept) for X-axis target. Defaults to Center (0.5).
        K: Camera Matrix.
    """
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
        R_fix = get_rotation_to_center_ray(K, target_u, target_v, (w, h))

    # Final Homography ---
    # Chain: K_new @ R_relative @ R_fix @ K_inv
    # We apply R_fix FIRST (closest to K_inv) to align the target vector to Z-axis in Camera Frame.
    # Then R_relative aligns that Z-axis (now containing the target) to World Down.
    H = K_new @ R_relative @ R_fix @ np.linalg.inv(K)

    # Vertical Flip Matrix
    flip_vertical = np.array([
        [1,  0,  0],
        [0, -1,  sf_target_shape[1]], 
        [0,  0,  1]
    ])
    
    H_final = flip_vertical @ H

    return cv2.warpPerspective(frame, H_final, sf_target_shape, borderMode=cv2.BORDER_REPLICATE, borderValue=(0, 0, 0))