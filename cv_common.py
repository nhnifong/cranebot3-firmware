import os
import cv2
from pupil_apriltags import Detector
import numpy as np
import time
from functools import lru_cache
from config import Config
import model_constants

# --- Configuration ---
config = Config()
mtx = config.intrinsic_matrix
distortion = config.distortion_coeff

# The marker IDs will correspond to the index in this list.
marker_names = [
    'origin',
    'gantry',
    'gamepad',
    'hamper',
    'trash',
]

# AprilTag images are typically downloaded, not generated in code.
# You can find printable PNGs for all standard families here:
# https://github.com/AprilRobotics/apriltag-imgs

# Define the physical size of any markers that are not the default size.
special_sizes = {
    'origin': 0.1680, # size in meters
}
default_marker_size = 0.0868 # The default side length of markers in meters

# The 'tag36h11' family is a good general-purpose choice.
# Other options include 'tag16h5', 'tag25h9', 'tagCircle21h7', etc.
# increase quad_decimate to improve speed at the cost of distance
detector = Detector(families="tag36h11", quad_decimate=1.0)

def locate_markers(im):
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
        marker_points = np.array([[-0.5, 0.5, 0],
                                  [0.5, 0.5, 0],
                                  [0.5, -0.5, 0],
                                  [-0.5, -0.5, 0]], dtype=np.float32)

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
            # From the viewer's perspective, the x-axis is to the right, y-axis down, and z-axis is into the tag.
            _, r, t = cv2.solvePnP(mp, corners, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
            
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

def project_pixels_to_floor(normalized_pixels, pose):
    """
    Batch projects normalized pixels (0-1) to the floor plane (Z=0).
    Assumes global 'mtx' and 'distortion' are defined.
    """
    # Scale 0-1 inputs to 1920x1200 and reshape for cv2 (N, 1, 2)
    pts = np.array(normalized_pixels, dtype=np.float64) * [1920, 1200]
    
    # Camera Algebra: Get Camera Center (C) and Rotated Rays (world frame)
    R = cv2.Rodrigues(np.array(pose[0]))[0]
    C = -R.T @ np.array(pose[1]).reshape(3, 1)
    
    # undistortPoints handles the batch. Reshape output to (2, N) and stack 1s -> (3, N)
    uv_norm = cv2.undistortPoints(pts.reshape(-1, 1, 2), mtx, distortion).reshape(-1, 2).T
    rays_world = R.T @ np.vstack((uv_norm, np.ones(uv_norm.shape[1])))

    # Intersect Z=0: s = -C_z / Ray_z
    # Use errstate to silence divide-by-zero warnings for parallel rays
    with np.errstate(divide='ignore', invalid='ignore'):
        s = -C[2] / rays_world[2]

    # Filter valid points (s > 0 means intersection is in front of camera)
    # We mask immediately to avoid invalid arithmetic on parallel rays
    valid_mask = (s > 0) & (s != np.inf)
    
    # Calculate P = C + s * Ray only for valid indices
    # Result shape starts as (3, M), we take first 2 rows (x,y) and transpose to (M, 2)
    valid_rays = rays_world[:, valid_mask]
    valid_s = s[valid_mask]
    
    floor_pts = C + valid_rays * valid_s
    return floor_pts[:2, :].T

# --- Precompute some inverted poses ---
gantry_april_inv = invert_pose(model_constants.gantry_april)
anchor_cam_inv = invert_pose(model_constants.anchor_camera)
gripper_imu_inv = invert_pose(model_constants.gripper_imu)

