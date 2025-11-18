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

def project_pixel_to_floor(pixel, pose):
    """
    Projects a 2D pixel coordinate onto the floor plane (Z=0).
    
    Args:
        pixel: Tuple (u, v) or list [u, v]. Should be in the 1920x1200 pixel space since that's what the mtx and distortion coeffs were collected with.
        pose: (Camera rotation vector, Camera position vector)
        
    Returns:
        (x, y) tuple of the point on the floor in world coordinates.
        Returns None if the ray never hits the floor (parallel or looking up).
    """
    rvec = np.array(pose[0], dtype=np.float64).reshape((3, 1))
    tvec = np.array(pose[1], dtype=np.float64).reshape((3, 1))
    
    # Undistort the point
    # cv2.undistortPoints expects input shape (N, 1, 2)
    pixel_arr = np.array([[[pixel[0], pixel[1]]]], dtype=np.float64)
    
    # This returns the point in "Normalized Camera Coordinates" (x', y')
    # where the ray direction in camera frame is (x', y', 1)
    # Note: We pass None for P (newCameraMatrix) to get normalized coords, not pixels
    undistorted_pt = cv2.undistortPoints(pixel_arr, mtx, distortion)
    
    x_norm = undistorted_pt[0, 0, 0]
    y_norm = undistorted_pt[0, 0, 1]
    
    # Define the Ray in Camera Coordinates
    # The ray starts at (0,0,0) and goes through (x_norm, y_norm, 1)
    ray_cam = np.array([[x_norm], [y_norm], [1.0]])

    # Transform the Ray to World Coordinates
    # World to Camera: P_cam = R * P_world + t
    # Camera to World: P_world = R^T * (P_cam - t)
    
    # Convert Rodrigues vector to Rotation Matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Calculate Camera Center in World Frame (The start of the ray)
    # C_world = -R^T * tvec
    R_inv = R.T
    cam_center_world = -R_inv @ tvec
    
    # Calculate Ray Direction in World Frame
    # d_world = R^T * d_cam
    ray_world = R_inv @ ray_cam
    
    # Find Intersection with Z=0 Plane
    # A point on the ray is P(s) = cam_center + s * ray_direction
    # We want the point where P_z(s) = 0
    # 0 = C_z + s * d_z  =>  s = -C_z / d_z
    
    C_z = cam_center_world[2, 0]
    d_z = ray_world[2, 0]
    
    # Check if ray is parallel to floor or pointing away
    if abs(d_z) < 1e-6:
        print("Ray is parallel to the floor!")
        return None
        
    s = -C_z / d_z
    print(f's=\n{s}')
    
    # If s < 0, the intersection is behind the camera or the ray is pointing up
    # Since C_z (camera height) is positive (2.455m), d_z must be negative (looking down)
    if s < 0:
        # A negative s means the intersection is behind the camera plane (or looking up)
        # We must confirm the ray is pointing toward the floor (d_z < 0)
        print("Intersection is invalid (ray pointing away or behind camera plane)!")
        return None

    # If s < 0, the intersection is behind the camera
    if s < 0:
        print("Intersection is behind the camera!")
        return None
        
    # Calculate X and Y intersections
    floor_x = cam_center_world[0, 0] + s * ray_world[0, 0]
    floor_y = cam_center_world[1, 0] + s * ray_world[1, 0]
    
    return (floor_x, floor_y)

# --- Precompute some inverted poses ---
gantry_april_inv = invert_pose(model_constants.gantry_april)
anchor_cam_inv = invert_pose(model_constants.anchor_camera)
gripper_imu_inv = invert_pose(model_constants.gripper_imu)

