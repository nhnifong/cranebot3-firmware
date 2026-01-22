import cv2
import numpy as np
import nf_robot.common.definitions as model_constants

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

# --- Precompute some inverted poses ---
gantry_april_inv = invert_pose(model_constants.gantry_april)
anchor_cam_inv = invert_pose(model_constants.anchor_camera)
gripper_imu_inv = invert_pose(model_constants.gripper_imu)

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