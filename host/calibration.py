import cv2
import cv2.aruco as aruco
import numpy as np
from time import time

from cv_common import cranebot_boards, cranebot_detectors, locate_board

def average_pose(poses):
    """
    Averages a list of Charuco board detection results to provide a more accurate pose.

    Args:
        poses: A list of tuples, where each tuple contains:
            - rvec: Rotation vector of the Charuco board.
            - tvec: Translation vector of the Charuco board.

    Returns:
        A tuple containing the averaged rvec and tvec
    """

    # Convert rotation vectors to rotation matrices
    rotation_matrices = []
    translation_vectors = []
    for rvec, tvec in poses:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        rotation_matrices.append(rotation_matrix)
        translation_vectors.append(tvec.reshape(3, 1))  # Ensure correct shape

    # Average the translation vectors
    average_translation_vector = np.mean(np.array(translation_vectors), axis=0)

    # Average the rotation matrices using singular value decomposition
    sum_of_rotation_matrices = np.zeros((3, 3))
    for rotation_matrix in rotation_matrices:
        sum_of_rotation_matrices += rotation_matrix
    average_intermediate_matrix = sum_of_rotation_matrices / len(poses)

    U, S, V = np.linalg.svd(average_intermediate_matrix)
    average_rotation_matrix = np.dot(U, V)

    # Ensure a proper rotation matrix (det should be close to 1)
    if np.linalg.det(average_rotation_matrix) < 0:
        average_rotation_matrix = -average_rotation_matrix

    # Convert the averaged rotation matrix back to a rotation vector
    average_rotation_vector, _ = cv2.Rodrigues(average_rotation_matrix)

    return average_rotation_vector, average_translation_vector.flatten()

def relative_pose(rvec1, tvec1, rvec2, tvec2):
    """
    Computes the pose of the second Charuco board relative to the first.

    Args:
        rvec1: Rotation vector of the first board.
        tvec1: Translation vector of the first board.
        rvec2: Rotation vector of the second board.
        tvec2: Translation vector of the second board.

    Returns:
        A tuple containing:
            - rvec_rel: Rotation vector of the second board relative to the first.
            - tvec_rel: Translation vector of the second board relative to the first.
        Returns None if there is an error.
    """
    try:
        # Convert rotation vectors to rotation matrices
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)

        # Calculate the relative rotation
        R_rel = np.dot(R2, R1.T)

        # Calculate the relative translation
        tvec_rel = tvec2.reshape(3, 1) - np.dot(R_rel, tvec1.reshape(3, 1))

        # Convert the relative rotation matrix back to a rotation vector
        rvec_rel, _ = cv2.Rodrigues(R_rel)

        return rvec_rel, tvec_rel.flatten() #flatten for consistency with input

    except Exception as e:
        print(f"Error in relative_pose: {e}")
        return None

def find_anchor_positions(n_anc = 3, n_to_capture = 10, timeout = 40)
    # initialize detections to a list of empty lists, one for each anchor
    detections = []
    for i in range(n_anc):
        detections.append([])
    # capture 10 images of the origin board from each anchor camera.
    start_time = time()
    while any([len(a)<n_to_capture for a in detections]):
        for anchor_index, dlist in enumerate(detections):
            if len(dlist) < n_to_capture:
                retval, rvec, tvec = locate_board(image, 'origin')
                if retval:
                    detections.append((rvec, tvec))
            if time() > start_time + timeout:
                raise TimeoutError("timed out trying to capture enough images of the origin board. Captured %r" % list(len(a) for a in detections))

    # average the positions for each camera.
    camera_pose = map(average_pose, detections)

    # invert the frame of reference. (find the pose of the anchor camera in the reference frame of the origin board)

    # return the positions

def calibrate():
    # find network addresses and identify boards that are part of the same robot
    # find the distortion coefficients of every camera
    # locate the anchors relative to the origin
    find_anchor_positions()
    # zero axes
    # save the definition of this robot to a file.
