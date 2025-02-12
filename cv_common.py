import cv2
import cv2.aruco as aruco
import numpy as np
import time

try:
    calib = np.load('calibration_data.npz')
    mtx = calib['intrinsic_matrix']
    distortion = calib['distCoeff']
except:
    # Intrinsic Matrix:
    mtx = np.array(
    [[1.37167317e+03, 0.00000000e+00, 5.50300368e+02],
     [0.00000000e+00, 1.37632021e+03, 3.16524984e+02],
     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # Distortion Coefficients: 
    distortion = np.array(
    [[ 0.01959627, 0.56390969, -0.00256158, -0.00496641, -0.93400812]])

# the ids are the index in the list
marker_names = [
    'origin',
    'gripper_front',
    'gripper_back',
    'gantry_front',
    'gantry_back',
    'bin_other',
]

aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
parameters = aruco.DetectorParameters()
parameters.minMarkerPerimeterRate = 0.04
parameters.maxMarkerPerimeterRate = 4.0
detector = aruco.ArucoDetector(aruco_dict, parameters)
marker_size = 0.09 # Length of ArUco marker in meters

def locate_markers(im):
    corners, ids, rejectedImgPoints = detector.detectMarkers(im)
    results = []
    if ids is not None:
        #estimate pose of every marker in the image
        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                  [marker_size / 2, marker_size / 2, 0],
                                  [marker_size / 2, -marker_size / 2, 0],
                                  [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

        for i,c in zip(ids, corners):
            _, r, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
            try:
                name = marker_names[i[0]]
                # this is meant to be json serializable
                results.append({
                    'n': name,
                    'r': r.tolist(),
                    't': t.tolist()})
            except IndexError:
                # saw something that's not part of my robot
                print(f'Unknown marker spotted with id {i}')
    return results

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

def invert_pose(pose):
    """
    Invert the frame of reference.
    If rvec and tvec represent the rotation and translation of a board relative to a camera
    return the the rotation and translation of the camera relative to the board.
    """
    rvec, tvec = pose
    R_cam_to_marker, _ = cv2.Rodrigues(rvec)  # Rotation matrix
    R_marker_to_cam = R_cam_to_marker.T  # Transpose for inverse rotation
    tvec_marker_to_cam = -np.dot(R_marker_to_cam, tvec)
    rvec_marker_to_cam, _ = cv2.Rodrigues(R_marker_to_cam)  # Back to rotation vector
    return rvec, tvec

    # positive Z points out of the face of the marker
    return rvec_marker_to_cam, tvec_marker_to_cam

def compose_poses(poses):
    """Composes a chain of relative poses into a single global pose.
    equivalent to the problem of locating an object in a 3d scene that has a chain of parent objects.

    Args:
        poses: A list of tuples, where each tuple contains (rvec, tvec)
               representing the pose transformation from the previous frame
               to the current frame.

    Returns:
        A tuple (rvec_global, tvec_global) representing the final pose
        in the reference frame of the 0th pose (often the global reference frame.)
        Returns None if the input is invalid.
    """
    if not poses:
        return None

    rvec_global = poses[0][0].copy()
    tvec_global = poses[0][1].copy()

    R_global, _ = cv2.Rodrigues(rvec_global) #Initial Rotation Matrix

    for rvec_relative, tvec_relative in poses[1:]:
        R_relative, _ = cv2.Rodrigues(rvec_relative)
        # Accumulate rotation
        R_global = np.dot(R_global, R_relative)
        # Accumulate translation
        tvec_global = np.dot(R_global, tvec_relative) + tvec_global

    rvec_global, _ = cv2.Rodrigues(R_global)  # Convert back to rotation vector
    return rvec_global, tvec_global

def generateMarkerImages():
    border_px = 40
    marker_side_px = 500
    cm = (marker_size/marker_side_px)*(marker_side_px+border_px*2)*100
    print('boards should be printed with a side length of %0.2f cm' % cm)
    for i, name in enumerate(marker_names):
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, i, marker_side_px)

        # white frame with black corner squares
        total_size = marker_side_px + 2 * border_px
        framed_image = np.ones((total_size, total_size), dtype=np.uint8) * 255

        # Place the marker image in the center
        framed_image[border_px:border_px + marker_side_px, border_px:border_px + marker_side_px] = marker_image

        # Draw black squares in the corners

        framed_image[:border_px, :border_px] = 0
        framed_image[-border_px:, -border_px:] = 0
        framed_image[:border_px, -border_px:] = 0
        framed_image[-border_px:, :border_px] = 0

        cv2.imwrite(os.path.join('boards',name+'.png'), framed_image)

if __name__ == "__main__":
    generateMarkerImages()
