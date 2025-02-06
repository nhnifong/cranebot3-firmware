import cv2
import cv2.aruco as aruco
import numpy as np
from time import time
from cv_common import locate_markers


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

    
def calibate_camera():
    #Input the number of board images to use for calibration (recommended: ~20)
    n_boards = 20
    #Input the number of squares on the board (width and height)
    board_w = 9
    board_h = 6
    # side length of one square in meters
    board_dim = 0.02381
    #Initializing variables
    board_n = board_w * board_h
    opts = []
    ipts = []
    npts = np.zeros((n_boards, 1), np.int32)
    intrinsic_matrix = np.zeros((3, 3), np.float32)
    distCoeffs = np.zeros((5, 1), np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    # prepare object points based on the actual dimensions of the calibration board
    # like (0,0,0), (25,0,0), (50,0,0) ....,(200,125,0)
    objp = np.zeros((board_h*board_w,3), np.float32)
    objp[:,:2] = np.mgrid[0:(board_w*board_dim):board_dim,0:(board_h*board_dim):board_dim].T.reshape(-1,2)

    #Loop through the images.  Find checkerboard corners and save the data to ipts.
    images_obtained = 0
    start_time = time()
    timeout = 60
    while images_obtained < n_boards and time() < (start_time + timeout):
    
        #Loading images
        # name = 'calibration/GOPR' + str(i+3701) + '.JPG'
        # print('Loading... ' + name)
        # image = cv2.imread(name)

        # obtain image
        capture_url = "http://192.168.1.146/capture?_cb={}"
        req = urllib.request.urlopen(capture_url.format(time()*1000))
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)

        #Convert to grayscale
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        #Find chessboard corners
        found, corners = cv2.findChessboardCorners(grey_image, (board_w,board_h),cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if found == True:
            #Add the "true" checkerboard corners
            opts.append(objp)

            #Improve the accuracy of the checkerboard corners found in the image and save them to the ipts variable.
            cv2.cornerSubPix(grey_image, corners, (20, 20), (-1, -1), criteria)
            ipts.append(corners)
            images_obtained += 1 
            print("images obtained {}/{}".format(images_obtained, n_boards))
    
    if images_obtained < n_boards:
        print("Timed out before obtaining enough images of the calibration board")
        return False
    
    print('Finished capturing images.')

    #Calibrate the camera
    print('Running Calibrations...')
    ret, intrinsic_matrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(opts, ipts, grey_image.shape[::-1],None,None)

    #Save matrices
    print('Intrinsic Matrix: ')
    print(str(intrinsic_matrix))
    print('Distortion Coefficients: ')
    print(str(distCoeff))

    #Save data
    print('Saving data file to calibration_data.npz')
    np.savez('calibration_data', distCoeff=distCoeff, intrinsic_matrix=intrinsic_matrix)
    print('Calibration complete')

    #Calculate the total reprojection error.  The closer to zero the better.
    tot_error = 0
    for i in range(len(opts)):
        imgpoints2, _ = cv2.projectPoints(opts[i], rvecs[i], tvecs[i], intrinsic_matrix, distCoeff)
        error = cv2.norm(ipts[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error

    print("Total reprojection error: ", tot_error/len(opts))
    return True

def calibrate_all():
    # find the distortion coefficients of every camera

    # locate the anchors relative to the origin
    anchor_poses = find_anchor_poses()

    # zero axes?

    # return calibration data that could be saved to a file.
    return {
        'anchor_ips': anchor_ips,
        'anchor_positions': anchor_positions,
    }
