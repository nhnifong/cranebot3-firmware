import cv2
import cv2.aruco as aruco
import numpy as np


names = [
	'origin',
	'gripper_front',
	'gripper_back',
	'gantry_front',
	'gantry_back',
	'anchor0',
	'anchor1',
	'anchor2',
	'anchor3',
	'bin_laundry',
	'bin_trash',
	'bin_toys',
	'bin_other',
]
cranebot_boards = {}
cranebot_detectors = {}

# Define ChArUco board parameters
SQUARE_LENGTH = 0.02  # Length of one square in meters
MARKER_LENGTH = 0.015 # Length of ArUco marker in meters
ROWS = 3           # Number of rows of squares
COLS = 3           # Number of columns of squares
N_MARKERS = 4      # Number of markers per board

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

for i,name in enumerate(names):
	board = aruco.CharucoBoard((ROWS, COLS), SQUARE_LENGTH, MARKER_LENGTH, dictionary, np.arange(i*N_MARKERS, (i+1)*N_MARKERS))
	detector = aruco.CharucoDetector(board)
	cranebot_boards[name] = board
	cranebot_detectors[name] = detector

def locate_board(im, name):
    charuco_corners, charuco_ids, marker_corners, marker_ids = cranebot_detectors[name].detectBoard(im)
    if (charuco_corners is not None and len(charuco_corners) > 0) and (marker_corners is not None and len(marker_corners) > 0):
    	return aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, cranebot_boards[0], camera_matrix, dist_coeffs, None, None)
    else:
    	return False, None, None