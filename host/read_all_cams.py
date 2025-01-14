import cv2
import cv2.aruco as aruco
import numpy as np
import glob

# Define ChArUco board parameters
SQUARE_LENGTH = 0.02  # Length of one square in meters
MARKER_LENGTH = 0.015 # Length of ArUco marker in meters
ROWS = 5           # Number of rows of squares
COLS = 7           # Number of columns of squares

# Create ChArUco board
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((3, 3), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
detector = aruco.CharucoDetector(board)

# Camera calibration parameters (replace with your calibrated values!)
camera_matrix = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32) # Example values
dist_coeffs = np.array([0.1, -0.01, 0.001, 0.0001, 0], dtype=np.float32) # Example Distortion coefficients

# Load images
image_files = glob.glob("images/*.jpg")  # Assumes images are in an "images" folder
all_corners = []
all_ids = []

for image_file in image_files:
    frame = cv2.imread(image_file)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    # corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(frame)

    if charuco_ids is not None and len(ids) > 0:
        if retval > 20: # Ensure we have enough corners to estimate pose
            # Estimate the pose of the ChArUco board
            rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs)

            # Draw the detected markers and axis
            aruco.drawDetectedMarkers(frame, corners, ids)
            aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1) # Axis length 0.1 meters

            print(f"Pose for {image_file}:")
            print("Rotation Vector (rvec):\n", rvec)
            print("Translation Vector (tvec):\n", tvec)

        else:
            print(f"Not enough ChArUco corners detected in {image_file} to estimate pose.")
    else:
        print(f"No ArUco markers detected in {image_file}")

    cv2.imshow(image_file, frame)

cv2.waitKey(0)
cv2.destroyAllWindows()