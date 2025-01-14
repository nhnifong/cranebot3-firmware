import cv2
import cv2.aruco as aruco
import numpy as np
import glob

# Define ChArUco board parameters
SQUARE_LENGTH = 0.02  # Length of one square in meters
MARKER_LENGTH = 0.015 # Length of ArUco marker in meters
ROWS = 3           # Number of rows of squares
COLS = 3           # Number of columns of squares

# Create ChArUco board
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((ROWS, COLS), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
detector = aruco.CharucoDetector(board)

gim = board.generateImage((500, 500))

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

    print(charuco_corners)
    print(charuco_ids)
    print(marker_corners)
    print(marker_ids)

    # Draw the detected markers and axis
    if len(marker_corners) > 0: 
        aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
    # aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1) # Axis length 0.1 meters

    if charuco_corners is not None and len(charuco_corners) > 0:
        aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (255, 0, 0));

        #estimate charuco board pose
        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None)
        print(f"Pose for {image_file}:")
        print("Rotation Vector (rvec):\n", rvec)
        print("Translation Vector (tvec):\n", tvec)

        if retval:
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, length=0.05, thickness=3)

    cv2.imshow(image_file, frame)

cv2.waitKey(0)
cv2.destroyAllWindows()