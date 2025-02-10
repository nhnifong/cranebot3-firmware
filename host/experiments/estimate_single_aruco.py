import cv2
import cv2.aruco as aruco
import numpy as np
import glob

MARKER_LENGTH = 0.015 # Length of ArUco marker in meters

# Intrinsic Matrix: 
camera_matrix = np.array(
[[1.55802968e+03, 0.00000000e+00, 8.58167917e+02],
 [0.00000000e+00, 1.56026885e+03, 6.28095370e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
 
# Distortion Coefficients: 
dist_coeffs = np.array(
[[ 3.40916628e-01, -2.38650897e+00, -8.85125582e-04, 3.34240054e-03, 4.69525036e+00]])


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
parameters = cv2.aruco.DetectorParameters()
parameters.minMarkerPerimeterRate = 0.04
parameters.maxMarkerPerimeterRate = 4.0

# Load images
image_files = glob.glob("images/foo_pxl_shrink.jpg")  # Assumes images are in an "images" folder
all_corners = []
all_ids = []

def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []

    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return np.array(rvecs), np.array(tvecs), trash

for image_file in image_files:
    frame = cv2.imread(image_file)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    print(corners)

    if ids is not None:
        #estimate pose of every marker in the image
        # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
        rvecs, tvecs, _ = estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)

        print(f"Pose for {image_file}:")
        print("Rotation Vector (rvec):\n", rvecs)
        print("Translation Vector (tvec):\n", tvecs)

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH)

    cv2.imwrite("images/out.jpg", frame)

# cv2.waitKey(0)
# cv2.destroyAllWindows()