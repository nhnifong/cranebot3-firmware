import cv2
import cv2.aruco as aruco
import numpy as np
import glob
from time import time
import urllib.request

from cv_common import cranebot_boards, cranebot_detectors

capture_url = "http://192.168.1.146/capture?_cb=%d"

# Intrinsic Matrix: 
camera_matrix = np.array(
[[1.55802968e+03, 0.00000000e+00, 8.58167917e+02],
 [0.00000000e+00, 1.56026885e+03, 6.28095370e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
 
# Distortion Coefficients: 
dist_coeffs = np.array(
[[ 3.40916628e-01, -2.38650897e+00, -8.85125582e-04, 3.34240054e-03, 4.69525036e+00]])

# Load images
image_files = glob.glob("images/messy3.jpg")  # Assumes images are in an "images" folder
all_corners = []
all_ids = []

for image_file in image_files:
    frame = cv2.imread(image_file)

    # capture_url = "http://192.168.1.146/capture?_cb={}"
    # req = urllib.request.urlopen(capture_url.format(time()*1000))
    # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    # frame = cv2.imdecode(arr, -1)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    # corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
    charuco_corners, charuco_ids, marker_corners, marker_ids = cranebot_detectors["origin"].detectBoard(frame)

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
        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, cranebot_boards["origin"], camera_matrix, dist_coeffs, None, None)
        # print(f"Pose for {image_file}:")
        print("Rotation Vector (rvec):\n", rvec)
        print("Translation Vector (tvec):\n", tvec)

        if retval:
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, length=0.05, thickness=3)

    cv2.imshow("foo", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()