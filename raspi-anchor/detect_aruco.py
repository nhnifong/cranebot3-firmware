import cv2
import cv2.aruco as aruco
import numpy as np
import time
from picamera2 import Picamera2

# Use the still configuration, which gives the full 4k resolution
picam2 = Picamera2()
#capture_config = picam2.create_still_configuration()
# full res is 4608x2592
capture_config = picam2.create_preview_configuration(main={"size": (2304, 1296), "format": "RGB888"})
picam2.configure(capture_config)
picam2.start()

# Intrinsic Matrix: 
mtx = np.array(
[[1.55802968e+03, 0.00000000e+00, 8.58167917e+02],
 [0.00000000e+00, 1.56026885e+03, 6.28095370e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Distortion Coefficients: 
distortion = np.array(
[[ 3.40916628e-01, -2.38650897e+00, -8.85125582e-04, 3.34240054e-03, 4.69525036e+00]])

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
                results.append((name, np.array(r), np.array(t)))
            except IndexError:
                # saw something that's not part of my robot
                print(f'Unknown marker spotted with id {i}')
    return results

while True:
    im = picam2.capture_array()
    print(locate_markers(im))
