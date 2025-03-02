import numpy as np
import cv2
import cv2.aruco as aruco
from cv_common import detector, mtx, distortion, origin_charuco_board, parameters
from time import sleep


charc_parameters = aruco.CharucoParameters()
charc_parameters.cameraMatrix = mtx
charc_parameters.distCoeffs = distortion
charc_parameters.tryRefineMarkers = True

charc_detector = aruco.CharucoDetector(origin_charuco_board, charc_parameters, parameters)

marker_size = 0.212 # size of charuco board in meters
corner_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                          [marker_size / 2, marker_size / 2, 0],
                          [marker_size / 2, -marker_size / 2, 0],
                          [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

import sys
video_uri = f'tcp://192.168.1.{sys.argv[1]}:8888'
print(f'Connecting to {video_uri}')
cap = cv2.VideoCapture(video_uri)
print(cap)
while True:
    ret, image = cap.read()
    if not ret:
        sleep(0.5)
        continue
    marker_corners, marker_ids, rejectedImgPoints = detector.detectMarkers(image)
    if marker_ids is not None:
        diamondCorners, diamondIds, _, _ = charc_detector.detectDiamonds(image, marker_corners, marker_ids)
        if len(diamondIds) > 0:
            aruco.drawDetectedDiamonds(image, diamondCorners, diamondIds)
            # estimate pose
            _, rvec, tvec = cv2.solvePnP(corner_points, diamondCorners[0][0], mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
            cv2.drawFrameAxes(image, mtx, distortion, rvec, tvec, 0.1)  # 0.1 is the axis length in meters
    image = cv2.resize(image, (2304, 1296),  interpolation = cv2.INTER_LINEAR)
    cv2.imshow("Detected ChArUco Board", image)
    cv2.waitKey(1)
cv2.destroyAllWindows()


(
    (np.array([[[1463.,  535.],
        [1496.,  537.],
        [1493.,  568.],
        [1459.,  565.]]], dtype=float32),),
    np.array([[10]], dtype=int32),
    (np.array([[[1463.,  535.],
        [1496.,  537.],
        [1493.,  568.],
        [1459.,  565.]]], dtype=float32),),
    np.array([[10]], dtype=int32)
)