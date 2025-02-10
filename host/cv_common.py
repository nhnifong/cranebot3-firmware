import cv2
import cv2.aruco as aruco
import numpy as np
import os

# Intrinsic Matrix: 
camera_matrix = np.array(
[[1.55802968e+03, 0.00000000e+00, 8.58167917e+02],
 [0.00000000e+00, 1.56026885e+03, 6.28095370e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
 
# Distortion Coefficients: 
dist_coeffs = np.array(
[[ 3.40916628e-01, -2.38650897e+00, -8.85125582e-04, 3.34240054e-03, 4.69525036e+00]])

# the ids are the index in the list
marker_names = {
    'origin',
    'gripper_front',
    'gripper_back',
    'gantry_front',
    'gantry_back',
    'bin_other',
}

marker_size = 0.09 # Length of ArUco marker in meters

aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
aruco_parameters = aruco.DetectorParameters()
# Minimum and maximum size that an aruco marker could be as a fraction of the image width
aruco_parameters.minMarkerPerimeterRate = 0.01
aruco_parameters.maxMarkerPerimeterRate = 8.0
detector = aruco.ArucoDetector(aruco_dict, aruco_parameters)

class Detection:
    def __init__(self, name, r, t):
        self.name = name
        self.rotation = r
        self.translation = t

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
                name = marker_names[i]
                results.append(Detection(name, np.array(r), np.array(t)))
            except IndexError:
                # saw something that's not part of my robot
                print(f'Unknown marker spotted with id {i}')
    return results

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

if __name__ == '__main__':
    generateMarkerImages()