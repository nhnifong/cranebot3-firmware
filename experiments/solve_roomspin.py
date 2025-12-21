import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from cv_common import *
from raspi_anchor_client import pose_from_det

rvec = None

input_image = cv2.imread('experiments/input_image.png')
quat = np.load('experiments/quat.npy')

# try to stabilize the image with roomspin = 0
im = stabilize_frame(input_image, quat, 0)

detections = locate_markers_gripper(im)
for det in detections:
	if det['n'] == 'origin':
		print('detected origin')
		rvec, tvec = pose_from_det(det)
		cv2.drawFrameAxes(im, K_new, distortion, rvec, tvec, 0.1)  # 0.1 is the axis length in meters

cv2.imshow("example", im)
if cv2.waitKey(0) & 0xFF == ord('q'):
    pass

euler_rot = Rotation.from_rotvec(rvec).as_euler('zyx')
print(f'euler rotation of origin card relative to stabilized gripper camera {euler_rot}')
roomspin = euler_rot[0]


# if we tried to stabilize the image again with the determined amount of spin, is it perfect?
im = stabilize_frame(input_image, quat, -roomspin)

cv2.imshow("example", im)
if cv2.waitKey(0) & 0xFF == ord('q'):
    pass