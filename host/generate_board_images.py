import cv2
import cv2.aruco as aruco
import os

from cv_common import cranebot_boards, cranebot_detectors, names

for name in names:
	board_image = cranebot_boards[name].generateImage((500, 500))
	cv2.imwrite(os.path.join('boards',name+'.png'), board_image)