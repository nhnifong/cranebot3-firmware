import cv2
from cv_common import cranebot_boards
import os

for key,board in cranebot_boards.items():
	print(key)
	img = board.generateImage((500,500))
	cv2.imwrite(os.path.join('boards',key+'.jpg'), img)