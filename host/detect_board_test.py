from cv_common import locate_board
import cv2

frame = cv2.imread('images/origin_medium.jpg')
if frame is not None:
    retval, rvec, tvec = locate_board(frame, 'origin')
    print(f"Found board: {retval}")
    print(f"Rotation Vector: {rvec}")
    print(f"Translation Vector: {tvec}")