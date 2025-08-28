import cv2
import numpy as np
from cv_common import locate_markers


cap = cv2.VideoCapture("tcp:192.168.1.153:8888")
if not cap.isOpened():
    print('no video stream available')
else:
    avg = 2.5
    while True:
        # blocks until a frame can be read, then decodes it. 
        ret, frame = cap.read()
        if ret:
            result = locate_markers(frame)
            for detection in result:
                print(f'{detection["n"]} distance {detection["t"][2][0]}')


"""
/usr/bin/rpicam-vid -t 0 --width=4608 --height=2592 --listen -o tcp://0.0.0.0:8888 --codec mjpeg --vflip --hflip --buffer-count=3 --autofocus-mode continuous


Provisional anchor points relative to origin card
[[-2.85086033 -2.85631199  2.43513633]
 [-3.29916513  2.30460651  2.47757824]
 [ 2.35525042 -2.54746161  2.53668309]
 [ 1.89739506  2.71192721  2.52582104]]


"""