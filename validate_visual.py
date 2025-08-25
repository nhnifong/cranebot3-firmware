import cv2
import numpy as np
from cv_common import locate_markers


cap = cv2.VideoCapture("tcp:192.168.1.151:8888")
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
                if detection['n'] == 'gantry_front':
                    distance = detection["t"][2][0]
                    avg = avg*0.9 + distance*0.1
                    print(f'gantry {avg}')


"""
/usr/bin/rpicam-vid -t 0 --width=4608 --height=2592 --listen -o tcp://0.0.0.0:8888 --codec mjpeg --vflip --hflip --buffer-count=3 --autofocus-mode continuous


obtained result from find_cal_params (array([[[-0.03700344,  0.33014234,  2.39665213],
        [-2.56569561, -2.36662633,  2.6380979 ]],

       [[ 0.15650809,  0.01848311,  0.88090839],
        [-3.22272033,  2.38765039,  2.6380979 ]],

       [[ 0.14986345, -0.19392029, -2.18131932],
        [ 2.57445305, -2.30818949,  2.6380979 ]],

       [[ 0.21895104, -0.05669679, -0.71766575],
        [ 2.11099143,  2.88491576,  2.6380979 ]]]), array([ -96.43921262,  -91.97409759, -114.77050435, -103.90828597]))
Task exception was never retrieved
future: <Task finished name='Task-30' coro=<AsyncObserver.full_auto_calibration() done, defined at /home/nhn/cranebot3-firmware/observer.py:308> exception=ValueError('cannot reshape array of size 24 into shape (2,3)')>
Traceback (most recent call last):
  File "/home/nhn/cranebot3-firmware/observer.py", line 398, in full_auto_calibration
    pose = result_params[i][0:5].reshape((2,3))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: cannot reshape array of size 24 into shape (2,3)



"""