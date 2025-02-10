import cv2
import cv2.aruco as aruco
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

camera = PiCamera()
rawCapture = PiRGBArray(camera)

# allow the camera to warmup
time.sleep(0.1)

# grab an image from the camera
camera.capture(rawCapture, format="bgr")
image = rawCapture.array

# display the image on screen and wait for a keypress
print(image.shape)