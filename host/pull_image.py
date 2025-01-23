import urllib.request
import numpy as np
import cv2
from time import time

# set to highest resolution
urllib.request.urlopen("http://192.168.1.146/control?var=framesize&val=13")
capture_url = "http://192.168.1.146/capture?_cb={}"
req = urllib.request.urlopen(capture_url.format(time()*1000))
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, -1)

cv2.imshow('random_title', img)
cv2.waitKey(0)
cv2.destroyAllWindows()