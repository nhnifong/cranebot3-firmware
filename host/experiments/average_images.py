import cv2
import numpy as np
from time import time, sleep
import urllib.request
from skimage import data, img_as_float
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform

def capture_OV2640():
    capture_url = "http://192.168.1.147/capture?_cb={}"
    req = urllib.request.urlopen(capture_url.format(time()*1000))
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    return cv2.imdecode(arr, -1)

# Load the first image as the reference
first = capture_OV2640()
reference_image = img_as_float(first)
aligned_images = [reference_image]

for i in range(5):
    # Load the current image
    current_image = img_as_float(capture_OV2640())

    # Calculate the shift using phase cross-correlation
    shift, error, diffphase = phase_cross_correlation(reference_image, current_image)

    # Create an inverse transformation matrix for translation
    tf = AffineTransform(translation=-shift) # Negative shift because it's an *inverse* map

    # Apply the transformation using inverse_map
    aligned_image = warp(current_image, inverse_map=tf, output_shape = reference_image.shape)

    print(i)
    aligned_images.append(aligned_image)
    sleep(2)

# Average the aligned images
averaged_image = np.mean(np.stack(aligned_images), axis=0)

cv2.imshow("first", first)
cv2.imshow("better", averaged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
vfc
vfc
vfcv
class