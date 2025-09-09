import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cv_common import locate_markers

def local_aruco_detection():
    """
    Open the camera and detect aruco markers. put any detections on the provided queue
    TODO this seems to chew up pretty much all the resources we have.
    consider cropping the image to the area we beleive there to be a marker in.
    """
    print("PiCamera detection process started")
    from picamera2 import Picamera2
    from libcamera import Transform, controls
    picam2 = Picamera2()
    pprint(picam2.sensor_modes) # investigate modes with cropped FOV
    # full res is 4608x2592
    # you can flip the image with transform=Transform(hflip=1, vflip=1)
    # capture_config = picam2.create_preview_configuration(main={"size": (2304, 1296), "format": "RGB888"})
    # running at full resolution takes pretty much all the RAM on the raspi zero even with just 1 framebuffer.
    # nearly every other process will get swapped out.
    capture_config = picam2.create_still_configuration(main={"size": (1920, 1080), "format": "RGB888"})
    # allow Picamera2 to choose an efficient size close to what we requested
    picam2.align_configuration(capture_config)
    picam2.configure(capture_config)
    picam2.start()
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "AfSpeed": controls.AfSpeedEnum.Fast}) 

    while True:
        im = picam2.capture_array()
        detections = locate_markers(im)
        if len(detections) > 0:
            for det in detections:
                print(det)
    print("PiCamera detection process ended")

local_aruco_detection()