from cv_common import locate_markers

def local_aruco_detection(outq, control_queue):
    """
    Open the camera and detect aruco markers. put any detections on the provided queue
    TODO this seems to chew up pretty much all the resources we have.
    consider cropping the image to the area we beleive there to be a marker in.
    """
    print("PiCamera detection process started")
    from picamera2 import Picamera2
    from libcamera import Transform
    picam2 = Picamera2()
    # pprint(picam2.sensor_modes) # investigate modes with cropped FOV
    # full res is 4608x2592
    # this is half res. seems it can still detect a 10cm aruco from about 2 meters at a rate of 30fps
    # you can flip the image with transform=Transform(hflip=1, vflip=1)
    capture_config = picam2.create_preview_configuration(main={"size": (2304, 1296), "format": "RGB888"})
    # allow Picamera2 to choose an efficient size close to what we requested
    picam2.align_configuration(capture_config)
    picam2.configure(capture_config)
    picam2.start()
    while True:
        if not control_queue.empty():
            if control_queue.get_nowait() == "STOP":
                break # exit loop, ending process

        sec = time.time()
        im = picam2.capture_array()
        detections = locate_markers(im)
        if len(detections) > 0:
            for det in detections:
                det['s'] = sec # add the time of capture to the detection
                outq.put(det)
    print("PiCamera detection process ended")


def dummyProcess(outq, control_queue):
    """
    A process that can act as a stand in for local_aruco_detection in situations where picamera is not installed/working
    """
    print("dummy process started")
    while True:
        if not control_queue.empty():
            if control_queue.get_nowait() == "STOP":
                break # exit loop, ending process
    print("dummy process ended")