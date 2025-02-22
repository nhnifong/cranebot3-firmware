from cv_common import locate_markers
import time
from pprint import pprint
import cv2

def local_aruco_detection(outq, control_queue):
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
    capture_config = picam2.create_still_configuration(main={"size": (4608, 2592), "format": "RGB888"})
    # allow Picamera2 to choose an efficient size close to what we requested
    picam2.align_configuration(capture_config)
    picam2.configure(capture_config)
    picam2.start()
    picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.1, "AfSpeed": controls.AfSpeedEnum.Fast}) 
    send_images = False
    send_detections = False
    while True:
        if not control_queue.empty():
            message = control_queue.get_nowait()
            if message == 'STOP':
                break # exit loop, ending process
            if message.startswith('MODE'):
                m = message.split(':')
                send_images = m[1] == 'True'
                send_detections = m[2] == 'True'
        if send_images or send_detections:
            sec = time.time()
            im = picam2.capture_array()
            if send_images:
                result, encoded_img = cv2.imencode('.jpg', im)  # Encode to memory buffer
                if result:
                    outq.put({'image':{'timestamp':sec, 'data':encoded_img.tobytes()}})
                else:
                    print(f"Encoding failed with extension {ext}")
            if send_detections:
                detections = locate_markers(im)
                if len(detections) > 0:
                    for det in detections:
                        det['s'] = sec # add the time of capture to the detection
                        outq.put({'detection':det})
        else:
            time.sleep(1)
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
