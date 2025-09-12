import av
import numpy as np
import cv2

# Adjust these options for your needs.
options = {
    'rtsp_transport': 'tcp',
    'fflags': 'nobuffer',
    'flags': 'low_delay',
    'fast': '1',
}

try:
    # Open the stream with explicit FFmpeg options
    container = av.open("tcp://192.168.1.157:8888", options=options, mode='r')

    stream = next(s for s in container.streams if s.type == 'video')

    for frame in container.decode(stream):
        # The frame object is a PyAV video frame.
        # You can convert it to a NumPy array for OpenCV/NumPy-based processing.
        img = frame.to_ndarray(format='bgr24')
        
        # Now you can hand off the 'img' numpy array to your processing pipeline.
        # print("Processed a frame.")
        
        cv2.imshow('stream', img)
        if cv2.waitKey(1) == ord('q'):
             break

finally:
    if 'container' in locals():
        container.close()

    cv2.destroyAllWindows()
