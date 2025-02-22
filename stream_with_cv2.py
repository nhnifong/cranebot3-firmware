import cv2
from cv_common import locate_markers
from multiprocessing import Pool, TimeoutError, Queue

# video streamed with the command from start_stream.sh
video_path = 'tcp://192.168.1.151:8888'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

queue = Queue()
with Pool(processes=6) as pool:
    while True:
        try:
            ret, frame = cap.read()
            if ret:
                async_result = pool.apply_async(locate_markers, (frame,), callback=print)
        except KeyboardInterrupt:
            break

# there is no information in the stream about the capture time of the frames.
# but we can get the frame number with this attribute,
# fnum = cap.get(cv2.CAP_PROP_POS_FRAMES)
# and if the host is watching the stdout of the rpicam-vid
# then it will see a line like #63 (7.18 fps) exp 26632.00 ag 1.12 dg 1.00
# every time it captures a frame, and it could recording the wall time for every line,
# and send it over the seperate websocket connection.
# the mean delay is roughly 0.7 seconds.