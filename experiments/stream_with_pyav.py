# stream with av but keep the compressed frame around

import av
import numpy as np
import time
import cv2
import sys

options = {
    'rtsp_transport': 'tcp',
    'fflags': 'nobuffer',
    'flags': 'low_delay',
    'fast': '1',
}

# container = av.open(f"tcp://192.168.1.{sys.argv[1]}:8888", options=options, mode='r')
container = av.open(f"udp://127.0.0.1:36182", options=options, mode='r')
stream = next(s for s in container.streams if s.type == 'video')
stream.thread_type = "SLICE"
fnum = 59 
lt = time.time()


for packet in container.demux(stream):
    for frame in packet.decode():
        # This is your raw, uncompressed image for Apriltag detection
        print(f'pts={frame.pts}')
        print(f'dts={frame.dts}')
        print(f'time={frame.time}')
        print(f'time_base={frame.time_base}')
        raw_image_ndarray = frame.to_ndarray(format='bgr24')
        
        now = time.time()
        fr = 1/(now-lt)
        lt = now

# /usr/bin/rpicam-vid -t 0 \
#   --width=1920 --height=1080 \
#   --listen -o tcp://0.0.0.0:8888 \
#   --codec h264 \
#   --vflip --hflip \
#   --autofocus-mode continuous