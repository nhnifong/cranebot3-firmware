# stream with av but keep the compressed frame around

import av
import numpy as np
import time
import cv2

options = {
    'rtsp_transport': 'tcp',
    'fflags': 'nobuffer',
    'flags': 'low_delay',
    'fast': '1',
}

container = av.open("tcp://192.168.1.157:8888", options=options, mode='r')
stream = next(s for s in container.streams if s.type == 'video')
stream.thread_type = "SLICE"
fnum = 59 
lt = time.time()


for packet in container.demux(stream):
    # This is your compressed JPEG data. It's a very cheap operation.
    jpeg_byte_data = bytes(packet)

    # sanity check that it's a real image
    foo = cv2.imdecode(np.frombuffer(jpeg_byte_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert foo is not None # this is asserting. Of course it is, the stream is h264 doh

    for frame in packet.decode():
        # This is your raw, uncompressed image for Apriltag detection
        raw_image_ndarray = frame.to_ndarray(format='bgr24')
        
        now = time.time()
        fr = 1/(now-lt)
        print(f'compressed {len(jpeg_byte_data)} b decompressed {raw_image_ndarray.shape} framerate {fr}')
        lt = now

# /usr/bin/rpicam-vid -t 0 \
#   --width=1920 --height=1080 \
#   --listen -o tcp://0.0.0.0:8888 \
#   --codec h264 \
#   --vflip --hflip \
#   --autofocus-mode continuous