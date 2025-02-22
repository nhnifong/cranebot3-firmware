#!/bin/bash
# full resolution
# encode each frame seperately as a jpg
# the camera is installed upside down, flip 180
# 5 buffers is the most we have space for at full resolution. this is the limiting factor to framerate.
# somtimes, there is room for 6 buffers, but if there isn't, the stream wont start.
# autofocus mode manual at position 0.1 means use a focal distance of 10 meters.
rpicam-vid -t 0 \
  --width=4608 --height=2592 \
  --listen -o tcp://0.0.0.0:8888 \
  --codec mjpeg \
  --vflip --hflip \
  --buffer-count=6 \
  --autofocus-mode manual --lens-position 0.1