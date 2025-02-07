#!/bin/bash
rpicam-vid -t 0 --width=2304 --height=1296 --inline --listen -o tcp://0.0.0.0:8888 --codec mjpeg
