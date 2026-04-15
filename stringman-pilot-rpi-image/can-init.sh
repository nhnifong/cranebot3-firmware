#!/bin/bash
# /usr/local/bin/can-init.sh

/sbin/ip link set can0 type can bitrate 1000000
/sbin/ip link set can0 up
/sbin/ip link set can0 txqueuelen 65536

sleep 0.5

exit 0