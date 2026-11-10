#!/bin/bash
# /usr/local/bin/can-init.sh

/sbin/ip link set can0 type can bitrate 1000000
/sbin/ip link set can0 up
# Kernel 6.18 rejects values above 32767 ("integer out of range"), leaving the default of 10,
# which a motor scan with nothing ACKing overflows (ENOBUFS).
/sbin/ip link set can0 txqueuelen 10000

sleep 0.5

exit 0