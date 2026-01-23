#!/bin/bash
set -e
# Grow partition 2 (rootfs) on mmcblk0
# returns 0 if changed, 1 if no change needed (we accept both)
growpart /dev/mmcblk0 2 || true
# Resize filesystem online
resize2fs /dev/mmcblk0p2
# Disable this service so it doesn't run again
systemctl disable resize-rootfs.service