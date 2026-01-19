#!/bin/bash
set -e

echo "--- Starting Stringman Component Setup ---"

# Create directory structure
mkdir -p /opt/robot

# Create Virtual Environment
# We create it with --system-site-packages so it can see python3-picamera2 
# (which is an apt package, difficult to compile via pip on Pi)
python3 -m venv --system-site-packages /opt/robot/env

# Install pip packages
# We use the venv's pip binary
/opt/robot/env/bin/pip install --upgrade pip
/opt/robot/env/bin/pip install stringman

# Install Systemd Service
# The 'files' directory in the layer is accessible relative to the script location 
# during build, or we copy from the mounted source.
# In rpi-image-gen, we usually copy files manually or use a specific file directive.
# Here we assume the tool mounts the layer source at a known path or we move files via the script.

# Copy service file (assuming files are available in context)
install -m 644 files/robot.service /etc/systemd/system/cranebot.service

# Enable the service
systemctl enable cranebot.service

# Apply custom boot config to disable login shell on uart line, enable hardware uart, and disable bluetooth.
# We backup the original and overwrite with ours
if [ -f /boot/firmware/config.txt ]; then
    mv /boot/firmware/config.txt /boot/firmware/config.txt.bak
fi
install -m 644 files/config.txt /boot/firmware/config.txt

echo "--- Stringman Component Setup Complete ---"