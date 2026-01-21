#!/bin/bash
set -e

# The first argument passed by mmdebstrap is the path to the rootfs
ROOTFS_DIR="$1"

if [ -z "$ROOTFS_DIR" ]; then
    echo "Error: ROOTFS_DIR not provided. This script must be run as a hook."
    exit 1
fi

echo "--- Starting Stringman Component Setup on $ROOTFS_DIR ---"

# Helper function to run commands inside the image
run_in_chroot() {
    chroot "$ROOTFS_DIR" /bin/bash -c "$1"
}

# Install NetworkManager Wifi Connection
echo "Installing Wifi Config from $FILES_DIR/default-wifi-connection.nmconnection"
# NetworkManager connections must be owned by root and have 600 permissions
install -m 600 -o root -g root "$FILES_DIR/default-wifi-connection.nmconnection" "$ROOTFS_DIR/etc/NetworkManager/system-connections/preconfigured-wifi.nmconnection"

# Create directory structure
# (We use mkdir on the host, targeting the directory inside the rootfs)
mkdir -p "$ROOTFS_DIR/opt/robot"

# Create Virtual Environment
# We run python INSIDE the image to create the venv
run_in_chroot "python3 -m venv --system-site-packages /opt/robot/env"

# Install pip packages
run_in_chroot "/opt/robot/env/bin/pip install --upgrade pip"
run_in_chroot "/opt/robot/env/bin/pip install \"nf_robot[pi]\""

# Install Systemd Service
# We copy from our layer files (on host) to the rootfs (on host)
# Note: 'files/' is relative to where the script is run from (the layer dir usually)
install -m 644 files/cranebot.service "$ROOTFS_DIR/etc/systemd/system/cranebot.service"

# Enable the service (by creating the symlink manually or using systemctl in chroot)
run_in_chroot "systemctl enable cranebot.service"

# Apply custom boot config
if [ -f "$ROOTFS_DIR/boot/firmware/config.txt" ]; then
    mv "$ROOTFS_DIR/boot/firmware/config.txt" "$ROOTFS_DIR/boot/firmware/config.txt.bak"
fi
install -m 644 files/config.txt "$ROOTFS_DIR/boot/firmware/config.txt"

echo "--- Stringman Component Setup Complete ---"