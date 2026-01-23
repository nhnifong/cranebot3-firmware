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
echo "Installing Wifi Config from preconfigured-wifi.nmconnection"
# NetworkManager connections must be owned by root and have 600 permissions
install -m 600 -o root -g root "preconfigured.nmconnection" "$ROOTFS_DIR/etc/NetworkManager/system-connections/preconfigured.nmconnection"

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
install -m 644 cranebot.service "$ROOTFS_DIR/etc/systemd/system/cranebot.service"

# Enable the service (by creating the symlink manually or using systemctl in chroot)
run_in_chroot "systemctl enable cranebot.service"

# Install a one time filesystem resize service on first boot to expand to fill the SD card
install -m 644 resize-rootfs.sh "$ROOTFS_DIR/usr/local/sbin/resize-rootfs.sh"
install -m 644 resize-rootfs.service "$ROOTFS_DIR/etc/systemd/system/resize-rootfs.service"
run_in_chroot "systemctl enable resize-rootfs.service"

# Apply custom boot config
if [ -f "$ROOTFS_DIR/boot/firmware/config.txt" ]; then
    mv "$ROOTFS_DIR/boot/firmware/config.txt" "$ROOTFS_DIR/boot/firmware/config.txt.bak"
fi
install -m 644 config.txt "$ROOTFS_DIR/boot/firmware/config.txt"

echo "--- Stringman Component Setup Complete ---"