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

# Give pi permission to run this service
run_in_chroot "chown -R pi:pi /opt/robot"
run_in_chroot "usermod -aG netdev pi"

# Create Polkit rule to allow 'netdev' group to manage NetworkManager
# This, combined with libpam-systemd, fixes the "property missing" error for non-root users.
mkdir -p "$ROOTFS_DIR/etc/polkit-1/rules.d"
cat <<EOF > "$ROOTFS_DIR/etc/polkit-1/rules.d/50-allow-netdev.rules"
polkit.addRule(function(action, subject) {
  if (action.id.indexOf("org.freedesktop.NetworkManager.") == 0 && subject.isInGroup("netdev")) {
    return polkit.Result.YES;
  }
});
EOF

# set up i2c Kernel Module
# We must load 'i2c-dev' to create the /dev/i2c-1 character device which is what raspi-config would do if you ran it interactively
# Without this, i2cdetect and python libraries cannot see the bus.
if [ ! -f "$ROOTFS_DIR/etc/modules" ]; then
    touch "$ROOTFS_DIR/etc/modules"
fi

if ! grep -q "i2c-dev" "$ROOTFS_DIR/etc/modules"; then
    echo "i2c-dev" >> "$ROOTFS_DIR/etc/modules"
fi

# Create missing udev rule for Camera DMA Heaps ---
# Minimal images lack the rule that lets 'video' group access /dev/dma_heap/*
# This fixes "Could not open any dma-buf provider" when running as non-root.
echo "Creating udev rules for DMA Heap access..."
mkdir -p "$ROOTFS_DIR/etc/udev/rules.d"
echo 'SUBSYSTEM=="dma_heap", GROUP="video", MODE="0660"' > "$ROOTFS_DIR/etc/udev/rules.d/99-camera-perms.rules"

# GPIO Access (Fixes "No access to /dev/mem" by enabling /dev/gpiomem for gpio group)
echo 'KERNEL=="gpiomem", GROUP="gpio", MODE="0660"' > "$ROOTFS_DIR/etc/udev/rules.d/99-gpio.rules"

# I2C Access (Fixes access to /dev/i2c-* for 'i2c' group)
echo 'KERNEL=="i2c-[0-9]*", GROUP="i2c", MODE="0660"' > "$ROOTFS_DIR/etc/udev/rules.d/99-i2c.rules"

# Install Systemd Service
install -m 644 cranebot.service "$ROOTFS_DIR/etc/systemd/system/cranebot.service"

# Enable the service (by creating the symlink manually or using systemctl in chroot)
run_in_chroot "systemctl enable cranebot.service"

# Install a one time filesystem resize service on first boot to expand to fill the SD card
install -m 755 resize-rootfs.sh "$ROOTFS_DIR/usr/local/sbin/resize-rootfs.sh"
install -m 644 resize-rootfs.service "$ROOTFS_DIR/etc/systemd/system/resize-rootfs.service"
run_in_chroot "systemctl enable resize-rootfs.service"

# Apply custom boot config
if [ -f "$ROOTFS_DIR/boot/firmware/config.txt" ]; then
    mv "$ROOTFS_DIR/boot/firmware/config.txt" "$ROOTFS_DIR/boot/firmware/config.txt.bak"
fi
install -m 644 config.txt "$ROOTFS_DIR/boot/firmware/config.txt"

# Disable Serial Console (UART Login) ---
CMDLINE="$ROOTFS_DIR/boot/firmware/cmdline.txt"

if [ -f "$CMDLINE" ]; then
    echo "Disabling serial console in cmdline.txt..."
    # Remove console=serial0,115200 (or ttyAMA0/ttyS0) to stop kernel messages going to UART
    # We match console= followed by serial0, ttyAMA0, or ttyS0, followed by baud rate
    sed -i -E 's/console=(serial0|ttyAMA0|ttyS0),[0-9]+ //g' "$CMDLINE"
    sed -i -E 's/console=(serial0|ttyAMA0|ttyS0),[0-9]+//g' "$CMDLINE"
else
    echo "Warning: $CMDLINE not found. Could not disable serial console kernel args."
fi

# Mask the systemd services that spawn login prompts on UART
# We mask all common Raspberry Pi UART identifiers to be sure.
echo "Masking serial getty services..."
run_in_chroot "systemctl mask serial-getty@ttyAMA0.service serial-getty@ttyS0.service serial-getty@serial0.service"

echo "Setting Wi-Fi Country Code..."
run_in_chroot "raspi-config nonint do_wifi_country US"

echo "--- Stringman Component Setup Complete ---"