#!/bin/bash

# This script generates a NetworkManager connection file for a new Wi-Fi network.
# It requires two arguments: SSID and Password.
# It must be run with sudo to write to /etc/ and set permissions.
# The microSD card of the raspi image must be inserted and it's roofs partition mounted at /media/you/rootfs 

# --- Configuration ---
FILE_NAME="end-user-wlan.nmconnection"
CONFIG_DIR="/media/$SUDO_USER/rootfs/etc/NetworkManager/system-connections"
FILE_PATH="$CONFIG_DIR/$FILE_NAME"
CONNECTION_ID="end-user-wlan"

# --- Argument Validation ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 \"<SSID>\" \"<Password>\""
    echo "Example: $0 \"My WiFi Name\" \"MyP@ssword!\""
    echo "Note: Remember to quote your arguments!"
    exit 1
fi

SSID=$1
PASSWORD=$2

# --- Prerequisite Checks ---
echo "Checking prerequisites..."

# 1. Check for root/sudo
if [ "$EUID" -ne 0 ]; then
  echo "Error: This script must be run as root (use sudo)."
  exit 1
fi

# 2. Check for uuidgen command
if ! command -v uuidgen &> /dev/null; then
    echo "Error: 'uuidgen' command not found. Please install the 'uuid-runtime' package."
    exit 1
fi

# 3. Check if target directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Target directory does not exist: $CONFIG_DIR"
    echo "Is your SD card's 'rootfs' partition mounted at /media/nhn/rootfs?"
    exit 1
fi

# --- File Generation ---
echo "Generating new UUID..."
NEW_UUID=$(uuidgen)

echo "Writing connection file to $FILE_PATH..."

# Use a Here-Document (cat << EOF) to write the file content.
cat << EOF > "$FILE_PATH"
[connection]
id=$CONNECTION_ID
uuid=$NEW_UUID
type=wifi
[wifi]
mode=infrastructure
ssid=$SSID
hidden=false
[ipv4]
method=auto
[ipv6]
addr-gen-mode=default
method=auto
[proxy]
[wifi-security]
key-mgmt=wpa-psk
psk=$PASSWORD
EOF

# --- Set Permissions ---
echo "Setting permissions to 600..."
chmod 600 "$FILE_PATH"

echo "---"
echo "Success! File created:"
echo "$FILE_PATH"
echo ""
echo "ID:     $CONNECTION_ID"
echo "SSID:   $SSID"
echo "UUID:   $NEW_UUID"