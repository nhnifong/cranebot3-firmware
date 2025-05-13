#!/bin/bash

APP_DIR="/home/$SUDO_USER/cranebot3-firmware"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="cranebot.service"
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME"

# Check for root privileges
if [[ $EUID -ne 0 ]]; then
  echo "This script must be run as root."
  exit 1
fi

apt install python3-picamera2 --no-install-recommends
# apt install imx500-all

if [[ ! -d "venv" ]]; then
  python3.11 -m venv --system-site-packages venv
fi
source venv/bin/activate
pip3 install -r requirements_raspi.txt
deactivate

# Create service file
cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=Cranebot Service
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=1
User=$SUDO_USER
WorkingDirectory=$APP_DIR
ExecStart=$VENV_DIR/bin/python3 component_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Make service file executable
chmod 644 "$SERVICE_FILE"

# Enable and start the service
systemctl enable "$SERVICE_NAME"
systemctl start "$SERVICE_NAME"

echo "Cranebot service installed"
systemctl status --no-pager "$SERVICE_NAME"
