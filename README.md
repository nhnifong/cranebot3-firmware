# cranebot3-firmware

Control code for a household robotic crane.

## [Build Guides and Documentation](https://nhnifong.github.io/neufangled-site-2/)

Purchase assembled robots or kits at [neufangled.com](https://neufangled.com)

## Installation of Robot Control Panel (end users)

Linux (python 3.11 or later)

    sudo apt install python3-dev python3-virtualenv python3-pip ffmpeg
    python -m virtualenv venv
    pip install "stringman[ui]"

Windows

Mac

### Run

Start control panel with ursina UI in lan mode

    python host/main.py

Or start headless in a mode that connects to remote telemetry

    python observer.py

## Installation of Robot Control Panel (developers)

    sudo apt install python3-dev python3-virtualenv python3-pip ffmpeg
    python -m virtualenv venv
    source venv/bin/activate
    pip install -e ".[ui,dev,pi]"

### Run tests

    pytest tests

## Raspberry Pi setup

`stringman-pilot-rpi-image` contains the configuration needed to build an SD card image for various stringman robot components.

Build the image with

    

After booting the robot component with the image for the first time, it will use it's camera to look for a wifi share QR code to get connected. You can produce a code with [qifi.org](htts://qifi.org)

Once the pi sees the code it will connect to the network and remember those settings. It should then be discoverable by the control panel via multicast DNS (Bonjour)

## Starting from a base image

Alternatively the software can be set up from a fresh raspberry pi lite 64 bit image.
After booting any raspberry pi from a fresh image, perform an update

    sudo apt update
    sudo apt full-upgrade

When starting with the lite raspi image, you will be missing the following, so install those.

    sudo apt install git python3-dev python3-virtualenv

Set the component type by uncommenting the appropriate line in server.conf

    nano server.conf

Install stringman

    git clone https://github.com/nhnifong/cranebot3-firmware.git && cd cranebot3-firmware
    chmod +x install.sh
    sudo ./install.sh

### additional settings for anchors

Setup for any raspberry pi that will be part of an anchor
Enable uart serial harware interface interactively.

    sudo raspi-config

In interface optoins, select serial port. disable the login shell, but enable hardware serial.

add the following lines lines to to `/boot/firmware/config.txt`  at the end this disables bluetooth, which would otherwise occupy the uart hardware.
Then reboot after this change

    enable_uart=1
    dtoverlay=disable-bt

### additional settings for gripper

Setup for the raspberry pi in the gripper with the inventor hat mini
Enable i2c

    sudo raspi-config nonint do_i2c 0

Add this line to `/boot/firmware/config.txt` just under `dtparam=i2c_arm=on` and reboot

    dtparam=i2c_baudrate=400000

## Rebuilding the raspberry pi image


## Training models


## Support this project

[Donate on Ko-fi](https://ko-fi.com/neufangled)