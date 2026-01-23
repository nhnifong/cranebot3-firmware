# nf_robot

Control code for the Stringman household robotic crane from Neufangled Robotics

## [Build Guides and Documentation](https://nhnifong.github.io/neufangled-site-2/)

Purchase assembled robots or kits at [neufangled.com](https://neufangled.com)

## Installation of stringman controler

Linux (python 3.11 or later)

    sudo apt install python3-dev python3-virtualenv python3-pip ffmpeg
    python -m virtualenv venv
    pip install "stringman[host]"

Start headless robot controller in a mode that connects to remote telemetry

    stringman-headless

For LAN mode, see [local_ui/README.md](local_ui/README.md)

## Installation of Robot Control Panel (developers)

    git clone https://github.com/nhnifong/cranebot3-firmware.git

    sudo apt install python3-dev python3-virtualenv python3-pip ffmpeg
    python -m virtualenv venv
    source venv/bin/activate
    pip install -e ".[host,dev,pi]"

### If you have an RTX 5090

    pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 torchcodec==0.6.0 --index-url https://download.pytorch.org/whl/cu129

### Run tests

    pytest tests

### Setting up a component

Robot components that boot from the `stringman-zero2w.img` image should begin looking for wifi share codes with their camera immediately. You can produce a code with [qifi.org](htts://qifi.org)

Once the pi sees the code it will connect to the network and remember those settings. It should then be discoverable by the control panel via multicast DNS (Bonjour)

## Starting from a base rpi image

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

### Additional settings for anchors

Setup for any raspberry pi that will be part of an anchor
Enable uart serial harware interface interactively.

    sudo raspi-config

In interface optoins, select serial port. disable the login shell, but enable hardware serial.

add the following lines lines to to `/boot/firmware/config.txt`  at the end this disables bluetooth, which would otherwise occupy the uart hardware.
Then reboot after this change

    enable_uart=1
    dtoverlay=disable-bt

### Additional settings for gripper

Setup for the raspberry pi in the gripper with the inventor hat mini
Enable i2c

    sudo raspi-config nonint do_i2c 0

Add this line to `/boot/firmware/config.txt` just under `dtparam=i2c_arm=on` and reboot

    dtparam=i2c_baudrate=400000

## Rebuilding the python module

within a venv install the build tools

    python3 -m pip install --upgrade build twine

Bump the version number in pyproject.toml
then at this repo's root, build the module. Artifacts will be in dist/

    python3 -m build

Upload to Pypi

    python3 -m twine upload dist/*

## Training models


## Support this project

[Donate on Ko-fi](https://ko-fi.com/neufangled)