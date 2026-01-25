# nf_robot

Control code for the Stringman household robotic crane from Neufangled Robotics

## [Build Guides and Documentation](https://nhnifong.github.io/neufangled-site-2/)

Purchase assembled robots or kits at [neufangled.com](https://neufangled.com)

## Installation of stringman controler

Linux (python 3.11 or later)

    sudo apt install python3-dev python3-virtualenv python3-pip ffmpeg
    python -m virtualenv venv
    pip install "nf_robot[host]"

Start headless robot controller in a mode that connects to remote telemetry.
The particular robot details will be read from/saved to bedroom.conf

    stringman-headless --config=bedroom.conf

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

Robot components that boot from the [`stringman-zero2w.img`](https://storage.googleapis.com/stringman-models/stringman-zero2w.img) (1.6GB) image should begin looking for wifi share codes with their camera immediately. You can produce a code with [qifi.org](htts://qifi.org)

Once the pi sees the code it will connect to the network and remember those settings. It should then be discoverable by the control panel via multicast DNS (Bonjour)

## Starting from a base rpi image

Alternatively the software can be set up from a fresh raspberry pi lite 64 bit image.
After booting any raspberry pi from a fresh image, perform an update

    sudo apt update -y && sudo apt full-upgrade -y -o Dpkg::Options::="--force-confold" && sudo apt install -y git python3-dev python3-virtualenv rpicam-apps i2c-tools

Clone the [cranebot-firmware](https://github.com/nhnifong/cranebot3-firmware) repo

    git clone https://github.com/nhnifong/cranebot3-firmware.git && cd cranebot3-firmware

Set the component type by uncommenting the appropriate line in server.conf

    nano server.conf

Install stringman

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

### QA scripts

Note that if you are proceeding to QA scripts right after doing the steps above you must reboot and then stop the service before running those scrips.

    sudo reboot now

log back in

    sudo systemctl stop cranebot.service

Run QA scripts for the specific component type as root

    sudo /opt/robot/env/bin/qa-anchor [power_anchor]
    sudo /opt/robot/env/bin/qa-gripper
    sudo /opt/robot/env/bin/qa-gripper-arp

These scripts both check whether everything is connected as it should be and in the case of anchors, set whether it is a power anchor or not.

## Training models

## Support this project

[Donate on Ko-fi](https://ko-fi.com/neufangled)