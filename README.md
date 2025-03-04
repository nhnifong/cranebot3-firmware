# cranebot3-firmware

Control code for a crane mounted household robotic crane consisting of a gripper than hangs from multiple lines
attached to spools in AI-camera equipped anchor points on the walls of a room.

## Desktop setup

    sudo apt install python3-pip python3-virtualenv
    python3 -m virtualenv venv
    source venv/bin/activate
    pip3 install -r requirements_desktop.txt

Start control panel with UI

    python3 host/main.py


requirements_desktop.txt includes the game engine, while requirements_raspi.txt is more lightweight and includes only the dependencies of the headless servers that run on the raspberry pi zeros

main.py is a graphical control panel made with ursina (a python game engine)
in it's current form, it requires a blender installation in order to load assets

    sudo apt-get install blender

## Raspberry Pi setup

After booting any raspberry pi from a fresh image, perform an update

    sudo apt update
    sudo apt full-upgrade

you may have to hit enter a few times during full-upgrade.

Since we require the picamera2 module and it can't be installed with pip, you have to install it with apt and create a virtualenv that can use site packages 

    sudo apt install python3-picamera2 imx500-all --no-install-recommends
    git clone https://github.com/nhnifong/cranebot3-firmware.git
    cd cranebot3-firmware
    python3 -m venv --system-site-packages venv
    source venv/bin/activate
    pip3 install -r requirements_raspi.txt

### Anchors

Setup for any raspberry pi that will be part of an anchor
Enable uart serial harware interface

    sudo raspi-config nonint do_serial_hw 0

add the following lines lines to to `/boot/firmware/config.txt`  at the end this disables bluetooth, which would otherwise occupy the uart hardware.
Then reboot after this change

    enable_uart=1
    dtoverlay=disable-bt

Start server

    python3 anchor/anchor_server.py

### Gripper

Setup for the raspberry pi in the gripper with the inventor hat mini
Enable i2c

    sudo raspi-config nonint do_i2c 0

Add this line to `/boot/firmware/config.txt` just under `dtparam=i2c_arm=on` and reboot

    dtparam=i2c_baudrate=400000

Start server

    python3 anchor/gripper_server.py




