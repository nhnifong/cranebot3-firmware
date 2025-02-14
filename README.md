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

for either the gripper or anchor server
add the following lines lines to to `/boot/firmware/config.txt` and reboot)

    enable_uart=1
    dtoverlay=disable-bt

Since we require the picamera2 module and it can't be installed with pip, you have to create a virtualenv that can use site packages 

    sudo apt install python3-picamera2 imx500-all --no-install-recommends
    python3 -m venv --system-site-packages venv
    source venv/bin/activate
    pip3 install -r requirements_raspi.txt

Start server

    python3 anchor/anchor_server.py

OR

    python3 anchor/gripper_server.py




