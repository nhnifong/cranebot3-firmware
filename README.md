# cranebot3-firmware
Control ode for a crane mounted household robotic gripper

gripper-sketch/ contains arduino ide project for flashing the gripper firmware.

required libraries:

 - AXP313A - https://github.com/cdjq/DFRobot_AXP313A
 - ESP32_New_ISR_Servo - https://github.com/khoih-prog/ESP32_New_ISR_Servo
 - DFRobot_BNO055 - https://github.com/DFRobot/DFRobot_BNO055
 - DFRobot_BMP280 - https://github.com/DFRobot/DFRobot_BMP280


Follow the Arduino IDE configuration steps for the DF Robot Firebeetle 2

  https://wiki.dfrobot.com/SKU_DFR0975_FireBeetle_2_Board_ESP32_S3#target_6

OR

  In case that link rots, install the esp32 board using this manager url https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
  
  Set the sketch board type to "DF Robot Firebeetle 2 ESP32-S3"
  
  Select the following settings from tools menu before burning
	
 - USB CDC on boot "Enabled"
 - Flash Size: "16MB (128Mb)"
 - Partition Scheme "16M Flash (3MB APP/9.9MB FATFS)"
 - PSRAM "OPI PSRAM"

host/ contains the desktop server which connects to the ESP32s over wifi and perform image analysis and control.

currently just a buch of broken prototype code.

    sudp apt install python3-pip python3-virtualenv
    python3 -m virtualenv venv
    source venv/bin/activate
    pip3 install -r requirements.txt

main.py is a graphical control panel made with ursina (a python game engine)
in it's current form, it requires a blender installation in order to load assets

    sudo apt-get install blender

raspi-anchor contains a server that is meant to run on the raspberry pi zero 2w inside the anchors
that are mounted to the walls.

add the following lines lines to to /boot/firmware/config.txt and reboot)

    enable_uart=1
    dtoverlay=disable-bt

It has it's own `requirements.txt` file.
after installing the requirements in a python3 virtual env on the raspi, run with

    python3 anchor_server.py