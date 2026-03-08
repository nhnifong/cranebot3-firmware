Damaio setup

Starting from raspberry pi lite image, set up CAN bus according to instructions from https://www.waveshare.com/wiki/RS485_CAN_HAT#CAN_Bus

    sudo nano /boot/firmware/config.txt

    dtparam=spi=on
    dtoverlay=mcp2515-can0,oscillator=12000000,interrupt=25,spimaxfrequency=2000000

    sudo reboot

reconnect with port 5000 forwarded

    ssh -L 5000:localhost:5000 pi@192.168.1.226

    sudo ip link set can0 up type can bitrate 1000000
    sudo ifconfig can0 txqueuelen 65536
    sudo ifconfig can0 up

    sudo apt-get install can-utils


create a venv and install python library from https://damiao-motor.jia-xie.com/#next-steps

    python3 -m venv venv
    source venv/bin/activate
    pip install damiao-motor

run gui

    damiao gui

In the GUI you must edit the feedback id and receive id and store the parameters to flash and then power cycle the motor for it to really take effect. 