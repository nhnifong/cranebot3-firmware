# control the MKS_SERVO42C stepper with the serial port (UART)
# the bluetooth has to be disabled on the raspi zero 2w
# requires the following lines to be written to /boot/firmware/config.txt (and reboot)
# enable_uart=1
# dtoverlay=disable-bt

import serial
from time import sleep

ser = serial.Serial ("/dev/ttyAMA0", 38400)    #Open port with baud rate

# ping motor
ping = b"\xe0\x3a\x1a"
print(ping)
ser.write(ping)
ser.flush()
# expect e0 01 e1 meaning it is enabled
b = ser.read(3)
print(repr(b))
