# control the MKS_SERVO42C stepper with the serial port (UART)
# i had to run sudo raspi-config and enable the Serial hardware
# need to find out out to do that non interactively

import serial
from time import sleep

ser = serial.Serial ("/dev/ttyS0", 38400)    #Open port with baud rate

# ping motor
ser.write(b"\0xe0\0x3a\0x1a")
# expect e0 01 e1 meaning it is enabled
b = ser.read(3)
print(repr(b))
