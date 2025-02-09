# control the MKS_SERVO42C stepper with the serial port (UART)
# the bluetooth has to be disabled on the raspi zero 2w
# requires the following lines to be written to /boot/firmware/config.txt (and reboot)
# enable_uart=1
# dtoverlay=disable-bt

import serial # note this is pyserial not the module named serial in pip
from time import sleep

PING = b'\x3a'
STOP = b'\xf7'
READ_ANGLE = b'\x36'

class MKSSERVO42C:
    def __init__(self):
        self.port = serial.Serial ("/dev/ttyAMA0", 38400)

    def ping(self):
        """
        Return true if the motor is awake and enabled
        """
        self._sendSingleByteCommand(PING)
        ans = self.port.read(3)
        return len(ans) == 3 and ans[1] == b'\x01'

    def stop(self):
        """
        Command the motor to stop immediately.
        Return true if the motor replied status ok
        """
        self._sendSingleByteCommand(STOP)
        ans = self.port.read(3)
        return len(ans) == 3 and ans[1] == b'\x01'

    def runConstantSpeed(self, speed):
        """
        Command the motor to run at a constant speed between -127 and +127
        Return true if the motor replied status ok
        """

        # the first bit is direction
        if speed > 0:
            first_bit = 128 # (line lengthening, top of spool moves towards the wall)
        else:
            first_bit = 0 # (line shortening, top of spool moves away from the wall)

        # the next 7 bits are speed
        combined = (min(speed, 127) + first_bit).to_bytes()

        message = b'\xe0\xf6' + combined
        message += self._calculateChecksum(message)
        self.port.write(message)
        ans = self.port.read(3)
        return len(ans) == 3 and ans[1] == b'\x01'

    def getShaftAngle(self):
        """
        Get the absolute shaft angle since boot
        return (status, result)
        """
        self._sendSingleByteCommand(READ_ANGLE)
        ans = self.port.read(6) # address byte, 32 bit integer, checksum byte
        if len(ans) != 6:
            return False, 0
        motor_angle = int.from_bytes(ans[1:5], byteorder='big', signed=False)
        # in the 0 direction, the angle reported by getMotorShaftAngle is decreasing.
        return True, motor_angle

    def _sendSingleByteCommand(self, b):
        """
        b must be a bytes object with a length of 1
        """
        message = b'\xe0' + b
        message += self._calculateChecksum(message)
        self.port.write(message)

    def _calculateChecksum(self, message):
        """
        the last (least signifigant) byte in the sum of all the bytes in the message
        """
        return (sum(message) & 255).to_bytes()

if __name__ == "__main__":
    motor = MKSSERVO42C()
    motor.runConstantSpeed(10)
    sleep(1)
    motor.stop()
