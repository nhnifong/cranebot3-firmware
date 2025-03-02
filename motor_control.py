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

DEFAULT_MICROSTEPS = 16
# number of angle ticks returned from READ_ANGLE command per revolution
ANGLE_RESOLUTION = 65535
# A speed of 1 is this many revs/sec
SPEED1_REVS = 30000.0/(DEFAULT_MICROSTEPS * 200)/60
# The maximum positive speed in revs/ per second.
MAX_SPEED = 127 * SPEED1_REVS

class MockSerial:
    def __init__(self):
        pass
    def read(self, l):
        return b'x01'*l
    def write(self, l):
        pass

class MKSSERVO42C:
    def __init__(self):
        self.port = serial.Serial ("/dev/ttyAMA0", 38400)
        self.port.timeout = 1
        self.port.write_timeout = 1
        # self.port = MockSerial()

    def ping(self):
        """
        Return true if the motor is awake and enabled
        """
        self._sendSingleByteCommand(PING)
        ans = self.port.read(3)
        return len(ans) == 3 and ans[1] == 1

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
        Command the motor to run at a constant speed in revolutions per second
        Return true if the motor replied status ok
        """

        # convert revs/sec to valid speed range (-127 +127)
        command_speed = max(-127, min(int(speed / SPEED1_REVS), 127))

        # the first bit is direction
        if command_speed > 0:
            first_bit = 128 # (line lengthening, top of spool moves towards the wall)
        else:
            first_bit = 0 # (line shortening, top of spool moves away from the wall)

        # the next 7 bits are speed
        combined = (min(abs(command_speed), 127) + first_bit).to_bytes(1, byteorder='big')

        message = b'\xe0\xf6' + combined
        message += self._calculateChecksum(message)
        self.port.write(message)
        ans = self.port.read(3)
        return len(ans) == 3 and ans[1] == b'\x01'

    def getShaftAngle(self):
        """
        Get the absolute shaft angle since boot in revolutions as a double precision float
        return (status, result)

        When spinning with negative speeds passed to runConstantSpeed, the angle reported by getMotorShaftAngle is decreasing.
        """
        self._sendSingleByteCommand(READ_ANGLE)
        ans = self.port.read(6) # address byte, 32 bit integer, checksum byte
        if len(ans) != 6:
            return False, 0
        motor_angle = int.from_bytes(ans[1:5], byteorder='big', signed=False)
        return True, float(motor_angle) / ANGLE_RESOLUTION

    def _sendSingleByteCommand(self, b):
        """
        b must be a bytes object with a length of 1
        """
        message = b'\xe0' + b
        message += self._calculateChecksum(message)
        self.port.write(message)
        self.port.flush()

    def _calculateChecksum(self, message):
        """
        the last (least signifigant) byte in the sum of all the bytes in the message
        """
        return (sum(message) & 255).to_bytes(1, byteorder='big')

    def getMaxSpeed(self):
        return MAX_SPEED

if __name__ == "__main__":
    motor = MKSSERVO42C()
    assert(motor.ping())
    for i in range(25):
        motor.runConstantSpeed(i)
        sleep(0.1)
    for i in range(25, 1, -1):
        motor.runConstantSpeed(-i)
        sleep(0.1)
    motor.stop()
