from motor_control import MKSSERVO42C
from math import pi
import asyncio
import time

DEFAULT_MICROSTEPS = 16
ANGLE_RESOLUTION = 65535
# A speed of 1 is this many revs/sec
SPEED1_REVS = 30000.0/(DEFAULT_MICROSTEPS * 200)/60;
SPOOL_DIAMETER_MM = 24
METER_PER_REV = SPOOL_DIAMETER_MM * pi * 0.001;
DATA_LEN = 1000
PE_TERM = 1.5
# maximum acceleration in meters of line per second squared
MAX_ACCEL = 8.0
LOOP_DELAY_S = 0.03

def constrain(value, minimum, maximum):
    return max(minimum, min(value, maximum))

class SpoolController:
    def __init__(self):
        self.motor = MKSSERVO42C()
        self.speed = 0
        # Meters of line that were spooled out when zeroAngle was set.
        self.lineAtStart = 1.9
        self.zeroAngle = 0
        # last set speed in motor units (-127 to 127)
        self.motor_speed = 0
        # record of line length. tuples of (time, meters)
        self.record = []
        # plan of desired line lengths
        self.desiredLine = []
        self.lastIndex = 0

    def setReferenceLength(self, length):
        """
        Provide an external observation of the current line length
        """
        self.lineAtStart = length
        self.zeroAngle = self.motor.getShaftAngle()

    def setPlan(self, plan):
        """
        set plan to an array of tuples of time and length
        """
        self.desiredLine = plan
        self.lastIndex = 0

    def popMeasurements(self):
        copy_record = self.record
        self.record = []
        return copy_record

    def currentLineLength(self):
        """
        return the curren time and current unspooled line in meters
        """
        success, angle = self.motor.getShaftAngle()
        l = METER_PER_REV * (float(angle) - self.zeroAngle) / ANGLE_RESOLUTION + self.lineAtStart
        # accumulate these so you can send them to the websocket
        row = (time.time(), l)
        self.record.append(row)
        return row

    def slowStop(self):
        direction = self.speed / self.speed
        while abs(self.speed) > 0:
            self.speed -= direction
            self.motor.runConstantSpeed(self.speed)
            time.sleep(0.05)
        self.motor.stop()

    def trackingLoop(self):
        """
        Constantly try to match the position and speed given in an array
        """
        while True:
            t, currentLen = self.currentLineLength()
            if len(self.desiredLine) == 0:
                if self.speed != 0:
                    self.slowStop()
                continue

            # Find the earliest entry in desiredLine that is still in the future.
            while self.desiredLine[self.lastIndex][0] <= t:
                self.lastIndex += 1

            if self.lastIndex >= DATA_LEN:
                if self.speed != 0:
                    self.slowStop()
                continue

            targetLen = self.desiredLine[self.lastIndex][1]
            position_err = targetLen - currentLen

            # What would the speed be between the two datapoints tha straddle the present?
            # Positive values mean line is lengthening
            # result is in meters of line per second
            targetSpeed = ((self.desiredLine[self.lastIdx][1] - self.desiredLine[self.lastIdx-1][1])
                / (self.desiredLine[self.lastIdx][0] - self.desiredLine[self.lastIdx-1][0]))
            currentSpeed = self.motor_speed * SPEED1_REVS * METER_PER_REV
            speed_err = targetSpeed - currentSpeed
            # If our positional error was zero, we could go exactly that speed.
            # if our position was behind the targetLen (line is lengthening, and we are shorter than targetLen),
            # (or line is shortening and we are longer than target len) then we need to go faster than targetSpeed to catch up
            # ideally we want to catch up in one step, but we have max acceleration constraints.
            aimSpeed = targetSpeed + position_err * PE_TERM; # meters of line per second

            wouldAccel = (aimSpeed - currentSpeed) / LOOP_DELAY_S
            if wouldAccel > MAX_ACCEL:
                aimSpeed = MAX_ACCEL * LOOP_DELAY_S + currentSpeed
            elif wouldAccel < -MAX_ACCEL:
                aimSpeed = -MAX_ACCEL * LOOP_DELAY_S + currentSpeed

            # take this speed
            self.speed = constrain(aimSpeed / meters_per_rev / speed1_revs, -127, 127)
            self.motor.runConstantSpeed(self.speed)

            time.sleep(LOOP_DELAY_S)

