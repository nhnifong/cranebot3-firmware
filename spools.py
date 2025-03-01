from math import pi
import asyncio
import time


DATA_LEN = 1000
PE_TERM = 1.5
# maximum acceleration in meters of line per second squared
MAX_ACCEL = 8.0
LOOP_DELAY_S = 0.03
# record line length every 10th iteration
REC_MOD = 10

def constrain(value, minimum, maximum):
    return max(minimum, min(value, maximum))

class SpoolController:
    def __init__(self, motor, empty_diameter, full_diameter, full_length):
        """
        Create a controller for a spool of line.
        empty_diameter_mm is the diameter of the spool in millimeters when no line is on it.
        full_diameter is the diameter in mm of the bulk of wrapped line when full_length meters of line are wrapped.
        line_capacity_m is the length of line in meters that is attached to this spool.
            if all of it were reeled in, the object at the end would reach the limit switch, if there is one.
        """
        self.motor = motor
        self.empty_diameter = empty_diameter
        self.full_diameter = full_diameter
        self.full_length = full_length
        # self.meters_per_rev = spool_diameter_mm * pi * 0.001;
        # last commanded motor speed in revs/sec
        self.speed = 0
        # Meters of line that were spooled out when zeroAngle was set.
        self.lineAtStart = 1.9
        self.meters_per_rev =  self.meters_per_rev(self.lineAtStart)
        # The angle of the shaft when setReferenceAngle was last called (in revolutions)
        self.zeroAngle = 0
        # record of line length. tuples of (time, meters)
        self.record = []
        # plan of desired line lengths
        self.desiredLine = []
        self.lastIndex = 0
        self.runSpoolLoop = True
        self.rec_loop_counter = 0

    def meters_per_rev(self, currentLenUnspooled):
        # interpolate between empty and full diamter based on how much line is on the spool
        fraction_wrapped = self.full_length / (self.full_length - currentLenUnspooled)
        current_diameter = (self.full_diameter - self.empty_diameter) * fraction_wrapped + self.empty_diameter
        return current_diameter * pi * 0.001;

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
        return the current time and current unspooled line in meters
        """
        success, angle = self.motor.getShaftAngle()
        l = self.meters_per_rev * (angle - self.zeroAngle) + self.lineAtStart
        # accumulate these so you can send them to the websocket
        row = (time.time(), l)
        if self.rec_loop_counter == REC_MOD:
            self.meters_per_rev =  self.meters_per_rev(l) # this also doesn't need to be updated at a high frequency.
            self.record.append(row)
            self.rec_loop_counter = 0
        self.rec_loop_counter += 1
        return row

    def fastStop(self):
        # fast stop is permanent.
        # it causes the trackingloop task to stop,
        # causing the websocket connection to close
        self.motor.stop()
        self.runSpoolLoop = False

    def trackingLoop(self):
        """
        Constantly try to match the position and speed given in an array
        """
        while self.runSpoolLoop:
            t, currentLen = self.currentLineLength()

            # Find the earliest entry in desiredLine that is still in the future.
            while self.desiredLine[self.lastIndex][0] <= t and self.lastIndex < len(self.desiredLine):
                self.lastIndex += 1

            # slow stop when there is no data to track
            if self.lastIndex >= len(self.desiredLine):
                if self.speed != 0:
                    direction = self.speed / self.speed
                    self.speed -= direction * 0.1
                    self.motor.runConstantSpeed(self.speed)
                time.sleep(LOOP_DELAY_S)
                continue

            targetLen = self.desiredLine[self.lastIndex][1]
            position_err = targetLen - currentLen

            # What would the speed be between the two datapoints tha straddle the present?
            # Positive values mean line is lengthening
            # result is in meters of line per second
            targetSpeed = ((self.desiredLine[self.lastIdx][1] - self.desiredLine[self.lastIdx-1][1])
                / (self.desiredLine[self.lastIdx][0] - self.desiredLine[self.lastIdx-1][0]))
            # change in line length in meters per second
            currentSpeed = self.speed * self.meters_per_rev
            speed_err = targetSpeed - currentSpeed
            # If our positional error was zero, we could go exactly that speed.
            # if our position was behind the targetLen (line is lengthening, and we are shorter than targetLen),
            # (or line is shortening and we are longer than target len) then we need to go faster than targetSpeed to catch up
            # ideally we want to catch up in one step, but we have max acceleration constraints.
            aimSpeed = targetSpeed + position_err * PE_TERM; # meters of line per second

            # limit the acceleration of the line
            wouldAccel = (aimSpeed - currentSpeed) / LOOP_DELAY_S
            if wouldAccel > MAX_ACCEL:
                aimSpeed = MAX_ACCEL * LOOP_DELAY_S + currentSpeed
            elif wouldAccel < -MAX_ACCEL:
                aimSpeed = -MAX_ACCEL * LOOP_DELAY_S + currentSpeed

            maxspeed = self.motor.getMaxSpeed()
            self.speed = constrain(aimSpeed / self.meters_per_rev, -maxspeed, maxspeed)
            self.motor.runConstantSpeed(self.speed)

            time.sleep(LOOP_DELAY_S)

