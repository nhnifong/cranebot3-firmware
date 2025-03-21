from math import pi
import asyncio
import time
import serial
import logging


DATA_LEN = 1000
PE_TERM = 1.5
# maximum acceleration in meters of line per second squared
MAX_ACCEL = 0.4
LOOP_DELAY_S = 0.03
# record line length every 10th iteration
REC_MOD = 10

def constrain(value, minimum, maximum):
    return max(minimum, min(value, maximum))

class SpoolController:
    def __init__(self, motor, empty_diameter, full_diameter, full_length, gear_ratio=1.0):
        """
        Create a controller for a spool of line.
        empty_diameter_mm is the diameter of the spool in millimeters when no line is on it.
        full_diameter is the diameter in mm of the bulk of wrapped line when full_length meters of line are wrapped.
        line_capacity_m is the length of line in meters that is attached to this spool.
            if all of it were reeled in, the object at the end would reach the limit switch, if there is one.
        gear_ratio refers to how many rotations the spool makes for one rotation of the motor shaft.
        """
        self.motor = motor
        self.empty_diameter = empty_diameter
        self.full_diameter = full_diameter
        self.full_length = full_length
        self.gear_ratio = gear_ratio
        # last commanded motor speed in revs/sec
        self.speed = 0
        # Meters of line that were spooled out when zeroAngle was set.
        self.lineAtStart = 0.5
        self.lastLength = 0.5
        self.meters_per_rev =  self.calc_meters_per_rev(self.lineAtStart)
        # The angle of the shaft when setReferenceAngle was last called (in revolutions)
        self.zeroAngle = 0
        # record of line length. tuples of (time, meters)
        self.record = []
        # plan of desired line lengths
        self.desiredLine = []
        self.lastIndex = 0
        self.runSpoolLoop = True
        self.rec_loop_counter = 0
        self.moveAllowed = True

        # when this bool is set, spool tracking will pause.
        self.spoolPause = False

    def calc_meters_per_rev(self, currentLenUnspooled):
        # interpolate between empty and full diamter based on how much line is on the spool
        fraction_wrapped = (self.full_length - currentLenUnspooled) / self.full_length
        current_diameter = (self.full_diameter - self.empty_diameter) * fraction_wrapped + self.empty_diameter
        return current_diameter * pi * 0.001 * self.gear_ratio

    def setReferenceLength(self, length):
        """
        Provide an external observation of the current line length
        """
        self.lineAtStart = length
        success, l = self.motor.getShaftAngle()
        if success:
            self.zeroAngle = l

    def setPlan(self, plan):
        """
        set plan to an array of tuples of time and length
        """
        self.desiredLine = plan
        self.lastIndex = 0

    def jogRelativeLen(self, rel):
        new_l = self.lastLength + rel
        self.setPlan([
            (time.time(), self.lastLength),
            (time.time()+2, new_l),
        ])

    def popMeasurements(self):
        copy_record = self.record
        self.record = []
        return copy_record

    def currentLineLength(self):
        """
        return the current time and current unspooled line in meters
        """
        success, angle = self.motor.getShaftAngle()
        if not success:
            logging.error("Could not read shaft angle from motor")
            return (time.time(), self.lastLength)
        self.lastLength = self.meters_per_rev * (angle - self.zeroAngle) + self.lineAtStart

        self.moveAllowed = True
        if self.lastLength < 0 or self.lastLength > self.full_length:
            logging.error(f"Bad length calculation! length={self.lastLength}, shaftAngle={angle}, zeroAngle={self.zeroAngle}, lineAtStart={self.lineAtStart}, meters_per_rev={self.meters_per_rev}")
            self.moveAllowed = False

        # accumulate these so you can send them to the websocket
        row = (time.time(), self.lastLength)
        if self.rec_loop_counter == REC_MOD:
            self.meters_per_rev =  self.calc_meters_per_rev(self.lastLength) # this also doesn't need to be updated at a high frequency.
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

    def pauseTrackingLoop(self):
        self.spoolPause = True

    def resumeTrackingLoop(self):
        self.spoolPause = False

    def trackingLoop(self):
        """
        Constantly try to match the position and speed given in an array

        """
        while self.runSpoolLoop:
            if self.spoolPause:
                time.sleep(0.2)
                continue
            try:
                t, currentLen = self.currentLineLength()

                # Find the earliest entry in desiredLine that is still in the future.
                while self.lastIndex < len(self.desiredLine) and self.desiredLine[self.lastIndex][0] <= t:
                    self.lastIndex += 1

                # slow stop when there is no data to track
                if self.lastIndex >= len(self.desiredLine):
                    if abs(self.speed) > 0:
                        self.speed *= 0.9
                        if abs(self.speed) < 0.2:
                            self.speed = 0
                        logging.debug('Slow stopping')
                        self.motor.runConstantSpeed(self.speed)
                    time.sleep(LOOP_DELAY_S)
                    continue

                targetLen = self.desiredLine[self.lastIndex][1]
                position_err = targetLen - currentLen

                # What would the speed be between the two datapoints tha straddle the present?
                # Positive values mean line is lengthening
                # result is in meters of line per second
                # if there is only one desired length, we can't find two that straddle the present and have to use current length and time
                if self.lastIndex == 0:
                    last_time = t
                    last_len = currentLen
                else:
                    last_time = self.desiredLine[self.lastIndex-1][0]
                    last_len = self.desiredLine[self.lastIndex-1][1]
                targetSpeed = ((self.desiredLine[self.lastIndex][1] - last_len)
                    / (self.desiredLine[self.lastIndex][0] - last_time))
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

                if self.moveAllowed:
                    self.motor.runConstantSpeed(self.speed)
                else:
                    logging.warning("would move at speed={self.speed} but length is invalid. calibrate length first.")

                time.sleep(LOOP_DELAY_S)
            except serial.serialutil.SerialTimeoutException:
                print('Lost serial contact with motor')
                break

