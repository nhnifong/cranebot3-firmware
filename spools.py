from math import pi
import asyncio
import time
import serial
import logging
import asyncio


DATA_LEN = 1000
PE_TERM = 1.5
# maximum acceleration in meters of line per second squared
MAX_ACCEL = 0.4
LOOP_DELAY_S = 0.03
# record line length every 10th iteration
REC_MOD = 10

# Constants used in tension equalization process
TENSION_SLACK_THRESH = 0.4 # kilograms. below this, line is assumed to be slack
TENSION_TIGHT_THRESH = 0.7 # kilograms. above this, line is assumed to be too tight.
MOTOR_SPEED_DURING_CALIBRATION = 1 # revolutions per second
MAX_LINE_CHANGE_DURING_CALIBRATION = 0.5 # meters
MKS42C_EXPECTED_ERR = 1.8 # degrees of expected angle error with no load per commanded rev/sec
MKS42C_TORQUE_FACTOR = 0.031 # kg-meters per degree of error. Factor for computing torque from the risidual angle error


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
        self.abort_equalize_tension = False
        self.live_tension_low_thresh = TENSION_SLACK_THRESH
        self.live_tension_high_thresh = TENSION_TIGHT_THRESH
        self.smoothed_tension = 0

        # when this bool is set, spool tracking will pause.
        self.spoolPause = False

    def calc_meters_per_rev(self, currentLenUnspooled):
        """meters of line change per motor revolution"""
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

        self.smoothed_tension = self.currentTension() * 0.2 + self.smoothed_tension * 0.8

        # accumulate these so you can send them to the websocket
        row = (time.time(), self.lastLength, self.smoothed_tension)
        if self.rec_loop_counter == REC_MOD:
            self.meters_per_rev =  self.calc_meters_per_rev(self.lastLength) # this also doesn't need to be updated at a high frequency.
            self.record.append(row)
            self.rec_loop_counter = 0
        self.rec_loop_counter += 1
        return row

    def currentTension(self):
        """return the line tension in kilograms of force.
        Only possible for the MKS42C
        """
        _, angleError = self.motor.getShaftError()
        baseline = MKS42C_EXPECTED_ERR * self.speed
        load_err = (angleError - baseline) * -1
        # The load on the line subtracts about 1 degree of angle error from the baseline per kilogram.
        torque = MKS42C_TORQUE_FACTOR * load_err
        return torque / self.meters_per_rev


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
                t, currentLen, tension = self.currentLineLength()

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

    async def equalizeSpoolTension(self, controllerApprovalEvent, sendUpdatesFunc):
        """Without tracking any particular length, reel the spool until the line tension is within a predefined range

        Pause spool tracking loop
        Measure the starting shaft angle
        measure shaft angle error. Very low errors indicate slackness, higher errors indicate tightness.
        While (angle error is not in the middle of it's range) and (calibration not aborted) and (line change is less than safe threshold):
            run the motor at a constant very low speed (positive for tight lines, negative for slack lines)
            Measure the shaft angle and compute the change in the amount of outspooled line since start
        report the value to the controller and remain paused.
        When the controller approves,
        Change the reference length by that amount.
        Unpause the tracking loop.

        Any lines that started slack will begin to reel in
        Any line that didn't start slack will begin to reel out.
        reeling (out or in) should immediately stop if
          1. the slackness threshold is crossed (we can see this locally)
        reeling out should also stop if
          2. Any other line that started slack has stopped (controller will inform us)
        """
        self.pauseTrackingLoop()
        # make sure the tracking loop is definitely paused. it's on another thread. 
        await asyncio.sleep(LOOP_DELAY_S * )
        self.motor.runConstantSpeed(0)
        self.speed = 0
        logging.info("Equalizing spool tension")
        self.abort_equalize_tension = False
        t, curLength, tension = self.currentLineLength()
        startLength = curLength
        line_delta = 0

        # reset thresholds to default
        self.live_tension_low_thresh = TENSION_SLACK_THRESH
        self.live_tension_high_thresh = TENSION_TIGHT_THRESH

        # decide initial speed. motor direction will not change during the loop
        if self.smoothed_tension < ERR_SLACK_THRESH:
            started_slack = True
            self.speed = -MOTOR_SPEED_DURING_CALIBRATION
        else:
            started_slack = False
            self.speed = MOTOR_SPEED_DURING_CALIBRATION # starts tight
        self.motor.runConstantSpeed(self.speed)
        is_slack = started_slack

        # wait for stop condition
        while ((started_slack == is_slack)
               and not self.abort_equalize_tension
               and abs(lineDelta < MAX_LINE_CHANGE_DURING_CALIBRATION)):
            await asyncio.sleep(1/30)
            # self.currentLineLength() causes length and tension to be calculated and recorded in a list that
            # is periodically flushed to the websocket by a task that is always running while the ws is connected
            t, curLength, tension = self.currentLineLength()
            print(f'curLength={curLength} tension={tension}')
            line_delta = curLength - startLength
            is_slack = self.smoothed_tension < self.live_tension_low_thresh

        # todo, verify by experiment
        # if you started tight but became slack, reel back in a small amount

        # inform controller that we hit a trigger and stopped.
        sendUpdatesFunc({
            'tension_seek_stopped': {
                'line_delta': line_delta,
                'started_slack': started_slack,
                'is_slack': is_slack,
            }
        })

        logging.info(f"Stopped equalization with tension={self.smoothed_tension} and a line delta of {lineDelta}")
        self.motor.runConstantSpeed(0)
        await asyncio.wait_for(controllerApprovalEvent.wait(), timeout=30)
        # the caller will have set this flag while we were waiting to indicate approval.
        if not self.abort_equalize_tension:
            # alter reference length but not zero angle.
            self.lineAtStart += line_delta
        self.desiredLine = []
        self.resumeTrackingLoop()