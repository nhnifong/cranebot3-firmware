import math
import asyncio
import time
import serial
import logging
import asyncio
import numpy as np

# values that can be overridden by the controller
default_conf = {
    # number of records of length to keep
    'DATA_LEN': 1000,
    # factor controlling how much positon error matters in the tracking loop.
    'PE_TERM': 1.5,
    # maximum acceleration in meters of line per second squared
    'MAX_ACCEL': 0.8,
    # sleep delay of tracking loop
    'LOOP_DELAY_S': 0.03,
    # record line length every x iterations of tracking loop
    'REC_MOD': 3,
}

class SpiralCalculator:
    def __init__(self, empty_diameter, full_diameter, full_length, gear_ratio, motor_orientation):
        self.empty_diameter = empty_diameter * 0.001 # millimeter to meters
        self.gear_ratio = gear_ratio # encoder rotations per spool rotation 
        # a negative motor orientation means that negative speeds make the line shorter.
        self.motor_orientation = motor_orientation
        self.zero_angle = 0

        # since line accumulates on the spool in a spiral, the amount of wrapped line is an exponential function of the spool angle.
        self.recalc_k_params(full_diameter, full_length)

    def set_zero_angle(self, zero_a):
        self.zero_angle = zero_a

    def recalc_k_params(self, full_diameter, full_length):
        self.full_diameter = full_diameter * 0.001
        self.full_length = full_length
        # since line accumulates on the spool in a spiral, the amount of wrapped line is an exponential function of the spool angle.
        self.diameter_diff = self.full_diameter - self.empty_diameter
        if self.diameter_diff > 0:
            self.k1 = (self.empty_diameter * self.full_length) / self.diameter_diff
            self.k2 = (math.pi * self.gear_ratio * self.diameter_diff) / self.full_length
        else:
            self.k1 = self.empty_diameter * self.full_length / 1e-9 # Avoid division by zero
            self.k2 = (math.pi * self.gear_ratio * 1e-9) / self.full_length

    def calc_za_from_length(self, length, angle):
        """ Given an observed length and current angle, what would the zero angle be, all other things being equal?"""
        # how many revs must the motor have turned since empty be to have this length
        spooled_length = self.full_length - length
        relative_angle = math.log(spooled_length / self.k1 + 1) / self.k2
        angle *= self.motor_orientation
        return angle - relative_angle

    def get_spooled_length(self, motor_angle_revs):
        relative_angle = self.motor_orientation * motor_angle_revs - self.zero_angle # shaft angle relative to zero_angle
        if self.diameter_diff == 0:
            return relative_angle * self.gear_ratio * math.pi * self.empty_diameter
        else:
            return self.k1 * (math.exp(self.k2 * relative_angle) - 1)

    def get_unspooled_length(self, motor_angle_revs):
        return self.full_length - self.get_spooled_length(motor_angle_revs)

    def get_unspool_rate(self, motor_angle_revs):
        relative_angle = self.motor_orientation * motor_angle_revs - self.zero_angle

        if self.diameter_diff == 0:
            return math.pi * self.empty_diameter * self.gear_ratio
        else:
            effective_spool_diameter =  self.empty_diameter + self.diameter_diff * (self.get_spooled_length(motor_angle_revs) / self.full_length)
            return math.pi * effective_spool_diameter * self.gear_ratio

class SpoolController:
    def __init__(self, motor, empty_diameter, full_diameter, full_length, conf, gear_ratio=1.0, tight_check_fn=None):
        """
        Create a controller for a spool of line.

        empty_diameter_mm is the diameter of the spool in millimeters when no line is on it.
        full_diameter is the diameter in mm of the bulk of wrapped line when full_length meters of line are wrapped.
        line_capacity_m is the length of line in meters that is attached to this spool.
            if all of it were reeled in, the object at the end would reach the limit switch, if there is one.
        gear_ratio refers to how many rotations the spool makes for one rotation of the motor shaft.
        tight_check_fn a function that can return true when the line is tight and false when it is slack
        """
        self.motor = motor
        self.tight_check_fn = tight_check_fn
        self.sc = SpiralCalculator(empty_diameter, full_diameter, full_length, gear_ratio, -1)

        self.conf = conf
        self.conf.update(default_conf)
        
        # last commanded motor speed in revs/sec
        self.speed = 0
        # speed of line change in meters/sec last sent from controller.
        self.aim_line_speed = 0
        # whether to track a length plan 'plan' or use aim_line_speed 'speed'
        self.tracking_mode = 'plan'
        self.last_length = 3.0
        self.last_angle = 0.0
        self.meters_per_rev = self.sc.get_unspool_rate(self.last_angle)
        self.record = []
        # plan of desired line lengths
        self.desired_line = []
        self.last_index = 0
        self.run_spool_loop = True
        self.rec_loop_counter = 0

        # when this bool is set, spool tracking will pause.
        self.spoolPause = False

    def setReferenceLength(self, length):
        """
        Provide an external observation of the current unspooled line length
        """
        if self.tight_check_fn is not None and not self.tight_check_fn():
            return # gantry position has no relationship to spool zero angle if the line is slack.
        success = False
        attempts = 0
        while not success and attempts < 10:
            success, angle = self.motor.getShaftAngle()
            attempts += 1
        if success:
            za = self.sc.calc_za_from_length(length, angle)
            self.sc.set_zero_angle(za)
            logging.debug(f'Zero angle estimate={za} revs. current value of {angle}, using reference length {length} m')
            # this affects the estimated current amount of wrapped wire
            self.meters_per_rev = self.sc.get_unspool_rate(angle)

    def _commandSpeed(self, speed):
        """command a specific speed from the motor."""
        if self.speed == speed:
            return
        self.speed = speed
        self.motor.runConstantSpeed(self.speed)

    def setPlan(self, plan):
        """
        Swith to plan tracking mode and set the plan to an array of tuples of time and length
        """
        self.tracking_mode = 'plan'
        self.desired_line = plan
        self.last_index = 0

    def setAimSpeed(self, lineSpeed):
        """Switch to speed tracking mode and set the aim speed in meters of line per second.
        negative values reel line in.
        """
        self.tracking_mode = 'speed'
        self.aim_line_speed = lineSpeed

    def jogRelativeLen(self, rel):
        new_l = self.last_length + rel
        self.setPlan([
            (time.time(), self.last_length),
            (time.time()+2, new_l),
        ])

    def popMeasurements(self):
        """Return up to DATA_LEN measurements. newest at the end."""
        copy_record = self.record
        self.record = []
        return copy_record

    def currentLineLength(self):
        """
        return the current time and current unspooled line in meters
        Also store the length in an array to be popped later.
        """
        success, angle = self.motor.getShaftAngle()
        if not success:
            logging.warning("Could not read shaft angle from motor")
            return (time.time(), self.last_length)

        if abs(angle - self.last_angle) > 1:
            logging.warning(f'motor moved more than 1 rev since last read, last_angle={self.last_angle} angle={angle} diff={angle - self.last_angle}')
        self.last_angle = angle

        self.last_length = self.sc.get_unspooled_length(angle)
        self.meters_per_rev = self.sc.get_unspool_rate(angle)
        currentLineSpeed = self.speed * self.meters_per_rev

        # accumulate these so you can send them to the websocket
        if self.tight_check_fn is None:
            row = (time.time(), self.last_length, currentLineSpeed)
        else:
            row = (time.time(), self.last_length, currentLineSpeed, self.tight_check_fn())

        if self.rec_loop_counter >= self.conf['REC_MOD']:
            self.record.append(row)
            self.rec_loop_counter = 0
        self.rec_loop_counter += 1
        return time.time(), self.last_length

    def fastStop(self):
        # fast stop is permanent.
        # it causes the trackingloop task to stop,
        # causing the websocket connection to close
        self.motor.stop()
        self.run_spool_loop = False

    def pauseTrackingLoop(self):
        self.spoolPause = True

    def resumeTrackingLoop(self):
        self.spoolPause = False

    def getAimSpeedFromPlan(self, t):
        """Compute a speed to aim for based on our position in the length plan

        Combines speed error with positional error to get a speed to aim for (pre-enforcement of max accel)
        """
        targetLen = self.desired_line[self.last_index][1]
        position_err = targetLen - self.last_length
        if abs(position_err) < 0.001:
            return 0

        # What would the speed be between the two datapoints tha straddle the present?
        # Positive values mean line is lengthening
        # result is in meters of line per second
        # if there is only one desired length, we can't find two that straddle the present and have to use current length and time
        if self.last_index == 0:
            time_A = t
            len_A = self.last_length
        else:
            time_A = self.desired_line[self.last_index-1][0]
            len_A = self.desired_line[self.last_index-1][1]
        targetSpeed = ((self.desired_line[self.last_index][1] - len_A) / (self.desired_line[self.last_index][0] - time_A))
        speed_err = targetSpeed - self.speed * self.meters_per_rev

        # If our positional error was zero, we could go exactly that speed.
        # if our position was behind the targetLen (line is lengthening, and we are shorter than targetLen),
        # (or line is shortening and we are longer than target len) then we need to go faster than targetSpeed to catch up
        # ideally we want to catch up in one step, but we have max acceleration constraints.
        return targetSpeed + position_err * self.conf['PE_TERM']; # meters of line per second

    def trackingLoop(self):
        """
        Constantly try to match the position and speed given in an array

        """
        while self.run_spool_loop:
            if self.spoolPause:
                time.sleep(0.2)
                continue
            try:
                t, currentLen = self.currentLineLength()

                # change in line length in meters per second
                if self.tracking_mode == 'plan':
                    # Find the earliest entry in desired_line that is still in the future.
                    while self.last_index < len(self.desired_line) and self.desired_line[self.last_index][0] <= t:
                        self.last_index += 1
                    # slow stop when there is no data to track
                    if self.last_index >= len(self.desired_line):
                        if abs(self.speed) > 0:
                            newspeed = self.speed * 0.9
                            if abs(newspeed) < 0.2:
                                newspeed = 0
                            logging.debug(f'Slow stopping {newspeed}')
                            self._commandSpeed(newspeed)
                        time.sleep(self.conf['LOOP_DELAY_S'])
                        continue
                    aimSpeed = self.getAimSpeedFromPlan(t)
                elif self.tracking_mode == 'speed':
                    aimSpeed = self.aim_line_speed
                else:
                    aimSpeed = 0

                # stop outspooling of line when not tight and switch is available (anchors hardware 4.5.5 and later)
                if aimSpeed > 0 and (self.tight_check_fn is not None) and (not self.tight_check_fn()):
                    logging.warning(f"would unspool at speed={aimSpeed} but switch shows line is not tight, and unspooling could slacken or tangle line.")
                    aimSpeed = 0

                # limit the acceleration of the line
                currentSpeed = self.speed * self.meters_per_rev
                wouldAccel = (aimSpeed - currentSpeed) / self.conf['LOOP_DELAY_S']
                if wouldAccel > self.conf['MAX_ACCEL']:
                    aimSpeed = self.conf['MAX_ACCEL'] * self.conf['LOOP_DELAY_S'] + currentSpeed
                elif wouldAccel < -self.conf['MAX_ACCEL']:
                    aimSpeed = -self.conf['MAX_ACCEL'] * self.conf['LOOP_DELAY_S'] + currentSpeed

                maxspeed = self.motor.getMaxSpeed()

                # convert speed to revolutions per second
                cspeed = np.clip(aimSpeed / self.meters_per_rev, -maxspeed, maxspeed)
                if abs(cspeed) < 0.02:
                    cspeed = 0
                self._commandSpeed(cspeed)

                time.sleep(self.conf['LOOP_DELAY_S'])
            except serial.serialutil.SerialTimeoutException:
                logging.critical('Lost serial contact with motor. This may require power cycling the motor controller.')
                break
        logging.info(f'Spool tracking loop stopped')
