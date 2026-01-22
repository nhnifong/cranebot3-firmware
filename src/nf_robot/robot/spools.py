import math
import asyncio
import time
import logging
import asyncio
import numpy as np

# values that can be overridden by the controller
default_conf = {
    # number of records of length to keep
    'DATA_LEN': 1000,
    # factor controlling how much positon error matters in the tracking loop.
    'PE_TERM': 2.0,
    # maximum acceleration in meters of line per second squared
    'MAX_ACCEL': 0.8,
    # sleep delay of tracking loop
    'LOOP_DELAY_S': 0.03,
    # record line length every x iterations of tracking loop
    'REC_MOD': 3,
    # default cruise speed in meters/sec for position moves
    'CRUISE_SPEED': 0.3,
    # deadband in meters to prevent jitter when close to target
    'DEADBAND': 0.005,
}

class SpiralCalculator:
    def __init__(self, empty_diameter, full_diameter, full_length, gear_ratio, motor_orientation):
        self.empty_diameter = empty_diameter * 0.001 # millimeter to meters
        self.gear_ratio = gear_ratio # spool rotations per encoder rotation 
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
        gear_ratio refers to how many rotations the spool makes for one rotation of the encoder.
        tight_check_fn a function that can return true when the line is tight and false when it is slack
        """
        self.motor = motor
        self.tight_check_fn = tight_check_fn
        self.sc = SpiralCalculator(empty_diameter, full_diameter, full_length, gear_ratio, -1)

        self.conf = conf
        self.conf.update(default_conf)
        
        # last commanded motor speed in revs/sec
        self.speed = 0
        
        # Mode switching: 'position' tracks target_length, 'speed' tracks aim_line_speed
        # must start in speed mode or we might zoom upon turning on.
        self.tracking_mode = 'speed' 
        
        # Position tracking state
        self.target_length = 3.0 # Meters
        self.cruise_speed = self.conf['CRUISE_SPEED']  # Meters/sec (max speed during position moves)
        
        # Speed tracking state (meters/sec)
        self.aim_line_speed = 0

        # Current state
        self.last_length = 3.0
        self.last_angle = 0.0
        self.meters_per_rev = self.sc.get_unspool_rate(self.last_angle)
        
        # Recording and Loops
        self.record = []
        self.run_spool_loop = True
        self.rec_loop_counter = 0

        # when this bool is set, spool tracking will pause.
        self.spoolPause = False

    def setReferenceLength(self, length):
        """ Provide an external observation of the current unspooled line length """
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
            # Sync target to reality to prevent sudden movement upon init
            if abs(self.last_length - length) > 0.5: 
                self.target_length = length 
                self.last_length = length
            # force tracking mode to speed so we don't zoom
            self.tracking_mode = 'speed'

    def _commandSpeed(self, speed):
        """ Command a specific speed from the motor. """
        if self.speed == speed:
            return
        self.speed = speed
        self.motor.runConstantSpeed(self.speed)

    def setTargetLength(self, length, cruise_speed=None):
        """
        Switch to position tracking mode.
        The spool will move to 'length' at 'cruise_speed', constrained by MAX_ACCEL.
        """
        self.tracking_mode = 'position'
        self.target_length = length
        if cruise_speed is not None:
            self.cruise_speed = abs(cruise_speed)

    def setAimSpeed(self, lineSpeed):
        """Switch to speed tracking mode and set the aim speed in meters of line per second.
        negative values reel line in.
        """
        self.tracking_mode = 'speed'
        self.aim_line_speed = lineSpeed

    def jogRelativeLen(self, rel):
        """
        Switch to position tracking mode
        and add a relative distance to the current target length.
        """
        self.tracking_mode = 'position'
        new_l = self.last_length + rel
        # Keep existing cruise speed, just update target
        self.setTargetLength(new_l)

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

    def trackingLoop(self):
        """
        Constantly try to match the position or speed targets.
        """
        while self.run_spool_loop:
            if self.spoolPause:
                time.sleep(0.2)
                continue

            t, currentLen = self.currentLineLength()
            
            # Determine the desired line speed based on mode
            aimSpeed = 0

            if self.tracking_mode == 'position':
                position_err = self.target_length - self.last_length
                
                # deadband to prevent jitter when extremely close to target
                if abs(position_err) < self.conf['DEADBAND']: 
                    aimSpeed = 0
                else:
                    # Simple Proportional controller clamped to cruise_speed.
                    # As we get closer (error decreases), the speed decreases (deceleration).
                    # PE_TERM determines how aggressively we brake. 
                    # High PE_TERM = Late braking, Low PE_TERM = Early braking.
                    calc_speed = position_err * self.conf['PE_TERM']
                    aimSpeed = np.clip(calc_speed, -self.cruise_speed, self.cruise_speed)

            elif self.tracking_mode == 'speed':
                aimSpeed = self.aim_line_speed

            # Stop outspooling of line when not tight and switch is available
            # This overrides both position and speed commands
            if aimSpeed > 0 and (self.tight_check_fn is not None) and (not self.tight_check_fn()):
                logging.warning(f"would unspool at speed={aimSpeed} but switch shows line is not tight.")
                aimSpeed = 0

            # limit the acceleration of the line (Physics constraint)
            currentSpeed = self.speed * self.meters_per_rev
            wouldAccel = (aimSpeed - currentSpeed) / self.conf['LOOP_DELAY_S']
            
            if wouldAccel > self.conf['MAX_ACCEL']:
                aimSpeed = self.conf['MAX_ACCEL'] * self.conf['LOOP_DELAY_S'] + currentSpeed
            elif wouldAccel < -self.conf['MAX_ACCEL']:
                aimSpeed = -self.conf['MAX_ACCEL'] * self.conf['LOOP_DELAY_S'] + currentSpeed

            maxspeed = self.motor.getMaxSpeed()

            # convert speed to revolutions per second
            cspeed = np.clip(aimSpeed / self.meters_per_rev, -maxspeed, maxspeed)
            
            # Minimum motor speed check so we go full quiet.
            if abs(cspeed) < 0.02:
                cspeed = 0
                
            self._commandSpeed(cspeed)

            time.sleep(self.conf['LOOP_DELAY_S'])
        logging.info(f'Spool tracking loop stopped')
