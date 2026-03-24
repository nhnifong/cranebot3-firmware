import math
import asyncio
import time
import logging
import numpy as np
from damiao_motor import DaMiaoMotor

from nf_robot.robot.spools import SpiralCalculator


# values that can be overridden by the controller
default_conf_dm = {
    # number of records of length to keep
    'DATA_LEN': 1000,
    # maximum acceleration of spools in radians per second squared
    'MAX_ACCEL': 0.01,
    # sleep delay of tracking loop
    'LOOP_FREQ_HZ': 50,
    # measured torque on a motor with orintation 1 and a line that is minimally taught. 
    'TARGET_TORQUE': -0.05,
    # default cruise speed in meters/sec for position moves
    'CRUISE_SPEED': 0.3,
    # factor controlling how torque is smoothed with ema.
    'SMOOTH_FACTOR': 0.08,
    # maximum safe line speed in meters per second
    'MAX_SAFE_LINE_SPEED': 2.0,
    # Proportional gain for the software-side position loop (meters error -> meters/sec)
    'POS_KP': 1.5,
    # Distance in meters within which we consider a jog "complete"
    'POS_DEADBAND': 0.005
}

class DamiaoSpoolController:
    """
    Spool controller intended for Damiao H-6215 motor
    return torque instead of a boolean tightness value
    there is no check tight function.
    motor is expected to be of type DaMiaoMotor
    direction is 1 or -1.
        should be set to 1 for the motor on the right when facing the front of the device.
        in other words should be 1 when negative motor commands reel in the line.
    """
    def __init__(self, motor:DaMiaoMotor, empty_diameter, full_diameter, full_length, config, direction):
        """
        Create a controller for a spool of line.

        empty_diameter_mm is the diameter of the spool in millimeters when no line is on it.
        full_diameter is the diameter in mm of the bulk of wrapped line when full_length meters of line are wrapped.
        line_capacity_m is the length of line in meters that is attached to this spool.
            if all of it were reeled in, the object at the end would reach the limit switch, if there is one.
        """
        self.motor = motor
        self.direction = direction
        self.sc = SpiralCalculator(empty_diameter, full_diameter, full_length, 1, direction)

        # config is the dictionary we should use and the object that will be updated by clients if any online reconfiguration occurs
        # that's why this appears backards
        self.conf = config
        self.conf.update(default_conf_dm)
        
        # Speed tracking state (meters/sec)
        self.aim_line_speed = 0
        # Jog command state
        self.target_length = None

        # Current state
        self.last_length = 3.0
        self.last_angle = 0.0
        self.last_tension  = 0.0
        self.meters_per_rev = self.sc.get_unspool_rate(self.last_angle)
        self.torque_err = 0
        
        # Absolute position tracking state
        self.last_raw_pos = None
        self.rev_offset = 0.0

        # Recording and Loops
        self.record = []
        self.run_spool_loop = True

    def _getAbsoluteAngle(self):
        """Get absolute motor angle in revolutions"""

    def setReferenceLength(self, length):
        """ Provide an external observation of the current unspooled line length """
        if self.torque_err > 0:
            return # gantry position has no relationship to spool zero angle if the line is slack.

        if length > 20 or length < 0:
            logging.warning(f'length ({length}) passed to setReferenceLength outside range [0,20]')
            return

        za = self.sc.calc_za_from_length(length, self.last_angle)
        self.sc.set_zero_angle(za)
        logging.debug(f'Zero angle estimate={za} revs. current value of {self.last_angle}, using reference length {length} m')
        # this affects the estimated amount of wrapped wire
        self.meters_per_rev = self.sc.get_unspool_rate(self.last_angle)

        # Sync target to reality if we were mid-jog
        if self.target_length is not None:
            self.target_length = length

    def setAimSpeed(self, lineSpeed):
        """Set the aim speed in meters of line per second.
        negative values reel line in.
        """
        self.target_length = None
        self.aim_line_speed = np.clip(lineSpeed, -self.conf['MAX_SAFE_LINE_SPEED'], self.conf['MAX_SAFE_LINE_SPEED'])

    def jog(self, delta_meters):
        """ 
        Change the unspooled length by delta_meters relative to current target 
        (or current length if no target active).
        """
        if self.target_length is None:
            self.target_length = self.last_length
        
        self.target_length += delta_meters
        logging.info(f"Jogging by {delta_meters}m. New target length: {self.target_length:.3f}m")

    def popMeasurements(self):
        """Return up to DATA_LEN measurements. newest at the end."""
        copy_record = self.record
        self.record = []
        return copy_record

    def fastStop(self):
        # fast stop is permanent.
        # it causes the trackingloop task to stop,
        # causing the websocket connection to close
        self.run_spool_loop = False

    def _update_absolute_angle(self, current_raw_rad):
        """
        Convert the wrapping motor position (-12.5 to +12.5 rad) into 
        continuous absolute revolutions.
        """
        full_range = 25.0
        half_range = full_range / 2.0

        if self.last_raw_pos is None:
            self.last_raw_pos = current_raw_rad
            return self.rev_offset + (current_raw_rad / (2 * math.pi))

        # We detect a rollover by looking for a massive instantaneous jump.
        diff = current_raw_rad - self.last_raw_pos
        
        if diff > half_range:
            # Wrapped from negative to positive; subtract the full range to keep it continuous.
            self.rev_offset -= (full_range / (2 * math.pi))
        elif diff < -half_range:
            # Wrapped from positive to negative; add the full range.
            self.rev_offset += (full_range / (2 * math.pi))

        self.last_raw_pos = current_raw_rad
        return self.rev_offset + (current_raw_rad / (2 * math.pi))

    def trackingLoop(self):
        """
        Constantly try to match the position or speed targets.
        """

        self.motor.enable()
        self.motor.ensure_control_mode("VEL")

        self.motor.set_acceleration(self.conf['MAX_ACCEL'])
        self.motor.set_deceleration(-self.conf['MAX_ACCEL'])

        self.motor.send_cmd_vel(target_velocity=0.0)
        
        start_time = time.time()
        last_time = start_time
        smooth_torque = 0
        smooth_mute = 1
        twopi = 2*math.pi 

        try:
            while self.run_spool_loop:
                loop_start = time.time()
                dt = loop_start - last_time
                if dt <= 0:
                    dt = 1e-4  # Prevent zero division errors if loop runs too fast
                last_time = loop_start
                
                # Read feedback from motor. flip if necessary.
                states = self.motor.get_states()
                motor_pos = self.direction * states.get('pos', 0.0) # radians from -12 to +12
                motor_vel = self.direction * states.get('vel', 0.0) # radians per second
                motor_torque = self.direction * states.get('torq', 0.0) # Newton-meters
                # we could also read status status_code, t_mos, t_rotor (temps)

                # Convert to absolute position in revolutions
                self.last_angle = self._update_absolute_angle(motor_pos)
                
                # low pass filter torque
                sf = self.conf['SMOOTH_FACTOR']
                smooth_torque = motor_torque * sf + smooth_torque * (1 - sf)

                # calculate line data from motor data
                self.last_length = self.sc.get_unspooled_length(self.last_angle)
                self.meters_per_rev = self.sc.get_unspool_rate(self.last_angle)
                current_line_speed = (motor_vel / twopi) * self.meters_per_rev
                self.last_tension = (-smooth_torque * twopi) / self.meters_per_rev

                # Position control logic (using during jog)
                if self.target_length is not None:
                    dist_err = self.target_length - self.last_length
                    # Proportional control for speed based on position error
                    self.aim_line_speed = dist_err * self.conf['POS_KP']
                    # Clamp to safe speeds
                    self.aim_line_speed = np.clip(self.aim_line_speed, 
                                                 -self.conf['CRUISE_SPEED'], 
                                                 self.conf['CRUISE_SPEED'])
                    # If we are within the deadband, stop and clear target
                    if abs(dist_err) < self.conf['POS_DEADBAND']:
                        self.target_length = None
                        self.aim_line_speed = 0

                # accumulate these. parent class will send them on the websocket at it's own rate
                row = (loop_start, self.last_length, current_line_speed, self.last_tension)
                self.record.append(row)

                # convert last commanded speed from motion controller in meters per second
                # to motor velocity in radians per second based on current circumfrence
                # let motor enforce acceleration limit
                wanted_motor_vel = self.aim_line_speed / self.meters_per_rev * twopi

                # prevent birdsnest by soft muting velocity when outspooling with no tension
                self.torque_err = smooth_torque - self.conf['TARGET_TORQUE']
                mute = 0 if (self.torque_err > 0 and wanted_motor_vel > 0) else 1
                smooth_mute = mute * sf + smooth_mute * (1 - sf)
                wanted_motor_vel *= smooth_mute
                
                self.motor.send_cmd_vel(target_velocity=wanted_motor_vel*self.direction)

                time_to_sleep = 1.0 / self.conf['LOOP_FREQ_HZ'] - (time.time() - loop_start)
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
            logging.info(f'Spool tracking loop stopped')

        finally:
            self.motor.disable()

