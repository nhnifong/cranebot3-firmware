import asyncio
from getmac import get_mac_address
import logging
from collections import deque
import time
import pickle
import os
import board
import busio
import json
import numpy as np
import math

from adafruit_mpu6050 import MPU6050 # accelerometer
from adafruit_vl53l1x import VL53L1X # rangefinder
from adafruit_ads1x15 import ADS1015, AnalogIn, ads1x15 # analog2digital converter for pressure

from nf_robot.robot.anchor_server import RobotComponentServer
from nf_robot.robot.simple_st3215 import SimpleSTS3215
from nf_robot.common.util import remap, clamp, PID

""" Server for Arpeggio Gripper

Hardware is a Raspberry pi zero 2W, Camera Module 3 Wide, and Stringman Gripper Hat.
"""

FINGER = 1
WRIST = 2
STEPS_PER_REV = 4096
GEAR_RATIO = 10/45 
FINGER_TRAVEL_DEG = 59 
FINGER_TRAVEL_STEPS = FINGER_TRAVEL_DEG / 360 / GEAR_RATIO * STEPS_PER_REV
DT = 1/60

# values that can be overridden by the controller
default_gripper_conf = {
    # (seconds) Time before a zero-speed command is assumed if no new commands arrive
    'ACTION_TIMEOUT': 0.2,
    # (dimensionless, 0-1) Low-pass filter smoothing factor for the raw force reading
    'FILTER_COEFF': 0.15,
    # (dimensionless) Proportional gain for the finger force PID controller
    'FINGER_PID_KP': 1.5,
    # (dimensionless) Derivative gain for the finger force PID controller
    'FINGER_PID_KD': 0.1,
    # (dimensionless) Integral gain for the finger force PID controller
    'FINGER_PID_KI': 0.05,
    # (normalized force, 0-1) Minimum error required to trigger a PID adjustment
    'FORCE_DEADBAND': 0.02,
    # (normalized voltage drop, 0-1) Pressure threshold to switch from position to force mode
    'FORCE_TRIGGER_THRESHOLD': 0.025,
    # (force/deg) Scaling factor mapping commanded finger speed to desired force increments
    'FORCE_RATE_MULTIPLIER': 1.0 / 200.0,
    # (normalized force, 0-1) The target force immediately applied upon entering force mode
    'INITIAL_DESIRED_FORCE': 0.08,
    # (raw motor units, 0-1000) The maximum allowed motor load before capping the normalized load contribution
    'MAX_SAFE_LOAD': 500,
    # (dimensionless, 0-1) The proportional weight of the pad pressure in the composite force calculation.
    # the weight allocated to the motor load reading is 1-this value
    'PRESSURE_WEIGHT': 0.7,
}


class GripperArpServer(RobotComponentServer):
    def __init__(self):
        super().__init__()
        self.conf.update(default_gripper_conf)
        # the observer identifies hardware by the service types advertised on zeroconf
        self.service_type = 'cranebot-gripper-arpeggio-service'

        self.stream_command = [
            "/usr/bin/rpicam-vid", "-t", "0", "-n",
            "--width=384", "--height=384",
            "--framerate", "100"
            "-o", "tcp://0.0.0.0:8888?listen=1",
            "--codec", "libav",
            "--libav-format", "mpegts",
            "--autofocus-mode", "continuous",
            "--low-latency",
            "--bitrate", "1200kbps"
        ]

        i2c = busio.I2C(board.SCL, board.SDA)

        self.rangefinder = VL53L1X(i2c)
        model_id, module_type, mask_rev = self.rangefinder.model_info
        logging.info(f'Rangefinder Model ID: 0x{model_id:0X} Module Type: 0x{module_type:0X} Mask Revision: 0x{mask_rev:0X}')
        self.rangefinder.distance_mode = 2 # LONG. results returned in centimeters.
        self.rangefinder.start_ranging()

        self.ads = ADS1015(i2c)
        self.pressure_sensor = AnalogIn(self.ads, ads1x15.Pin.A0)

        self.imu = MPU6050(i2c)

        self.motors = SimpleSTS3215()
        self.motors.configure_multiturn(WRIST)

        # the superclass, RobotComponentServer, assumes the presense of this attribute
        self.spooler = None

        unique = ''.join(get_mac_address().split(':'))
        self.service_name = self.service_type + '.' + unique

        self.desired_finger_angle = 0
        self.desired_wrist_angle = 0
        
        self.last_simple_wrist_angle = None
        self.unrolled_wrist_angle = 0
        self.wrist_step_offset = 0
            
        self.desired_finger_speed = 0
        self.desired_wrist_speed = 0

        self.time_last_commanded_finger_speed = 0
        self.time_last_commanded_wrist_speed = 0

        self.last_time_imu = time.time()

        self.last_gyro = np.zeros(2)
        self.filtered_alpha = np.zeros(2)

        # pendulum constants for 53cm pole
        L = 0.4526 # 0.53
        g = 9.81
        # omega is the constant angular frequency of the pendulum.
        # It represents the speed that the system progresses along it's cycle radians per second.
        self.omega = np.sqrt(g / L)
        
        # The state matrix representing the sin curves being fitted to the gyro measurements.
        # Row 0: X-axis swing, Row 1: Y-axis swing.
        # Col 0: Velocity (Sine component), Col 1: Phase tracker (Cosine component).
        self.state = np.zeros((2, 2))

        # observation_gain defines how much real sensor data to average into the model each time step
        self.observation_gain = 0.1 

        # Finger pressure PID controller
        self.fingerpid = PID(self.conf['FINGER_PID_KP'], self.conf['FINGER_PID_KI'], self.conf['FINGER_PID_KD'], DT)
        self.last_finger_data = None

        self.filtered_force = 0.0
        self.in_force_mode = False
        self.desired_force = 0.0
        
        # State tracking for the motor overload safety feature
        self.finger_torque_reenable_time = 0.0
        self.wrist_torque_reenable_time = 0.0

        # defaults for persistent values
        self.finger_open_pos = -1000
        self.finger_closed_pos = 1000
        self.saved_unrolled_wrist_angle = 0
        self.saved_finger_angle = 0

        self.motor_loop_pause = False

        # taps to trigger wifi reset
        self.taps = deque(maxlen=5)
        self.was_pressed = False


        if os.path.exists('arp_gripper_state.json'):
            try:
                with open('arp_gripper_state.json', 'r') as f:
                    d = json.load(f)
                    self.finger_open_pos = d['finger_open_pos']
                    self.finger_closed_pos = d['finger_closed_pos']
                    self.saved_unrolled_wrist_angle = d.get('unrolled_wrist_angle', 0)
                    self.saved_finger_angle = d.get('finger_angle', 0)
            except (json.JSONDecodeError, EOFError):
                os.remove('arp_gripper_state.json')

    def getWristAngle(self):
        # motor only reports it's position within one revolution.
        # even though you can command multi turns from it.
        # return a value between 0 and 1080, which is the same range we accept commands in.
        # 540 is the neutral position.
        wrist_data = self.motors.get_feedback(WRIST)
        simple_angle = wrist_data['position'] / STEPS_PER_REV * 360

        # Anchor continuous tracking to the motor's actual physical position at boot
        if self.last_simple_wrist_angle is None:
            self.last_simple_wrist_angle = simple_angle
            
            # Assume the physical joint moved less than half a turn while powered off
            # to mathematically reconstruct the continuous multi-turn angle from the saved state
            error = (simple_angle - self.saved_unrolled_wrist_angle + 180) % 360 - 180
            self.unrolled_wrist_angle = self.saved_unrolled_wrist_angle + error

            # Calculate the step offset between Python's continuous unrolled frame 
            # and the motor's hardware encoder frame (which wraps 0-4095 at boot).
            # This is critical to prevent zooming/spin-up on the first command.
            expected_steps = self.unrolled_wrist_angle / 360 * STEPS_PER_REV
            self.wrist_step_offset = expected_steps - wrist_data['position']

            return clamp(self.unrolled_wrist_angle, 0, 1080)

        # Unroll simple_angle against desired_wrist_angle instead of last_simple_wrist_angle.
        # Since desired_wrist_angle tracks intent rigidly at 60Hz, this is immune to aliasing
        # caused by read/CPU stutters (which resulted in unrolled_wrist_angle drifting out of bounds).
        error = (simple_angle - self.desired_wrist_angle + 180) % 360 - 180
        self.unrolled_wrist_angle = self.desired_wrist_angle + error
        
        self.last_simple_wrist_angle = simple_angle
        return clamp(self.unrolled_wrist_angle, 0, 1080)

    def getFingerAngle(self):
        # updated in get_current_grip_force which is called in pid loop at 60 hz
        if self.last_finger_data is not None:
            return remap(self.last_finger_data['position'], self.finger_open_pos, self.finger_closed_pos, -90, 90)
        else:
            return 0

    def readOtherSensors(self):
        t = time.time()
        finger_angle = self.getFingerAngle()
        wrist_angle = self.getWristAngle()

        self.update['grip_sensors'] = {
            'time': t,
            'fing_v': self.filtered_force,
            'fing_a': finger_angle,
            'wrist_a': wrist_angle,
            'dforce': self.desired_force if self.in_force_mode else 0,
        }

        if self.rangefinder.data_ready:
            distance = self.rangefinder.distance
            # If the floor is out of range, distance is None
            if distance:
                self.rangefinder.clear_interrupt()
                self.update['grip_sensors']['range'] = distance / 100

    def checkMotorLoad(self, finger_data, wrist_data):
        # Prevent repetitive triggering by checking if the re-enable timer is already running
        if finger_data['load'] < 1000 and finger_data['load'] > self.conf['MAX_SAFE_LOAD'] and not self.finger_torque_reenable_time:
            logging.warning(f"Finger motor load ({finger_data['load']}) exceeds limit. Disabling torque for 1s.")
            self.motors.torque_enable(FINGER, False)
            self.finger_torque_reenable_time = time.time() + 1.0
            
            if self.in_force_mode:
                self.desired_force = self.conf['INITIAL_DESIRED_FORCE']

        if wrist_data['load'] < 1000 and wrist_data['load'] > self.conf['MAX_SAFE_LOAD'] and not self.wrist_torque_reenable_time:
            logging.warning(f"Wrist motor load ({wrist_data['load']}) exceeds limit. Disabling torque for 1s.")
            self.motors.torque_enable(WRIST, False)
            self.wrist_torque_reenable_time = time.time() + 1.0

    def get_current_grip_force(self):
        self.last_finger_data = self.motors.get_feedback(FINGER)
        
        # Values greater than 1000 represent load in the opposite direction, which is irrelevant for grip force
        raw_load = self.last_finger_data['load'] if self.last_finger_data['load'] <= 1000 else 0
        norm_load = min(raw_load / self.conf['MAX_SAFE_LOAD'], 1.0)
        
        # Force Sensitive Resistors (FSRs) drop resistance logarithmically with applied force, 
        # causing a highly non-linear voltage curve (massive drop on light touch, slow drop on heavy press).
        # Applying an exponent > 1 compresses the overly sensitive light-touch region and linearizes the usable force proxy.
        norm_pressure = clamp((max(0.0, 3.3 - self.pressure_sensor.voltage) / 3.3) ** 2.5, 0.0, 1.0)
        
        # Low-pass filter mitigates sensor noise and prevents the PID derivative term from causing severe jitter
        weighted_sum = (norm_pressure * self.conf['PRESSURE_WEIGHT']) + (norm_load * (1-self.conf['PRESSURE_WEIGHT']))
        self.filtered_force = (self.conf['FILTER_COEFF'] * weighted_sum) + ((1 - self.conf['FILTER_COEFF']) * self.filtered_force)
        
        return self.filtered_force, norm_pressure

    def startOtherTasks(self):
        # any tasks started here must stop on their own when self.run_server goes false
        umtask = asyncio.create_task(self.updateMotors())
        return [umtask]

    async def updateMotors(self):
        try:
            # runs at startup of server
            self.motors.torque_enable(FINGER, True)
            self.motors.torque_enable(WRIST, True)

            # initialize with current positions to prevent sudden moves
            self.desired_wrist_angle = self.saved_unrolled_wrist_angle
            logging.info(f'wrist angle at startup = {self.desired_wrist_angle}')
            self.desired_finger_angle = self.saved_finger_angle
            logging.info(f'finger angle at startup = {self.desired_finger_angle}')

            last_movement_time = time.time()
            
            # State tracking to detect what actually changed in the current loop
            last_sent_finger_angle = self.desired_finger_angle
            last_sent_wrist_angle = self.desired_wrist_angle

            while self.run_server:
                now = time.time()
                
                # Halt routine updates if the calibration script requested motor control
                if self.motor_loop_pause:
                    await asyncio.sleep(0.1)
                    continue

                # Restore torque after safety timeout expires
                if self.finger_torque_reenable_time and now >= self.finger_torque_reenable_time:
                    logging.info("Safety timeout expired. Re-enabling finger motor torque.")
                    self.motors.torque_enable(FINGER, True)
                    self.finger_torque_reenable_time = 0.0
                    
                if self.wrist_torque_reenable_time and now >= self.wrist_torque_reenable_time:
                    logging.info("Safety timeout expired. Re-enabling wrist motor torque.")
                    self.motors.torque_enable(WRIST, True)
                    self.wrist_torque_reenable_time = 0.0

                # Actions are only valid for a short time.
                if now > self.time_last_commanded_finger_speed + self.conf['ACTION_TIMEOUT']:
                    self.desired_finger_speed = 0
                if now > self.time_last_commanded_wrist_speed + self.conf['ACTION_TIMEOUT']:
                    self.desired_wrist_speed = 0

                # update wrist
                self.desired_wrist_angle  = clamp(self.desired_wrist_angle + self.desired_wrist_speed * DT, 0, 1080)
                
                wrist_changed = False
                if last_sent_wrist_angle != self.desired_wrist_angle:
                    self.motors.set_position(WRIST, (self.desired_wrist_angle / 360 * STEPS_PER_REV) - self.wrist_step_offset)
                    last_sent_wrist_angle = self.desired_wrist_angle
                    wrist_changed = True

                # update fingers
                current_force, current_pressure = self.get_current_grip_force()
                
                # check for rising edges in pad pressure
                self.countFingerPresses(current_pressure)
                
                # Check for safety overload conditions on every loop iteration
                # We fetch wrist data here as well since get_current_grip_force only pulls finger data
                wrist_data = self.motors.get_feedback(WRIST)
                self.checkMotorLoad(self.last_finger_data, wrist_data)

                if not self.in_force_mode:
                    pa = self.desired_finger_angle
                    self.desired_finger_angle = clamp(self.desired_finger_angle + self.desired_finger_speed * DT, -90, 90)
                    if abs(self.desired_finger_speed) > 0:
                        fa = self.getFingerAngle()
                    
                    # Enter force mode dynamically upon contact while closing
                    if current_pressure > self.conf['FORCE_TRIGGER_THRESHOLD'] and self.desired_finger_speed > 0:
                        self.in_force_mode = True
                        self.desired_force = self.conf['INITIAL_DESIRED_FORCE']
                        self.fingerpid._error_sum = 0
                
                if self.in_force_mode:
                    # Modulate desired force based on finger speed commands
                    self.desired_force += self.desired_finger_speed * DT * self.conf['FORCE_RATE_MULTIPLIER']
                    
                    # Revert to position mode if commanded below zero force
                    if self.desired_force < 0:
                        self.in_force_mode = False
                        self.desired_force = 0
                        self.desired_finger_angle = self.getFingerAngle()
                    else:
                        self.desired_force = clamp(self.desired_force, 0.0, 1.0)
                        self.fingerpid.setpoint = self.desired_force
                        
                        if abs(self.desired_force - current_force) >= self.conf['FORCE_DEADBAND']:
                            self.desired_finger_angle = clamp(self.desired_finger_angle + self.fingerpid.calculate(current_force), -90, 90)

                finger_changed = False
                if last_sent_finger_angle != self.desired_finger_angle:
                    motorpos = remap(self.desired_finger_angle, -90, 90, self.finger_open_pos, self.finger_closed_pos)
                    self.motors.set_position(FINGER, motorpos)
                    last_sent_finger_angle = self.desired_finger_angle
                    finger_changed = True
                    
                # Record time logic for tracking inactivity to save layout state
                if finger_changed or wrist_changed:
                    last_movement_time = now
                elif now - last_movement_time > 5.0:
                    # Triggers exactly once per stop since properties sync immediately after write
                    if self.unrolled_wrist_angle != self.saved_unrolled_wrist_angle or self.desired_finger_angle != self.saved_finger_angle:
                        self.saved_unrolled_wrist_angle = self.unrolled_wrist_angle
                        self.saved_finger_angle = self.desired_finger_angle

                        with open('arp_gripper_state.json', 'w') as f:
                            json.dump({
                                'finger_open_pos': getattr(self, 'finger_open_pos', 0),
                                'finger_closed_pos': getattr(self, 'finger_closed_pos', 0),
                                'unrolled_wrist_angle': self.saved_unrolled_wrist_angle,
                                'finger_angle': self.saved_finger_angle
                            }, f)
                
                await asyncio.sleep(DT)
        except Exception as e:
            logging.exception("problem in motor tracking loop")

    async def process_imu(self, ws):
        """
        Observe gyro and fit a sin curve for use in active swing cancellation
        """
        while True:
            now = time.time()
            dt = now - self.last_time_imu
            self.last_time_imu = now

            # Get current angular velocity (rad/s)
            # MPU6050 returns (gx, gy, gz) in rad/s
            current_gyro = np.array(self.imu.gyro[:2])

            step_angle = self.omega * dt
            c_step, s_step = np.cos(step_angle), np.sin(step_angle)
            # Rotation matrix to keep the virtual pendulum 'swinging' in sync with time.
            self.state = self.state @ np.array([[c_step, -s_step], [s_step, c_step]])

            # Correct the Sine component (Col 0) using the actual Gyro reading.
            self.state[:, 0] += self.observation_gain * (current_gyro - self.state[:, 0])

            # The state of the model only needs to be sent to the client at the regular rate
            self.update['sm'] = self.state.tolist()
            self.update['st'] = self.last_time_imu

            # To use this information, the motion controller should evaluate the derivative of the model at future times
            # based on expected latency in order to obtain a prediction of the angular acceleration in X and Y.
            # the compensatory velocity to apply to the marker box is proportional to the inverse of that angular acceleration.
            # the compensation must be rotated to account for the wrist.

            await asyncio.sleep(1/100)

    def setFingerSpeed(self, deg_per_second):
        self.time_last_commanded_finger_speed = time.time()
        self.desired_finger_speed = deg_per_second

    def setWristSpeed(self, deg_per_second):
        self.time_last_commanded_wrist_speed = time.time()
        self.desired_wrist_speed = deg_per_second
            
    def setFingers(self, angle):
        self.in_force_mode = False
        self.desired_finger_angle = clamp(angle, -90, 90)
            
    def setWrist(self, angle):
        # Accept an angle in degrees between 0 and 1080 (3 revolutions)
        self.desired_wrist_angle = clamp(angle, 0, 1080)

    async def findTouchPoint(self):
        pos = self.motors.get_position(FINGER)
        # open a few degrees in case fingers were already touching.
        rel = 200
        self.motors.set_position(FINGER, pos + rel)
        await asyncio.sleep(0.5)
        data = self.motors.get_feedback(FINGER)

        # confirm no pressure on finger pad
        v = self.pressure_sensor.voltage
        assert v > 3, "Voltage too low on finger pad. Is pressure sensor connected?"

        # slowly close until the fingerpad voltage drops below 2V
        start = time.time()
        load = 0
        while v > 3.0 and time.time() < start+16:
            # logging.info(f'self.motors.set_position(FINGER, {pos + rel})')
            self.motors.set_position(FINGER, pos + rel)
            rel -= 20

            # you cannot command negative positions from these servos (though they will report negative positions in feedback)
            # so as the fingers get more closed, we may cross that point and need to reset midpoint again to move it off the edge of our working range.
            if pos+rel < 0:
                self.motors.set_speed(FINGER, 0)
                self.motors.torque_enable(FINGER, False)
                await asyncio.sleep(0.05)
                self.motors.reset_encoder_to_midpoint(FINGER)
                await asyncio.sleep(0.05)
                pos = self.motors.get_position(FINGER)
                rel = 0
                logging.info(f'reset midpoint position is now {pos}')

            await asyncio.sleep(0.05)
            v = self.pressure_sensor.voltage
            data = self.motors.get_feedback(FINGER)
            load = data["load"]
            if load < 1000: # ignore load values over 1000, they indicate force in the other direction
                if load>450:
                    self.motors.torque_enable(FINGER, False)
                    raise RuntimeError("motor load too high while no finger pressure detected")
        self.motors.set_speed(FINGER, 0)

        touch_pos = self.motors.get_position(FINGER)
        logging.info(f"Motor encoder position at finger touch = {touch_pos}")
        return touch_pos

    async def measureFingerContact(self):
        try:
            # pause the motor control loop
            self.motor_loop_pause = True
            # measure the motor angle where the fingers touch.
            logging.info(f"Calibrating finger servo...")
            self.motors.reset_encoder_to_midpoint(FINGER)

            touch_pos = await self.findTouchPoint()

            self.finger_closed_pos = touch_pos
            self.finger_open_pos = self.finger_closed_pos + FINGER_TRAVEL_STEPS
            self.saved_finger_angle = 90 
            self.desired_finger_angle = 90

            # Put the midpoint somwhere actually in the middle because we need nearly the full 4096 range
            self.motors.set_position(FINGER, touch_pos + 1800)
            await asyncio.sleep(2)
            self.motors.reset_encoder_to_midpoint(FINGER)
            touch_pos = await self.findTouchPoint()

            self.finger_closed_pos = touch_pos
            self.finger_open_pos = self.finger_closed_pos + FINGER_TRAVEL_STEPS
            self.saved_finger_angle = 90 
            self.desired_finger_angle = 90

            with open('arp_gripper_state.json', 'w') as f:
                json.dump({
                    'finger_open_pos': self.finger_open_pos,
                    'finger_closed_pos': self.finger_closed_pos,
                    'unrolled_wrist_angle': self.saved_unrolled_wrist_angle,
                    'finger_angle': self.saved_finger_angle
                }, f)

            # re-open to a relaxed position
            self.setFingers(70)

        except Exception as e:
            logging.exception("problem in finger calibration task")
        finally:
            self.motor_loop_pause = False
            self.update['finger_contact_calibration_complete'] = None

    async def processOtherUpdates(self, update, tg):
        if 'set_finger_angle' in update:
            self.setFingers(float(update['set_finger_angle']))
        if 'set_wrist_angle' in update:
            self.setWrist(float(update['set_wrist_angle']))
        if 'set_finger_speed' in update:
            self.setFingerSpeed(float(update['set_finger_speed']))
        if 'set_wrist_speed' in update:
            self.setWristSpeed(float(update['set_wrist_speed']))
        if 'measure_finger_contact' in update:
            asyncio.create_task(self.measureFingerContact())
        if 'identify' in update:
            self.identify()

    def identify(self):
        """ make a noise """
        self.motor_loop_pause = True
        pos = self.motors.get_position(FINGER)
        # open and close a few degrees
        self.motors.set_position(FINGER, pos + 60)
        time.sleep(0.2)
        self.motors.set_position(FINGER, pos)
        self.motor_loop_pause = False

    def countFingerPresses(self, pressure):
        """ Detect rising edge in finger pressure.
        If there is no client and it occurs five times in two seconds, set the reset wifi event from the parent class"""
        if self.reset_wifi_event is None:
            return
        pressed = pressure > 0.2

        if pressed and not self.was_pressed:
            # rising edge has been detected. 
            self.taps.append(time.time())
            # how many taps have occured in the past two seconds?
            tap_count = len(list(filter(lambda t: t>time.time()-2, self.taps)))
            # trigger special behavior. event is watched in anchor_server.py
            if tap_count == 5:
                self.reset_wifi_event.set()
        self.was_pressed = pressed


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    gs = GripperArpServer()
    asyncio.run(gs.main())