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

from nf_robot.robot.component_server import RobotComponentServer
from nf_robot.robot.simple_st3215 import SimpleSTS3215
from nf_robot.common.util import remap, clamp, PID
import nf_robot.common.definitions as model_constants

"""Server for the Arpeggio gripper: a pi zero 2W, Camera Module 3 Wide, and Stringman
Gripper Hat."""

FINGER = 1
WRIST = 2
STEPS_PER_REV = 4096
GEAR_RATIO = 10/45 
FINGER_TRAVEL_DEG = 59 
FINGER_TRAVEL_STEPS = FINGER_TRAVEL_DEG / 360 / GEAR_RATIO * STEPS_PER_REV
DT = 1/60
GRAVITY = 9.81
# Ceiling on the raw gyro backlog, in samples: a minute of the 100Hz IMU loop. Only fills
# while the host has recording on, and it drains every send, so this is just a bound on
# what one stalled connection can hold.
RAW_GYRO_MAX_SAMPLES = 6000
# How many samples one message carries, so a backlog drains over several sends rather than
# in one outsized frame.
RAW_GYRO_PER_MESSAGE = 200

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
    'FORCE_RATE_MULTIPLIER': 0.007,
    # (normalized force, 0-1) The target force immediately applied upon entering force mode
    'INITIAL_DESIRED_FORCE': 0.08,
    # (raw motor units, 0-1000) The maximum allowed motor load before capping the normalized load contribution (finger)
    'MAX_SAFE_LOAD': 500,
    'MAX_SAFE_WRIST_LOAD': 900,
    # (dimensionless, 0-1) weight of pad pressure in the composite force; motor load gets
    # the remaining 1-this
    'PRESSURE_WEIGHT': 0.7,
    # Which component_server.stream_modes entry the camera runs, from gripper_stream_modes
    # below. A running stream restarts to pick up a change.
    'STREAM_MODE': 'gripper_control',
    # (m) effective pole length for the swing model. Only used until the host connects and
    # sends the length its config.gripper.pole_type calls for, so this is the older pole:
    # a gripper that is never told is one on a robot too old to have the field.
    'POLE_LENGTH': model_constants.pole_length_abs500
}


# module level, so it can be read without constructing a GripperArpServer, whose __init__
# opens the I2C bus
stream_command = [
    "/usr/bin/rpicam-vid", "-t", "0", "-n",
    # the full-FOV 2304x1296 sensor mode scaled to 684x384 (16:9), which keeps the whole
    # wide field of view instead of the centre crop a square output would force
    "--mode", "2304:1296:10",
    "--width=684", "--height=384",
    "--framerate=60",
    "-o", "tcp://0.0.0.0:8888?listen=1&tcp_nodelay=1",
    "--codec", "libav",
    "--libav-format", "mpegts",
    "--autofocus-mode", "continuous",
    "--low-latency",
    "--bitrate", "1200kbps"
]

# Modes a gripper accepts, its normal one first. Both keep the sensor mode above, so
# capture mode is more pixels of the same scene rather than a different framing - which is
# what lets one camera calibration cover both.
gripper_stream_modes = ('gripper_control', 'gripper_capture')


class GripperArpServer(RobotComponentServer):
    def __init__(self):
        super().__init__()
        self.conf.update(default_gripper_conf)
        self.stream_modes = gripper_stream_modes
        # the observer identifies hardware by the service types advertised on zeroconf
        self.service_type = 'cranebot-gripper-arpeggio-service'

        self.stream_command = stream_command

        i2c = busio.I2C(board.SCL, board.SDA)

        self.rangefinder = VL53L1X(i2c)
        model_id, module_type, mask_rev = self.rangefinder.model_info
        logging.info(f'Rangefinder Model ID: 0x{model_id:0X} Module Type: 0x{module_type:0X} Mask Revision: 0x{mask_rev:0X}')
        self.rangefinder.distance_mode = 2 # LONG, reports centimeters
        self.rangefinder.start_ranging()

        self.ads = ADS1015(i2c)
        self.pressure_sensor = AnalogIn(self.ads, ads1x15.Pin.A0)

        self.imu = MPU6050(i2c)

        self.motors = SimpleSTS3215()
        self.motors.configure_multiturn(WRIST)

        # RobotComponentServer expects this attribute; a gripper has no spool
        self.spooler = None

        unique = ''.join(get_mac_address().split(':'))
        self.service_name = self.service_type + '.' + unique

        self.desired_finger_angle = 0
        self.desired_wrist_angle = 0
        
        self.last_simple_wrist_angle = None
        self.unrolled_wrist_angle = 0
        self.wrist_step_offset = 0

        # set while resetWrist/untwistWrist reposition the wrist; other wrist commands are
        # ignored until they finish
        self.wrist_busy = False
            
        self.desired_finger_speed = 0
        self.desired_wrist_speed = 0

        self.time_last_commanded_finger_speed = 0
        self.time_last_commanded_wrist_speed = 0

        self.last_time_imu = time.time()

        self.last_gyro = np.zeros(2)
        self.filtered_alpha = np.zeros(2)

        # the pendulum's angular frequency, rad/s. Recomputed when the client changes
        # POLE_LENGTH (see config_updated).
        self.omega = np.sqrt(GRAVITY / self.conf['POLE_LENGTH'])

        # sin curves fitted to the gyro. Row 0 X swing, row 1 Y; col 0 velocity (sine),
        # col 1 phase tracker (cosine).
        self.state = np.zeros((2, 2))

        # Raw gyro samples held for the host, only while it asks for them ('record_gyro').
        # The fitted model above cannot answer what the swing frequency actually is - it is
        # built assuming omega - so measuring the pole needs the untouched readings.
        self.record_gyro = False
        self.raw_gyro = deque(maxlen=RAW_GYRO_MAX_SAMPLES)

        # how much of each real gyro reading to average into the model per step
        self.observation_gain = 0.1

        self.fingerpid = PID(self.conf['FINGER_PID_KP'], self.conf['FINGER_PID_KI'], self.conf['FINGER_PID_KD'], DT)
        self.last_finger_data = None

        self.filtered_force = 0.0
        self.in_force_mode = False
        self.desired_force = 0.0
        
        # when the overload cutout lets each motor have torque back
        self.finger_torque_reenable_time = 0.0
        self.wrist_torque_reenable_time = 0.0

        # defaults for persistent values
        self.finger_open_pos = -1000
        self.finger_closed_pos = 1000
        self.saved_unrolled_wrist_angle = 0
        self.saved_finger_angle = 0
        # Whole turns that have fallen outside the [0, 1080] window across boots (see
        # getWristAngle), tracking cumulative cable twist. The cable tolerates about +/-30.
        self.total_wrist_turns = 0

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
                    self.total_wrist_turns = d.get('total_wrist_turns', 0)
            except (json.JSONDecodeError, EOFError):
                os.remove('arp_gripper_state.json')

    def save_state(self):
        with open('arp_gripper_state.json', 'w') as f:
            json.dump({
                'finger_open_pos': getattr(self, 'finger_open_pos', 0),
                'finger_closed_pos': getattr(self, 'finger_closed_pos', 0),
                'unrolled_wrist_angle': self.saved_unrolled_wrist_angle,
                'finger_angle': self.saved_finger_angle,
                'total_wrist_turns': self.total_wrist_turns,
            }, f)

    async def resetWrist(self):
        """Re-establish 540 as the wrist's neutral position.

        Turns one revolution negative before turning one positive, so the wire ends up
        with no net twist, then zeroes the offsets.
        """
        if self.wrist_busy:
            logging.info('resetWrist requested while wrist busy; ignoring')
            return
        self.wrist_busy = True
        a = self.getWristAngle()
        self.setWrist(a - 360)
        await asyncio.sleep(4)
        self.motor_loop_pause = True
        self.motors.reset_encoder_to_midpoint(WRIST)
        await asyncio.sleep(0.1)
        wrist_data = self.motors.get_feedback(WRIST)
        simple_angle = wrist_data['position'] / STEPS_PER_REV * 360
        logging.info(f'after encoder reset, wrist reports {simple_angle}. should be 180. moving one rev positive to 540')
        self.last_simple_wrist_angle = 180.0
        self.unrolled_wrist_angle = 180.0
        self.wrist_step_offset = 0.0
        self.setWrist(180)
        self.motor_loop_pause = False
        await asyncio.sleep(0.1)
        self.setWrist(540)
        end = time.time() + 4
        while time.time() < end:
            logging.info(self.getWristAngle())
            await asyncio.sleep(0.05)
        a = self.getWristAngle()
        logging.info(f'Resest wrist. should be 540. ({a})')
        self.wrist_busy = False

    def _nearest_multiple_of_360(self, angle, direction):
        """Nearest multiple of 360 to `angle`, reached by moving in `direction`
        (+1 increases angle, -1 decreases), clamped to the [0, 1080] command range."""
        if direction > 0:
            target = (math.floor(angle / 360) + 1) * 360
            if target > 1080:
                target -= 360
        else:
            target = (math.ceil(angle / 360) - 1) * 360
            if target < 0:
                target += 360
        return clamp(target, 0, 1080)

    async def _untwistOneTurn(self, direction):
        """Turn the wrist one revolution in `direction` (+1 or -1), relieving one turn of
        cable twist.

        Works by two encoder midpoint resets, each re-defining the current physical
        position as 180 degrees without moving. unrolled_wrist_angle lands on 180, usually
        not where it started, but its value mod 360 changes by the same amount the wrist
        physically turned, so room heading tracking stays right. wrist_step_offset ends at
        0 with the encoder's zero shifted a full revolution.
        """
        for _ in range(2):
            target = self._nearest_multiple_of_360(self.unrolled_wrist_angle, direction)
            self.setWrist(target)
            await asyncio.sleep(2)

            self.motor_loop_pause = True
            self.motors.reset_encoder_to_midpoint(WRIST)
            await asyncio.sleep(0.1)
            self.last_simple_wrist_angle = 180.0
            self.unrolled_wrist_angle = 180.0
            self.wrist_step_offset = 0.0
            # follow the reset here too, or the motor loop would see a stale target and
            # drive the wrist back to it
            self.desired_wrist_angle = 180.0
            self.last_sent_wrist_angle = 180.0
            self.motor_loop_pause = False
            await asyncio.sleep(0.1)

    async def untwistWrist(self, turns=None):
        """Turn the wrist through whole revolutions to relieve accumulated cable twist.

        `turns` is how many to remove from total_wrist_turns, defaulting to all of them.
        Other wrist commands are ignored until this finishes.
        """
        if self.wrist_busy:
            logging.info('untwistWrist requested while wrist busy; ignoring')
            return
        if turns is None:
            turns = self.total_wrist_turns
        logging.info(f'untwistWrist {turns} turns requested.')
        turns = int(turns)
        if turns == 0:
            self.update['untwist_complete'] = {'turns_done': 0, 'total_wrist_turns': self.total_wrist_turns}
            return

        direction = -1 if turns > 0 else 1
        n = abs(turns)

        self.wrist_busy = True
        try:
            for _ in range(n):
                await self._untwistOneTurn(direction)
                self.total_wrist_turns += direction
                self.update['total_wrist_turns'] = self.total_wrist_turns
                self.save_state()
        finally:
            self.wrist_busy = False
            self.update['untwist_complete'] = {'turns_done': n, 'total_wrist_turns': self.total_wrist_turns}

    def getWristAngle(self):
        """Wrist angle in [0, 1080], the same three-revolution range commands use, 540
        being neutral. The motor only reports position within one revolution, so the extra
        turns are tracked here.

        At boot the encoder knows the physical angle mod 360 but not which of the three
        revolutions it belonged to, so unrolled_wrist_angle is re-anchored down by 0, 360
        or 720 depending on where it was at shutdown. Either way the full command range is
        available with no startup motion and the angle mod 360 - the heading the camera
        math needs - survives.

        PREFER SHUTTING DOWN WITH THE ANGLE IN [0, 360), the only range that shifts by
        zero. A shift re-centers [0, 1080] on a different 3-turn slice of physical
        rotation, so neutral drifts off-center, and repeated shifts the same way across
        power cycles can walk the cable toward its ~30 turn twist limit without any single
        session ever leaving [0, 1080]. So park at, say, 180 rather than neutral 540.
        """
        wrist_data = self.motors.get_feedback(WRIST)
        simple_angle = wrist_data['position'] / STEPS_PER_REV * 360

        # first read after boot: rebuild the multi-turn angle from saved state, assuming
        # the joint moved less than half a turn while powered off
        if self.last_simple_wrist_angle is None:
            self.last_simple_wrist_angle = simple_angle

            error = (simple_angle - self.saved_unrolled_wrist_angle + 180) % 360 - 180
            self.unrolled_wrist_angle = self.saved_unrolled_wrist_angle + error

            # gap between our continuous frame and the motor's encoder frame, which wraps
            # 0-4095 at boot
            expected_steps = self.unrolled_wrist_angle / 360 * STEPS_PER_REV
            offset = expected_steps - wrist_data['position']

            # Close that gap in whole revolutions rather than by moving the motor, so every
            # angle in [0, 1080] maps to a non-negative motor position this session.
            revolutions = round(offset / STEPS_PER_REV)
            self.unrolled_wrist_angle -= revolutions * 360
            self.saved_unrolled_wrist_angle -= revolutions * 360
            self.wrist_step_offset = 0.0

            if revolutions != 0:
                # keep total_wrist_turns*360 + unrolled_wrist_angle, the absolute physical
                # rotation, unchanged by the re-anchoring
                self.total_wrist_turns += revolutions
                self.update['total_wrist_turns'] = self.total_wrist_turns
                self.save_state()

            return clamp(self.unrolled_wrist_angle, 0, 1080)

        # Unrolled against desired_wrist_angle, not the last reading: intent advances
        # rigidly at 60Hz, so a read or CPU stutter cannot alias a turn and walk
        # unrolled_wrist_angle out of bounds.
        error = (simple_angle - self.desired_wrist_angle + 180) % 360 - 180
        self.unrolled_wrist_angle = self.desired_wrist_angle + error
        
        self.last_simple_wrist_angle = simple_angle
        return clamp(self.unrolled_wrist_angle, 0, 1080)

    def getFingerAngle(self):
        # last_finger_data comes from get_current_grip_force, in the 60Hz motor loop
        if self.last_finger_data is not None:
            return remap(self.last_finger_data['position'], self.finger_open_pos, self.finger_closed_pos, -90, 90)
        else:
            return 0

    def getAngleFromVertical(self):
        """Unsigned degrees between the gripper's down axis and true vertical, from the
        accelerometer: ~0 hanging straight down, ~90 horizontal.

        Carries no tilt direction, and only means anything while the gripper is roughly
        still - the accelerometer cannot tell gravity from motion.
        """
        ax, ay, az = self.imu.acceleration  # m/s^2
        horizontal = math.sqrt(ax * ax + ay * ay)
        return math.degrees(math.atan2(horizontal, az))

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
            # None when the floor is out of range
            if distance:
                self.rangefinder.clear_interrupt()
                self.update['grip_sensors']['range'] = distance / 100

        # drained here rather than in process_imu so a whole send interval's samples travel
        # together, keeping every one of them instead of only the last write before a send
        if self.raw_gyro:
            batch = [self.raw_gyro.popleft() for _ in range(min(len(self.raw_gyro), RAW_GYRO_PER_MESSAGE))]
            self.update['gyro_record'] = batch

    def checkMotorLoad(self, finger_data, wrist_data):
        """Cut torque for a second on either motor that is overloaded."""
        # a running re-enable timer means this already fired; don't stack cutouts
        if finger_data['load'] < 1000 and finger_data['load'] > self.conf['MAX_SAFE_LOAD'] and not self.finger_torque_reenable_time:
            logging.warning(f"Finger motor load ({finger_data['load']}) exceeds limit. Disabling torque for 1s.")
            self.motors.torque_enable(FINGER, False)
            self.finger_torque_reenable_time = time.time() + 1.0
            
            if self.in_force_mode:
                self.desired_force = self.conf['INITIAL_DESIRED_FORCE']

        if wrist_data['load'] < 1000 and wrist_data['load'] > self.conf['MAX_SAFE_WRIST_LOAD'] and not self.wrist_torque_reenable_time and not self.wrist_busy:
            logging.warning(f"Wrist motor load ({wrist_data['load']}) exceeds limit. Disabling torque for 1s.")
            self.motors.torque_enable(WRIST, False)
            self.wrist_torque_reenable_time = time.time() + 1.0

    def get_current_grip_force(self):
        """(filtered composite grip force, raw normalized pad pressure), both 0-1."""
        self.last_finger_data = self.motors.get_feedback(FINGER)

        # over 1000 means load in the opening direction, which is not grip force
        raw_load = self.last_finger_data['load'] if self.last_finger_data['load'] <= 1000 else 0
        norm_load = min(raw_load / self.conf['MAX_SAFE_LOAD'], 1.0)

        # The FSR's resistance falls logarithmically with force - a big voltage drop on a
        # light touch, very little on a hard press - so the exponent flattens the
        # oversensitive light end into something usable as a force proxy.
        norm_pressure = clamp((max(0.0, 3.3 - self.pressure_sensor.voltage) / 3.3) ** 2.5, 0.0, 1.0)

        # low-pass, or sensor noise reaches the PID's derivative term as jitter
        weighted_sum = (norm_pressure * self.conf['PRESSURE_WEIGHT']) + (norm_load * (1-self.conf['PRESSURE_WEIGHT']))
        self.filtered_force = (self.conf['FILTER_COEFF'] * weighted_sum) + ((1 - self.conf['FILTER_COEFF']) * self.filtered_force)
        
        return self.filtered_force, norm_pressure

    def startOtherTasks(self):
        # any tasks started here must stop on their own when self.run_server goes false
        umtask = asyncio.create_task(self.updateMotors())
        return [umtask]

    async def updateMotors(self):
        """The 60Hz loop that owns both motors: applies commanded speeds, runs the finger
        force controller, and saves state once movement stops."""
        try:
            self.motors.torque_enable(FINGER, True)
            self.motors.torque_enable(WRIST, True)

            # reconcile wrist tracking with the motor's boot position before commanding
            # anything, which anchors wrist_step_offset at 0
            self.getWristAngle()

            # start from where the hardware already is, so nothing lurches
            self.desired_wrist_angle = self.saved_unrolled_wrist_angle
            logging.info(f'wrist angle at startup = {self.desired_wrist_angle}')
            self.desired_finger_angle = self.saved_finger_angle
            logging.info(f'finger angle at startup = {self.desired_finger_angle}')

            last_movement_time = time.time()

            # what was last written to each motor, so the loop only sends real changes
            last_sent_finger_angle = self.desired_finger_angle
            self.last_sent_wrist_angle = self.desired_wrist_angle

            while self.run_server:
                now = time.time()
                
                # something else (calibration, untwist) has taken the motors
                if self.motor_loop_pause:
                    await asyncio.sleep(0.1)
                    continue

                if self.finger_torque_reenable_time and now >= self.finger_torque_reenable_time:
                    logging.info("Safety timeout expired. Re-enabling finger motor torque.")
                    self.motors.torque_enable(FINGER, True)
                    self.finger_torque_reenable_time = 0.0
                    
                if self.wrist_torque_reenable_time and now >= self.wrist_torque_reenable_time:
                    logging.info("Safety timeout expired. Re-enabling wrist motor torque.")
                    self.motors.torque_enable(WRIST, True)
                    self.wrist_torque_reenable_time = 0.0

                # a speed command expires, so a dropped connection stops the motors
                if now > self.time_last_commanded_finger_speed + self.conf['ACTION_TIMEOUT']:
                    self.desired_finger_speed = 0
                if now > self.time_last_commanded_wrist_speed + self.conf['ACTION_TIMEOUT']:
                    self.desired_wrist_speed = 0

                # update wrist
                self.desired_wrist_angle  = clamp(self.desired_wrist_angle + self.desired_wrist_speed * DT, 0, 1080)
                
                wrist_changed = False
                if self.last_sent_wrist_angle != self.desired_wrist_angle:
                    self.motors.set_position(WRIST, (self.desired_wrist_angle / 360 * STEPS_PER_REV) - self.wrist_step_offset)
                    self.last_sent_wrist_angle = self.desired_wrist_angle
                    wrist_changed = True

                # update fingers
                current_force, current_pressure = self.get_current_grip_force()
                
                self.countFingerPresses(current_pressure)

                # get_current_grip_force only reads the finger motor
                wrist_data = self.motors.get_feedback(WRIST)
                self.checkMotorLoad(self.last_finger_data, wrist_data)

                if not self.in_force_mode:
                    pa = self.desired_finger_angle
                    self.desired_finger_angle = clamp(self.desired_finger_angle + self.desired_finger_speed * DT, -90, 90)
                    if abs(self.desired_finger_speed) > 0:
                        fa = self.getFingerAngle()
                    
                    # touching something while closing hands the fingers to the force
                    # controller, so the operator's speed command becomes a force command
                    if current_pressure > self.conf['FORCE_TRIGGER_THRESHOLD'] and self.desired_finger_speed > 0:
                        self.in_force_mode = True
                        self.desired_force = self.conf['INITIAL_DESIRED_FORCE']
                        self.fingerpid._error_sum = 0

                if self.in_force_mode:
                    self.desired_force += self.desired_finger_speed * DT * self.conf['FORCE_RATE_MULTIPLIER']

                    # commanding below zero force is how the operator lets go
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
                    
                # Persist the pose 5s after motion stops, so a power cut leaves the saved
                # angles right. Writes once per stop: the saved values match immediately.
                if finger_changed or wrist_changed:
                    last_movement_time = now
                elif now - last_movement_time > 5.0:
                    if self.unrolled_wrist_angle != self.saved_unrolled_wrist_angle or self.desired_finger_angle != self.saved_finger_angle:
                        self.saved_unrolled_wrist_angle = self.unrolled_wrist_angle
                        self.saved_finger_angle = self.desired_finger_angle

                        self.save_state()
                
                await asyncio.sleep(DT)
        except Exception as e:
            logging.exception("problem in motor tracking loop")

    def config_updated(self, changed):
        if 'POLE_LENGTH' in changed:
            self.omega = np.sqrt(GRAVITY / self.conf['POLE_LENGTH'])

    async def process_imu(self, ws):
        """Fit a swinging pendulum to the gyro at 100Hz and publish it for the host's
        swing cancellation.

        Sending the model rather than raw gyro lets the host project it forward to cover
        control latency; see arp_gripper_client.compute_swing_correction.
        """
        while True:
            now = time.time()
            dt = now - self.last_time_imu
            self.last_time_imu = now

            current_gyro = np.array(self.imu.gyro[:2]) # rad/s

            if self.record_gyro:
                self.raw_gyro.append([now, float(current_gyro[0]), float(current_gyro[1])])

            # advance the virtual pendulum's phase to now, then pull the velocity
            # component toward what the gyro actually reads
            step_angle = self.omega * dt
            c_step, s_step = np.cos(step_angle), np.sin(step_angle)
            self.state = self.state @ np.array([[c_step, -s_step], [s_step, c_step]])
            self.state[:, 0] += self.observation_gain * (current_gyro - self.state[:, 0])

            self.update['sm'] = self.state.tolist()
            self.update['st'] = self.last_time_imu

            await asyncio.sleep(1/100)

    def setFingerSpeed(self, deg_per_second):
        self.time_last_commanded_finger_speed = time.time()
        self.desired_finger_speed = deg_per_second

    def setWristSpeed(self, deg_per_second):
        self.time_last_commanded_wrist_speed = time.time()
        self.desired_wrist_speed = deg_per_second
            
    def setFingers(self, angle):
        self.in_force_mode = False
        self.desired_force = 0
        self.desired_finger_angle = clamp(angle, -90, 90)
            
    def setWrist(self, angle):
        # degrees in [0, 1080], three revolutions
        self.desired_wrist_angle = clamp(angle, 0, 1080)

    async def findTouchPoint(self):
        """Close the fingers slowly until the pad reads pressure, returning that encoder
        position. Raises if the motor loads up with no pressure, which means something is
        jammed between the fingers."""
        pos = self.motors.get_position(FINGER)
        # open a few degrees in case the fingers were already touching
        rel = 200
        self.motors.set_position(FINGER, pos + rel)
        await asyncio.sleep(0.5)
        data = self.motors.get_feedback(FINGER)

        # confirm no pressure on finger pad
        v = self.pressure_sensor.voltage
        if v < 2.2:
            logging.info("Voltage too low on finger pad ({v}). Is pressure sensor connected?")

        start = time.time()
        load = 0
        while v > 2.2 and time.time() < start+16:
            self.motors.set_position(FINGER, pos + rel)
            rel -= 20

            # These servos accept no negative position commands, though they report them,
            # so closing far enough runs off the end of the range; re-centre to continue.
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
            if load < 1000: # over 1000 is load in the opening direction
                if load>450:
                    self.motors.torque_enable(FINGER, False)
                    raise RuntimeError("motor load too high while no finger pressure detected")
        self.motors.set_speed(FINGER, 0)

        touch_pos = self.motors.get_position(FINGER)
        logging.info(f"Motor encoder position at finger touch = {touch_pos}")
        return touch_pos

    async def measureFingerContact(self):
        """Calibrate finger_open_pos/finger_closed_pos by feeling for where the fingers
        meet, twice: the first pass moves the encoder midpoint into the middle of the
        range, the second measures against it."""
        try:
            self.motor_loop_pause = True
            logging.info(f"Calibrating finger servo...")
            self.motors.reset_encoder_to_midpoint(FINGER)

            touch_pos = await self.findTouchPoint()

            self.finger_closed_pos = touch_pos
            self.finger_open_pos = self.finger_closed_pos + FINGER_TRAVEL_STEPS
            self.saved_finger_angle = 90 
            self.desired_finger_angle = 90

            # the fingers need nearly the whole 4096 range, so centre the midpoint
            self.motors.set_position(FINGER, touch_pos + 1800)
            await asyncio.sleep(2)
            self.motors.reset_encoder_to_midpoint(FINGER)
            touch_pos = await self.findTouchPoint()

            self.finger_closed_pos = touch_pos
            self.finger_open_pos = self.finger_closed_pos + FINGER_TRAVEL_STEPS
            self.saved_finger_angle = 90
            self.desired_finger_angle = 90

            self.save_state()

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
        if 'set_wrist_angle' in update and not self.wrist_busy:
            self.setWrist(float(update['set_wrist_angle']))
        if 'set_finger_speed' in update:
            self.setFingerSpeed(float(update['set_finger_speed']))
        if 'set_wrist_speed' in update and not self.wrist_busy:
            self.setWristSpeed(float(update['set_wrist_speed']))
        if 'measure_finger_contact' in update:
            asyncio.create_task(self.measureFingerContact())
        if 'query_angle_from_vertical' in update:
            self.update['angle_from_vertical'] = self.getAngleFromVertical()
        if 'identify' in update:
            self.identify()
        if 'reset_wrist' in update and not self.wrist_busy:
            asyncio.create_task(self.resetWrist())
        if 'untwist' in update and not self.wrist_busy:
            asyncio.create_task(self.untwistWrist(update['untwist']))
        if 'record_gyro' in update:
            self.record_gyro = bool(update['record_gyro'])
            if self.record_gyro:
                # a new recording starts empty, so whatever the last one left behind
                # cannot show up in it. Stopping leaves the backlog to finish draining.
                self.raw_gyro.clear()
            logging.info(f'raw gyro recording {"on" if self.record_gyro else "off"}')

    def identify(self):
        """Twitch the fingers, so an operator can tell which gripper this is."""
        self.motor_loop_pause = True
        pos = self.motors.get_position(FINGER)
        self.motors.set_position(FINGER, pos + 60)
        time.sleep(0.2)
        self.motors.set_position(FINGER, pos)
        self.motor_loop_pause = False

    def countFingerPresses(self, pressure):
        """Five taps on the finger pad within two seconds sets the wifi reset event.

        The pad is the only input the gripper has when it cannot reach a host, which is
        exactly when the wifi needs resetting. component_server.watch_for_reset acts on
        the event, and ignores it if a client is connected, so a grasp cannot trip it.
        """
        if self.reset_wifi_event is None:
            return
        pressed = pressure > 0.2

        if pressed and not self.was_pressed:
            self.taps.append(time.time())
            tap_count = len(list(filter(lambda t: t>time.time()-2, self.taps)))
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