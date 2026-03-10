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
ACTION_TIMEOUT = 0.2

# --- TUNED CONSTANTS ---
FILTER_COEFF = 1.0
FINGER_PID_KP = 2.0
FINGER_PID_KD = 1.0
FINGER_PID_KI = 0.05
FORCE_DEADBAND = 0.02
# -----------------------

MAX_SAFE_LOAD = 750
PRESSURE_WEIGHT = 0.8
LOAD_WEIGHT = 0.2

enable_force_pid = False

# values that can be overridden by the controller
default_gripper_conf = {
}


class GripperArpServer(RobotComponentServer):
    def __init__(self, mock_motor=None):
        super().__init__()
        self.conf.update(default_gripper_conf)
        # the observer identifies hardware by the service types advertised on zeroconf
        self.service_type = 'cranebot-gripper-arpeggio-service'

        self.stream_command = [
            "/usr/bin/rpicam-vid", "-t", "0", "-n",
            "--width=1920", "--height=1080",
            "-o", "tcp://0.0.0.0:8888?listen=1",
            "--codec", "libav",
            "--libav-format", "mpegts",
            "--autofocus-mode", "continuous",
            "--low-latency",
            "--bitrate", "2000kbps"
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
        self.fingerpid = PID(FINGER_PID_KP, FINGER_PID_KI, FINGER_PID_KD, DT)
        self.last_finger_data = None

        # -1.0 to 0 is position (Open to Neutral), 0 to 1.0 is force.
        self.finger_cmd = -1.0
        self.filtered_force = 0

        # try to read the physical positions of winch and finger last written to disk.
        # For the gripper, there's a good change nothing has moved since power down.
        try:
            with open('arp_gripper_state.json', 'r') as f:
                d = json.load(f)
                self.finger_open_pos = d['finger_open_pos']
                self.finger_closed_pos = d['finger_closed_pos']
                self.saved_unrolled_wrist_angle = d.get('unrolled_wrist_angle', 0)
                self.saved_finger_angle = d.get('finger_angle', 0)
        except FileNotFoundError:
            logging.error('Calibration data not found. run qa-gripper-arp')
        except EOFError:
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

        # Accumulate the shortest-path delta between consecutive readings to track multi-turn rotations
        self.unrolled_wrist_angle += (simple_angle - self.last_simple_wrist_angle + 180) % 360 - 180
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
        pressure_v = remap(self.pressure_sensor.voltage, 3.3, 0, 0, 1)

        self.update['grip_sensors'] = {
            'time': t,
            'fing_v': pressure_v,
            'fing_a': finger_angle,
            'wrist_a': wrist_angle,
        }

        if self.rangefinder.data_ready:
            distance = self.rangefinder.distance
            # If the floor is out of range, distance is None
            if distance:
                self.rangefinder.clear_interrupt()
                self.update['grip_sensors']['range'] = distance / 100

    def get_current_grip_force(self):
        # Read both finger motor load and pad pressure and map them to a force reading between 0 and 1
        self.last_finger_data = self.motors.get_feedback(FINGER)
        raw_load = self.last_finger_data['load']
        if raw_load > 1000:
            raw_load = 0 # vals greater than 1000 represent load in the opposite direction - resisting finger opening. ignore here.
        
        norm_pressure = remap(self.pressure_sensor.voltage, 3.3, 0, 0, 1)
        norm_load = min(raw_load / MAX_SAFE_LOAD, 1.0)
        
        # Weighted Sum
        x = (norm_pressure * PRESSURE_WEIGHT) + (norm_load * LOAD_WEIGHT)
        
        # Sigmoid Mapping
        # Standard Sigmoid: f(x) = 1 / (1 + exp(-k * (x - x0)))
        # To map a 0-1 input to a roughly 0-1 output, we shift/scale:
        k = 10  # Steepness of the curve
        x0 = 0.5 # Midpoint
        force_proxy = 1 / (1 + math.exp(-k * (x - x0)))
        
        # Apply Low-Pass Filter to smooth out motor noise/inertia spikes
        self.filtered_force = (FILTER_COEFF * force_proxy) + ((1 - FILTER_COEFF) * self.filtered_force)
        
        return self.filtered_force


    def startOtherTasks(self):
        # any tasks started here must stop on their own when self.run_server goes false
        umtask = asyncio.create_task(self.updateMotors())
        return [umtask]

    async def updateMotors(self):
        # runs at startup of server
        self.motors.torque_enable(FINGER, True)
        self.motors.torque_enable(WRIST, True)

        # initialize with current positions to prevent sudden moves
        self.desired_wrist_angle = self.getWristAngle()
        logging.info(f'wrist angle at startup = {self.desired_wrist_angle}')
        self.desired_finger_angle = self.getFingerAngle()
        logging.info(f'finger angle at startup = {self.desired_finger_angle}')

        
        last_movement_time = time.time()

        # wrist moves based on a commanded speed or position from the client
        # finger moves based on a commanded grip force. grip force is a composite of pad pressure and motor load.

        while self.run_server:
            now = time.time()

            # update wrist
            wrist_before = self.desired_wrist_angle
            # Actions are only valid for a short time.
            if now > self.time_last_commanded_wrist_speed + ACTION_TIMEOUT:
                self.desired_wrist_speed = 0
            # alter the desired position
            self.desired_wrist_angle  = clamp(self.desired_wrist_angle + self.desired_wrist_speed * DT, 0, 1080)
            if wrist_before != self.desired_wrist_angle:
                target_pos = self.desired_wrist_angle / 360 * STEPS_PER_REV
                self.motors.set_position(WRIST, target_pos)

            finger_before = self.desired_finger_angle
            if not enable_force_pid:
                self.desired_finger_angle = remap(self.finger_cmd, -1.0, 1.00, -90, 90)
            else:
                if self.finger_cmd < 0:
                    # position mode: Map -1.0 to 0.0 into -90 to 0 degrees
                    self.desired_finger_angle = remap(self.finger_cmd, -1.0, 0, -90, 0)
                    # Reset PID state so it doesn't "jump" when switching to force mode
                    self.fingerpid._error_sum = 0
                        
                else:
                    # force mode: Map 0.0 to 1.0 into desired grip force
                    self.fingerpid.setpoint = self.finger_cmd
                    current_force = self.get_current_grip_force()
                    
                    error = abs(self.finger_cmd - current_force)
                    if error < FORCE_DEADBAND:
                        adjustment = 0
                    else:
                        logging.info(f'current_force {current_force:0.3f}')
                        adjustment = self.fingerpid.calculate(current_force)

                    self.foo = (current_force, adjustment) 
                    self.desired_finger_angle = clamp(self.desired_finger_angle + adjustment, -90, 90)

            if finger_before != self.desired_finger_angle:
                target_pos = remap(self.desired_finger_angle, -90, 90, self.finger_open_pos, self.finger_closed_pos)
                self.motors.set_position(FINGER, target_pos)
                
            # Record time logic for tracking inactivity to save layout state
            if finger_before != self.desired_finger_angle or wrist_before != self.desired_wrist_angle:
                last_movement_time = now
            elif now - last_movement_time > 5.0:
                # Triggers exactly once per stop since properties sync immediately after write
                if self.unrolled_wrist_angle != self.saved_unrolled_wrist_angle or self.desired_finger_angle != self.saved_finger_angle:
                    self.saved_unrolled_wrist_angle = self.unrolled_wrist_angle
                    self.saved_finger_angle = self.desired_finger_angle
                    logging.info(f'saving unrolled wrist angle of {self.saved_unrolled_wrist_angle}')

                    with open('arp_gripper_state.json', 'w') as f:
                        json.dump({
                            'finger_open_pos': getattr(self, 'finger_open_pos', 0),
                            'finger_closed_pos': getattr(self, 'finger_closed_pos', 0),
                            'unrolled_wrist_angle': self.saved_unrolled_wrist_angle,
                            'finger_angle': self.saved_finger_angle
                        }, f)
            
            await asyncio.sleep(DT)

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
        """
        Increment/Decrement the unified command value.
        Positive deg_per_second increases force command.
        Negative deg_per_second decreases force, eventually moving into open-position mode.
        """
        self.time_last_commanded_finger_speed = time.time()
        # Scale the speed to the -1.0 to 1.0 range
        self.finger_cmd = clamp(self.finger_cmd + deg_per_second * DT / 90, -1.0, 1.0)

    def setWristSpeed(self, deg_per_second):
        self.time_last_commanded_wrist_speed = time.time()
        self.desired_wrist_speed = deg_per_second
            
    def setFingers(self, angle):
        # Override to set direct angle if needed, maps to the cmd range
        angle = clamp(angle, -90, 90)
        if angle <= 0:
            self.finger_cmd = remap(angle, -90, 0, -1.0, 0.0)
        else:
            # If commanding a positive angle manually, treat it as a force request
            self.finger_cmd = remap(angle, 0, 90, 0.0, 1.0)
            
    def setWrist(self, angle):
        # Accept an angle in degrees between 0 and 1080 (3 revolutions)
        angle = clamp(angle, 0, 1080)
        self.desired_wrist_angle = angle
        target_pos = self.desired_wrist_angle / 360 * 4096
        self.motors.set_position(WRIST, target_pos)

    async def processOtherUpdates(self, update, tg):
        if 'set_finger_angle' in update:
            self.setFingers(float(update['set_finger_angle']))
        if 'set_wrist_angle' in update:
            self.setWrist(float(update['set_wrist_angle']))
        if 'set_finger_speed' in update:
            self.setFingerSpeed(float(update['set_finger_speed']))
        if 'set_wrist_speed' in update:
            self.setWristSpeed(float(update['set_wrist_speed']))



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    gs = GripperArpServer()
    asyncio.run(gs.main())
