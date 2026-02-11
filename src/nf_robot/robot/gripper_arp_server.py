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
# import adafruit_bno08x
# from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_vl53l1x import VL53L1X # rangefinder
from adafruit_ads1x15 import ADS1015, AnalogIn, ads1x15 # analog2digital converter for pressure

from nf_robot.robot.anchor_server import RobotComponentServer
from nf_robot.robot.simple_st3215 import SimpleSTS3215
from nf_robot.common.util import remap, clamp

""" Server for Arpeggio Gripper

Hardware is a Raspberry pi zero 2W, Camera Module 3 Wide, and Stringman Gripper Hat.

the gripper hat has a a2d converter with connected finger pressure sensor,
BNO085 imu on the i2c bus,
a half duplex smart servo comm circuit with two connected st3215 servos
laser rangefinder

the rpi zero 2w's hardware i2c bus may not play nice with the bno085
but this can be avoided with a software i2c bus
dtparam=i2c_arm=off
dtoverlay=i2c-gpio,bus=1,i2c_gpio_sda=2,i2c_gpio_scl=3,i2c_gpio_delay_us=2

"""

FINGER = 1
WRIST = 2
STEPS_PER_REV = 4096
GEAR_RATIO = 10/45 # a finger lever makes this many revolutions per revolution of the drive gear
FINGER_TRAVEL_DEG = 59 # actually 60 but need small margin of space at wide open. 
FINGER_TRAVEL_STEPS = FINGER_TRAVEL_DEG / 360 / GEAR_RATIO * STEPS_PER_REV
DT = 1/60
ACTION_TIMEOUT = 0.2


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
            "--vflip", "--hflip",
            "--autofocus-mode", "continuous",
            "--low-latency",
            "--bitrate", "2000kbps"
        ]

        i2c = busio.I2C(board.SCL, board.SDA)
        # self.imu = BNO08X_I2C(i2c, address=0x4b)
        # self.imu.enable_feature(adafruit_bno08x.BNO_REPORT_ROTATION_VECTOR)

        self.rangefinder = VL53L1X(i2c)
        model_id, module_type, mask_rev = self.rangefinder.model_info
        logging.info(f'Rangefinder Model ID: 0x{model_id:0X} Module Type: 0x{module_type:0X} Mask Revision: 0x{mask_rev:0X}')
        self.rangefinder.distance_mode = 2 # LONG. results returned in centimeters.
        self.rangefinder.start_ranging()

        self.ads = ADS1015(i2c)
        self.pressure_sensor = AnalogIn(self.ads, ads1x15.Pin.A0)

        self.motors = SimpleSTS3215()
        self.motors.configure_multiturn(WRIST)

        # the superclass, RobotComponentServer, assumes the presense of this attribute
        self.spooler = None

        unique = ''.join(get_mac_address().split(':'))
        self.service_name = self.service_type + '.' + unique

        self.desired_finger_angle = 0
        self.desired_wrist_angle = 0
            
        self.desired_finger_speed = 0
        self.desired_wrist_speed = 0

        self.time_last_commanded_finger_speed = 0
        self.time_last_commanded_wrist_speed = 0

        # try to read the physical positions of winch and finger last written to disk.
        # For the gripper, there's a good change nothing has moved since power down.
        try:
            with open('arp_gripper_state.json', 'r') as f:
                d = json.load(f)
                self.finger_open_pos = d['finger_open_pos']
                self.finger_closed_pos = d['finger_closed_pos']
        except FileNotFoundError:
            pass
        except EOFError: # corruption
            os.remove('offsets.pickle')

    def readOtherSensors(self):
        # 1.35 ms to read data from both motors with two synchronous calls
        t = time.time()
        finger_data = self.motors.get_feedback(FINGER)
        wrist_data = self.motors.get_feedback(WRIST)

        finger_angle = remap(finger_data['position'], self.finger_open_pos, self.finger_closed_pos, -90, 90)
        pressure_v = remap(self.pressure_sensor.voltage, 3.3, 0, 0, 1)

        # motor only reports it's position within one revolution.
        # even though you can command multi turns from it.
        # return a value between 0 and 1080, which is the same range we accept commands in.
        # 540 is the neutral position.
        # the only way to relate this measurement to an absolute multi turn position is to reference our last command.

        # Convert raw data to 0-360 degrees
        simple_angle = wrist_data['position'] / 4096 * 360
        # Calculate the difference between where we want to be and where the motor says it is
        # within a single revolution context.
        # gives the shortest distance to the target.
        error = (simple_angle - (self.desired_wrist_angle % 360) + 180) % 360 - 180
        # Subtracting that shortest-path error from the desired angle aligns 
        # the multi-turn value with the motor's actual physical orientation.
        wrist_angle = self.desired_wrist_angle - error
        #Clamp to 0-1080 range
        wrist_angle = clamp(wrist_angle, 0, 1080)

        self.update['grip_sensors'] = {
            'time': t,
            # 'quat': self.imu.quaternion,
            'fing_v': pressure_v,
            'fing_a': finger_angle,
            'wrist_a': wrist_angle,
            # range added below
        }

        if self.rangefinder.data_ready:
            distance = self.rangefinder.distance
            # If the floor is out of range, distance is None
            if distance:
                self.rangefinder.clear_interrupt()
                self.update['grip_sensors']['range'] = distance / 100

        self.checkMotorLoad(finger_data, wrist_data)

    def checkMotorLoad(self, finger_data, wrist_data):
        """
        Check recently read data for overload conditions and act on it.
        TODO, we need to experiment and find some more sensible behavior here, as well as to have a reset mechanism.
        """
        MAX_LOAD = 750 # Motor returns a value between 0 and 1000.
        # but sometimes values are over 1000 in which case they should be ignored
        if finger_data['load'] < 1000 and finger_data['load'] > MAX_LOAD:
            logging.warning(f"Finger motor load ({finger_data['load']}) exceeds limit ({MAX_LOAD}). motor disabled")
            self.motors.torque_enable(FINGER, False)
        if wrist_data['load'] < 1000 and wrist_data['load'] > MAX_LOAD:
            logging.warning(f"Finger motor load ({wrist_data['load']}) exceeds limit ({MAX_LOAD}). motor disabled")
            self.motors.torque_enable(WRIST, False)


    def startOtherTasks(self):
        # any tasks started here must stop on their own when self.run_server goes false
        umtask = asyncio.create_task(self.updateMotors())
        return [umtask]

    async def updateMotors(self):
        while self.run_server:
            now = time.time()

            finger_before = self.desired_finger_angle
            wrist_before = self.desired_wrist_angle
            
            # Actions are only valid for a short time.
            if now > self.time_last_commanded_finger_speed + ACTION_TIMEOUT:
                self.desired_finger_speed = 0
            if now > self.time_last_commanded_wrist_speed + ACTION_TIMEOUT:
                self.desired_wrist_speed = 0

            # alter the desired position
            self.desired_finger_angle = clamp(self.desired_finger_angle + self.desired_finger_speed * DT, -90, 90)
            self.desired_wrist_angle  = clamp(self.desired_wrist_angle + self.desired_wrist_speed * DT, 0, 1080)

            if finger_before != self.desired_finger_angle:
                target_pos = remap(self.desired_finger_angle, -90, 90, self.finger_open_pos, self.finger_closed_pos)
                self.motors.set_position(FINGER, target_pos)

            if wrist_before != self.desired_wrist_angle:
                target_pos = self.desired_wrist_angle / 360 * 4096
                self.motors.set_position(WRIST, target_pos)
            
            await asyncio.sleep(DT)

    def setFingerSpeed(self, deg_per_second):
        self.time_last_commanded_finger_speed = time.time()
        self.desired_finger_speed = deg_per_second

    def setWristSpeed(self, deg_per_second):
        self.time_last_commanded_wrist_speed = time.time()
        self.desired_wrist_speed = deg_per_second
            
    def setFingers(self, angle):
        # use same finger "angle" range as previous gripper. translate internally.
        # -90 is wide open, and 90 is closed tight.
        # 
        angle = clamp(angle, -90, 90)
        self.desired_finger_angle = angle
        target_pos = remap(self.desired_finger_angle, -90, 90, self.finger_open_pos, self.finger_closed_pos)
        self.motors.set_position(FINGER, target_pos)
            
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
