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


# values that can be overridden by the controller
default_gripper_conf = {
}


class GripperArpServer(RobotComponentServer):
    def __init__(self, mock_motor=None):
        super().__init__()
        self.conf.update(default_gripper_conf)
        # the observer identifies hardware by the service types advertised on zeroconf
        self.service_type = 'cranebot-gripper-arpeggio-service'

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

        # the superclass, RobotComponentServer, assumes the presense of this attribute
        self.spooler = None

        unique = ''.join(get_mac_address().split(':'))
        self.service_name = self.service_type + '.' + unique

        self.desired_finger_angle = 0
        self.desired_wrist_angle = 0

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
        # if necessary this function could be faster by waiting on all this IO at the same time.

        # 1.35 ms to read data from both motors with two synchronous calls
        t = time.time()
        finger_data = self.motors.get_feedback(FINGER)
        wrist_data = self.motors.get_feedback(WRIST)

        finger_angle = remap(finger_data['position'], self.finger_open_pos, self.finger_closed_pos, -90, 90)
        wrist_angle = remap(wrist_data['position'], 0, 4000, 0, 360)
        pressure_v = remap(self.pressure_sensor.voltage, 3.3, 0, 0, 1)

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
        return []
            
    def setFingers(self, angle):
        # use same finger "angle" range as previous gripper. translate internally.
        # -90 is wide open, and 90 is closed tight.
        # 
        self.desired_finger_angle = angle
        target_pos = remap(self.desired_finger_angle, -90, 90, self.finger_open_pos, self.finger_closed_pos)
        self.motors.set_position(FINGER, target_pos)

            
    def setWrist(self, angle):
        # Accept an angle in degrees.
        self.desired_wrist_angle = angle
        target_pos = self.desired_wrist_angle / 360 * 4000
        self.motors.set_position(WRIST, target_pos)

    async def processOtherUpdates(self, update, tg):
        if 'set_finger_angle' in update:
            self.setFingers(clamp(float(update['set_finger_angle']), -90, 90))
        if 'set_wrist_angle' in update:
            self.setWrist(float(update['set_wrist_angle']))

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    gs = GripperArpServer()
    asyncio.run(gs.main())
