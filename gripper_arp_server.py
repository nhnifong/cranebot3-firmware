import asyncio
from anchor_server import RobotComponentServer
from getmac import get_mac_address
import logging
from collections import deque
from util import remap, clamp
import time
import pickle
import os
import board
import busio
# import adafruit_bno08x
# from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_vl53l1x import VL53L1X # rangefinder
from simple_st3215 import SimpleSTS3215
from adafruit_ads1x15 import ADS1015, AnalogIn, ads1x15 # analog2digital converter for pressure

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

        # i2c = busio.I2C(board.SCL, board.SDA)
        # self.imu = BNO08X_I2C(i2c, address=0x4b)
        # self.imu.enable_feature(adafruit_bno08x.BNO_REPORT_ROTATION_VECTOR)

        self.rangefinder = VL53L1X(i2c)
        model_id, module_type, mask_rev = self.rangefinder.model_info
        logging.info(f'Rangefinder Model ID: 0x{model_id:0X} Module Type: 0x{module_type:0X} Mask Revision: 0x{mask_rev:0X}')
        self.rangefinder.distance_mode = 2 # LONG. results returned in centimeters.
        self.rangefinder.start_ranging()

        self.ads = ADS1015(i2c)
        self.pressure_sensor = AnalogIn(ads, ads1x15.Pin.A0)

        self.motors = SimpleSTS3215(port='/dev/serial0', timeout=0.5)

        # the superclass, RobotComponentServer, assumes the presense of this attribute
        self.spooler = None

        unique = ''.join(get_mac_address().split(':'))
        self.service_name = self.service_type + '.' + unique

        self.last_finger_angle = 0
        self.desired_finger_angle = 0

        # try to read the physical positions of winch and finger last written to disk.
        # For the gripper, there's a good change nothing has moved since power down.
        try:
            with open('offsets.pickle', 'rb') as f:
                d = pickle.load(f)
                self.last_finger_angle = d['last_finger_angle']
                self.desired_finger_angle = d['last_finger_angle']
        except FileNotFoundError:
            pass
        except EOFError: # corruption
            os.remove('offsets.pickle')

    def readOtherSensors(self):

        self.update['grip_sensors'] = {
            'time': time.time(),
            # 'quat': self.imu.quaternion,
            'fing_v': self.hat.gpio_pin_value(PRESSURE_PIN),
            # we don't have an encoder that tells us the true finger angle. fing_a is only what it was last commanded to be.
            # this could be easily remedied by using a smart servo or by adding another mouse wheel encoder to the IHM's B port
            'fing_a': self.last_finger_angle,
        }

        if self.rangefinder.data_ready:
            distance = self.rangefinder.distance
            # If the floor is out of range, distance is None
            if distance:
                self.rangefinder.clear_interrupt()
                self.update['grip_sensors']['range'] = distance / 100

    def readPressure(self):
        """
        Report finger pressure as a value beween 0 and 1

        Voltage at no pressure will be close to 3.3 but may require calibration.
        Voltage decreased to as little as 1v with a strong grip.
        """
        return remap(self.pressure_sensor.voltage, 3.3, 0, 0, 1)


    def startOtherTasks(self):
        # any tasks started here must stop on their own when self.run_server goes false
        t2 = asyncio.create_task(self.saveOffsets())
        return [t2]

    async def saveOffsets(self):
        """Periodically save winch length and finger position to disk so we don't recal after power out"""
        i=0
        while self.run_server:
            await asyncio.sleep(1) # less sleep so this won't hold up server shutdown
            i+=1
            if i==30:
                i=0
                with open('offsets.pickle', 'wb') as f:
                    f.write(pickle.dumps({
                          'last_finger_angle': self.last_finger_angle
                        }))

    def setFingers(self, angle):
        # use same finger "angle" range as previous gripper. translate internally.
        self.desired_finger_angle = angle
        target_pos = remap(self.desired_finger_angle, -90, 90, 0, 4000) 
        self.motors.set_position(FINGER_MOTOR_ID, target_pos)

    async def processOtherUpdates(self, update, tg):
        if 'zero_winch_line' in update:
            tg.create_task(self.performZeroWinchLine())
        if 'set_finger_angle' in update:
            self.setFingers(clamp(float(update['set_finger_angle']), -90, 90))

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    gs = GripperArpServer()
    asyncio.run(gs.main())
