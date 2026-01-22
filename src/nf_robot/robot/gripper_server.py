import asyncio
from inventorhatmini import InventorHATMini, SERVO_1, SERVO_2, ADC
from ioexpander import IN_PU
from ioexpander.common import PID, clamp
from getmac import get_mac_address
import logging
from collections import deque
import time
import pickle
import os
import board
import busio
import adafruit_bno08x
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_vl53l1x import VL53L1X

from nf_robot.robot.anchor_server import RobotComponentServer
from nf_robot.robot.spools import SpoolController

# the rpi zero 2w's hardware i2c bus may not play nice with the bno085
# but this can be avoided with a software i2c bus
# dtparam=i2c_arm=off
# dtoverlay=i2c-gpio,bus=1,i2c_gpio_sda=2,i2c_gpio_scl=3,i2c_gpio_delay_us=2

# the speed will not increase at settings beyond this value
WINCH_MAX_SPEED = 43
# at or below this the motor does not spin
WINCH_DEAD_ZONE = 4
# this constant is obtained from the injora website. assumes the motor is driven at 6 volts.
WINCH_MAX_RPM = 1.0166
# converts from speed setting to rpm. the speed relationship is probably close to linear, but I have not confirmed
SPEED1_REVS = WINCH_MAX_RPM / (WINCH_MAX_SPEED - WINCH_DEAD_ZONE)
# gpio pin of pressure sensing resistor
PRESSURE_PIN = 0
# gpio pin of limit switch. 0 is pressed
LIMIT_SWITCH_PIN = 1

# The UUIDs for the UART service and its transmit characteristic
# These must match the UUIDs on the ESP32-S3 firmware
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# values that can be overridden by the controller
default_gripper_conf = {
    # PID values for pressure loop
    'POS_KP': 4.0,
    'POS_KI': 0.0,
    'POS_KD': 0.022,
    # update rate of finger pressure PID loop in updates per second
    'UPDATE_RATE': 40,
    # voltage of pressure sensor at ideal grip pressure
    'TARGET_HOLDING_PRESSURE': 1.3,
    # threshold just abolve the lowest pressure voltage we expect to read
    'PRESSURE_MIN': 0.1,
    # The total servo value change per second below which we say it has stabilized.
    'MEAN_SERVO_VAL_CHANGE_THRESHOLD': 3,
    # The servo value at which the fingers press against eachother empty with TARGET_HOLDING_PRESSURE
    'FINGER_TOUCH': 80,
    # max open servo value
    'OPEN': -80,
}

class GripperSpoolMotor():
    """
    Motor interface for gripper spools motor with the same methods as MKSSERVO42C
    Currently based on the injora 360 deg 35kg open loop winch servo 
    https://www.injora.com/products/injora-injs035-360-35kg-waterproof-digital-servo-360-steering-winch-wheel-for-rc
    but using a mouse wheel encoder for position feedback.
    """
    def __init__(self, hat):
        self.servo = hat.servos[SERVO_1]
        self.hat = hat
        self.run = True

    def ping(self):
        return True

    def stop(self):
        self.servo.value(0)

    def runConstantSpeed(self, speed):
        # in revolutions per second
        command_speed = max(-127, min(int(SPEED1_REVS * speed), 127))

        if speed == 0:
            command_speed = 0
        elif speed > 0:
            command_speed = speed / SPEED1_REVS + WINCH_DEAD_ZONE
        elif speed < 0:
            command_speed = speed / SPEED1_REVS - WINCH_DEAD_ZONE
        self.servo.value(command_speed)

    def getShaftAngle(self):
        # in revolutions
        # we assume that an encoder has been conected to the motot A port, even if there is no motor
        return True, self.hat.encoders[0].revolutions()

    def getShaftError(self):
        return False, 0 # unsupported

    def getMaxSpeed(self):
        return 1.0166


class RaspiGripperServer(RobotComponentServer):
    def __init__(self, mock_motor=None):
        super().__init__()
        self.conf.update(default_gripper_conf)
        self.service_type = 'cranebot-gripper-service'

        self.hat = InventorHATMini(init_leds=False)
        self.hand_servo = self.hat.servos[SERVO_2]
        self.hat.gpio_pin_mode(PRESSURE_PIN, ADC) # pressure resistor
        self.hat.gpio_pin_mode(LIMIT_SWITCH_PIN, IN_PU)

        i2c = busio.I2C(board.SCL, board.SDA)
        self.imu = BNO08X_I2C(i2c, address=0x4b)
        self.imu.enable_feature(adafruit_bno08x.BNO_REPORT_ROTATION_VECTOR)

        self.rangefinder = VL53L1X(i2c)
        model_id, module_type, mask_rev = self.rangefinder.model_info
        logging.info(f'Rangefinder Model ID: 0x{model_id:0X} Module Type: 0x{module_type:0X} Mask Revision: 0x{mask_rev:0X}')
        self.rangefinder.distance_mode = 2 # LONG. results returned in centimeters.
        self.rangefinder.start_ranging()

        if mock_motor is not None:
            self.motor = mock_motor
        else:
            self.motor = GripperSpoolMotor(self.hat)

        # the superclass, RobotComponentServer, assumes the presense of this attribute
        self.spooler = SpoolController(
            self.motor,
            gear_ratio=36/9,
            empty_diameter=20,
            full_diameter=24,
            full_length=1.93,
            conf=self.conf
        )

        unique = ''.join(get_mac_address().split(':'))
        self.service_name = 'cranebot-gripper-service.' + unique

        self.last_finger_angle = 0
        self.desired_finger_angle = 0
        self.finger_speed = 0 # degrees per second
        self.past_val_rates = deque(maxlen=self.conf['UPDATE_RATE'])

        # try to read the physical positions of winch and finger last written to disk.
        # For the gripper, there's a good change nothing has moved since power down.
        try:
            with open('offsets.pickle', 'rb') as f:
                winch, finger = pickle.load(f)
                self.spooler.setReferenceLength(winch)
                self.last_finger_angle = finger
                self.desired_finger_angle =finger
        except FileNotFoundError:
            pass
        except EOFError: # corruption
            os.remove('offsets.pickle')

        # a mode in which the finger tries to automatically hold a given pressure
        # mode will switch to False if a "set_finger_angle" update is received and
        # switch back to True if a "grip" update is received.
        self.use_finger_loop = True

    def readOtherSensors(self):

        self.update['grip_sensors'] = {
            'time': time.time(),
            'quat': self.imu.quaternion,
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


    def startOtherTasks(self):
        # any tasks started here must stop on their own when self.run_server goes false
        t1 = asyncio.create_task(self.fingerLoop())
        t2 = asyncio.create_task(self.saveOffsets())
        return [t1, t2]

    async def saveOffsets(self):
        """Periodically save winch length and finger position to disk so we don't recal after power out"""
        i=0
        while self.run_server:
            await asyncio.sleep(1) # less sleep so this won't hold up server shutdown
            i+=1
            if i==30:
                i=0
                with open('offsets.pickle', 'wb') as f:
                    f.write(pickle.dumps((self.spooler.last_length, self.last_finger_angle)))

    async def fingerLoop(self):
        """
        Main control loop for fingers.
        Makes finger angle more smoothly track a target.
        We know the angle can't change faster than about 90 deg/s
        We both want self.last_finger_angle to be an accurate reflection of where it is, 
        and we want to apply some acceleration limit
        """
        running_delay = 0.03
        max_accel = 800 # degrees per second squared
        while self.run_server:
            if not self.use_finger_loop:
                await asyncio.sleep(0.2)
                continue
            # smoothly track desired finger angle
            angle_error = self.desired_finger_angle - self.last_finger_angle
            want_speed = clamp(angle_error, -10, 10) * 15 # 150 deg/sec max speed
            speed_error = want_speed - self.finger_speed
            self.finger_speed += clamp(speed_error, -max_accel*running_delay, max_accel*running_delay)
            self.last_finger_angle += clamp(self.finger_speed * running_delay, -90, 90)
            self.hand_servo.value(self.last_finger_angle)
            await asyncio.sleep(running_delay)

    async def performZeroWinchLine(self):
        logging.info(f'Zeroing winch line {self.hat.gpio_pin_value(LIMIT_SWITCH_PIN)}')
        self.spooler.pauseTrackingLoop()
        try:
            while self.hat.gpio_pin_value(LIMIT_SWITCH_PIN) == 1 and self.run_server:
                self.motor.runConstantSpeed(-1)
                await asyncio.sleep(0.03)
                # TODO, also stop this loop if encoder shows the spool isn't rotating
                # spoolangle = self.hat.encoders[0].revolutions()
                # if abs(spoolangle - lastspoolangle) < (however much it should turn in 0.03 seconds):
                #     raise RuntimeError("Spool became stuck while zeroing winch line")
            self.spooler.setReferenceLength(0.01) # 1 cm
            self.update['winch_zero_success'] = True
        except Exception as e:
            self.update['winch_zero_success'] = False
            raise e
        finally:
            # stop motor even if task throws exception
            self.motor.runConstantSpeed(0)
            self.spooler.resumeTrackingLoop()

    async def processOtherUpdates(self, update, tg):
        if 'zero_winch_line' in update:
            tg.create_task(self.performZeroWinchLine())
        if 'set_finger_angle' in update:
            self.use_finger_loop = True
            self.desired_finger_angle = clamp(float(update['set_finger_angle']), -90, 90)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    gs = RaspiGripperServer()
    asyncio.run(gs.main())
