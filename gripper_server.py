import asyncio
from anchor_server import RobotComponentServer
from inventorhatmini import InventorHATMini, SERVO_1, SERVO_2, ADC
from ioexpander import IN_PU
from ioexpander.common import PID, clamp
from spools import SpoolController
from getmac import get_mac_address
import logging
from collections import deque
import time

import board
import busio
import adafruit_bno08x
from adafruit_bno08x.i2c import BNO08X_I2C

# these two constants are obtained experimentally
# the speed will not increase at settings beyond this value
WINCH_MAX_SPEED = 43
# at or below this the motor does not spin
WINCH_DEAD_ZONE = 4
# this constant is obtained from the injora website. assumes the motor is driven at 6 volts.
WINCH_MAX_RPM = 1.0166
# converts from speed setting to rpm. the speed relationship is probably close to linear, but I have not confirmed
SPEED1_REVS = WINCH_MAX_RPM / (WINCH_MAX_SPEED - WINCH_DEAD_ZONE)
# gpio pin of infrared rangefinder
RANGEFINDER_PIN = 0
# gpio pin of pressure sensing resistor
PRESSURE_PIN = 1
# gpio pin of limit switch. 0 is closed
LIMIT_SWITCH_PIN = 2
# PID values for pressure loop
POS_KP = 5.0
POS_KI = 0.0
POS_KD = 0.022
# update rate of finger pressure PID loop in updates per second
UPDATE_RATE = 40
# voltage of pressure sensor at ideal grip pressure
TARGET_HOLDING_PRESSURE = 0.6
# threshold just abolve the lowest pressure voltage we expect to read
PRESSURE_MIN = 0.5
# The total servo value change per second below which we say it has stabilized.
MEAN_SERVO_VAL_CHANGE_THRESHOLD = 3
# The servo value at which the fingers press against eachother empty with TARGET_HOLDING_PRESSURE
FINGER_TOUCH = 80
# max open servo value
OPEN = -80

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
        pass

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

    def getMaxSpeed(self):
        return 1.0166


class RaspiGripperServer(RobotComponentServer):
    def __init__(self):
        super().__init__()
        self.name_prefix = 'raspi-gripper-'
        self.service_type = 'cranebot-gripper-service'

        self.hat = InventorHATMini(init_leds=False)
        self.hand_servo = self.hat.servos[SERVO_2]
        self.hat.gpio_pin_mode(RANGEFINDER_PIN, ADC) # infrared range
        self.hat.gpio_pin_mode(PRESSURE_PIN, ADC) # pressure resistor

        i2c = busio.I2C(board.SCL, board.SDA)
        self.imu = BNO08X_I2C(i2c)
        self.imu.enable_feature(adafruit_bno08x.BNO_REPORT_ROTATION_VECTOR)
        self.imu.enable_feature(adafruit_bno08x.BNO_REPORT_LINEAR_ACCELERATION)

        # when false, open and release the object
        # when true, repeatedly try to grasp the object
        self.tryHold = False
        self.tryHoldChanged = asyncio.Event()

        self.motor = GripperSpoolMotor(self.hat)

        # the superclass, RobotComponentServer, assumes the presense of this attribute
        self.spooler = SpoolController(self.motor, empty_diameter=20, full_diameter=36, full_length=1)

        unique = ''.join(get_mac_address().split(':'))
        self.service_name = 'cranebot-gripper-service.' + unique

        self.last_value = 0
        self.past_val_rates = deque(maxlen=UPDATE_RATE)
        self.holding = False
        self.holdPressure = False

    def readOtherSensors(self):
        # 5cm - 2.3v
        # 10cm - 2.0v
        # 15cm - 1.5v
        # 20cm - 1.15v

        # 22cm - 1.8v
        # the measurement is not thrown off by the fingers being closed at all
        self.update['IR range'] = self.hat.gpio_pin_value(RANGEFINDER_PIN)*(-11)+33
        self.update['imu'] = {
            'accel': [time.time(), *self.imu.linear_acceleration],
            'quat': self.imu.quaternion
        }


    def startOtherTasks(self):
        # any tasks started here must stop on their own when self.run_server goes false
        asyncio.create_task(self.fingerLoop())

    async def holdPressurePid(self, target_v):
        """
        control the hand servo to hold the voltage on the pressure pin at the target
        """
        voltage_pid = PID(POS_KP, POS_KI, POS_KD, 1/UPDATE_RATE)
        voltage_pid.setpoint = target_v
        pos = OPEN
        while self.holdPressure and self.tryHold:
            # get the current pressure
            voltage = self.hat.gpio_pin_value(PRESSURE_PIN)
            # run pid calcucaltion. it tells you how much to move
            val = voltage_pid.calculate(voltage)
            pos += val
            logging.debug(f'calculated pid value {val}, servo pos = {pos}')
            # set servo position
            self.hand_servo.value(clamp(pos,-90,90))
            # record the absolute value change to know if it is stabilizing
            self.past_val_rates.append(abs(pos - self.last_value))
            self.last_value = pos
            await asyncio.sleep(1/UPDATE_RATE)

    async def readStableFingerValue(self):
        # wait for value to stabilize
        # todo, what if the client commands tryHold=False while we are in this loop
        # this might also need a timeout.
        while sum(self.past_val_rates)/UPDATE_RATE > MEAN_SERVO_VAL_CHANGE_THRESHOLD:
            await asyncio.sleep(0.25)
        return self.last_value


    async def fingerLoop(self):
        """
        Main control loop for fingers.

        The gripper has explicit states, open and trying to hold something.
        In the trying to hold something state, called 'hold' for short, the grip will repeatedly alternate between two more states,
        closing to maintain a pressure, and opening again
            during the maintain pressure state, set the desired grip pressure to maybe 500g.
            A pid loop then attempts to find the servo value that results in this pressure.
            In this loop. when we see the servo value stabilize,
            If it is low, like 65, this means something is held. we always report whether somethig is held back to the client on the websocket.
            stay in this state.
            if it was high, like 85, this means no object was grasped and the fingers are pressing against eachother.
            pause the pressure pid loop, and set to fully open, sleeping long enough for it to fully open.
            wait up to a certain timeout for the object to be in the sweet spot, and reenter the hold pressure mode.
        
        TODO: the camera may also be a way of determining whether somehting is held.
        Either by doing something in opencv with a reference image of a closed, empty gripper,
        or by looking at the output tensor of the AI camera 
        """
        while self.run_server:
            # repeatedly try to grasp the object
            while self.tryHold:
                logging.info(f'tryHold={self.tryHold} holding={self.holding}')
                if not self.holding:
                    # wait for the target to be in the sweet spot
                    pass
                    # Start gripping
                    logging.info(f'Close grip and maintain pressure')
                    self.holdPressure = True
                    asyncio.create_task(self.holdPressurePid(TARGET_HOLDING_PRESSURE))
                    await asyncio.sleep(0.5)
                finger_val = await self.readStableFingerValue()
                logging.info(f'Finger stable at {finger_val} with mean absolute change of {sum(self.past_val_rates)/UPDATE_RATE} over the last second')
                logging.info(f'pressure pad voltage = {self.hat.gpio_pin_value(PRESSURE_PIN)}')
                # look where it stabilized
                if finger_val < FINGER_TOUCH:
                    # object is present
                    # putting anything in the self.update dict means it will get flushed to the websocket
                    self.holding = True
                    self.update['holding'] = True
                    await asyncio.sleep(0.25)
                    # stay in the loop, checking stable finger position
                else:
                    # grasped nothing. stop the pid loop, open the grip, and wait one second.
                    # We also reach this if the object slipped out, and the value restabilized with the fingers touching.
                    logging.info(f'Fingers closed on nothing. self.holding was {self.holding}')
                    self.holdPressure = False
                    self.hand_servo.value(OPEN)
                    self.holding = False
                    self.update['holding'] = False
                    # consider sending a count of the number of times we failed to grasp.
                    # by the time we wake, we expect the fingers to be open, and the estimator to have decided whether tryHold should still be true
                    await asyncio.sleep(2.0)

            else:
                # the else condition runs if the loop completes without a break statement or an exception
                # this should occur only when a websocket update was received setting self.tryHold to false.
                # open completely.
                logging.info(f'Grip commanded open.')
                self.hand_servo.value(OPEN)
                # do not leave the station until the passengers have fully departed.
                while self.hat.gpio_pin_value(PRESSURE_PIN) > PRESSURE_MIN:
                    await asyncio.sleep(0.05)\
                # now you can tell the controller its ok to move to the next destination.
                self.update['holding'] = False
                # stay in this state until commanded to do otherwise.
                await self.tryHoldChanged.wait()
                self.tryHoldChanged.clear()

    async def performZeroWinchLine(self):
        self.spooler.pauseTrackingLoop()
        while self.hat.gpio_pin_value(LIMIT_SWITCH_PIN) == 0 and self.run_server:
            self.motor.runConstantSpeed(-1)
            await asyncio.sleep(0.03)
        self.spooler.setReferenceLength(0.01) # 1 cm
        self.spooler.resumeTrackingLoop()


    def processOtherUpdates(self, update):
        if 'grip' in update:
            logging.info(f'setting grip {update["grip"]}')
            if update['grip'] == 'open':
                self.tryHold = False
            elif update['grip'] == 'closed':
                self.tryHold = True
                self.tryHoldChanged.set()
        if 'zero_winch_line' in update:
            self.performZeroWinchLine()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    gs = RaspiGripperServer()
    asyncio.run(gs.main())
