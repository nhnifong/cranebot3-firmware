import asyncio
from anchor_server import RobotComponentServer
from inventorhatmini import InventorHATMini, SERVO_1, SERVO_2, ADC
from ioexpander import IN_PU
from spools import SpoolController
from getmac import get_mac_address

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
        # command_speed = max(-127, min(int(SPEED1_REVS * speed), 127))

        if speed == 0:
            command_speed = 0
        elif speed > 0:
            command_speed = speed / SPEED1_REVS + WINCH_DEAD_ZONE
        elif speed < 0:
            command_speed = speed / SPEED1_REVS - WINCH_DEAD_ZONE
        # self.servo.value(command_speed)

    def getShaftAngle(self):
        # in revolutions
        # we assume that an encoder has been conected to the motot A port, even if there is no motor
        return True, self.hat.encoders[0].degrees()/360

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
        self.shouldBeFingersClosed = False

        # the superclass, RobotComponentServer, assumes the presense of this attribute
        self.spooler = SpoolController(GripperSpoolMotor(self.hat), empty_diameter=20, full_diameter=36, full_length=1)

        unique = ''.join(get_mac_address().split(':'))
        self.service_name = 'cranebot-gripper-service.' + unique

    def readAnalog(self):
        # 5cm - 2.3v
        # 10cm - 2.0v
        # 15cm - 1.5v
        # 20cm - 1.15v
        # the measurement is not thrown off by the fingers being closed at all
        voltage = self.hat.gpio_pin_value(RANGEFINDER_PIN)

    def spoolTrackingLoop(self):
        # return the spool tracking function
        return self.spooler.trackingLoop

    def fingerLoop(self):
        """
        Main control loop for fingers
        if we wish to be holding somehting right now, command fingers closed, and maintain pressure.
        maintain an estimate at all times of whether we are successfully holding something.

        without actually training a network, holding something would be indicated by
         * finger pressure being high
         * high enough elapsed time since we started closing the fingers
         * rangefinger reading is low and constant despite moving relative to the floor
        
        the camera may also be a way of determining whether somehting is held.
        Either by doing something in opencv with a reference image of a closed, empty gripper,
        or by looking at the output tensor of the AI camera 
        """
        while self.run_client:
            if self.shouldBeFingersClosed:
                # todo: in gripper closed mode, hold pressure constant
                pass
                # self.hand_servo.value(90)
            else:
                pass
                # self.hand_servo.value(-90)
            voltage = hat.gpio_pin_value(PRESSURE_PIN)
            # putting anything in the self.update dict means it will get flushed to the websocket
            self.update['holding'] = voltage > 1.5

    def processOtherUpdates(self, update):
        if 'grip' in update:
            if update['grip'] == 'open':
                self.shouldBeFingersClosed = False
            elif update['grip'] == 'closed':
                self.shouldBeFingersClosed = True


if __name__ == "__main__":
    gs = RaspiGripperServer()
    asyncio.run(gs.main())
