import asyncio
from anchor_server import RobotComponentServer
from inventorhatmini import InventorHATMini, SERVO_1

# these two constants are obtained experimentally
# the speed will not increase at settings beyond this value
WINCH_MAX_SPEED = 43
# at or below this the motor does not spin
WINCH_DEAD_ZONE = 4
# this constant is obtained from the injora website. assumes the motor is driven at 6 volts.
WINCH_MAX_RPM = 1.0166
# converts from speed setting to rpm. the speed relationship is probably close to linear, but I have not confirmed
SPEED1_REVS = WINCH_MAX_RPM / (WINCH_MAX_SPEED - WINCH_DEAD_ZONE)

class GripperSpoolMotor():
    """
    Motor interface for gripper spools motor with the same methods as MKSSERVO42C
    Currently based on the injora 360 deg 35kg open loop winch servo 
    https://www.injora.com/products/injora-injs035-360-35kg-waterproof-digital-servo-360-steering-winch-wheel-for-rc
    but using a mouse wheel encoder for position feedback.
    """
    def __init__(self, hat):
        self.servo = self.hat.servos[SERVO_1]
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
        self.servo.value(command_speed)

    def getShaftAngle(self):
        # in revolutions
        # we assume that an encoder has been conected to the motot A port, even if there is no motor
        return self.hat.encoders[0].degrees()/360

    def getMaxSpeed(self):
        return 1.0166


class RaspiGripperServer(RobotComponentServer):
    def __init__(self):
        super().__init__()
        self.name_prefix = 'raspi-gripper-'
        self.service_type = 'cranebot-gripper-service'

        self.hat = InventorHATMini(init_leds=False)
        # self.spool_servo = board.servos[SERVO_1]
        # self.hand_servo = board.servos[SERVO_2]
        self.hat.gpio_pin_mode(0, ADC) # infrared range
        self.hat.gpio_pin_mode(1, ADC) # pressure resistor
        self.shouldBeFingersClosed = False

        # the superclass, RobotComponentServer, assumes the presense of this attribute
        self.spooler = SpoolController(GripperSpoolMotor(self.hat), spool_diameter_mm=19.71)

    def readAnalog(self)
        voltage = board.gpio_pin_value(0)

    def getSpoolMeasurements(self):
        return self.spooler.popMeasurements()

    def stopMotors(self):
        self.spooler.fastStop()

    def spoolTrackingLoop(self)
        # return the spool tracking function
        return self.spooler.trackingLoop

    def fingerLoop(self):
        """
        Main control loop for fingers
        if we wish to be holding somehting right now, command fingers closed, and maintain pressure.
        maintain an estimate at all times of whether we are successfully holding something.
        """
        while self.run:
            if self.shouldBeFingersClosed:
                # todo: in gripper closed mode, hold pressure constant
                self.hand_servo.value(0)
            else:
                self.hand_servo.value(180)

    def processOtherUpdates(self, update):
        if 'grip' in update:
            if update['grip'] == 'open':
                self.shouldBeFingersClosed = False
            elif update['grip'] == 'closed':
                self.shouldBeFingersClosed = True


if __name__ == "__main__":
    gs = RaspiGripperServer()
    asyncio.run(gs.main())
