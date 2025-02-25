import asyncio
from anchor_server import RobotComponentServer
from inventorhatmini import InventorHATMini, SERVO_1


class GripperSpoolMotor():
    """
    Motor interface for gripper spools motor with the same methods as MKSSERVO42C
    Currently based on the injora 360 deg 35kg winch motor
    """
    def __init__(self):
        pass

    def ping(self):
        pass

    def stop(self):
        pass

    def runConstantSpeed(self, speed):
        pass

    def getShaftAngle(self):
        pass

    def getMaxSpeed(self):
        pass


class RaspiGripperServer(RobotComponentServer):
    def __init__(self):
        super().__init__()
        self.name_prefix = 'raspi-gripper-'
        self.service_type = 'cranebot-gripper-service'

        self.hat = InventorHATMini(init_leds=False)
        self.spool_servo = board.servos[SERVO_1]
        self.hand_servo = board.servos[SERVO_2]
        self.shouldBeFingersClosed = False

        # the superclass assumes the presense of this attribute
        self.spooler = SpoolController(GripperSpoolMotor(), spool_diameter_mm=24)

    def getSpoolMeasurements(self):
        return self.spooler.popMeasurements()

    def stopMotors(self):
        self.spooler.fastStop()

    def spoolTrackingLoop(self)
        return self.spooler.trackingLoop

    def fingerLoop(self):
        """
        Main control loop for fingers
        if we wish to be holding somehting right now, command fingers closed, and maintain pressure.
        maintain an estimate at all times of whether we are successfully holding something.
        """
        if self.shouldBeFingersClosed:
            self.hand_servo.value(0)
        else:
            self.hand_servo.value(180)

    def processOtherUpdates(self, update):
        if 'grip' in update:
            if update['grip'] = 'open':
                self.shouldBeFingersClosed = False
            elif update['grip'] = 'closed':
                self.shouldBeFingersClosed = True


if __name__ == "__main__":
    ras = RaspiAnchorServer()
    asyncio.run(ras.main())