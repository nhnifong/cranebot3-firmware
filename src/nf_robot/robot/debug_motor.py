import time

class DebugMotor():
    def __init__(self):
        self.speed = 0
        self.position = 40.0
        self.last_check = time.time()
        self.accelLimit = None

    def ping(self):
        return True

    def stop(self):
        print('stop')

    def runConstantSpeed(self, speed):
        if self.accelLimit is not None and abs(speed - self.speed) > self.accelLimit:
            raise VlueError(f"Tried to change debug motor speed too abruptly from {self.speed} to {speed}")
        self.speed = speed
        # print(f'runConstantSpeed({speed})')

    def getShaftAngle(self):
        now = time.time()
        elapsed = now - self.last_check
        self.last_check = now
        self.position += self.speed * elapsed
        # print(f'position={self.position} revs')
        return (True, self.position)

    def getShaftError(self):
        return (True, 0)

    def getMaxSpeed(self):
        return 200.0

    def setAccelLimit(self, limit):
        self.accelLimit = limit