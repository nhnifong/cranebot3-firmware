import time

class DebugMotor():
    def __init__(self):
        self.speed = 0
        self.position = 40.0
        self.last_check = time.time()
        pass

    def ping(self):
        return True

    def stop(self):
        print('stop')

    def runConstantSpeed(self, speed):
        self.speed = speed
        print(f'runConstantSpeed({speed})')

    def getShaftAngle(self):
        now = time.time()
        elapsed = now - self.last_check
        self.last_check = now
        self.position += self.speed * elapsed
        print(f'position={self.position} revs')
        return (True, self.position)

    def getShaftError(self):
        return (True, 0)

    def getMaxSpeed(self):
        return 2.0