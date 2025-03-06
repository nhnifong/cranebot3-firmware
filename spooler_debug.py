import asyncio
import time
from spools import SpoolController


class DebugMotor():
    def __init__(self):
        self.speed = 0
        self.position = 4.0
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

    def getMaxSpeed(self):
        return 2.0


async def main():
    spooler = SpoolController(DebugMotor(), empty_diameter=20, full_diameter=20, full_length=10)
    spool_task = asyncio.create_task(asyncio.to_thread(spooler.trackingLoop))
    await asyncio.sleep(0.5)
    # spooler.jogRelativeLen(-0.15)
    spooler.setReferenceLength(1.0)
    t = time.time()
    # move 10 cm in 1 second
    plan = [[t + (i/6), 1.0+i*0.01]
        for i in range(6)]
    spooler.setPlan(plan)
    await spool_task

asyncio.run(main())