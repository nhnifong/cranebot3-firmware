import asyncio
import time
from spools import SpoolController
from debug_motor import DebugMotor

async def main():
    spooler = SpoolController(DebugMotor(), empty_diameter=25, full_diameter=54, full_length=9, gear_ratio=16/40)
    print(f'spooler gear ratio = {spooler.gear_ratio}')
    spooler.setReferenceLength(2.4)
    print(f'spooler debug k1_over_k2={spooler.k1_over_k2}, k2={spooler.k2}')
    print('')
    print(f'spooled length = {spooler._get_spooled_length(40.0)}')
    print(f'unspooled length = {spooler.get_unspooled_length(40.0)}')
    print(f'unspool rate = {spooler.get_unspool_rate(40.0)}')
    print('')
    print(f'spooled length = {spooler._get_spooled_length(50.0)}')
    print(f'unspooled length = {spooler.get_unspooled_length(50.0)}')
    print(f'unspool rate = {spooler.get_unspool_rate(50.0)}')
    print('')
    return
    spool_task = asyncio.create_task(asyncio.to_thread(spooler.trackingLoop))
    await asyncio.sleep(0.5)
    # spooler.jogRelativeLen(-0.15)
    t = time.time()
    # move 10 cm in 1 second
    plan = [[t + (i/6), 1.0+i*0.01]
        for i in range(6)]
    spooler.setPlan(plan)
    result = await spool_task

asyncio.run(main())