import asyncio
from ursina import *

app = Ursina()

async def long_running_task():
    print("Starting long task...")
    await asyncio.sleep(2)  # Simulate an I/O-bound task
    print("Long task finished")
    ursina.application.invoke(setattr, cube, 'color', color.blue) #safe update of cube.color

def start_async_task():
    asyncio.ensure_future(long_running_task())

cube = Entity(model='cube', color=color.red, collider='box')
button = Button(text='Start Async Task', on_click=start_async_task)

app.run()