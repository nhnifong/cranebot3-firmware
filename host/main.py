import numpy as np
import sys
import asyncio
import argparse
import logging
import threading
import time
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
from position_estimator import CDPR_position_estimator
from calibration import calibrate_all

from ursina import (
    Ursina,
    Entity,
    Button,
    EditorCamera,
    color,
    DirectionalLight,
    Vec3,
    Text,
    Mesh,
    invoke,
)
from ursina.shaders import (
    lit_with_shadows_shader,
    unlit_shader,
)

# try:
#   params = np.load("calibratrion_data.npz")
# except IOError:
#   params = calibrate_all();
#   np.savez('calibration_data', params**)

# horizon_seconds = 10 # amount of time over which to observe and model position
# observer = Observer(horizon_seconds, len(params['anchor_positions']))
# estimator = CDPR_position_estimator(observer, params['anchor_positions'])

# def continuous_position_estimation():
#   # this loop should run every time there's a new measurement but no faster than a certain rate.
#   # But that's probably identical to running it as fast as possible.
#   while True:
#       print(estimator.estimate())
#       estimator.move_to_present()

# position_thread = threading.Thread(target=continuous_position_estimation)

# def signal_handler(sig, frame):
#     position_thread.join()
#     sys.exit(0)

# signal.signal(signal.SIGINT, signal_handler)
# position_thread.start()

app = Ursina()

# add the charuco board that represents the room origin
# when a charuco board is located, it's origin is it's top left corner.
square = Entity(model='quad', position=(0.03, 0, -0.03), rotation=(90,0,0), color=color.white, scale=(0.06, 0.06))  # Scale in meters
square.texture = 'origin.jpg'
# add a 1cm sphere to clarify where the game origin is
origin_sphere = Entity(model='sphere', position=(0,0,0), color=color.orange, scale=(0.01), shader=unlit_shader)  # Scale in meters

#show a very large floor
square = Entity(model='quad', position=(0, -0.05, 0), rotation=(90,0,0), color=color.brown, scale=(10, 10))  # Scale in meters

# this models units are in mm, but the game units are meters
anchor_color = (0.9, 0.9, 0.9, 1.0)
def add_anchor(pos, rot=(0,  0,0)):
    return Entity(model='anchor', color=anchor_color, scale=0.001, position=pos, rotation=(0,  0,0), shader=lit_with_shadows_shader)
anchor1 = add_anchor((-2,2, 3))
anchor2 = add_anchor(( 2,2, 3))
anchor3 = add_anchor(( 0,2,-2), rot=(0,180,0))

gantry = Entity(model='gantry', color=(0.4, 0.4, 0.0, 1.0), scale=0.001, position=(0,1,1), rotation=(0,0,0), shader=lit_with_shadows_shader)

gripper = Entity(model='gripper_closed', color=(0.3, 0.3, 0.9, 1.0), scale=0.001, position=(0,0.3,1), rotation=(-90,0,0), shader=lit_with_shadows_shader)

def draw_line(point_a, point_b):
    line = Entity(model=Mesh(vertices=[point_a, point_b], mode='line', thickness=2), color=color.light_gray)
draw_line(anchor1.position, gantry.position)
draw_line(anchor2.position, gantry.position)
draw_line(anchor3.position, gantry.position)

draw_line(gantry.position, gripper.position)

light = DirectionalLight(shadows=True)
light.look_at(Vec3(1,-1,1))

args = None

def on_button_click():
    pass
    # gripper.color = color.random_color()

foo_button = Button(text='Foo', color=color.azure, position=(-.7, 0), scale=(.3, .1), on_click=on_button_click)
EditorCamera()

def notify_connected_bots_change(available_bots={}):
    offs = 0
    for server,info in available_bots.items():
        text_entity = Text(server, world_scale=16, position=(-0.1, -0.4 + offs))
        offs -= 0.03

cranebot_service_name = 'cranebot-service'
# listener for updating the list of available robot component servers
# keep track of unique components with the server attribute of the 
class CranebotListener(ServiceListener):
    def __init__(self):
        super().__init__()
        self.available_bots = {}

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print(f"Service {name} updated")
        info = zc.get_service_info(type_, name)
        if name.split(".")[1] == cranebot_service_name:
            if info.server is not None and info.server != '':
                self.available_bots[info.server] = info
                invoke(notify_connected_bots_change, self.available_bots)

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print(f"Service {name} removed")
        info = zc.get_service_info(type_, name)
        del self.available_bots[info.server]

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        print(f"Service {name} added, service info: {info}")
        if name.split(".")[1] == cranebot_service_name:
            if info.server is not None and info.server != '':
                self.available_bots[info.server] = info
                invoke(notify_connected_bots_change, self.available_bots)

run_discovery_task = True
def service_discovery_task():
    zeroconf = Zeroconf()
    listener = CranebotListener()
    browser = ServiceBrowser(zeroconf, "_http._tcp.local.", listener)
    print('Started service discovery task')
    while run_discovery_task:
        time.sleep(0.1)
    zeroconf.close()

discovery_thread = threading.Thread(target=service_discovery_task, daemon=True)
discovery_thread.start()

app.run()