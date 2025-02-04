import numpy as np
from scipy.interpolate import BSpline
import sys
import threading
import time
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

class ControlPanelUI:
    def __init__(self):
        self.app = Ursina()

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

        self.gripper = Entity(model='gripper_closed', color=(0.3, 0.3, 0.9, 1.0), scale=0.001, position=(0,0.3,1), rotation=(-90,0,0), shader=lit_with_shadows_shader)

        def draw_line(point_a, point_b):
            line = Entity(model=Mesh(vertices=[point_a, point_b], mode='line', thickness=2), color=color.light_gray)
        draw_line(anchor1.position, gantry.position)
        draw_line(anchor2.position, gantry.position)
        draw_line(anchor3.position, gantry.position)

        draw_line(gantry.position, self.gripper.position)

        light = DirectionalLight(shadows=True)
        light.look_at(Vec3(1,-1,1))

        foo_button = Button(text='Foo', color=color.azure, position=(-.7, 0), scale=(.3, .1), on_click=self.on_button_click)
        EditorCamera()

    def on_button_click(self):
        self.gripper.color = color.random_color()


    def notify_connected_bots_change(self, available_bots={}):
        offs = 0
        for server,info in available_bots.items():
            text_entity = Text(server, world_scale=16, position=(-0.1, -0.4 + offs))
            offs -= 0.03

    def receive_updates(self, min_to_ui_q):
        while True:
            updates = min_to_ui_q.get()
            if 'knots' in updates:
                self.knots = updates['knots']
            if 'spline_degree' in updates:
                self.spline_degree = updates['spline_degree']
            if 'minimization_step_seconds' in updates:
                pass  #updates['minimization_step_seconds']
            if 'gripper_path' in updates:
                self.gripper_pos_spline = BSpline(self.knots, updates['gripper_path'], self.spline_degree, True)
            if 'gantry_path' in updates:
                self.gantry_pos_spline = BSpline(self.knots, updates['gantry_path'], self.spline_degree, True)


    def start(self):
        self.app.run()

def start_ui(min_to_ui_q):
    cpui = ControlPanelUI()

    estimator_update_thread = threading.Thread(target=cpui.receive_updates, args=(min_to_ui_q, ), daemon=True)
    estimator_update_thread.start()

    cpui.start();