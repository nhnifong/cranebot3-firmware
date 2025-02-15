import numpy as np
from scipy.interpolate import BSpline
import sys
import threading
import time
from position_estimator import CDPR_position_estimator
from scipy.spatial.transform import Rotation
from functools import partial
from cv_common import invert_pose, compose_poses
from math import pi
import atexit

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
    application,
    window,
)
from ursina.shaders import (
    lit_with_shadows_shader,
    unlit_shader,
)

# ursina considers +Y up. all the other processes, such as the position estimator consider +Z up. 
def swap_yz(vec):
    return (vec[0], vec[2], vec[1])

class SplineMovingEntity(Entity):
    """
    An entity that moves it's position along a spline function
    """
    def __init__(self, model, color, scale, position, rotation, shader, spline_func, **kwargs):
        super().__init__(
            model=model,
            color=color,
            scale=scale,
            position=position,
            rotation=rotation,
            shader=shader,
            **kwargs
        )
        self.spline_func = spline_func
        self.time_domain = (333,444)

    def setParams(self, func, domain):
        self.spline_func = func
        self.time_domain = domain

    def update(self):
        if self.spline_func is not None:
            t = (time.time() - self.time_domain[0]) / (self.time_domain[1] - self.time_domain[0])
            self.position = swap_yz(self.spline_func(t))

class ControlPanelUI:
    def __init__(self, to_pe_q, to_ob_q):
        self.app = Ursina()
        self.to_pe_q = to_pe_q
        self.to_ob_q = to_ob_q

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
        self.anchors = []
        self.anchors.append(add_anchor((-2,2, 3)))
        self.anchors.append(add_anchor(( 2,2, 3)))
        self.anchors.append(add_anchor(( -1,2,-2), rot=(0,180,0)))
        self.anchors.append(add_anchor(( -2,2,-2), rot=(0,180,0)))

        self.spline_curves = {}

        self.gantry = SplineMovingEntity(
            model='gantry',
            color=(0.4, 0.4, 0.0, 1.0),
            scale=0.001,
            position=(0,1,1),
            rotation=(0,0,0),
            shader=lit_with_shadows_shader,
            spline_func=None)

        self.gripper = SplineMovingEntity(
            model='gripper_closed',
            color=(0.3, 0.3, 0.9, 1.0),
            scale=0.001,
            position=(0,0.3,1),
            rotation=(-90,0,0),
            shader=lit_with_shadows_shader,
            spline_func=None)

        def draw_line(point_a, point_b):
            line = Entity(model=Mesh(vertices=[point_a, point_b], mode='line', thickness=2), color=color.light_gray)
        for a in self.anchors:
            draw_line(a.position, self.gantry.position)

        draw_line(self.gantry.position, self.gripper.position)

        light = DirectionalLight(shadows=True)
        light.look_at(Vec3(1,-1,1))

        self.calibration_button = Button(
            text='Calibrate Anchor Locations',
            color=color.azure,
            position=(-.7, 0), scale=(.1, .033),
            on_click=self.on_calibration_button_click)
        # start in pose calibration mode. TODO need to do this only if any of the four anchor clients boots up but can't find it's file
        # maybe you shouldn't read those files in the clients
        self.calibration_mode = 'pose'

        self.rot = [0,0,0]
        self.anchor_cam_inv = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))
        self.xbut = Button(
            text='x',
            color=color.azure,
            position=(-.7, 0.1), scale=(.1, .033),
            on_click=partial(self.xxx, i=0))
        self.xbut = Button(
            text='y',
            color=color.azure,
            position=(-.7, 0.2), scale=(.1, .033),
            on_click=partial(self.xxx, i=1))
        self.xbut = Button(
            text='z',
            color=color.azure,
            position=(-.7, 0.3), scale=(.1, .033),
            on_click=partial(self.xxx, i=2))

        EditorCamera()

    # renders a function that returns 3D points in a domain from 0 to 1
    # the y and z coordinates are swapped
    def render_curve(self, curve_function, name):
        model = Mesh(
                vertices=[Vec3(swap_yz(curve_function(time))) for time in np.linspace(0,1,32)],
                mode='line',
                thickness=2)
        if name in self.spline_curves:
            self.spline_curves[name].model = model
        else:
            self.spline_curves[name] = Entity(model=model, color=color.lime)

    def xxx(self, i):
        self.rot[i] += 0.5
        if self.rot[i] >= 2.0:
            self.rot[i] = 0
        self.anchor_cam_inv = invert_pose(compose_poses([
            (np.array([0.045625, -0.034915, 0.004762], dtype=float), np.array([0,0,0], dtype=float)),
            (np.array([0,0,0], dtype=float), np.array([self.rot[0]*pi,0,0], dtype=float)),
            (np.array([0,0,0], dtype=float), np.array([0,self.rot[0]*pi,0], dtype=float)),
            (np.array([0,0,0], dtype=float), np.array([0,0,self.rot[0]*pi], dtype=float)),
        ]))
        print(self.rot)
        print(f'anchor cam inv = {self.anchor_cam_inv}')


    def on_calibration_button_click(self):
        if self.calibration_mode == "pose":
            self.calibration_mode = "run"
            self.calibration_button.text = "enter calibration mode"
            self.calibration_buttoncolor=color.green,

        elif self.calibration_mode == "run":
            self.calibration_mode = "pose"
            self.calibration_button.text = "Calibrate Anchor Locations"
            self.calibration_buttoncolor=color.azure,

        self.to_ob_q.put({'set_calibration_mode': self.calibration_mode})


    def notify_connected_bots_change(self, available_bots={}):
        offs = 0
        for server,info in available_bots.items():
            text_entity = Text(server, world_scale=16, position=(-0.1, -0.4 + offs))
            offs -= 0.03

    def receive_updates(self, min_to_ui_q):
        while True:
            updates = min_to_ui_q.get()
            if 'STOP' in updates:
                application.quit()

            if 'knots' in updates:
                self.knots = updates['knots']

            if 'spline_degree' in updates:
                self.spline_degree = updates['spline_degree']

            if 'minimization_step_seconds' in updates:
                pass  #updates['minimization_step_seconds']

            if 'time_domain' in updates:
                # the time domain in unix seconds over which the gripper and gantry splines are defined.
                self.time_domain = updates['time_domain']

            if 'gripper_path' in updates:
                gripper_pos_spline = BSpline(self.knots, updates['gripper_path'], self.spline_degree, True)
                self.render_curve(gripper_pos_spline, 'gripper_spline')
                self.gripper.setParams(gripper_pos_spline, self.time_domain)

            if 'gantry_path' in updates:
                gantry_pos_spline = BSpline(self.knots, updates['gantry_path'], self.spline_degree, True)
                self.render_curve(gantry_pos_spline, 'gantry_spline')
                self.gantry.setParams(gantry_pos_spline, self.time_domain)

            if 'anchor_pose' in updates:
                apose = updates['anchor_pose']
                anchor_num = apose[0]
                self.anchors[anchor_num].position = swap_yz(apose[1][1])
                ps = compose_poses([apose[1], self.anchor_cam_inv])
                print(f'ps = {ps}')
                self.anchors[anchor_num].quaternion = tuple(Rotation.from_rotvec(np.array(swap_yz(
                    ps[0]
                ))).as_quat())

    def start(self):
        self.app.run()


def update():
    print('ursina called update')

def start_ui(to_ui_q, to_pe_q, to_ob_q):
    """
    Entry point to be used when starting this from main.py with multiprocessing
    """
    cpui = ControlPanelUI(to_pe_q, to_ob_q)

    # use simple threading here. ursina has it's own loop that conflicts with asyncio
    estimator_update_thread = threading.Thread(target=cpui.receive_updates, args=(to_ui_q, ), daemon=True)
    estimator_update_thread.start()

    def stop_other_processes():
        print("UI window closed. stopping other processes")
        to_ui_q.put({'STOP':None}) # stop our own listening thread too
        to_pe_q.put({'STOP':None})
        to_ob_q.put({'STOP':None})

    # ursina has no way to tell us when the window is closed. but this python module can do it.
    atexit.register(stop_other_processes)

    cpui.start();

if __name__ == "__main__":
    from multiprocessing import Queue
    to_ui_q = Queue()
    to_pe_q = Queue()
    to_ob_q = Queue()
    start_ui(to_ui_q, to_pe_q, to_ob_q)