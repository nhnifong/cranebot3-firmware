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
from panda3d.core import LQuaternionf
from itertools import permutations

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
    Pipe,
    Quad,
    invoke,
    application,
    window,
)
from ursina.shaders import (
    lit_with_shadows_shader,
    unlit_shader,
)

# the color for the lines that connect the anchors, gantry, and gripper
line_color = color.black

# ursina considers +Y up. all the other processes, such as the position estimator consider +Z up. 
def swap_yz(vec):
    return (vec[0], vec[2], vec[1])

# Transforms a rodrigues rotation vector ummmm its magic don't ask
def fix_rot(vec):
    return (vec[1], -vec[2], vec[0])

def draw_line(point_a, point_b):
    return Mesh(vertices=[point_a, point_b], mode='line')
    # return Pipe(
    #     path=[point_a, point_b],
    #     thicknesses=(0.01, 0.01),
    #     cap_ends=False,
    # )

class SplineMovingEntity(Entity):
    """
    An entity that moves it's position along a spline function
    """
    def __init__(self, ui, model, color, scale, position, rotation, shader, spline_func, **kwargs):
        super().__init__(
            model=model,
            color=color,
            scale=scale,
            position=position,
            rotation=rotation,
            shader=shader,
            **kwargs
        )
        self.is_gantry = (model == 'gantry')
        self.ui = ui
        self.spline_func = spline_func
        self.time_domain = (333,444)

    def setParams(self, func, domain):
        self.spline_func = func
        self.time_domain = domain

    def update(self):
        if self.spline_func is not None:
            t = (time.time() - self.time_domain[0]) / (self.time_domain[1] - self.time_domain[0])
            self.position = swap_yz(self.spline_func(t))

            if self.is_gantry:
                # update the lines between the gantry and the other things
                for anchor_num in range(self.ui.n_anchors):
                    self.ui.lines[anchor_num].model = draw_line(self.ui.anchors[anchor_num].position, self.position)
                self.ui.vert_line.model = draw_line(self.ui.gripper.position, self.position)


class ControlPanelUI:
    def __init__(self, datastore, to_pe_q, to_ob_q):
        self.app = Ursina()
        self.datastore = datastore
        self.to_pe_q = to_pe_q
        self.to_ob_q = to_ob_q
        self.n_anchors = datastore.n_cables

        # add the charuco board that represents the room origin
        # when a charuco board is located, it's origin is it's top left corner.
        square = Entity(model='quad', position=(0.03, 0, -0.03), rotation=(90,0,0), color=color.white, scale=(0.06, 0.06))  # Scale in meters
        square.texture = 'origin.jpg'
        # add a 1cm sphere to clarify where the game origin is
        origin_sphere = Entity(model='sphere', position=(0,0,0), color=color.orange, scale=(0.01), shader=unlit_shader)  # Scale in meters

        sphereX = Entity(model='sphere', position=(1,0,0), color=color.red, scale=(0.1), shader=unlit_shader)
        sphereY = Entity(model='sphere', position=(0,1,0), color=color.green, scale=(0.1), shader=unlit_shader)
        sphereZ = Entity(model='sphere', position=(0,0,1), color=color.blue, scale=(0.1), shader=unlit_shader)

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
            ui=self,
            model='gantry',
            color=(0.4, 0.4, 0.0, 1.0),
            scale=0.001,
            position=(0,1,1),
            rotation=(0,0,0),
            shader=lit_with_shadows_shader,
            spline_func=None)

        self.gripper = SplineMovingEntity(
            ui=self,
            model='gripper_closed',
            color=(0.3, 0.3, 0.9, 1.0),
            scale=0.001,
            position=(0,0.3,1),
            rotation=(-90,0,0),
            shader=lit_with_shadows_shader,
            spline_func=None)

        self.lines = []
        for a in self.anchors:
            self.lines.append(Entity(model=draw_line(a.position, self.gantry.position), color=line_color, shader=unlit_shader))

        self.vert_line = Entity(model=draw_line(self.gantry.position, self.gripper.position), color=line_color, shader=unlit_shader)

        light = DirectionalLight(shadows=True)
        light.look_at(Vec3(1,-1,1))

        self.calibration_button = Button(
            text="Enter calibration mode",
            color=color.azure,
            position=(-0.7, -0.45), scale=(.35, .033),
            on_click=self.on_calibration_button_click)
        # start in pose calibration mode. TODO need to do this only if any of the four anchor clients boots up but can't find it's file
        # maybe you shouldn't read those files in the clients
        self.calibration_mode = 'run'

        EditorCamera()

    # renders a function that returns 3D points in a domain from 0 to 1
    # the y and z coordinates are swapped
    def render_curve(self, curve_function, name):
        model = Pipe(
            path=[Vec3(swap_yz(curve_function(time))) for time in np.linspace(0,1,32)],
            thicknesses=(0.01, 0.01),
            cap_ends=False,
        )
        if name in self.spline_curves:
            self.spline_curves[name].model = model
        else:
            self.spline_curves[name] = Entity(model=model, color=color.lime, shader=unlit_shader)

    def on_calibration_button_click(self):
        if self.calibration_mode == "pose":
            self.calibration_mode = "run"
            self.calibration_button.text = "Enter calibration mode"
            # self.calibration_button.color=color.green,

        elif self.calibration_mode == "run":
            self.calibration_mode = "pose"
            self.calibration_button.text = "Freeze Anchor Locations"
            # self.calibration_button.color=color.azure,

        self.to_ob_q.put({'set_calibration_mode': self.calibration_mode})

    def render_gripper_ob(self,):
        """
        Display a visual indication of aruco based gripper observations
        """
        while True:
            print("render gripper ob")
            gripper_pose = self.datastore.gripper_pose.deepCopy()
            for row in gripper_pose:
                sphereX = Entity(model='sphere', position=(row[4],row[6],row[5]), color=color.white, scale=(0.1), shader=lit_with_shadows_shader)
            time.sleep(15)

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
                # print(f"minimization_step_seconds = {updates['minimization_step_seconds']}")
                pass

            if 'time_domain' in updates:
                # the time domain in unix seconds over which the gripper and gantry splines are defined.
                self.time_domain = updates['time_domain']

            if 'gripper_path' in updates:
                control_points = np.array(list(map(swap_yz, updates['gripper_path'])))
                gripper_pos_spline = BSpline(self.knots, control_points, self.spline_degree, True)
                self.render_curve(gripper_pos_spline, 'gripper_spline')
                self.gripper.setParams(gripper_pos_spline, self.time_domain)

            if 'gantry_path' in updates:
                control_points = np.array(list(map(swap_yz, updates['gantry_path'])))
                gantry_pos_spline = BSpline(self.knots, control_points, self.spline_degree, True)
                self.render_curve(gantry_pos_spline, 'gantry_spline')
                self.gantry.setParams(gantry_pos_spline, self.time_domain)

            if 'anchor_pose' in updates:
                apose = updates['anchor_pose']
                anchor_num = apose[0]
                self.anchors[anchor_num].position = swap_yz(apose[1][1])
                self.anchors[anchor_num].quaternion = LQuaternionf(*list(Rotation.from_rotvec(np.array(fix_rot(apose[1][0]))).as_quat()))

    def start(self):
        self.app.run()


def update():
    # seems like this only happens when this file is __main__
    print('ursina called update')

def start_ui(datastore, to_ui_q, to_pe_q, to_ob_q):
    """
    Entry point to be used when starting this from main.py with multiprocessing
    """
    cpui = ControlPanelUI(datastore, to_pe_q, to_ob_q)

    # use simple threading here. ursina has it's own loop that conflicts with asyncio
    estimator_update_thread = threading.Thread(target=cpui.receive_updates, args=(to_ui_q, ), daemon=True)
    estimator_update_thread.start()

    # rgo = threading.Thread(target=cpui.render_gripper_ob, daemon=True)
    # rgo.start()

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