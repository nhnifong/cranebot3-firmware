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

from ursina import *
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

def create_wall(p1, p2, height=1):
    """Creates a vertical quad wall between two points."""
    p1.y = 0
    p2.y = 0
    center = (p1 + p2) / 2
    center.y = height/2
    direction = (p2 - p1).normalized()
    up = Vec3(0, 1, 0)

    # Calculate the right vector (perpendicular to direction and up)
    right = direction.cross(up).normalized()

    # Calculate the scale (width) of the wall.  Adjust as needed.
    width = (p2 - p1).length()

    # Create the quad
    wall = Entity(
        model='quad',
        position=center,
        # rotation=Quaternion.look_at(direction, up), # Key for correct rotation
        scale = (width, height),
        texture='vertical_gradient',
        color=(1.0,0.358,0.0,0.5),
        shader=unlit_shader,
        double_sided=True,
    )
    wall.look_at(center+right);
    return wall

class SplineMovingEntity(Entity):
    """
    An entity that moves it's position along a spline function
    """
    def __init__(self, ui, spline_func, model, **kwargs):
        super().__init__(
            model=model,
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
            self.position = self.spline_func(t)

            if self.is_gantry:
                # update the lines between the gantry and the other things
                for anchor_num in range(self.ui.n_anchors):
                    self.ui.lines[anchor_num].model = draw_line(self.ui.anchors[anchor_num].position, self.position)
                self.ui.vert_line.model = draw_line(self.ui.gripper.position, self.position)


class Gripper(SplineMovingEntity):
    def __init__(self, ui, spline_func, **kwargs):
        super().__init__(
            ui=ui,
            spline_func=spline_func,
            collider='box', # for mouse interaction
            **kwargs
        )
        self.label_offset = (0.00, 0.04)
        self.label = Text(
            color=(0.1,0.1,0.1,1.0),
            text=f"Gripper\nNot Detected",
            scale=0.7,
        )
        self.last_ob_render = time.time()


    def setStatus(self, status):
        self.label.text = f"Anchor {self.num} {status}"

    def update(self):
        super().update()
        self.label.position = world_position_to_screen_position(self.position) + self.label_offset
        if time.time() > self.last_ob_render+0.5:
            self.ui.render_gripper_ob()
            self.last_ob_render = time.time()

    def on_mouse_enter(self):
        self.color = anchor_color_selected

    def on_mouse_exit(self):
        self.color = anchor_color

    def on_click(self):
        self.wp = WindowPanel(
        title=f"Gripper Controls",
        content=(
            Button(text='Open', color=color.gold, text_color=color.black),
            Button(text='Show video feed', color=color.gold, text_color=color.black),
            Button(text='Autofocus', color=color.gold, text_color=color.black),
            Button(text='Stop Spool Motor', color=color.gold, text_color=color.black),
            Button(text='Reel in 20cm', color=color.gold, text_color=color.black),
            Button(text='Reel out 20cm', color=color.gold, text_color=color.black),
            Button(text='Sleep', color=color.gold, text_color=color.black),
            ),
        popup=True
        )


anchor_color = (0.8, 0.8, 0.8, 1.0)
anchor_color_selected = (0.9, 0.9, 1.0, 1.0)
class Anchor(Entity):
    def __init__(self, num, position, rotation=(0,0,0)):
        super().__init__(
            position=position,
            rotation=rotation,
            model='anchor',
            color=anchor_color,
            scale=0.001,
            shader=lit_with_shadows_shader,
            collider='box'
        )
        self.num = num
        self.label_offset = (0.00, 0.04)

        self.label = Text(
            color=(0.1,0.1,0.1,1.0),
            text=f"Anchor {self.num}\nNot Detected",
            scale=0.7,
        )

    def setStatus(self, status):
        self.label.text = f"Anchor {self.num} {status}"

    def update(self):
        self.label.position = world_position_to_screen_position(self.position) + self.label_offset

    def on_mouse_enter(self):
        self.color = anchor_color_selected

    def on_mouse_exit(self):
        self.color = anchor_color

    def on_click(self):
        self.wp = WindowPanel(
        title=f"Anchor {self.num}",
        content=(
            Button(text='Show video feed', color=color.gold, text_color=color.black),
            Button(text='Autofocus', color=color.gold, text_color=color.black),
            Button(text='Stop Spool Motor', color=color.gold, text_color=color.black),
            Button(text='Reel in 20cm', color=color.gold, text_color=color.black),
            Button(text='Reel out 20cm', color=color.gold, text_color=color.black),
            Button(text='Sleep', color=color.gold, text_color=color.black),
            ),
        popup=True
        )


class Floor(Entity):
    def __init__(self, **kwargs):
        super().__init__(
            collider='box',
            **kwargs
        )
        self.target = Entity(
            model='quad',
            position=(0, 0, 0),
            rotation=(90,0,0),
            color=color.white,
            scale=(0.5, 0.5),
            texture='green_target.png',
        )

    def on_click(self,):
        print(mouse.world_point)
        # send message to position estimator with desired future position

    def update(self,):
        if mouse.hovered_entity == self:
            # Get the intersection point in world coordinates
            self.target.position = mouse.world_point

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

        # sphereX = Entity(model='sphere', position=(1,0,0), color=color.red, scale=(0.1), shader=unlit_shader)
        # sphereY = Entity(model='sphere', position=(0,1,0), color=color.green, scale=(0.1), shader=unlit_shader)
        # sphereZ = Entity(model='sphere', position=(0,0,1), color=color.blue, scale=(0.1), shader=unlit_shader)

        #show a very large floor
        floor = Floor(model='quad', position=(0, -0.05, 0), rotation=(90,0,0), color=(0.35,0.22,0.18,1.0), scale=(10, 10), shader=lit_with_shadows_shader)  # Scale in meters

        # anchor_color = (0.9, 0.9, 0.9, 1.0)
        # def add_anchor(pos, rot=(0,  0,0)):
        #     # this models units are in mm, but the game units are meters
        #     return Entity(model='anchor', color=anchor_color, scale=0.001, position=pos, rotation=(0,  0,0), shader=lit_with_shadows_shader)
        self.anchors = []
        self.anchors.append(Anchor(0, (-2,2, 3)))
        self.anchors.append(Anchor(1, ( 2,2, 3)))
        self.anchors.append(Anchor(2, ( -1,2,-2), rotation=(0,180,0)))
        self.anchors.append(Anchor(3, ( -2,2,-2), rotation=(0,180,0)))

        self.spline_curves = {}

        self.gantry = SplineMovingEntity(
            ui=self,
            spline_func=None,
            model='gantry',
            color=(0.4, 0.4, 0.0, 1.0),
            scale=0.001,
            position=(0,1,1),
            rotation=(0,0,0),
            shader=lit_with_shadows_shader,
        )

        self.gripper = Gripper(
            ui=self,
            spline_func=None,
            model='gripper_closed',
            color=(0.3, 0.3, 0.9, 1.0),
            scale=0.001,
            position=(0,0.3,1),
            rotation=(-90,0,0),
            shader=lit_with_shadows_shader,
        )

        self.lines = []
        for a in self.anchors:
            self.lines.append(Entity(model=draw_line(a.position, self.gantry.position), color=line_color, shader=unlit_shader))

        self.vert_line = Entity(model=draw_line(self.gantry.position, self.gripper.position), color=line_color, shader=unlit_shader)

        # the color is how you control the brightness
        DirectionalLight(position=(1, 10, 1), shadows=True, rotation=(45, -5, 5), color=(0.8,0.8,0.8,1))
        AmbientLight(color=(0.8,0.8,0.8,1))
        # light.look_at(Vec3(1,-1,1))

        self.calibration_button = Button(
            text="Enter calibration mode",
            color=color.azure,
            position=(-0.7, -0.45), scale=(.35, .033),
            on_click=self.on_calibration_button_click)
        # start in pose calibration mode. TODO need to do this only if any of the four anchor clients boots up but can't find it's file
        # maybe you shouldn't read those files in the clients
        self.calibration_mode = 'run'

        # draw the robot work area boundaries with walls that have a gradient that reaches up from the ground and fades to transparent.
        # between every pair of anchors, draw a horizontal line. if all the other anchors' horizontal positions are on one side of that line, proceed
        # make a vertical quad that passes through that line and apply the fade texture to it.
        create_wall(self.anchors[0].position, self.anchors[1].position)
        create_wall(self.anchors[1].position, self.anchors[2].position)
        create_wall(self.anchors[2].position, self.anchors[3].position)
        create_wall(self.anchors[3].position, self.anchors[0].position)

        # img = cv2.cvtColor(cv2.imread('images/cap_0.jpg'), cv2.COLOR_BGR2RGB)
        # im_pil = Image.fromarray(img)
        # im_pil = im_pil.convert("RGBA")
        # texture = Texture(im_pil)
        # texture = Texture('images/cap_0.jpg')
        self.camview = Entity(model='quad', scale=(2*1.777777, 2), position=(0,4,0))
        self.camview.enabled = False

        self.go_quads = []
        self.max_go_quads = 100
        self.go_quad_next = 0

        Sky(color=color.light_gray)
        EditorCamera()

    # renders a function that returns 3D points in a domain from 0 to 1
    # the y and z coordinates are swapped
    def render_curve(self, curve_function, name):
        try:
            model = Pipe(
                path=[Vec3(tuple(curve_function(time))) for time in np.linspace(0,1,32)],
                thicknesses=(0.01, 0.01),
                cap_ends=False,
            )
        except:
            # Sometimes the verts are very close together and it breaks this constructor, and in that case
            # it's fine to just not update the model
            return
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

    def render_gripper_ob_inner(self, row):
        if len(self.go_quads) < self.max_go_quads:
            self.go_quads.append(Entity(
                model='cube',
                position=(row[4],row[6],row[5]),
                color=color.white, scale=(0.03),
                shader=unlit_shader))
        else:
            self.go_quads[self.go_quad_next].position = (row[4],row[6],row[5])
            self.go_quad_next = (self.go_quad_next+1)%self.max_go_quads

    def render_gripper_ob(self,):
        """
        Display a visual indication of aruco based gripper observations
        """
        gripper_pose = self.datastore.gantry_pose.deepCopy()
        for row in gripper_pose:
            self.render_gripper_ob_inner(row)

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

            if 'pil_image' in updates:
                print('received pil image in UI')
                pili = updates['pil_image']
                self.camview.texture = Texture(pili.convert("RGBA"))


    def start(self):
        self.app.run()

def start_ui(datastore, to_ui_q, to_pe_q, to_ob_q):
    """
    Entry point to be used when starting this from main.py with multiprocessing
    """
    cpui = ControlPanelUI(datastore, to_pe_q, to_ob_q)

    # use simple threading here. ursina has it's own loop that conflicts with asyncio
    estimator_update_thread = threading.Thread(target=cpui.receive_updates, args=(to_ui_q, ), daemon=True)
    estimator_update_thread.start()

    rgo = threading.Thread(target=cpui.render_gripper_ob, daemon=True)
    rgo.start()

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
    from data_store import DataStore
    datastore = DataStore(horizon_s=10, n_cables=4)
    to_ui_q = Queue()
    to_pe_q = Queue()
    to_ob_q = Queue()
    start_ui(datastore, to_ui_q, to_pe_q, to_ob_q)