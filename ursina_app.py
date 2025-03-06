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
from position_estimator import default_weights, weight_names, find_intersection
from cv_common import average_pose, compose_poses
import model_constants

from ursina import *
from ursina.shaders import (
    lit_with_shadows_shader,
    unlit_shader,
)
from ursina.prefabs.dropdown_menu import DropdownMenu, DropdownMenuButton

# the color for the lines that connect the anchors, gantry, and gripper
line_color = color.black

# user readable mode names
mode_names = {
    'run':   'Run Normally',
    'pause': 'Pause/Observe',
    'pose':  'Find Anchor Postions',
}
mode_descriptions = {
    'run':   'Movement continuously follows the green spline. Goal positions are selected automatically or can be added with the mouse',
    'pause': 'WASD-QE moves the gantry, scroll moves the winch line. Space toggles the grip. Spline fitting from observation occurs but has no effect.',
    'pose':  'Place the origin card on the floor in the center of the room. Anchor positions are continuously estimated from this card.',
}
detections_format_str = 'Detections/sec {val:.2f}'
video_latency_format_str = 'Video latency {val:.2f} s'
video_framerate_format_str = 'Avg framerate {val:.2f} fps'

ds = 0.1 # direct movement speed in meters per second
key_behavior = {
    # key: (axis, speed)
    'a': (0, -ds),
    'd': (0, ds),
    'w': (1, ds),
    's': (1, -ds),
    'q': (2, -ds),
    'e': (2, ds),

    'a up': (0, 0),
    'd up': (0, 0),
    'w up': (1, 0),
    's up': (1, 0),
    'q up': (2, 0),
    'e up': (2, 0),
}

def input(key):
    pass

# ursina considers +Y up. all the other processes, such as the position estimator consider +Z up. 
def swap_yz(vec):
    return (vec[0], vec[2], vec[1])

# Transforms a rodrigues rotation vector ummmm its magic don't ask
def fix_rot(vec):
    # return (vec[1], -vec[2], vec[0])
    return (vec[0], vec[2], vec[1])

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
            scale=0.6,
        )
        # self.last_ob_render = time.time()


    def setStatus(self, status):
        self.label.text = f"Gripper\n{status}"

    def update(self):
        super().update()
        self.label.position = world_position_to_screen_position(self.position) + self.label_offset
        # if time.time() > self.last_ob_render+0.5:
        #     self.ui.render_gripper_ob()
        #     self.last_ob_render = time.time()

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
            Button(text='Reel in 20cm', color=color.orange, text_color=color.black, on_click=self.reel_manual),
            Button(text='Reel out 20cm', color=color.orange, text_color=color.black, on_click=self.reel_manual),
            Button(text='Sleep', color=color.gold, text_color=color.black),
            ),
        popup=True
        )

    def reel_manual(self, delta_meters):
        self.wp.enabled = False


anchor_color = (0.8, 0.8, 0.8, 1.0)
anchor_color_selected = (0.9, 0.9, 1.0, 1.0)
class Anchor(Entity):
    def __init__(self, num, to_ob_q, position, rotation=(0,0,0)):
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
        self.to_ob_q = to_ob_q
        self.pose = np.array((rotation, position))

        self.label = Text(
            color=(0.1,0.1,0.1,1.0),
            text=f"Anchor {self.num}\nNot Detected",
            scale=0.6,
        )

    def setStatus(self, status):
        self.label.text = f"Anchor {self.num}\n{status}"

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
            Button(text='Reel in 20cm', color=color.orange, text_color=color.black,
                on_click=partial(self.reel_manual, -0.2)),
            Button(text='Reel out 20cm', color=color.orange, text_color=color.black,
                on_click=partial(self.reel_manual, 0.2)),
            Button(text='Sleep', color=color.gold, text_color=color.black),
            ),
        popup=True
        )

    def reel_manual(self, delta_meters):
        self.wp.enabled = False
        self.to_ob_q.put({'jog_spool':{'anchor':self.num, 'rel':delta_meters}})        


class Floor(Entity):
    def __init__(self, **kwargs):
        super().__init__(
            collider='box',
            **kwargs
        )
        self.alt = 0.3 # altitude of blue circle in meters
        self.target = Entity(
            model='quad',
            position=(0, 0, 0),
            rotation=(90,0,0),
            color=color.white,
            scale=(0.5, 0.5, 0.5),
            texture='green_target.png',
        )
        self.pipe = Entity(
            model=Pipe(
                path=[(0,0,0), (0,self.alt*2,0)], # have to multiply by 2 because it inherited the scale of it's parent
                thicknesses=(0.01, 0.01),
                cap_ends=True),
            rotation=(-90,0,0),
            parent=self.target,
            color=color.white,
        )
        self.circle = Entity(
            model='quad',
            position=(0, 0, -self.alt*2),
            color=color.white,
            scale=(0.2, 0.2, 0.2),
            parent=self.target,
            texture='blue_circle.png',
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
        self.time_domain = (1,2)
        self.grip_closed = True
        self.direction = np.array([0,0,0], dtype=float)
        self.could_be_moving = False

        # only matters when this file is main
        global input
        input = self.input

        # start in pose calibration mode. TODO need to do this only if any of the four anchor clients boots up but can't find it's file
        # maybe you shouldn't read those files in the clients
        self.calibration_mode = 'pause'

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
        self.anchors.append(Anchor(0, to_ob_q, (-2,2, 3)))
        self.anchors.append(Anchor(1, to_ob_q, ( 2,2, 3)))
        self.anchors.append(Anchor(2, to_ob_q, ( -1,2,-2), rotation=(0,180,0)))
        self.anchors.append(Anchor(3, to_ob_q, ( -2,2,-2), rotation=(0,180,0)))

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

        # show a visualization of goal positions
        self.goals = [Entity(
                position=(0,0,0),
                model='sphere', # todo use a map marker thing
                color=color.azure,
                scale=0.1,
                shader=lit_with_shadows_shader,
                enabled=False,
            ) for i in range(8)]

        # the color is how you control the brightness
        DirectionalLight(position=(1, 10, 1), shadows=True, rotation=(45, -5, 5), color=(0.8,0.8,0.8,1))
        AmbientLight(color=(0.8,0.8,0.8,1))
        # light.look_at(Vec3(1,-1,1))

        self.stop_button = Button(
            text="STOP",
            color=color.red,
            position=(0.83, -0.45), scale=(.10, .033),
            on_click=self.on_stop_button)

        # draw the robot work area boundaries with walls that have a gradient that reaches up from the ground and fades to transparent.
        # between every pair of anchors, draw a horizontal line. if all the other anchors' horizontal positions are on one side of that line, proceed
        # make a vertical quad that passes through that line and apply the fade texture to it.
        # create_wall(self.anchors[0].position, self.anchors[1].position)
        # create_wall(self.anchors[1].position, self.anchors[2].position)
        # create_wall(self.anchors[2].position, self.anchors[3].position)
        # create_wall(self.anchors[3].position, self.anchors[0].position)

        # img = cv2.cvtColor(cv2.imread('images/cap_0.jpg'), cv2.COLOR_BGR2RGB)
        # im_pil = Image.fromarray(img)
        # im_pil = im_pil.convert("RGBA")
        # texture = Texture(im_pil)
        # texture = Texture('images/cap_0.jpg')
        self.camview = Entity(model='quad', scale=(2*1.777777, 2), position=(0,4,0))
        self.camview.enabled = False

        self.go_quads = []
        self.max_go_quads = 200
        self.go_quad_next = 0

        self.modePanel = Panel(model='quad', z=99, 
            color=(0.1,0.1,0.1,1.0),
            position=(0,-0.48),
            scale=(3,0.1),
        )
        self.modePanelLine1y = -0.44
        self.modePanelLine2y = -0.46
        self.modePanelLine3y = -0.48

        self.error = Text(
            color=color.red,
            position=(0.1,self.modePanelLine3y),
            text="error",
            scale=0.6,
            enabled=False,
        )

        self.mode_text = Text(
            color=(0.0,1.0,0.3,1.0),
            position=(-0.1,self.modePanelLine1y),
            text=mode_names[self.calibration_mode],
            scale=0.7,
            enabled=True,
        )

        self.mode_descrip_text = Text(
            color=(0.9,0.9,0.9,1.0),
            position=(-0.45,self.modePanelLine2y),
            text=mode_descriptions[self.calibration_mode],
            scale=0.6,
            enabled=True,
        )

        self.detections_text = Text(
            color=(0.9,0.9,0.9,1.0),
            position=(-0.805,self.modePanelLine1y),
            text=detections_format_str.format(val=0),
            scale=0.5,
            enabled=True,
        )

        self.video_latency_text = Text(
            color=(0.9,0.9,0.9,1.0),
            position=(-0.805,self.modePanelLine2y),
            text=video_latency_format_str.format(val=0),
            scale=0.5,
            enabled=True,
        )

        self.video_framerate_text = Text(
            color=(0.9,0.9,0.9,1.0),
            position=(-0.805,self.modePanelLine3y),
            text=video_framerate_format_str.format(val=0),
            scale=0.5,
            enabled=True,
        )

        # make a little video monitor icon for each camera status.
        x = -0.869
        y = -0.483
        offx = 0.042
        offy = 0.037
        self.vid_status = [Panel(model='quad', z=91, 
            color=color.white,
            texture='vid_out.png',
            position=position,
            scale=(0.01998,0.01498),
        ) for position in [(x,y), (x+offx,y), (x,y+offy), (x+offx,y+offy), ((x+offx/2,y+offy/2))]]

        # position estimator controls
        slider_row_h = 0.025
        self.sliders = [ThinSlider(
            text=weight_names[i],
            dynamic=True,
            scale=0.5,
            position=(0.62, -0.15 - slider_row_h*i),
            max=10,
            value=default_weights[i],
            on_value_changed=partial(self.change_weight, i),
        ) for i in range(len(default_weights))]
        for thing in self.sliders:
            thing.bg.color = color.black
            thing.knob.color = color.black
            thing.label.color = color.black
            thing.knob.text_color = color.black

        # self.direct_move_indicator = Entity(model='arrow', color=color.lime, scale=(1, 1, 1), position=(0,0.5,0))

        DropdownMenu('Menu', buttons=(
            DropdownMenu('Mode', buttons=(
                DropdownMenuButton(mode_names['run'], on_click=partial(self.set_mode, 'run')),
                DropdownMenuButton(mode_names['pause'], on_click=partial(self.set_mode, 'pause')),
                DropdownMenuButton(mode_names['pose'], on_click=partial(self.set_mode, 'pose')),
                )),
            DropdownMenuButton('Calibrate Line Lengths', on_click=self.calibrate_lines),
            ))

        Sky(color=color.light_gray)
        EditorCamera()

    def input(self, key):
        if key == 'space':
            self.grip_closed = not self.grip_closed
            self.to_ob_q.put({'set_grip': self.grip_closed})

        if key in key_behavior:
            axis, speed = key_behavior[key]
            self.direction[axis] = speed

    def change_weight(self, index):
        self.to_pe_q.put({'weight_change': (index, self.sliders[index].value)})

    def show_error(self, error):
        self.error.text = error
        self.error.enabled = True

    def clear_error(self):
        self.error.enabled = False

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

    def set_mode(self, mode):
        self.calibration_mode = mode
        self.mode_text.text = mode_names[mode]
        self.mode_descrip_text.text = mode_descriptions[mode]
        self.to_ob_q.put({'set_run_mode': self.calibration_mode})

    def calibrate_lines(self):
        # calibrate line lengths
        # Assume we have been stopped for a while.
        if self.calibration_mode != 'pause':
            self.error.text = "Line calibration can only be performed while in Pause/Observe mode"
            self.error.enabled = True
            return
        print('Do line calibration')
        # average recent gantry poses.
        poses = self.datastore.gantry_pose.deepCopy()
        gantry_pose = average_pose(poses[:,1:].reshape(-1,2,3))[:3]
        lengths = [3.79,4.94,2.95,4.08] 
        # for i, anchor in enumerate(self.anchors):
        #     grommet_pose = compose_poses([anchor.pose, model_constants.anchor_grommet])[:3]
        #     distance = np.linalg.norm(grommet_pose[1] - gantry_pose[1])
        #     lengths[i] = distance - 0.02 # to make up for the distance between the gantry origin and it's grommets, which are all symmetric
        # display a confirmation dialog
        fmt = '{:.3f}'
        self.line_cal_confirm = WindowPanel(
            title=f"Calculated Anchor Line Lengths",
            content=(
                Text(text='Distances in meters'),
                InputField(default_value=fmt.format(lengths[0])),
                InputField(default_value=fmt.format(lengths[1])),
                InputField(default_value=fmt.format(lengths[2])),
                InputField(default_value=fmt.format(lengths[3])),
                Button(text='Confirm', color=color.azure,  on_click=self.finish_calibration),
                ),
            popup=True
            )

    def finish_calibration(self):
        # read the lengths that the user may have modified
        try:
            lengths = [float(self.line_cal_confirm.content[1].text) for i in range(4)]
        except ValueError:
            return
        self.to_ob_q.put({'do_line_calibration': lengths})
        self.line_cal_confirm.enabled = False

    def on_stop_button(self):
        self.to_ob_q.put({'set_run_mode':'pause'})

    def render_gripper_ob(self, row, color):
        if len(self.go_quads) < self.max_go_quads:
            self.go_quads.append(Entity(
                model='cube',
                position=(row[4],row[6],row[5]),
                color=color, scale=(0.03),
                shader=unlit_shader))
        else:
            self.go_quads[self.go_quad_next].position = (row[4],row[6],row[5])
            self.go_quads[self.go_quad_next].color = color
            self.go_quad_next = (self.go_quad_next+1)%self.max_go_quads

    def periodic_actions(self):
        """
        Run certain actions at a rate slightly less than, and independent of the framerate.
        """
        # Display a visual indication of aruco based gripper observations
        time.sleep(2)
        while True:
            gantry_pose = self.datastore.gantry_pose.deepCopy()
            for row in gantry_pose:
                invoke(self.render_gripper_ob, row, color.white)
            gripper_pose = self.datastore.gripper_pose.deepCopy()
            for row in gripper_pose:
                invoke(self.render_gripper_ob, row, color.light_gray)

            # send a direct move command in pause mode
            if self.calibration_mode == 'pause':
                invoke(self.direct_move)
            
            time.sleep(0.25)

    def notify_connected_bots_change(self, available_bots={}):
        offs = 0
        for server,info in available_bots.items():
            text_entity = Text(server, world_scale=16, position=(-0.1, -0.4 + offs))
            offs -= 0.03

    def get_simplified_position(self):
        """
        Calculate a gantry position based solely on the last line record from each anchor and the anchor poses
        """
        lengths = []
        anchor_positions = []
        for i, alr in enumerate(self.datastore.anchor_line_record):
            lengths.append(alr.getLast()[-1])
            anchor_positions.append(swap_yz(self.anchors[i].position))
        if sum(lengths) == 0:
            invoke(self.show_error, "Must be connected and perform line calibration before using direct movement")
            return anchor_positions, False, None
        anchor_positions = np.array(anchor_positions)
        lengths = np.array(lengths)
        result = find_intersection(anchor_positions, lengths)
        return anchor_positions, result.x, result.success

    def direct_move(self):
        """
        Send planned anchor lines to the robot that would move the gantry in a straight line
        in the direction of self.direction for a small amount of time.
        """
        move_duration = 1 # seconds
        if self.calibration_mode != 'pause':
            return
        if sum(self.direction) == 0:
            if self.could_be_moving:
                # immediately cancel whatever remains of the movement
                self.to_ob_q.put({'slow_stop_all': None})
                # set a flag so we don't do this constantly, even though it would be harmless.
                self.could_be_moving = False
            return
        anchor_positions, start, success = self.get_simplified_position()
        if not success:
            print('could not obtain position from line lengths')
            return
        # TODO invoke a function that will visually indicate the position and direction
        # calculate a few time intervals in the near future
        times = np.linspace(0, move_duration, 6, dtype=np.float64).reshape(-1, 1)
        # where we want the gantry to be at the time intervals
        gantry_positions = self.direction * times + start
        # represent as absolute times
        times = times + time.time()
        # the anchor line lengths if the gantry were at those positions
        # format as an array of times and lengths, one array for each anchor
        future_anchor_lines = np.array([
            np.column_stack([
                times,
                np.linalg.norm(gantry_positions - pos, axis=1)])
            for pos in anchor_positions])
        # send it
        self.to_ob_q.put({
            'future_anchor_lines': {'sender':'ui', 'data':future_anchor_lines},
        })
        self.could_be_moving = True

    def receive_updates(self, min_to_ui_q):
        while True:
            # queue.get has to happen in a thread, because it blocks
            updates = min_to_ui_q.get()
            # but processing the update needs to happen in the ursina loop, because it will modify a bunch of entities.
            invoke(self.process_update, updates)

    def process_update(self, updates):
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
            self.anchors[anchor_num].pose = apose[1]
            self.anchors[anchor_num].position = swap_yz(apose[1][1])
            self.anchors[anchor_num].quaternion = LQuaternionf(*list(Rotation.from_rotvec(np.array(fix_rot(apose[1][0]))).as_quat()))

        if 'pil_image' in updates:
            print('received pil image in UI')
            pili = updates['pil_image']
            self.camview.texture = Texture(pili.convert("RGBA"))

        if 'connection_status' in updates:
            status = updates['connection_status']
            print(status)
            if status['websocket'] == 0:
                user_status_str = 'Not Detected'
            elif status['websocket'] == 1:
                user_status_str = 'Connecting...'
            else:
                user_status_str = 'Online'

            if status['video']:
                vidstatus_tex = 'vid_ok.png'
            else:
                vidstatus_tex = 'vid_out.png'

            if 'anchor_num' in status:
                self.anchors[status['anchor_num']].setStatus(user_status_str)
                self.vid_status[status['anchor_num']].texture = vidstatus_tex
            elif 'gripper' in status:
                self.gripper.setStatus(user_status_str)
                self.vid_status[4].texture = vidstatus_tex

        if 'vid_stats' in updates:
            stats = updates['vid_stats']
            self.detections_text.text = detections_format_str.format(val=stats['detection_rate'])
            self.video_latency_text.text = video_latency_format_str.format(val=stats['video_latency'])
            self.video_framerate_text.text = video_framerate_format_str.format(val=stats['video_framerate'])

        if 'goal_points' in updates:
            for i, gp in enumerate(updates['goal_points']):
                if i == len(self.goals):
                    break
                self.goals[i].enabled = True
                self.goals[i].position = swap_yz(gp)
            # disable the rest
            for i in range(len(updates['goal_points']), len(self.goals)):
                self.goals[i].enabled = False

        if 'input' in updates:
            self.input(updates['input'])


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

    rgo = threading.Thread(target=cpui.periodic_actions, daemon=True)
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