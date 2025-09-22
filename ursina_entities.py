from ursina import *
from ursina.shaders import (
    lit_with_shadows_shader,
    unlit_shader,
)
import numpy as np
from cv_common import invert_pose, compose_poses
import model_constants
from scipy.spatial.transform import Rotation
from functools import partial
import time

# ursina considers +Y up. all the other processes, such as the position estimator consider +Z up. 
def swap_yz(vec):
    return (vec[0], vec[2], vec[1])

# Transforms a rodrigues rotation vector into an ursina euler rotation tuple in degrees
def to_ursina_rotation(rvec):
    euler = Rotation.from_rotvec(rvec).as_euler('xyz', degrees=True)
    return (-euler[0], -euler[2], euler[1])

droop = (np.linspace(-1,1,20)**2-1)*0.2 # droop of 0.2 meters
def draw_line_slack(point_a, point_b):
    verts = np.linspace(point_a, point_b, 20)
    verts[:,1] += droop
    return Mesh(vertices=verts, mode='line')

def draw_line(point_a, point_b, slack=False):
    if slack:
        return draw_line_slack(point_a, point_b)
    else:
        return Mesh(vertices=[point_a, point_b], mode='line')

MAX_JOG_SPEED = 0.3

gripper_color = (0.9, 0.20, 0.34, 1.0)

class Gantry(Entity):
    def __init__(self, ui, to_ob_q, **kwargs):
        super().__init__(**kwargs)
        self.ui = ui
        self.to_ob_q = to_ob_q
        self.last_update_t = time.time()

        # position and velocity in the z up coordinate space, not the ursina space
        self.zup_pos = np.zeros(3)
        self.zup_vel = np.zeros(3)

        self.slack = [False, False, False, False]

    def set_position_velocity(self, pos, vel):
        self.zup_pos = pos
        self.zup_vel = vel
        self.position = swap_yz(pos)
        # height of the mouse target reticule, not the height of the floor
        # self.ui.floor.set_reticule_height(pos[2])

    def set_slack_vis(self, slack):
        """slack is an array of bools for each anchor. true meaning the line should be drawn with a droop"""
        self.slack = slack

    def update(self):
        now = time.time()
        elapsed = now - self.last_update_t
        self.zup_pos = self.zup_pos + self.zup_vel * elapsed
        self.position = swap_yz(self.zup_pos)
        self.last_update_t = now
        self.redraw_wires()

    def redraw_wires(self):
        # update the lines between the gantry and the other things
        for anchor_num in range(self.ui.n_anchors):
            self.ui.lines[anchor_num].model = draw_line(self.ui.anchors[anchor_num].position, self.position, slack=self.slack[anchor_num])
        self.ui.vert_line.model = draw_line(self.ui.gripper.position, self.position)

class Gripper(Entity):
    def __init__(self, ui, to_ob_q, **kwargs):
        super().__init__(
            collider='box', # for mouse interaction
            model='gripper_body',
            scale=0.001,
            color=gripper_color,
            **kwargs
        )
        self.ui = ui,
        self.to_ob_q = to_ob_q
        self.closed = False
        self.label_offset = (0.00, 0.04)
        self.label = Text(
            color=(0.1,0.1,0.1,1.0),
            text=f"Gripper\nNot Detected",
            scale=0.6,
            enabled=False,
        )
        self.left_finger = Entity(
            parent=self,
            model='gripper_finger',
            position=(-25,41,-10.5),
            color=(0.0, 0.2, 0.0, 1.0),
        )
        self.right_finger = Entity(
            parent=self,
            model='gripper_finger',
            position=(25,41,-10.5),
            scale=(-1,1,1),
            color=(0.0, 0.2, 0.0, 1.0),
        )
        self.remap = {
            'r down': 'up arrow down',
            'r up': 'up arrow up',
            'r hold': 'up arrow hold',
            'f down': 'down arrow down',
            'f up': 'down arrow up',
            'f hold': 'down arrow hold',
        }
        # winch line jog speed
        self.jog_speed = 0
        self.vid_visible = False
        self.ip_address = None

    def setStatus(self, status):
        self.label.text = f"Gripper\n{status}"

    def setPose(self, pose):
        """
        Pose is a tuple containing a rotation vector (rodruiges) and a position in the z-up space
        """
        self.rotation = to_ursina_rotation(pose[0])
        self.position = swap_yz(pose[1])

    def on_mouse_enter(self):
        self.color = anchor_color_selected

    def on_mouse_exit(self):
        self.color = gripper_color

    def on_click(self):
        self.wp = WindowPanel(
        title=f"Gripper at {self.ip_address}",
        content=(
            Button(text='Open (Space)', color=color.gold, text_color=color.black),
            Button(text='Show video feed', color=color.gold, text_color=color.black,
                on_click=self.toggle_vid_feed),
            Button(text='Stop Spool Motor', color=color.gold, text_color=color.black),
            Button(text='Manual Spool Control', color=color.blue, text_color=color.white,
                on_click=self.open_manual_spool_control),
            ),
        popup=True,
        )

    def open_manual_spool_control(self):
        self.wp.enabled = False
        self.jog_speed = 0.1
        self.manual_spool_wp = WindowPanel(
            title="Manual Spool Control",
            content=(
                Text(text="Use buttons or Up/Down arrow keys to control spool."),
                # Button(text='Reel in 5cm', color=color.orange, text_color=color.black,
                #     on_click=partial(self.reel_manual, -0.05)),
                # Button(text='Reel out 5cm', color=color.orange, text_color=color.black,
                #     on_click=partial(self.reel_manual, 0.05)),
            ),
            popup=True,
        )

    def input(self, key):
        canMove = hasattr(self, "manual_spool_wp") and self.manual_spool_wp.enabled
        mkey = key
        if key in self.remap:
            canMove = True
            mkey = self.remap[key]


        if canMove:
            print(f'Gripper client input key {key} remapped to {mkey} and movement enabled')
            if mkey == 'up arrow':
                self.reel_manual(-self.jog_speed)
            elif mkey == 'up arrow hold':
                if self.jog_speed < MAX_JOG_SPEED:
                    self.jog_speed += 0.005
                self.reel_manual(-self.jog_speed)
            elif mkey == 'up arrow up':
                self.jog_speed = 0.1
                self.to_ob_q.put({'slow_stop_one':{'id':'gripper'}})


            elif mkey == 'down arrow':
                self.reel_manual(self.jog_speed)
            elif mkey == 'down arrow hold':
                if self.jog_speed < MAX_JOG_SPEED:
                    self.jog_speed += 0.005
                self.reel_manual(self.jog_speed)
            elif mkey == 'down arrow up':
                self.jog_speed = 0.1
                self.to_ob_q.put({'slow_stop_one':{'id':'gripper'}})

    def reel_manual(self, metersPerSecond):
        print(f'jog speed {metersPerSecond}')
        self.to_ob_q.put({'jog_spool':{'gripper':None, 'speed':metersPerSecond}})


    def toggleClosed(self):
        self.closed = not self.closed
        self.to_ob_q.put({'set_grip': self.closed})
        self.setAppearanceClosed(self.closed)

    def setAppearanceClosed(self, closed):
        if closed:
            self.left_finger.rotation = (0,0,0)
            self.right_finger.rotation = (0,0,0)
        else:
            self.left_finger.rotation = (0,0,60)
            self.right_finger.rotation = (0,0,-60)

    def toggle_vid_feed(self):
        self.vid_visible = not self.vid_visible
        self.to_ob_q.put({'toggle_previews':{'gripper':None, 'status':self.vid_visible}})


anchor_color = (0.8, 0.8, 0.8, 1.0)
anchor_color_selected = (0.9, 0.9, 1.0, 1.0)
class Anchor(Entity):
    def __init__(self, num, to_ob_q, position, rotation=(0,0,0)):
        super().__init__(
            position=position,
            rotation=rotation,
            model='anchor',
            color=anchor_color,
            scale=1,
            shader=lit_with_shadows_shader,
            collider='box'
        )
        self.num = num
        self.label_offset = (0.00, 0.04)
        self.to_ob_q = to_ob_q
        self.pose = np.array((rotation, position))
        self.ip_address = None

        self.label = Text(
            color=(0.1,0.1,0.1,1.0),
            text=f"Anchor {self.num}\nNot Detected",
            scale=0.5,
        )

        self.empty = Entity(
            scale=1,
            # rotation=(-35,-90,180),
            rotation=to_ursina_rotation(compose_poses([
                model_constants.anchor_camera,
                (np.array([pi/2,0,0], dtype=float), np.array([0,0,0], dtype=float))
                ])[0]),
            parent=self)

        self.camview = Entity(
            model='quad',
            scale=(2, 2/1.777777),
            position=(0,0,2),
            texture='cap_38.jpg',
            shader=unlit_shader,
            parent=self.empty,
            enabled=False)

        # flag for doing something once
        self.hasSetImage = False

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
        title=f"Anchor {self.num} at {self.ip_address}",
        content=(
            Button(text='Show video feed', color=color.gold, text_color=color.black,
                on_click=self.toggle_vid_feed),
            Button(text='Autofocus', color=color.gold, text_color=color.black),
            Button(text='Stop Spool Motor', color=color.gold, text_color=color.black),
            Button(text='Manual Spool Control', color=color.blue, text_color=color.white,
                on_click=self.open_manual_spool_control),
            Button(text='Reference Load', color=color.gold, text_color=color.black,
                on_click=self.open_ref_load_dialog),
            ),
        popup=True
        )

    def open_manual_spool_control(self):
        self.wp.enabled = False
        self.jog_speed = 0.1
        self.manual_spool_wp = WindowPanel(
            title="Manual Spool Control",
            content=(
                Text(text="Use buttons or Up/Down arrow keys to control spool."),
                # Button(text='Reel in 5cm', color=color.orange, text_color=color.black,
                #     on_click=partial(self.reel_manual, -0.05)),
                # Button(text='Reel out 5cm', color=color.orange, text_color=color.black,
                #     on_click=partial(self.reel_manual, 0.05)),
            ),
            popup=True,
        )

    def input(self, key):
        if hasattr(self, "manual_spool_wp") and self.manual_spool_wp.enabled:
            print(key)
            if key == 'up arrow':
                self.reel_manual(-self.jog_speed)
            elif key == 'up arrow hold':
                if self.jog_speed < MAX_JOG_SPEED:
                    self.jog_speed += 0.005
                self.reel_manual(-self.jog_speed)
            elif key == 'up arrow up':
                self.jog_speed = 0.1
                self.to_ob_q.put({'slow_stop_one':{'id':self.num}})


            elif key == 'down arrow':
                self.reel_manual(self.jog_speed)
            elif key == 'down arrow hold':
                if self.jog_speed < MAX_JOG_SPEED:
                    self.jog_speed += 0.005
                self.reel_manual(self.jog_speed)
            elif key == 'down arrow up':
                self.jog_speed = 0.1
                self.to_ob_q.put({'slow_stop_one':{'id':self.num}})



    def reel_manual(self, metersPerSecond):
        self.wp.enabled = False
        print(f'jog speed {metersPerSecond}')
        self.to_ob_q.put({'jog_spool':{'anchor':self.num, 'speed':metersPerSecond}})

    def open_ref_load_dialog(self):
        self.ref_load_wp = WindowPanel(
            title="Measure Reference Load",
            content=(
                Text(text="""Calibrate tension measurement by hanging a known weight
on the line between 0.3 and 1.0 kg.
Enter the actual weight in kg."""),
                InputField(default_value="0.5"),
                Button(text='Calibrate', color=color.orange, text_color=color.black,
                    on_click=self.measure_ref_load),
            ),
            popup=True,
        )

    def measure_ref_load(self):
        self.ref_load_wp.enabled = False
        load = float(self.ref_load_wp.content[1].text)
        self.to_ob_q.put({'measure_ref_load': {
            'anchor_num': self.num,
            'load': load
            }})

    def toggle_vid_feed(self):
        self.camview.enabled = not self.camview.enabled
        self.to_ob_q.put({'toggle_previews':{'anchor':self.num, 'status':self.camview.enabled}})


class Floor(Entity):
    def __init__(self, app, **kwargs):
        super().__init__(
            collider='box',
            **kwargs
        )
        self.app = app
        self.alt = 0.3 # altitude of blue circle in meters
        self.button = None
        self.button_offset = (0.02, -0.03)
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

        self.camview = Entity(
            model='quad',
            scale=(2, 1/1.777777),
            position=(0,1,-1),
            rotation=(0,90,0),
            texture='cap_38.jpg',
            shader=unlit_shader,
            parent=self,
            enabled=True)
        self.hasSetImage = False

    def set_reticule_height(self, height):
        self.alt = height
        self.circle.position[2] = -self.alt*2
        self.pipe.model=Pipe(
                path=[(0,0,0), (0,self.alt*2,0)],
                thicknesses=(0.01, 0.01),
                cap_ends=True)

    def on_click(self,):
        print(mouse.world_point)
        if self.button is None:
            # put a goal point indicator here and then the user to click it to confirm
            self.button = Button(
                color=rgb(0.1,0.8,0.1),
                text_color=color.black,
                text="Confirm",
                scale=(0.05, 0.02),
                text_size=0.5,
                on_click=self.button_confirm
            )
        else:
            self.button = None

    def update(self):
        if self.button is None:
            if mouse.hovered_entity == self:
                # Get the intersection point in world coordinates
                self.target.position = mouse.world_point
        else:
            self.button.position = world_position_to_screen_position(self.circle.world_position) + self.button_offset

    def button_confirm(self):
        self.button.enabled = False
        self.button = None
        gantry_goal = swap_yz(self.circle.world_position)
        self.app.to_ob_q.put({'gantry_goal_pos': gantry_goal})

class GoalPoint(Entity):
    def __init__(self, position, **kwargs):
        super().__init__(
            position=position,
            rotation=(-90,0,0),
            model='map_marker',
            color=color.azure,
            scale=0.075,
            shader=lit_with_shadows_shader,
            **kwargs
        )
        self.atime = time.time()+10
        self.fmt = 'ETA {val:.2f}'
        self.label_offset = (-0.02, -0.03)
        self.label = Text(
            color=(0.1,0.1,0.1,1.0),
            text=f"",
            scale=0.5,
        )

    def update(self):
        self.label.position = world_position_to_screen_position(self.position) + self.label_offset
        self.label.text = self.fmt.format(val=(time.time()-self.atime))

class EntityPool:
    def __init__(self, max_items, new_entity_fn):
        self.entities = []
        self.max_items = max_items
        self.next = 0
        self.new_entity_fn = new_entity_fn

    def replace(self, item_list, update_entity_fn):
        """show up to max_items from items_list and nothing more
        update_entity_fn(a, b) will be called for each item in item_list, where a is the item and b is the entity to be updated
        disable anything else in the pool that was not updated by this call
        """
        i = 0
        while i < min(len(item_list), self.max_items):
            # show an item with this entity
            if len(self.entities) <= i:
                self.entities.append(self.new_entity_fn())
            update_entity_fn(item_list[i], self.entities[i])
            self.entities[i].enabled = True
            i += 1
        # disable whatever remains
        while i < len(self.entities):
            self.entities[i].enabled = False
            i += 1
        self.next = 0

    def add(self, update_entity_fn):
        """show a new item, reusing the oldest one if needed"""
        if len(self.entities) < self.max_items:
            self.entities.append(self.new_entity_fn())
            update_entity_fn(self.entities[-1])
            self.entities[-1].enabled = True
        else:
            update_entity_fn(self.entities[self.next])
            self.entities[-1].enabled = True
            self.next = (self.next + 1) % self.max_items

class ThinSliderLog2(ThinSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # feedback for the errors of the last minimizer step
        self.errorbar = Entity(
            parent=self.bg,
            model=Quad(
                radius=0.01,
                segments=3),
            origin_x=-0.5,
            scale=(0.5, 0.035),
            color=color.red,
            z=0.2,
        )

        self._update_text()

    def setErrorBar(self, value):
        """value expected in range of 0 to 1"""
        self.errorbar.scale = (value/2, 0.035)

    def finalValue(self):
        return 2**(self.value-5)

    def _update_text(self):
        self.knob.text_entity.text = str(round(self.finalValue(), 3))

class IndicatorSphere(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, 
            model='sphere',
            position=(0,0,0),
            scale=(0.06),
            shader=unlit_shader)
        self.zup_pos = np.zeros(3)
        self.zup_vel = np.zeros(3)
        self.last_update_t = time.time()

    def set_position_velocity(self, pos, vel):
        """set position and velocity in the z-up space"""
        self.zup_pos = pos
        self.zup_vel = vel
        self.position = swap_yz(self.zup_pos)

    def update(self):
        now = time.time()
        elapsed = now - self.last_update_t
        self.zup_pos = self.zup_pos + self.zup_vel * elapsed
        self.position = swap_yz(self.zup_pos)
        self.last_update_t = now

# consider completely removing this Entity.
# move input handling and sending of the gantry_dir_sp message somewhere else.
# make the commanded velocity indicator work like the visual position indicator.
# as it is now this is not even an accurate visualization of what velocity is being commanded.
class DirectMoveGantryTarget(Entity):
    """A visual indicator and manager of gantry direct movement commands"""

    def __init__(self, app, *args, **kwargs):
        super().__init__(*args, **kwargs, 
            model='sphere',
            position=(0,0,0),
            color=color.cyan,
            scale=(0.1),
            shader=unlit_shader)
        self.app = app
        self.speed = 0.25
        self.last_update_t = time.time()

        # game pad analog movement vector (left stick and triggers)
        self.analog_dir = [0,0,0]

        # last actual commanded gantry velocity from observer in z-up coordinate space
        self.last_commanded_vel = np.zeros(3)

    def reset(self):
        self.last_move_vec = None

    def direct_move(self, speed=0.25):
        """
        Send speeds that would move the gantry in a straight line
        from where it is, towards the indicated goal point, at the given speed.
        positions are given in z-up coordinate system.
        """
        if sum(self.analog_dir) != 0:
            # game pad controller
            vector = np.array(self.analog_dir)
            mag = np.linalg.norm(vector)
            vector = vector / mag
            speed = mag * speed
        elif sum(self.app.direction) != 0:
            # keyboard
            vector = self.app.direction
            vector = vector / np.linalg.norm(vector)
        else:
            return

        print(f'direct move {vector} {speed}')
        self.app.to_ob_q.put({
            'gantry_dir_sp': {
                'direction':vector,
                'speed':speed,
            }
        })

        self.speed = speed # in meters per second

    def update(self):
        # these are available from the update function of any enabled entity
        net_trigger  = held_keys['gamepad right trigger'] - held_keys['gamepad left trigger']
        self.analog_dir = [held_keys['gamepad left stick x'], held_keys['gamepad left stick y'], net_trigger]

        # update the indicated direct move (cyan ball)
        now = time.time()
        elapsed = now - self.last_update_t
        p = self.position
        p += Vec3(
            self.last_commanded_vel[0] * elapsed,
            self.last_commanded_vel[2] * elapsed,
            self.last_commanded_vel[1] * elapsed,
        )
        self.position = p
        # self.enabled = (sum(self.app.direction) > 0)

        self.last_update_t = now

        