from ursina import *
from ursina.shaders import (
    lit_with_shadows_shader,
    unlit_shader,
)
from ursina.prefabs.health_bar import HealthBar
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

def mapval(x, in_min, in_max, out_min, out_max):
    return (float(x) - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

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
        # laser range indicator
        self.lrange = Entity(
            parent=self,
            model='cube',
            scale=(100,25,100),
            color=color.red,
            position=(0,-250,0),
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

    def setLaserRange(self, distance_m):
        self.lrange.position = (0,-distance_m*1000)

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

    def setFingerAngle(self, commanded_angle):
        """
        Sets the appearance of the finger angle.
        commanded_angle is int the range (-90, 90) This is the value that was commanded of the inventor hat mini
        this function translates it into the pilot hardware gripper's physical angle 
        """
        phys_angle = mapval(commanded_angle, -90, 90, 60, 0)
        self.left_finger.rotation = (0,0,phys_angle)
        self.right_finger.rotation = (0,0,-phys_angle)

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
            Button(text='Manual Spool Control', color=color.blue, text_color=color.white,
                on_click=self.open_manual_spool_control),
            # Add: tighten, stream logs 
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

    def toggle_vid_feed(self):
        self.camview.enabled = not self.camview.enabled
        self.to_ob_q.put({'toggle_previews':{'anchor':self.num, 'status':self.camview.enabled}})

GAMEPAD_GRIP_DEG_PER_SEC = 90
GAMEPAD_WINCH_METER_PER_SEC = 0.2

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
            enabled=False)
        self.hasSetImage = False

        # state for gamepad processing
        self.last_update_t = time.time()
        self.last_send_t = time.time()
        self.finger_angle = 0
        self.smooth_winch_speed = 0
        self.last_action = np.zeros(6)
        self.start_was_held = False
        self.dpad_up_was_held = False
        self.dpad_left_was_held = False
        self.seat_orbit_mode = True
        # assumed position of the person holding the gamepad. updated regularly with messages to ui_q using apriltag
        self.gp_pos = np.array([-1.3, 1.9])

        self.you = Entity(
            model='sphere',
            color=color.red,
            scale=0.05,
            shader=lit_with_shadows_shader,
            position=[self.gp_pos[0], 1, self.gp_pos[1]],
        )

    def set_gp_pos(self, pos):
        self.gp_pos = pos
        self.you.position = [self.gp_pos[0], 1, self.gp_pos[1]]

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

        #  =========== collect input from attatched gamepad ===========
        # these are available from the update function of any enabled entity, the floor just happens to be a singular always enabled entity.
        # inputs are sent to observer if anything changed

        # net trigger, vertical motion (right up, left down)
        net_trigger  = held_keys['gamepad right trigger'] - held_keys['gamepad left trigger']

        # left stick is lateral motion
        vector = np.array([held_keys['gamepad left stick x'], held_keys['gamepad left stick y'], net_trigger])

        if self.seat_orbit_mode:
            # left stick x orbits the seat clockwise
            # left stick y moves away from the seat
            gantry_pos_xy = np.array(self.app.gantry.zup_pos[:2])
            vec_to_object = gantry_pos_xy - self.gp_pos
            distance = np.linalg.norm(vec_to_object)
            # We can only orbit if we aren't at the exact center.
            if distance > 1e-6:
                radial_dir = vec_to_object / distance
                tangential_dir = np.array([radial_dir[1], -radial_dir[0]])
                orbit_vec_2d = (vector[0] * tangential_dir) + (vector[1] * radial_dir)
                vector = np.array([orbit_vec_2d[0], orbit_vec_2d[1], net_trigger])

        mag = np.linalg.norm(vector)
        speed = 0
        if mag > 0:
            vector = vector / mag
            speed = 0.25 * mag

        # hold a, close grip further. hold b, open grip further
        grip_change = 0
        if held_keys['gamepad a']:
            grip_change = GAMEPAD_GRIP_DEG_PER_SEC
        elif held_keys['gamepad b']:
            grip_change = -GAMEPAD_GRIP_DEG_PER_SEC
        now = time.time()
        self.finger_angle += np.clip(grip_change * (now - self.last_update_t), -90, 90)
        self.last_update_t = now

        # hold y, winch up. hold x, winch down
        line_speed = 0
        if held_keys['gamepad y']:
            line_speed = -GAMEPAD_WINCH_METER_PER_SEC
        elif held_keys['gamepad x']:
            line_speed = GAMEPAD_WINCH_METER_PER_SEC
        self.smooth_winch_speed = self.smooth_winch_speed*0.95 + line_speed*0.05
        if abs(self.smooth_winch_speed) < 0.0005:
            self.smooth_winch_speed = 0

        # control other actions

        # Episode control: can send any of
        # episode_start_stop, rerecord_episode, stop_recording

        # Start - start/stop recording episode. detect rising edge
        start_held = bool(held_keys['gamepad start'])
        if start_held and start_held != self.start_was_held:
            self.app.to_ob_q.put({'episode_ctrl': ['episode_start_stop']})
        self.start_was_held = start_held

        # D pad up - tension all lines and recalibrate lenths. detect rising edge
        dpad_up_held = bool(held_keys['gamepad dpad up'])
        if dpad_up_held and dpad_up_held != self.dpad_up_was_held:
            print('tension lines command from gamepad')
            self.app.to_ob_q.put({'tension_lines': None})
        self.dpad_up_was_held = dpad_up_held

        # D pad left - Run quick calibration. detect rising edge
        dpad_left_held = bool(held_keys['gamepad dpad left'])
        if dpad_left_held and dpad_left_held != self.dpad_left_was_held:
            self.app.to_ob_q.put({'half_cal': None})
        self.dpad_left_was_held = dpad_left_held

        act = np.array([*vector, speed, self.smooth_winch_speed, self.finger_angle])
        if not np.array_equal(act, self.last_action) or (now > (self.last_send_t + 0.2) and sum(vector) != 0):
            self.app.to_ob_q.put({
                'gamepad': {
                    'winch': self.smooth_winch_speed,
                    'finger': self.finger_angle,
                    'dir': vector,
                    'speed': speed,
                }
            })
            self.last_action = act
            self.last_send_t = now

        # -1.29516723  1.89840947  0.88209196

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
    def __init__(self, model='sphere', *args, **kwargs):
        super().__init__(*args, **kwargs,
            position=(0,0,0),
            scale=(0.06),
            shader=unlit_shader,
            model=model)
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

class VelocityArrow(Entity):
    """A debugging arrow entity to visualize a velocity vector."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, 
            model='arrow',
            scale=1000, # because its' a child of gantry which has a scale of 1/1000 for some reason
            **kwargs)
        self.zup_vel = np.zeros(3)

    def set_velocity(self, vel):
        """Sets the velocity vector to visualize in z-up space."""
        self.zup_vel = np.array(vel)
        magnitude = np.linalg.norm(self.zup_vel)

        # If magnitude is zero, hide the arrow
        if magnitude < 1e-6:
            self.enabled = False
            return
        self.enabled = True

        self.scale = [100, 100, 1000*magnitude]
        yup_vel = swap_yz(self.zup_vel / magnitude)
        self.look_at(yup_vel)
        

class PopMessage(WindowPanel):
    def __init__(self):
        self.popup_text_field = Text('Default message', wordwrap=34, origin=(0, 0))
        super().__init__(
            title='System Message', 
            content=[
                self.popup_text_field,
                Button('Dismiss', color=color.azure, on_click=self.close_popup) 
            ],
            enabled=False, # Start as hidden
        )

    def close_popup(self):
        self.enabled = False

    def show_message(self, message):
        destroy(self.popup_text_field) 
        self.popup_text_field = Text(message, wordwrap=34, origin=(0, 0))
        self.content[0] = self.popup_text_field
        self.layout()
        self.enabled = True

class CalFeedback(Entity):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.title = Text(
            'Auto Calibration',
            position=(-0.3, 0.50),
            scale=(0.9, 0.9),
            enabled=False,
        )
        self.status = Text(
            'Preparing',
            position=(-0.3, 0.47),
            scale=(0.6, 0.6),
            enabled=False,
        )
        self.bar = HealthBar(
            bar_color=color.lime.tint(-.25),
            roundness=.5,
            show_text=False,
            max_value=100,
            value=0,
            scale=(.5, 0.05),
            position=(-0.3, 0.45),
            enabled=False,
        )
        self.button = Button(
            'Cancel',
            color=color.gold, text_color=color.black,
            on_click=self.cancel,
            scale=(.12, 0.04),
            position=(0.3, 0.43),
            enabled=False,
        )
        self.last_update_t = time.time()
        self.dont_go_past = 100
        self.speed = 1

    def start(self):
        self.last_update_t = time.time()
        self.dont_go_past = 1
        self.speed = 1
        self.bar.value = 0
        self.title.enabled = True
        self.status.enabled = True
        self.bar.enabled = True
        self.button.enabled = True

    def cancel(self):
        self.title.enabled = False
        self.status.enabled = False
        self.bar.enabled = False
        self.button.enabled = False
        self.app.on_stop_button()

    def handle_message(self, upd):
        self.status.text = upd['message']
        self.bar.value = upd['progress']
        # make the bar move at this many percentage points per second up to a limit
        self.speed = upd['speed']
        self.dont_go_past = upd['dont_go_past']

    def update(self):
        if self.bar.enabled:
            now = time.time()
            elapsed = now - self.last_update_t
            self.bar.value = min(self.bar.value + self.speed * elapsed, self.dont_go_past)
            self.last_update_t = now