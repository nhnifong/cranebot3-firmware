from ursina import *
from ursina.shaders import (
    lit_with_shadows_shader,
    unlit_shader,
)
from ursina.prefabs.health_bar import HealthBar
import numpy as np
from cv_common import *
import model_constants
from scipy.spatial.transform import Rotation
from functools import partial
import time
from PIL import Image
from ssh_launch import launch_ssh_terminal
from generated.nf import telemetry, control, common
from util import *
import av

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

# color scheme of targets based on status
STATUS_COLORS = {
    telemetry.TargetStatus.SEEN: color.white,
    telemetry.TargetStatus.SELECTED: color.azure,
    telemetry.TargetStatus.PICKED_UP: color.gold,
}

class Gantry(Entity):
    def __init__(self, ui, **kwargs):
        super().__init__(**kwargs)
        self.ui = ui
        self.last_update_t = time.time()

        # position and velocity in the z up coordinate space, not the ursina space
        self.zup_pos = np.zeros(3)
        self.zup_vel = np.zeros(3)

        self.slack = [False, False, False, False]

    def set_position_velocity(self, pos, vel):
        self.zup_pos = pos
        self.zup_vel = vel
        self.position = swap_yz(pos)

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
    def __init__(self, ui, **kwargs):
        super().__init__(
            collider='box', # for mouse interaction
            model='gripper_body',
            scale=0.001,
            color=gripper_color,
            **kwargs
        )
        self.ui = ui,
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
        self.open = True

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
            Button(text='Open SSH Terminal', color=color.gold, text_color=color.black,
                on_click=partial(launch_ssh_terminal, self.ip_address)),
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
                self.ui.send_ob(jog_spool.JogSpool(is_gripper=True, speed=0))


            elif mkey == 'down arrow':
                self.reel_manual(self.jog_speed)
            elif mkey == 'down arrow hold':
                if self.jog_speed < MAX_JOG_SPEED:
                    self.jog_speed += 0.005
                self.reel_manual(self.jog_speed)
            elif mkey == 'down arrow up':
                self.jog_speed = 0.1
                self.ui.send_ob(jog_spool.JogSpool(is_gripper=True, speed=0))

    def reel_manual(self, metersPerSecond):
        self.ui.send_ob(jog_spool.JogSpool(is_gripper=True, speed=metersPerSecond))

    def setFingerAngle(self, commanded_angle):
        """
        Sets the appearance of the finger angle.
        commanded_angle is int the range (-90, 90) This is the value that was commanded of the inventor hat mini
        this function translates it into the pilot hardware gripper's physical angle 
        """
        phys_angle = mapval(commanded_angle, -90, 90, 60, 0)
        self.left_finger.rotation = (0,0,phys_angle)
        self.right_finger.rotation = (0,0,-phys_angle)

    def toggle_closed(self):
        """ Send a command to open or close the gripper """
        self.open = not self.open
        self.ui.send_ob(move=control.CombinedMove(finger = -30 if self.open else 85))


anchor_color = (0.8, 0.8, 0.8, 1.0)
anchor_color_selected = (0.9, 0.9, 1.0, 1.0)
class Anchor(Entity):
    def __init__(self, num, ui, pose):
        super().__init__(
            position=swap_yz(pose[1]),
            rotation=to_ursina_rotation(pose[0]),
            model='anchor',
            color=anchor_color,
            scale=1,
            shader=lit_with_shadows_shader,
            collider='box'
        )
        self.num = num
        self.label_offset = (0.00, 0.04)
        self.ui = ui
        self.pose = pose # we expect caller to just set this for us.
        self.anchor_cam_pose = compose_poses([pose, model_constants.anchor_camera])
        self.ip_address = None

        self.label = Text(
            color=(0.1,0.1,0.1,1.0),
            text=f"Anchor {self.num}\nNot Detected",
            scale=0.5,
        )

        # an entity with the rotation of the anchor camera.
        # entities can use this as a parent to position themselves in the camera's coordinate space.
        self.empty = Entity(
            scale=1,
            rotation=to_ursina_rotation(compose_poses([
                model_constants.anchor_camera,
                (np.array([pi/2,0,0], dtype=float), np.array([0,0,0], dtype=float)) # extra 90 degree look up
                ])[0]),
            parent=self)

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
            Button(text='Open SSH Terminal', color=color.gold, text_color=color.black,
                on_click=partial(launch_ssh_terminal, self.ip_address)),
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
                self.ui.send_ob(jog_spool.JogSpool(is_gripper=False, anchor_num=self.num, speed=0))


            elif key == 'down arrow':
                self.reel_manual(self.jog_speed)
            elif key == 'down arrow hold':
                if self.jog_speed < MAX_JOG_SPEED:
                    self.jog_speed += 0.005
                self.reel_manual(self.jog_speed)
            elif key == 'down arrow up':
                self.jog_speed = 0.1
                self.ui.send_ob(jog_spool.JogSpool(is_gripper=False, anchor_num=self.num, speed=0))


    def reel_manual(self, metersPerSecond):
        self.wp.enabled = False
        self.ui.send_ob(jog_spool.JogSpool(is_gripper=False, anchor_num=self.num, speed=metersPerSecond))

    def toggle_vid_feed(self):
        pass # todo make this toggle the window in the main app

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

        # Target lock state and confirm buttons
        self.locked = False
        self.confirm_btn = None
        self.cancel_btn = None
        self.button_offset = (0.02, -0.03)

        # state for gamepad processing
        self.last_update_t = time.time()
        self.last_send_t = time.time()
        self.finger_angle = 0
        self.smooth_winch_speed = 0
        self.last_action = np.zeros(6)
        self.start_was_held = False
        self.dpad_up_was_held = False
        self.dpad_left_was_held = False
        self.dpad_down_was_held = False
        self.dpad_right_was_held = False
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
        # gp for gamepad
        self.gp_pos = pos
        self.you.position = [self.gp_pos[0], 1, self.gp_pos[1]]

    # def on_click(self):
    #     if not self.locked:
    #         self.lock_target()

    def lock_target(self):
        self.locked = True
        # Create UI buttons
        self.confirm_btn = Button(
            parent=camera.ui,
            color=color.azure, text_color=color.white, text="Confirm",
            scale=(0.12, 0.05), text_size=0.75, 
            on_click=self.button_confirm
        )
        self.cancel_btn = Button(
            parent=camera.ui,
            color=color.red, text_color=color.white, text="Cancel",
            scale=(0.12, 0.05), text_size=0.75,
            on_click=self.button_cancel
        )
        # Force position update immediately so they don't jump 
        self.update_buttons()

    def button_cancel(self):
        self.unlock_target()

    def unlock_target(self):
        self.locked = False
        if self.confirm_btn:
            destroy(self.confirm_btn)
            self.confirm_btn = None
        if self.cancel_btn:
            destroy(self.cancel_btn)
            self.cancel_btn = None

    def button_confirm(self):
        # TODO add target to queue at swap_yz(self.target.world_position)
        self.unlock_target()

    def update_buttons(self):
        if self.confirm_btn:
            # We use world_position_to_screen_position to place UI elements over 3D objects
            screen_pos = self.target.screen_position
            # Screen pos is (0,0) center, range -0.5 to 0.5
            # We add a small offset to put buttons near the target
            self.confirm_btn.position = screen_pos + self.button_offset
            self.cancel_btn.position = screen_pos + self.button_offset + Vec2(0, -0.06)

    def update(self):
        # Target Selection Logic
        if not self.locked:
            if mouse.hovered_entity == self:
                self.target.position = mouse.world_point
        else:
            self.update_buttons()

        events = [] # control events

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
            vec_to_object = gantry_pos_xy - self.gp_pos[:2]
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
            events.append(control.ControlItem(episode_control=control.EpControl(events=['episode_start_stop'])))
        self.start_was_held = start_held

        # D pad up - tension all lines and recalibrate lenths. detect rising edge
        dpad_up_held = bool(held_keys['gamepad dpad up'])
        if dpad_up_held and dpad_up_held != self.dpad_up_was_held:
            print('tension lines command from gamepad')
            events.append(control.ControlItem(command=control.CommonCommand(name=control.Command.TIGHTEN_LINES)))
        self.dpad_up_was_held = dpad_up_held

        # D pad left - Run quick calibration. detect rising edge
        dpad_left_held = bool(held_keys['gamepad dpad left'])
        if dpad_left_held and dpad_left_held != self.dpad_left_was_held:
            events.append(control.ControlItem(command=control.CommonCommand(name=control.Command.HALF_CAL)))
        self.dpad_left_was_held = dpad_left_held

        # D pad right - grasp
        dpad_right_held = bool(held_keys['gamepad dpad right'])
        if dpad_right_held and dpad_right_held != self.dpad_right_was_held:
            events.append(control.ControlItem(command=control.CommonCommand(name=control.Command.GRASP)))
        self.dpad_right_was_held = dpad_right_held

        # D pad down - stop
        dpad_down_held = bool(held_keys['gamepad dpad down'])
        if dpad_down_held and dpad_down_held != self.dpad_down_was_held:
            events.append(control.ControlItem(command=control.CommonCommand(name=control.Command.STOP_ALL)))
        self.dpad_down_was_held = dpad_down_held

        act = np.array([*vector, speed, self.smooth_winch_speed, self.finger_angle])
        if not np.array_equal(act, self.last_action) or (now > (self.last_send_t + 0.2) and np.linalg.norm(vector) > 1e-3):
            events.append(control.ControlItem(move=control.CombinedMove(
                direction = fromnp(vector),
                speed = speed,
                finger = self.finger_angle,
                winch = self.smooth_winch_speed,
            )))
            self.last_action = act
            self.last_send_t = now

        if len(events)>0:
            self.app.send_ob(events=events)

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

class CamPreview(Entity):
    def __init__(self, cam_scale, name, anchor, floor, app):
        super().__init__()
        self.anchor = anchor
        self.floor = floor
        self.app = app
        self.heatmap = None
        self.cam_scale = cam_scale

        # 16:9 aspect ratio for the camera content
        if self.anchor is None:
            self.aspect_ratio = 1.0
            self.total_width = cam_scale / 1.777777
            self.total_height = cam_scale / 1.777777
        else:
            self.aspect_ratio = 1.777777
            self.total_width = cam_scale
            self.total_height = cam_scale / self.aspect_ratio

        # These ratios come from the size of the hole in the rounded frame texture
        self.content_width = self.total_width * (618 / 640)
        self.content_height = self.total_height * (338 / 360)
        
        # Setup the Camera Quad (The Content)
        # This sits in the background (z=0)
        self.cam = Entity(
            parent=self,
            model='quad',
            scale=(self.content_width, self.content_height),
            texture='waiting_for_video.png', # Placeholder
            shader=unlit_shader,
            color=color.white,
            z=0.01, # Slightly behind the frame
            collider='box',
            on_click=self.vid_clicked,
            on_mouse_exit=self.on_mouse_exit,
        )
        
        # overlay
        self.frame = Entity(
            parent=self,
            model='quad',
            texture="rounded_frame.png",
            scale=(self.total_width, self.total_height),
            color=color.white, # Color is baked into the texture
            z=0,
        )

        self.label = Text(
            parent=self,
            text=name,
            origin=(-0.5, -0.5),              # Anchor: Bottom Left of text
            x=-self.total_width / 2,
            y=(self.total_height / 2) + 0.003,  # Position: Top Edge of frame + Padding
            scale=0.8,                          # Text scale (relative to parent)
            color=(0.1,0.1,0.1,1.0)
        )

        self.total_height + 0.02

        self.little_point = Entity(
            parent=self,
            model='circle',
            color=color.green,
            scale=(0.02, 0.01),
            enabled=False,
        )

        # Diagnostic Arrow
        self.prediction_arrow = Entity(
            parent=self,
            model='arrow',
            color=color.cyan,
            enabled=False,
            position=(0,0,-0.2),
            # Initialize scale to thin proportions, Z will be overridden by magnitude
            scale=(0.015, 0.015, 1)
        )

        self.target_in_view = Entity(
            parent=self.label,
            model='quad',
            color=color.black,
            position=(0.25, 0.01, 0),
            scale=(0.05, 0.02),
            enabled=(self.anchor is None),
        )

        self.holding_something = Entity(
            parent=self.label,
            model='quad',
            color=color.gold,
            position=(0.32, 0.01, 0),
            scale=(0.05, 0.02),
            enabled=(self.anchor is None),
        )

        if self.anchor is not None:
            # visualization of identified targets as stroked squares
            self.target_squares = [Entity(
                parent=self,
                model=Mesh( # you must make a different mesh for each one
                    vertices=[(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0), (-0.5, -0.5, 0)],
                    mode='line',
                    thickness=1
                ),
                color=color.yellow,
                scale=(0.01, 0.01),
                enabled=False,
                z=-1,
            ) for i in range(20)]


        # indicates whether the texture has been allocated
        self.haveSetImage = False

    def set_predictions(self, data: telemetry.GripCamPredictions):
        """
        Takes a 2D vector in [-1, 1] range (e.g. from a joystick) and draws 
        a thin arrow from the center of the view.
        If vec is None or magnitude is ~0, hides the arrow.
        """
        vec = [data.move_x, data.move_y]
        valid_target_in_view = data.prob_target_in_view
        gripper_holding = data.prob_holding # purely visual estimate of whether the gripper is holding somehting.

        self.target_in_view.alpha = valid_target_in_view
        self.holding_something.alpha = gripper_holding

        if vec is None:
            self.prediction_arrow.enabled = False
            return

        # Map normalized [-1,1] input to local physical dimensions of the content view
        # Assuming vec[0] is X (Right+) and vec[1] is Y (Up+)
        target_x = mapval(clamp(vec[0], -1, 1), -1, 1, -self.content_width/2, self.content_width/2)
        target_y = mapval(clamp(vec[1], -1, 1), 1, -1, -self.content_height/2, self.content_height/2)

        # Vector length in local physical units
        physical_magnitude = Vec2(target_x, target_y).length() * 0.5

        if physical_magnitude < 1e-3:
            self.prediction_arrow.enabled = False
            return

        self.prediction_arrow.enabled = True

        # Set length based on magnitude. Keep thickness thin and constant.
        # The default Ursina arrow model is 1 unit long along its Z-axis.
        self.prediction_arrow.scale_z = physical_magnitude
        self.prediction_arrow.scale_x = 0.015 # Constant thin width
        self.prediction_arrow.scale_y = 0.015 # Constant thin height

        # Orientation using look_at:
        # Define the target point relative to the arrow's own position within the parent space.
        # We keep the Z coordinate the same as the arrow's Z to keep it flat in that plane.
        target_point_in_parent_space = Vec3(target_x, target_y, self.prediction_arrow.z)
        # Point the arrow's Z-axis (its length) towards the target point.
        self.prediction_arrow.look_at(target_point_in_parent_space)

    def setHeatmap(self, heatmap):
        """Store a heatmap image in BGR pixel format."""
        self.heatmap = heatmap

    def setImage(self, image):
        """
        Expects an image as an np.ndarray in a BGR pixel format, right side up.
        """
        if not self.haveSetImage:
            # we only need to do this the first time, so the allocated texture is the right size
            # even though this method of updating a texture exists, it's horribly slow.
            self.cam.texture = Texture(Image.fromarray(image))
            self.haveSetImage = True

        if self.app.show_heatmap and self.heatmap is not None:
            overlay = cv2.addWeighted(image, 0.8, self.heatmap, 0.4, 0)
        else:
            overlay = image

        # setRamImage seems to require the image to be flipped vertically and have a BGRA pixel format.
        overlay = cv2.cvtColor(cv2.flip(overlay, 0), cv2.COLOR_BGR2BGRA)
        self.cam.texture._texture.setRamImage(overlay)

    def update(self):
        if self.anchor and not self.floor.locked:
            if self.cam.hovered:
                # Convert Mouse World Point to Local Space of the Cam Quad
                local_point = mouse.world_point - self.cam.world_position

                dx = mouse.world_point.x - self.cam.world_position.x
                dy = mouse.world_point.y - self.cam.world_position.y
                u = clamp((dx / self.cam.world_scale_x) + 0.5, 0, 1)
                v = clamp((-dy / self.cam.world_scale_y) + 0.5, 0, 1)
                
                self.render_projected_point((u, v))

                # we want to use this coordinate to indicate in the UI what this would look like in other views
                # move the floor's reticule to the point where that ray intersects the floor
                floor_pos = project_pixels_to_floor(np.array([[u, v]]), self.anchor.anchor_cam_pose)
                if len(floor_pos) == 1:
                    pos2d = floor_pos[0]
                    self.floor.target.position = [pos2d[0], 0, pos2d[1]]

                    # draw a line directly from this anchor's camera to the floor reticule position.
                    self.app.camlines[self.anchor.num].enabled = True
                    self.app.camlines[self.anchor.num].model = draw_line(
                        self.anchor.empty.world_position, self.floor.target.world_position)

                    # update the view of this point in other cameras
                    for key, other_cam in self.app.cam_views.items():
                        if key is None or other_cam is self:
                            continue
                        # in any other cameras which are enabled, project that floor position back into their UV coords
                        uv_coords = project_floor_to_pixels(floor_pos, other_cam.anchor.anchor_cam_pose)
                        if len(uv_coords) == 1:
                            other_cam.render_projected_point(uv_coords[0])

    def on_mouse_exit(self):
        if not self.floor.locked:
            if self.anchor is not None:
                # Cleanup lines
                if self.app.camlines.get(self.anchor.num):
                    self.app.camlines[self.anchor.num].enabled = False
            
                # Cleanup ALL little points
                for other_cam in self.app.cam_views.values():
                    other_cam.little_point.enabled = False

    def render_projected_point(self, uv_point):
        self.little_point.enabled = True
        self.little_point.x = mapval(clamp(uv_point[0], 0, 1), 0, 1, -self.content_width/2, self.content_width/2)
        self.little_point.y = mapval(clamp(uv_point[1], 0, 1), 1, 0, -self.content_height/2, self.content_height/2)

    def set_target_points(self, item: telemetry.TargetList):
        """
        Given a list of targets, project them to this camera's view,
        and any that would be visible draw them as squares of the appropriate color
        """
        if self.anchor is None:
            return
        ts_index = 0
        for target in item.targets:
            floor_pos = np.array([[target.position.x, target.position.y]])
            # in any other cameras which are enabled, project that floor position back into their UV coords
            uv_coords = project_floor_to_pixels(floor_pos, self.anchor.anchor_cam_pose)
            if len(uv_coords) == 1:
                uv = uv_coords[0]
                ts = self.target_squares[ts_index]
                ts.color = STATUS_COLORS.get(target.status, color.white)
                ts.x=mapval(clamp(uv[0], 0, 1), 0, 1, -self.content_width/2, self.content_width/2)
                ts.y=mapval(clamp(uv[1], 0, 1), 1, 0, -self.content_height/2, self.content_height/2)
                ts.enabled = True
                ts_index += 1
        # disable the rest
        while ts_index < len(self.target_squares):
            self.target_squares[ts_index].enabled = False
            ts_index += 1

    def vid_clicked(self):
        # Only allow locking if this is an anchor (not generic/gripper) AND not already locked
        if self.anchor and not self.floor.locked:
            self.floor.lock_target()

    def connect_to_stream(self, uri):
        """
        Connect to a stream that provides the base video frames
        Must be run in it's own thread.
        """

        options = {
            'rtsp_transport': 'udp',
            'fflags': 'nobuffer',
            'flags': 'low_delay',
            'fast': '1',
        }
        print(f'connecting to {uri}')
        try:
            container = av.open(uri, options=options, mode='r')
            stream = next(s for s in container.streams if s.type == 'video')
            stream.thread_type = "SLICE"
            for packet in container.demux(stream):
                for frame in packet.decode():
                    arr = frame.to_ndarray(format='bgr24')
                    invoke(self.setImage, arr, delay=0.0001)
        except Exception as e:
            print(f'unable to connect to {uri} {e}')

class ActionList(Entity):
    def __init__(self):
        super().__init__()
        self.task_name = None
        self.target_list = []

        self.background = Entity(
            parent=camera.ui,
            model=Quad(radius=0.02, aspect=0.5),
            texture='panel_grad',
            origin=(-0.58, 0.7),      # Anchor to top-left with padding
            color=(0.1,0.1,0.1, 0.3), # translucent
            scale=(0.3, 0.73),
            position=window.top_left,
            z=3
        )

        self.cursor_pos = window.top_left + (0.052, -0.2)

        self.task_label = Text(
            parent=camera.ui,
            text="Task: None",
            origin=(-0.5, -0.5),              # Anchor: Bottom Left of text
            position=self.cursor_pos,
            scale=0.7,                          # Text scale (relative to parent)
            color=color.white,
            # z=2
        )
        self.cursor_pos[1] -= self.task_label.height

        self.targets_label = Text(
            parent=camera.ui,
            text="Targets: User 0 AI 0",
            origin=(-0.5, -0.5),              # Anchor: Bottom Left of text
            position=self.cursor_pos,
            scale=0.7,                          # Text scale (relative to parent)
            color=color.white,
            # z=2
        )
        self.cursor_pos[1] -= self.targets_label.height
        self.cursor_pos[1] -= 0.03 # blank line
        # self.cursor_pos[0] += 0.05 # indent
        self.list_start_cursor_pos = Vec2(self.cursor_pos) # deep copy
        self.target_labels = []

    def set_task_name(self, name):
        self.task_name = names
        self.task_label.text = f"Task: {name}"

    def set_target_list(self, targetlist: telemetry.TargetList):
        """
        see get_queue_snapshot() in target_queue.py for source data format
        """
        for tl in self.target_labels:
            destroy(tl)
        self.target_labels = []
        self.target_list = targetlist.targets

        count_user = 0
        count_ai = 0
        self.cursor_pos = Vec2(self.list_start_cursor_pos)
        for i, target in enumerate(self.target_list):
            if target.source == 'user':
                count_user += 1
            else:
                count_ai += 1
            x = target.position.x
            y = target.position.y
            tl = Text(
                parent=camera.ui,
                text=f"({target.source}) {target.id[:6]} at ({x:0.2f}, {y:0.2f})",
                origin=(-0.5, -0.5),              # Anchor: Bottom Left of text
                position=self.cursor_pos,
                scale=0.6,                          # Text scale (relative to parent)
                color=STATUS_COLORS.get(target.status, color.white),
            )
            self.cursor_pos[1] -= tl.height
            self.target_labels.append(tl)

        self.targets_label.text = f'Targets: user {count_user} ai {count_ai}'
