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

def draw_line(point_a, point_b):
    return Mesh(vertices=[point_a, point_b], mode='line')

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
                self.redraw_wires()

    def redraw_wires(self):
        # update the lines between the gantry and the other things
        for anchor_num in range(self.ui.n_anchors):
            self.ui.lines[anchor_num].model = draw_line(self.ui.anchors[anchor_num].position, self.position)
        self.ui.vert_line.model = draw_line(self.ui.gripper.position, self.position)


MAX_JOG_SPEED = 0.3

class Gripper(SplineMovingEntity):
    def __init__(self, ui, spline_func, to_ob_q, **kwargs):
        super().__init__(
            ui=ui,
            spline_func=spline_func,
            collider='box', # for mouse interaction
            model='gripper_body',
            scale=0.001,
            color=(0.9, 0.9, 0.9, 1.0),
            **kwargs
        )
        self.closed = False
        self.to_ob_q = to_ob_q
        self.label_offset = (0.00, 0.04)
        self.label = Text(
            color=(0.1,0.1,0.1,1.0),
            text=f"Gripper\nNot Detected",
            scale=0.6,
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


    def setStatus(self, status):
        self.label.text = f"Gripper\n{status}"

    def update(self):
        super().update()
        self.label.position = world_position_to_screen_position(self.position) + self.label_offset

    def on_mouse_enter(self):
        self.color = anchor_color_selected

    def on_mouse_exit(self):
        self.color = anchor_color

    def on_click(self):
        self.wp = WindowPanel(
        title=f"Gripper Controls",
        content=(
            Button(text='Open (Space)', color=color.gold, text_color=color.black),
            Button(text='Show video feed', color=color.gold, text_color=color.black),
            Button(text='Stop Spool Motor', color=color.gold, text_color=color.black),
            Button(text='Manual Spool Control', color=color.blue, text_color=color.white,
                       on_click=self.open_manual_spool_control),
            ),
        popup=True,
        )

    def open_manual_spool_control(self):
        self.wp.enabled = False
        self.jog_speed = 0.01
        self.manual_spool_wp = WindowPanel(
            title="Manual Spool Control",
            content=(
                Text(text="Use buttons or Up/Down arrow keys to control spool."),
                Button(text='Reel in 5cm', color=color.orange, text_color=color.black,
                    on_click=partial(self.reel_manual, -0.05)),
                Button(text='Reel out 5cm', color=color.orange, text_color=color.black,
                    on_click=partial(self.reel_manual, 0.05)),
            ),
            popup=True,
        )

    def input(self, key):
        if hasattr(self, "manual_spool_wp") and self.manual_spool_wp.enabled:
            if key == 'up arrow down':
                self.reel_manual(-self.jog_speed)
            elif key == 'up arrow hold':
                if self.jog_speed < MAX_JOG_SPEED:
                    self.jog_speed += 0.01
                self.reel_manual(-self.jog_speed)
            elif key == 'up arrow up':
                self.jog_speed = 0.1
                self.to_ob_q.put({'slow_stop_one':{'id':'gripper'}})


            elif key == 'down arrow down':
                self.reel_manual(self.jog_speed)
            elif key == 'down arrow hold':
                if self.jog_speed < MAX_JOG_SPEED:
                    self.jog_speed += 0.01
                self.reel_manual(self.jog_speed)
            elif key == 'down arrow up':
                self.jog_speed = 0.1
                self.to_ob_q.put({'slow_stop_one':{'id':'gripper'}})

    def reel_manual(self, delta_meters):
        self.to_ob_q.put({'jog_spool':{'gripper':None, 'rel':delta_meters}})


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
        title=f"Anchor {self.num}",
        content=(
            Button(text='Show video feed', color=color.gold, text_color=color.black),
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
        self.jog_speed = 0.01
        self.manual_spool_wp = WindowPanel(
            title="Manual Spool Control",
            content=(
                Text(text="Use buttons or Up/Down arrow keys to control spool."),
                Button(text='Reel in 5cm', color=color.orange, text_color=color.black,
                    on_click=partial(self.reel_manual, -0.05)),
                Button(text='Reel out 5cm', color=color.orange, text_color=color.black,
                    on_click=partial(self.reel_manual, 0.05)),
            ),
            popup=True,
        )

    def input(self, key):
        if hasattr(self, "manual_spool_wp") and self.manual_spool_wp.enabled:
            if key == 'up arrow down':
                self.reel_manual(-self.jog_speed)
            elif key == 'up arrow hold':
                if self.jog_speed < MAX_JOG_SPEED:
                    self.jog_speed += 0.01
                self.reel_manual(-self.jog_speed)
            elif key == 'up arrow up':
                self.jog_speed = 0.1
                self.to_ob_q.put({'slow_stop_one':{'id':self.num}})


            elif key == 'down arrow down':
                self.reel_manual(self.jog_speed)
            elif key == 'down arrow hold':
                if self.jog_speed < MAX_JOG_SPEED:
                    self.jog_speed += 0.01
                self.reel_manual(self.jog_speed)
            elif key == 'down arrow up':
                self.jog_speed = 0.1
                self.to_ob_q.put({'slow_stop_one':{'id':self.num}})

    def reel_manual(self, delta_meters):
        self.wp.enabled = False
        self.to_ob_q.put({'jog_spool':{'anchor':self.num, 'rel':delta_meters}})

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

class GoalPoint(Entity):
    def __init__(self, **kwargs):
        super().__init__(
            position=(0,0,0),
            rotation=(-90,0,0),
            model='map_marker',
            color=color.azure,
            scale=0.075,
            shader=lit_with_shadows_shader,
            enabled=True,
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
        self.speed = 0.1

        # expected seconds of latency between when we calculate a movement and when the anchors start to act on it.
        self.latency = 0.05

        self.last_move_start_pos = None # numpy array in z-up coordinate space
        self.last_move_start_time = None # float seconds since epoch
        self.last_move_duration = None # float seconds
        self.last_move_vec = None # numpy array in z-up coordinate space

    def estimatePosition(self, t):
        """Estimate the gantry's position at the given time and the last length plan sent"""
        elapsed_time = t - self.last_move_start_time
        if elapsed_time <= 0:
            return self.last_move_start_pos
        elif elapsed_time >= self.last_move_duration:
            # If the time is after the move ended, the position is the end position
            return self.last_move_start_pos + self.last_move_vec
        else:
            # Calculate the fraction of the move that has been completed
            fraction_complete = elapsed_time / self.last_move_duration
            # Estimate the current position by interpolating along the move vector
            return self.last_move_start_pos + self.last_move_vec * fraction_complete

    def reset(self):
        self.last_move_vec = None

    def direct_move(self, speed=0.1):
        """
        Send planned line lengths to the robot that would move the gantry in a straight line
        from where it is, to the indicated goal point, at the given speed.
        positions are given in z-up coordinate system.
        """
        now = time.time()
        expected_rcv_time = now + self.latency
        goal = np.array(swap_yz(self.position))

        if self.last_move_vec is None:
            self.last_move_start_pos = np.array(swap_yz(self.app.line_pos_sphere.position))
        else:
            self.last_move_start_pos = self.estimatePosition(expected_rcv_time)
        self.last_move_start_time = expected_rcv_time
        self.last_move_vec = goal - self.last_move_start_pos
        self.last_move_duration = np.linalg.norm(self.last_move_vec) / speed # seconds

        print(f'Direct move vector {self.last_move_vec}')

        # calculate regular intervals
        intervals = np.linspace(0, 1, 6, dtype=np.float64).reshape(-1, 1)
        # where we want the gantry to be at the time intervals
        gantry_positions = self.last_move_vec * intervals + self.last_move_start_pos
        print(f'first gant pos = {gantry_positions[0]}')
        # represent as absolute times
        times = intervals * self.last_move_duration + self.last_move_start_time
        # find the anchor line lengths if the gantry were at those positions
        # format as an array of times and lengths, one array for each anchor
        future_anchor_lines = np.array([
            np.column_stack([
                times,
                np.linalg.norm(gantry_positions - np.array(swap_yz(a.position)), axis=1)])
            for a in self.app.anchors])
        # send it
        self.app.to_ob_q.put({
            'future_anchor_lines': {
                'sender':'ui',
                'data':future_anchor_lines,
                'creation_time': now,
            }
        })

    def update(self):
        # update the indicated goal position for gantry
        p = self.position
        p += Vec3(
            self.app.direction[0] * self.speed,
            self.app.direction[2] * self.speed,
            self.app.direction[1] * self.speed,
        )
        self.position = p

        