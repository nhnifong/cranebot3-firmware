import numpy as np
from scipy.interpolate import BSpline
import sys
from direct.stdpy import threading # panda3d drop in replacement that is compatible with it's event loop
import time
from functools import partial
from cv_common import invert_pose, compose_poses
from math import pi
import atexit
from panda3d.core import LQuaternionf, Point2
from cv_common import average_pose, compose_poses
import model_constants
from config import Config

from ursina import *
from ursina.shaders import (
    lit_with_shadows_shader,
    unlit_shader,
)
from ursina.prefabs.dropdown_menu import DropdownMenu, DropdownMenuButton
from ursina_entities import * # my ursina objects
from websockets.sync.client import connect as websocket_connect_sync
from generated.nf import telemetry, control, common
from util import *

window.borderless = False

# the color for the lines that connect the anchors, gantry, and gripper
line_color = color.black

# user readable mode names
mode_names = {
    'run':   'Run Pick-and-Drop',
    'pause': 'Pause/Observe',
    'train': 'Accept Teleop Connections',
}
mode_descriptions = {
    'run':   'Continuously alternate between seeking out an object to grab and dropping it int the marked bin.',
    'pause': 'WASD-QE moves the gantry, RF moves the winch line. Space toggles the grip. Run calibration in this mode.',
    'train': 'Use gamepad to record episodes to LeRobot dataset or start and stop policy-driven modement.',
}
detections_format_str = 'Detections/sec {val:.2f}'
video_latency_format_str = 'Video latency {val:.2f} s'
video_framerate_format_str = 'Avg framerate {val:.2f} fps'
estimate_age_format_str = 'Position est. latency {val:.2f} s'

conn_status_strings = {
    telemetry.ConnStatus.NOT_DETECTED: 'Not Detected',
    telemetry.ConnStatus.CONNECTING: 'Connecting...',
    telemetry.ConnStatus.CONNECTED: 'Online',
}

key_behavior = {
    # key: (axis, speed)
    'a': (0, -1),
    'd': (0, 1),
    'w': (1, 1),
    's': (1, -1),
    'q': (2, -1),
    'e': (2, 1),

    'a up': (0, 0),
    'd up': (0, 0),
    'w up': (1, 0),
    's up': (1, 0),
    'q up': (2, 0),
    'e up': (2, 0),
}

def input(key):
    pass

# what color to show a solid depending on how many cameras it was seen by
solid_colors = {
    2: (1.0, 0.951, 0.71, 1.0),
    3: (1.0, 0.684, 0.71, 0.5),
    4: (1.0, 0.417, 0.71, 1.0),
}

def update_go_quad(position, color, e):
    e.position = position
    e.color = color

class ControlPanelUI:
    def __init__(self):
        self.app = Ursina(
            fullscreen=False,
            borderless=False,
            title="Stringman Control Panel",
        )
        
        # --- Core State ---
        self.config = Config()
        self.n_anchors = len(self.config.anchors)
        self.direction = np.zeros(3, dtype=float) # direction of currently commanded keyboard movement
        self.websocket = None

        # --- Setup Methods ---
        self._setup_scene_and_lighting()
        self._create_robot_entities()
        self._create_shape_tracker_entities()
        self._create_hud_panels()
        self._create_menus()

    def update_layout(self):
        # Calculate dimensions for the UI panel
        panel_width = window.aspect_ratio * (1 - self.split)
        
        # Update horizontal position of all panel items
        # Center of panel = (Right Edge) - (Half Panel Width)
        center_x = (window.aspect_ratio / 2) - (panel_width / 2)
        
        for item in self.right_panel_items:
            item.x = center_x

    def _setup_scene_and_lighting(self):
        Sky(color=color.light_gray)
        # the color of this light is how you control the brightness
        DirectionalLight(position=(2, 20, 1), shadows=True, rotation=(35, -5, 5), color=(0.8,0.8,0.8,1))
        AmbientLight(color=(0.8,0.8,0.8,1))
        EditorCamera(position=(1.8804024, 0.66652613, -0.1954718), rotation=(30.175781, 18.576408, 0))
        # Show a large floor. this is an active entity that moves a reticule with mouse input
        self.floor = Floor(
            app=self,
            model='quad',
            position=(0, -0.05, 0),
            rotation=(90,0,0),
            color=(0.35,0.22,0.18,1.0),
            scale=(10, 10), # Scale in meters
            shader=lit_with_shadows_shader
        ) 

    def _create_robot_entities(self):
        # Create gantry, gripper, anchors, lines, etc.
        self.anchors = []
        for i, a in enumerate(self.config.anchors):
            anch = Anchor(i, self, pose=a.pose)
            anch.pose = a.pose
            self.anchors.append(anch)

        self.gantry = Gantry(
            ui=self,
            model='gantry',
            color=(0.9, 0.9, 0.9, 1.0),
            scale=0.001,
            position=(0,1,1),
            rotation=(0,0,0),
            shader=lit_with_shadows_shader,
        )

        self.gripper = Gripper(
            ui=self,
            position=(0,0.3,1),
            shader=lit_with_shadows_shader,
        )

        # lines representing the fishing line and tether that connect the anchors to the gantry
        self.lines = []
        for a in self.anchors:
            self.lines.append(Entity(model=draw_line(a.position, self.gantry.position), color=line_color, shader=unlit_shader))
        self.vert_line = Entity(model=draw_line(self.gantry.position, self.gripper.position), color=line_color, shader=unlit_shader)

        # lines representing the mouse cursor projected from a camera to the floor
        self.camlines = {}
        for key in self.config.preferred_cameras:
            self.camlines[key] = Entity(model=draw_line([0,0,0], [1,1,1]), color=color.green, shader=unlit_shader, enabled=False)

        # debug indicators of the visual and hang based position and velocity estimates
        self.debug_indicator_visual = IndicatorSphere(color=color.red)
        self.debug_indicator_hang = IndicatorSphere(color=color.blue)

        # indicator of the last commanded gantry velocity, whether it was commanded manually or by some automatic process.
        self.commanded_velocity_indicator = VelocityArrow(color=color.cyan, parent=self.gantry, enabled=False)

        # show a visualization of goal positions
        self.goal_marker = GoalPoint([0,0,0], enabled=False)

        self.walls = [Entity(
            model='quad',
            texture='vertical_gradient',
            color=(0.0, 1.0, 0.0, 0.1),
            shader=unlit_shader,
            double_sided=True
            ) for i in range(4)]
        self.redraw_walls()

        # these cubes indicate the 3d position of gantry aruco detections
        self.go_quads = EntityPool(100, lambda: Entity(
                model='cube',
                color=color.white, scale=(0.03),
                shader=unlit_shader))

    def toggle_heatmaps(self):
        self.show_heatmap = not self.show_heatmap
        self.heatmap_button.color = color.gold if self.show_heatmap else color.gray
        self.heatmap_button.highlight_color = self.heatmap_button.color.tint(.2) 

    def _create_hud_panels(self):
        self.split = 0.75

        # Flow management for right panel
        self.right_panel_items = []
        # y position of next item to be added to right panel
        self.cursor_y = 0.4

        self.show_heatmap = True
        self.heatmap_button = Button(
            text="Heatmaps",
            color=color.gray,
            text_color=color.black,
            scale=(.15, .033),
            on_click=self.toggle_heatmaps
        )
        self.add_side_panel_item(self.heatmap_button)
        self.cursor_y -= 0.1

        self.cam_views = {}
        cam_scale = 0.4
        for key in self.config.preferred_cameras: # show only two anchors and the gripper for now
            c = CamPreview(
                cam_scale=0.4,
                name=(f'Anchor {key} camera' if key is not None else 'Gripper camera'),
                anchor=(self.anchors[key] if key is not None else None),
                floor=self.floor,
                app=self,
            )
            self.add_side_panel_item(c)
            self.cam_views[key] = c
        self.update_layout()

        # start pickup button
        self.start_pickup_button = Button(
            text="Start Pickup",
            parent=camera.ui,
            color=rgb(0.027, 0.530, 0.256),
            text_color=color.black,
            origin=(-0.5, 0.5),      # Anchor to top-left
            position=window.top_left + (0.021, -0.1),
            scale=(.15, .033),
            on_click=partial(self.simple_command, control.Command.PICK_AND_DROP))

        # left panel (action list)
        self.action_list = ActionList()

        # Setup the Bottom UI Panel Background
        self.bottom_ui_panel = Entity(
            parent=camera.ui,
            model='quad',
            color=(0.1,0.1,0.1,1.0),
            # origin=(0.5, 0.5),
            # position=window.top_right,
            position=(0,-0.48),
            scale=(3,0.1),
            z=2
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

        # TODO use this and mode description to describe the current status instead of mode, which no longer exists
        # status could be "Preparing" "Ready" and "Cleanup"
        self.mode_text = Text(
            color=(0.0,1.0,0.3,1.0),
            position=(-0.1,self.modePanelLine1y),
            text=mode_names['pause'],
            scale=0.7,
            enabled=True,
        )

        self.mode_descrip_text = Text(
            color=(0.9,0.9,0.9,1.0),
            position=(-0.45,self.modePanelLine2y),
            text=mode_descriptions['pause'],
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

        self.estimate_age_text = Text(
            color=(0.9,0.9,0.9,1.0),
            position=(-0.45,self.modePanelLine3y),
            text=estimate_age_format_str.format(val=0),
            scale=0.5,
            enabled=True,
        )

        # make a little video monitor icon for each camera status.
        x = -0.869
        y = -0.483
        offx = 0.042
        offy = 0.037
        self.vid_status = [Panel(
            model='quad',
            color=color.white,
            texture='vid_out.png',
            position=position,
            scale=(0.01998,0.01498),
            z=1
        ) for position in [(x,y), (x+offx,y), (x,y+offy), (x+offx,y+offy), ((x+offx/2,y+offy/2))]]
        for i,vs_panel in enumerate(self.vid_status):
            vs_panel.numbertext = Text(
                color=color.white,
                position=vs_panel.position + (-0.005, 0.005),
                text=str(i),
                scale=0.5,
            )

        self.stop_button = Button(
            text="STOP",
            color=color.red,
            position=(0.83, -0.45), scale=(.10, .033),
            on_click=self.on_stop_button)

        # reusable window panel for containing a popup message
        self.pop_message = PopMessage()

        # panel for showing the gamepad controls
        self.gamepad_window = Sprite(
            parent=camera.ui,
            texture='gamepad_diagram.png',
            origin=(0, 0),
            ppu=800,
            # z=0,
            enabled=False, # Start disabled
            collider='box',
            on_click=self.toggle_gamepad_window,
        )

    def add_side_panel_item(self, item, padding=0.027):
        # Parent to the screen, not the panel, to avoid distortion
        item.parent = camera.ui
        item.origin = (0, 0)
        item.z = -1  # Ensure it renders in front of the panel
        
        # Set vertical position based on cursor
        item.y = self.cursor_y
        
        # Move cursor down for the next item
        height = item.scale[1]
        if hasattr(item, 'total_height'):
            height = item.total_height
        self.cursor_y -= (height + padding)
        
        # Store for horizontal alignment updates
        self.right_panel_items.append(item)

    def _create_shape_tracker_entities(self):
        # create entities that are used to visualize the internal state of the shape tracker and 3d hull contruction.
        self.prisms = EntityPool(80, lambda: Entity(
            color=(1.0, 1.0, 1.0, 0.1),
            shader=lit_with_shadows_shader))
        self.solids = EntityPool(100, lambda: Entity(
            color=(1.0, 1.0, 0.5, 1.0),
            shader=lit_with_shadows_shader))

    def _create_menus(self):
        # Setup the DropdownMenu
        DropdownMenu('Menu', z=-10, buttons=(
            DropdownMenuButton('Tension all lines (D-up)', on_click=partial(self.simple_command, control.Command.TIGHTEN_LINES)),
            DropdownMenuButton('Run Full Calibration', on_click=partial(self.simple_command, control.Command.FULL_CAL)),
            DropdownMenuButton('Run Quick Calibration (D-left)', on_click=partial(self.simple_command, control.Command.HALF_CAL)),
            DropdownMenuButton('Zero Gripper Winch Line', on_click=partial(self.simple_command, control.Command.ZERO_WINCH)),
            DropdownMenuButton('Horizontal Move Test', on_click=partial(self.simple_command, control.Command.HORIZONTAL_CHECK)),
            DropdownMenuButton('Gamepad Controls', on_click=self.toggle_gamepad_window),
            DropdownMenuButton('Collect Gripper Images', on_click=partial(self.simple_command, control.Command.COLLECT_GRIPPER_IMAGES)),
            ))

    def toggle_gamepad_window(self):
        self.gamepad_window.enabled = not self.gamepad_window.enabled

    def redraw_walls(self):
        # draw the robot work area boundaries with walls that have a gradient that reaches up from the ground and fades to transparent.
        # between every pair of anchors, draw a horizontal line. if all the other anchors' horizontal positions are on one side of that line, proceed
        # make a vertical quad that passes through that line and apply the fade texture to it.
        order = [0,1,3,2]
        height = 1.5

        for i in range(4):
            p1 = self.anchors[order[i]].position
            p2 = self.anchors[order[(i+1)%4]].position
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
            self.walls[i].position = center
            self.walls[i].scale = (width, height)
            self.walls[i].look_at(center+right);


    def input(self, key):
        if key == 'space':
            self.gripper.toggle_closed()
        elif key == 'c':
            self.simple_command('half_cal')
        elif key == 'escape' and self.gamepad_window.enabled:
            self.gamepad_window.enabled = False

        was_dir = self.direction

        if key in key_behavior:
            axis, speed = key_behavior[key]
            self.direction[axis] = speed
            
            print(f'key = "{key}" direction = {self.direction}')
            if sum(self.direction) == 0:
                # immediately cancel whatever remains of the movement
                self.send_ob(command=control.CommonCommand(name=control.Command.STOP_ALL))
            else:
                # normalize and send
                vector = self.direction
                vector = vector / np.linalg.norm(vector)
                self.send_ob(move=control.CombinedMove(
                    direction=common.Vec3(*vector),
                    speed=0.25,
                ))

                # a move has been initiated. but sometimes ursina doesn't get the up key that would end the move.
                # this can be destructive for a CDPR, so we must ensure every move stops eventually.
                self.last_positive_input_time = time.time()
                invoke(self.maybe_end_move, delay=2)

    def maybe_end_move(self):
        if time.time()-2 > self.last_positive_input_time:
            print('Ending direct move with a timeout because ursina missed a key-up')
            self.direction = np.zeros(3, dtype=float)
            self.send_ob(command=control.CommonCommand(name=control.Command.STOP_ALL))

    def show_error(self, error):
        self.error.text = error
        self.error.enabled = True

    def clear_error(self):
        self.error.enabled = False

    def on_stop_button(self):
        self.send_ob(command=control.CommonCommand(name=control.Command.STOP_ALL))

    def render_gantry_ob(self, row, color):
        self.go_quads.add(partial(update_go_quad, row, color))

    def send_ob(self, events=None, **kwargs):
        """
        Ensure that the given control item is sent to every connected UI
        keyword args are passed directly to control item, so you can construct one like this
        """
        if events is None:
            events = [control.ControlItem(**kwargs)]
        batch = control.ControlBatchUpdate(
            robot_id="0",
            updates=events
        )
        to_send = bytes(batch)
        # synchronous, and we're on a different thread, but I'm pretty sure websockets does this in thread safe way
        self.websocket.send(to_send)

    def simple_command(self, cmd: control.Command):
        self.send_ob(command=control.CommonCommand(name=cmd))

    def receive_updates(self):
        # threading websocket api used because asyncio loop incompatible with ursina
        try:
            print('Connecting to local observer process')
            with websocket_connect_sync('ws://localhost:4245') as websocket:
                self.websocket = websocket
                # iterator ends when websocket closes.
                for message in websocket:
                    batch = telemetry.TelemetryBatchUpdate().parse(message)
                    self.robot_id = batch.robot_id
                    for update in batch.updates:
                        # but processing the update needs to happen in the ursina loop, because it will modify a bunch of entities.
                        invoke(self.process_update, update, delay=0.0001)
        except ConnectionRefusedError:
            print('Connection refused at ws://localhost:4245 Local observer not running.\nRun with main.py to start both observer and UI')
            invoke(application.quit, delay=0.0001)

    def process_update(self, item: telemetry.TelemetryItem):
        if item.pos_estimate:
            self._handle_pos_estimate(item.pos_estimate)
        if item.pos_factors_debug:
            self._handle_pos_factors(item.pos_factors_debug)
        if item.gantry_sightings:
            self._handle_gantry_sightings(item.gantry_sightings)
        if item.new_anchor_poses:
            self._handle_new_anchor_poses(item.new_anchor_poses)
        if item.component_conn_status:
            self._handle_component_conn_status(item.component_conn_status)
        if item.vid_stats:
            self._handle_vid_stats(item.vid_stats)
        if item.named_position:
            self._handle_named_position(item.named_position)
        if item.last_commanded_vel:
            self._handle_last_commanded_vel(item.last_commanded_vel)
        if item.pop_message:
            self._handle_pop_message(item.pop_message)
        if item.grip_sensors:
            self._handle_grip_sensors(item.grip_sensors)
        if item.grip_cam_preditions:
            self._handle_grip_cam_preditions(item.grip_cam_preditions)
        if item.target_list:
            self._handle_target_list(item.target_list)

        # if 'preview_image' in updates:
        #     pili = updates['preview_image']
        #     # note that self.cam_views is a dict keyd by anchor num and 'None' is the key of the gripper cam view
        #     self.cam_views[pili['anchor_num']].setImage(pili['image'])

        # if 'heatmap' in updates:
        #     pili = updates['heatmap']
        #     self.cam_views[pili['anchor_num']].setHeatmap(pili['image'])

    def _handle_pos_estimate(self, item: telemetry.PositionEstimate):
        self.gantry.set_position_velocity(tonp(item.gantry_position), tonp(item.gantry_velocity))
        self.gripper.setPose((tonp(item.gripper_pose.rotation), tonp(item.gripper_pose.position)))
        estimate_age = time.time() - item.data_ts
        self.estimate_age_text.text = estimate_age_format_str.format(val=estimate_age)
        self.gantry.set_slack_vis(item.slack)

    def _handle_pos_factors(self, item: telemetry.PositionFactors):
        self.debug_indicator_visual.set_position_velocity(tonp(item.visual_pos), tonp(item.visual_vel))
        self.debug_indicator_hang.set_position_velocity(tonp(item.hanging_pos), tonp(item.hanging_vel))

    def _handle_gantry_sightings(self, item: telemetry.GantrySightings):
        for s in item.sightings:
            self.render_gantry_ob((s.x, s.z, s.y), color.white)

    def _handle_new_anchor_poses(self, item: telemetry.AnchorPoses):
        for anchor_num, pose in enumerate(item.poses):
            apose = (tonp(pose.rotation), tonp(pose.position))
            self.anchors[anchor_num].enabled = True
            self.anchors[anchor_num].pose = apose
            self.anchors[anchor_num].position = swap_yz(apose[1])
            self.anchors[anchor_num].rotation = to_ursina_rotation(apose[0])
        self.gantry.redraw_wires()
        self.redraw_walls()

    def _handle_component_conn_status(self, item: telemetry.ComponentConnStatus):
        user_status_str = conn_status_strings.get(item.websocket_status, 'Unknown')

        if item.video_status == telemetry.ConnStatus.CONNECTED:
            vidstatus_tex = 'vid_ok.png'
            number_color = color.black
        elif item.video_status == telemetry.ConnStatus.CONNECTING:
            user_status_str += '\nConnecting to video...'
            vidstatus_tex = 'vid_out.png'
            number_color = color.white
        else:
            vidstatus_tex = 'vid_out.png'
            number_color = color.white

        if item.is_gripper:
            self.gripper.setStatus(user_status_str)
            self.vid_status[4].texture = vidstatus_tex
            self.vid_status[4].numbertext.color = number_color
            self.gripper.ip_address = item.ip_address
        else:
            self.anchors[item.anchor_num].setStatus(user_status_str)
            self.vid_status[item.anchor_num].texture = vidstatus_tex
            self.vid_status[item.anchor_num].numbertext.color = number_color
            self.anchors[item.anchor_num].ip_address = item.ip_address

    def _handle_vid_stats(self, item: telemetry.VidStats):
        self.detections_text.text = detections_format_str.format(val=item.detection_rate)
        self.video_latency_text.text = video_latency_format_str.format(val=item.video_latency)
        self.video_framerate_text.text = video_framerate_format_str.format(val=item.video_framerate)

    def _handle_named_position(self, item: telemetry.NamedObjectPosition):
        # gantry_goal_pos - blue map pin
        # gamepad - red cube
        # hamper - yellow cube
        # TODO use more intuitive models for these
        if 'gantry_goal_marker' == item.name:
            if item.position is not None:
                s = item.position
                self.goal_marker.position = (s.x, s.z, s.y)
                self.goal_marker.enabled = True
            else:
                self.goal_marker.enabled = False
        if 'gamepad' == item.name:
            self.floor.set_gp_pos(tonp(item.position))

    def _handle_last_commanded_vel(self, item: telemetry.CommandedVelocity):
        self.commanded_velocity_indicator.set_velocity(tonp(item.velocity))

    def _handle_pop_message(self, item: telemetry.Popup):
        self.pop_message.show_message(item.message)

    def _handle_grip_sensors(self, item: telemetry.GripperSensors):
        self.gripper.setLaserRange(item.grip_sensors.range)
        self.gripper.setFingerAngle(item.grip_sensors.angle)

    def _handle_grip_cam_preditions(self, item: telemetry.GripCamPredictions):
        self.cam_views[None].set_predictions(item)

    def _handle_target_list(self, item: telemetry.TargetList):
        # use this to re-write the panel on the left that indicates active targets
        self.action_list.set_target_list(item)

    def start(self):
        self.app.run()

    def end_update_thread(self):
        if self.websocket:
            self.websocket.close()

def start_ui(register_input):
    """
    Entry point to be used when starting this from main.py with multiprocessing
    """
    cpui = ControlPanelUI()
    register_input(cpui)

    # use simple threading here. ursina has it's own loop that conflicts with asyncio
    receive_updates_thread = threading.Thread(target=cpui.receive_updates, daemon=True)
    receive_updates_thread.start()

    def stop_others():
        cpui.end_update_thread()
        # print("UI window closed. stopping other processes")

    # ursina has no way to tell us when the window is closed. but this python module can do it.
    atexit.register(stop_others)

    cpui.start()

if __name__ == "__main__":
    def register_input_2(cpui):
        global input
        input = cpui.input
    start_ui(register_input_2)
