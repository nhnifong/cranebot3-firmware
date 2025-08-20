import numpy as np
from scipy.interpolate import BSpline
import sys
from direct.stdpy import threading # panda3d drop in replacement that is compatible with it's event loop
import time
from functools import partial
from cv_common import invert_pose, compose_poses
from math import pi
import atexit
from panda3d.core import LQuaternionf
from cv_common import average_pose, compose_poses
import model_constants
from PIL import Image
from config import Config

from ursina import *
from ursina.shaders import (
    lit_with_shadows_shader,
    unlit_shader,
)
from ursina.prefabs.dropdown_menu import DropdownMenu, DropdownMenuButton
from ursina_entities import * # my ursina objects

window.borderless = False

# the color for the lines that connect the anchors, gantry, and gripper
line_color = color.black

# user readable mode names
mode_names = {
    'run':   'Run Normally',
    'pause': 'Pause/Observe',
    'pose':  'Locate Anchors',
}
mode_descriptions = {
    'run':   'Movement continuously follows the green spline. Goal positions are selected automatically or can be added with the mouse',
    'pause': 'WASD-QE moves the gantry, RF moves the winch line. Space toggles the grip. Spline fitting from observation occurs but has no effect.',
    'pose':  'Place the origin card on the floor in the center of the room. Anchor positions are estimated from this card.',
}
detections_format_str = 'Detections/sec {val:.2f}'
video_latency_format_str = 'Video latency {val:.2f} s'
video_framerate_format_str = 'Avg framerate {val:.2f} fps'
estimate_age_format_str = 'Position est. latency {val:.2f} s'

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

def update_from_trimesh(tm, entity):
    entity.model = Mesh(vertices=tm.vertices[:, [0, 2, 1]].tolist(), triangles=tm.faces.tolist())

def update_from_trimesh_with_color(color, tm, entity):
    entity.color = color
    update_from_trimesh(tm, entity)

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
    def __init__(self, to_ob_q):
        self.app = Ursina(fullscreen=False, borderless=False)

        self.to_ob_q = to_ob_q
        self.n_anchors = 4 # todo grab from config
        self.time_domain = (1,2)
        self.direction = np.zeros(3, dtype=float)
        self.run_periodic_actions = True

        # start in pose calibration mode. TODO need to do this only if any of the four anchor clients boots up but can't find it's file
        # maybe you shouldn't read those files in the clients
        self.calibration_mode = 'pause'

        # an indicator of where the user wants the gantry to be during direct moves.
        self.dmgt = DirectMoveGantryTarget(self)

        # debug indicators of the visual and hang based position and velocity estimates
        self.debug_indicator_visual = IndicatorSphere(color=color.red)
        self.debug_indicator_hang = IndicatorSphere(color=color.blue)

        #show a very large floor
        self.floor = Floor(
            app=self,
            model='quad',
            position=(0, -0.05, 0),
            rotation=(90,0,0),
            color=(0.35,0.22,0.18,1.0),
            scale=(10, 10), # Scale in meters
            shader=lit_with_shadows_shader
        ) 

        self.config = Config()
        self.anchors = []
        for i, a in enumerate(self.config.anchors):
            anch = Anchor(i, to_ob_q, position=swap_yz(a.pose[1]), rotation=to_ursina_rotation(a.pose[0]))
            anch.pose = a.pose
            self.anchors.append(anch)

        self.gantry = Gantry(
            ui=self,
            to_ob_q=self.to_ob_q,
            model='gantry',
            color=(0.9, 0.9, 0.9, 1.0),
            scale=0.001,
            position=(0,1,1),
            rotation=(0,0,0),
            shader=lit_with_shadows_shader,
        )

        self.gripper = Gripper(
            ui=self,
            to_ob_q=self.to_ob_q,
            position=(0,0.3,1),
            shader=lit_with_shadows_shader,
        )

        self.lines = []
        for a in self.anchors:
            self.lines.append(Entity(model=draw_line(a.position, self.gantry.position), color=line_color, shader=unlit_shader))

        self.vert_line = Entity(model=draw_line(self.gantry.position, self.gripper.position), color=line_color, shader=unlit_shader)

        # show a visualization of goal positions
        self.goal_marker = GoalPoint([0,0,0], enabled=False)

        # the color is how you control the brightness
        DirectionalLight(position=(2, 20, 1), shadows=True, rotation=(35, -5, 5), color=(0.8,0.8,0.8,1))
        AmbientLight(color=(0.8,0.8,0.8,1))
        # light.look_at(Vec3(1,-1,1))

        self.stop_button = Button(
            text="STOP",
            color=color.red,
            position=(0.83, -0.45), scale=(.10, .033),
            on_click=self.on_stop_button)

        self.walls = [Entity(
            model='quad',
            texture='vertical_gradient',
            color=(0.0, 1.0, 0.0, 0.1),
            shader=unlit_shader,
            double_sided=True
            ) for i in range(4)]
        self.redraw_walls()

        self.go_quads = EntityPool(100, lambda: Entity(
                model='cube',
                color=color.white, scale=(0.03),
                shader=unlit_shader))

        self.hypo_anchors = [
            Entity(
                model='anchor',
                color=c,
                shader=unlit_shader,
                enabled=False,
            ) for c in [
                (1.0, 0.0, 0.0, 0.5),
                (1.0, 1.0, 0.0, 0.5),
                (0.0, 1.0, 0.0, 0.5),
                (0.0, 1.0, 1.0, 0.5),
            ]]

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
        self.vid_status = [Panel(model='quad', z=91, 
            color=color.white,
            texture='vid_out.png',
            position=position,
            scale=(0.01998,0.01498),
        ) for position in [(x,y), (x+offx,y), (x,y+offy), (x+offx,y+offy), ((x+offx/2,y+offy/2))]]
        for i,vs_panel in enumerate(self.vid_status):
            vs_panel.numbertext = Text(
                color=color.white,
                position=vs_panel.position + (-0.005, 0.005),
                text=str(i),
                scale=0.5,
            )

        self.prisms = EntityPool(80, lambda: Entity(
            color=(1.0, 1.0, 1.0, 0.1),
            shader=lit_with_shadows_shader))
        self.solids = EntityPool(100, lambda: Entity(
            color=(1.0, 1.0, 0.5, 1.0),
            shader=lit_with_shadows_shader))


        DropdownMenu('Menu', buttons=(
            DropdownMenu('Mode', buttons=(
                DropdownMenuButton(mode_names['run'], on_click=partial(self.set_mode, 'run')),
                DropdownMenuButton(mode_names['pause'], on_click=partial(self.set_mode, 'pause')),
                DropdownMenuButton(mode_names['pose'], on_click=partial(self.set_mode, 'pose')),
                )),
            DropdownMenuButton('Estimate line lengths', on_click=self.calibrate_lines),
            DropdownMenuButton('Tension all lines', on_click=partial(self.simple_command, 'tension_lines')),
            DropdownMenuButton('Run Full Calibration', on_click=partial(self.simple_command, 'full_cal')),
            DropdownMenuButton('Run Zero-Angle Calibration', on_click=partial(self.simple_command, 'half_cal')),
            DropdownMenuButton('Figure-8 motion test', on_click=partial(self.simple_command, 'fig-8')),
            DropdownMenu('Simulated Data', buttons=(
                DropdownMenuButton('Disable', on_click=partial(self.set_simulated_data_mode, 'disable')),
                DropdownMenuButton('Circle', on_click=partial(self.set_simulated_data_mode, 'circle')),
                DropdownMenuButton('Point to Point', on_click=partial(self.set_simulated_data_mode, 'point2point')),
                )),
            ))

        Sky(color=color.light_gray)
        EditorCamera()

    def set_simulated_data_mode(self, mode):
        self.to_ob_q.put({'set_simulated_data_mode': mode})

    def simple_command(self, command):
        self.to_ob_q.put({command: None})

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
            self.gripper.toggleClosed()

        was_dir = self.direction

        if key in key_behavior:
            axis, speed = key_behavior[key]
            self.direction[axis] = speed
            
            # the key change results in no commanded movement of the goal point
            print(f'key = "{key}" direction = {self.direction}')
            if sum(self.direction) == 0:
                # immediately cancel whatever remains of the movement
                self.to_ob_q.put({'slow_stop_all': None})
                self.dmgt.reset()
            else:
                # a move will be initiated. but sometimes ursina doesn't get the up key that would end the move.
                # this can be destructive for a CDPR, so we must ensure every move stops eventually.
                self.last_positive_input_time = time.time()
                invoke(self.maybe_end_move, delay=2)

    def maybe_end_move(self):
        if time.time()-2 > self.last_positive_input_time:
            print('Ending direct move with a timeout because ursina missed a key-up')
            self.direction = np.zeros(3, dtype=float)
            self.to_ob_q.put({'slow_stop_all': None})
            self.dmgt.reset()

    def show_error(self, error):
        self.error.text = error
        self.error.enabled = True

    def clear_error(self):
        self.error.enabled = False

    def set_mode(self, mode):
        self.calibration_mode = mode
        self.mode_text.text = mode_names[mode]
        self.mode_descrip_text.text = mode_descriptions[mode]
        self.to_ob_q.put({'set_run_mode': self.calibration_mode})
        if mode=='pose':
            self.confirm_poses_bt = Button(
                text='Confirm',
                color=color.green,
                text_color=color.black,
                on_click=self.finish_locate_anchors,
                position=(0.5,-0.45),
                scale=(.10, .033))
            for h in self.hypo_anchors:
                h.pose = None

    def finish_locate_anchors(self):
        self.calibration_mode = 'pause'
        self.mode_text.text = mode_names['pause']
        self.mode_descrip_text.text = mode_descriptions['pause']
        aposes = {}
        # copy the positions of the hypothetical anchors to the real anchors
        for i in range(4):
            if self.hypo_anchors[i].pose is None:
                print(f'Anchor Camera {i} did not obtain enough observations of the origin card to determine its location')
                continue
            aposes[i] = self.hypo_anchors[i].pose
            self.hypo_anchors[i].enabled = False
            self.anchors[i].pose = self.hypo_anchors[i].pose
            self.anchors[i].position = self.hypo_anchors[i].position
            self.anchors[i].rotation = self.hypo_anchors[i].rotation
        self.gantry.redraw_wires()
        self.redraw_walls()
        # observer will write the config for us
        self.to_ob_q.put({
            'confirm_anchors': aposes,
            'set_run_mode': self.calibration_mode
            })
        # hide button
        self.confirm_poses_bt.enabled = False


    def calibrate_lines(self):
        # calibrate line lengths
        # Assume we have been stopped for a while.
        if self.calibration_mode != 'pause':
            self.error.text = "Line calibration can only be performed while in Pause/Observe mode"
            self.error.enabled = True
            return
        print('Do line calibration')
        # use visual gantry position.
        gantry_position = self.debug_indicator_visual.zup_pos
        lengths = [3.79,4.94,2.95,4.08] 
        for i, anchor in enumerate(self.anchors):
            # index 1 in a pose tuple is the position.
            lengths[i] = np.linalg.norm(anchor.pose[1] - gantry_position)
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
            lengths = [float(self.line_cal_confirm.content[1+i].text) for i in range(4)]
        except ValueError:
            return
        self.to_ob_q.put({'do_line_calibration': lengths})
        self.line_cal_confirm.enabled = False

    def on_stop_button(self):
        self.to_ob_q.put({'slow_stop_all':None})

    def render_gantry_ob(self, row, color):
        self.go_quads.add(partial(update_go_quad, row, color))

    def periodic_actions(self):
        """
        Run certain actions at a rate slightly less than, and independent of the framerate.
        """
        time.sleep(2)
        while self.run_periodic_actions:

            if sum(self.direction) == 0:
                self.dmgt.position = self.gantry.position
            else:
                # self.dmgt.position will be updated in it's own update function at the framerate.
                # at this slightly lower rate, command the bot to move towards the goal point.
                self.dmgt.direct_move()

            time.sleep(1/10)

    def notify_connected_bots_change(self, available_bots={}):
        offs = 0
        for server,info in available_bots.items():
            text_entity = Text(server, world_scale=16, position=(-0.1, -0.4 + offs))
            offs -= 0.03

    def receive_updates(self, min_to_ui_q):
        while True:
            try:
                # queue.get has to happen in a thread, because it blocks
                updates = min_to_ui_q.get()
                # but processing the update needs to happen in the ursina loop, because it will modify a bunch of entities.
                invoke(self.process_update, updates, delay=0.0001)
            except (OSError, EOFError, TypeError):
                # sometimes when closing the app, this thread gets left hanging because that queue is gone
                return


    def process_update(self, updates):
        if 'STOP' in updates:
            application.quit()

        if 'minimizer_stats' in updates:
            estimate_age = time.time() - updates['minimizer_stats']['data_ts']
            self.estimate_age_text.text = estimate_age_format_str.format(val=estimate_age)

        if 'pos_estimate' in updates:
            p = updates['pos_estimate']
            self.gantry.set_position_velocity(p['gantry_pos'], p['gantry_vel'])
            self.gantry.set_slack_vis(p['slack_lines'])
            self.gripper.setPose(p['gripper_pose'])
            # p['slack_lines']

        if 'pos_factors_debug' in updates:
            p = updates['pos_factors_debug']
            self.debug_indicator_visual.set_position_velocity(p['visual_pos'], p['visual_vel'])
            self.debug_indicator_hang.set_position_velocity(p['hang_pos'], p['hang_vel'])

        if 'gantry_observation' in updates:
            invoke(self.render_gantry_ob, swap_yz(updates['gantry_observation']), color.white, delay=0.0001)

        if 'anchor_pose' in updates:
            # if you get this message while in pose mode, it's a hypothetical pose not yet confirmed by the user.
            if self.calibration_mode == 'pose':
                apose = updates['anchor_pose']
                print(apose)
                anchor_num = apose[0]
                self.hypo_anchors[anchor_num].enabled = True
                self.hypo_anchors[anchor_num].pose = apose[1]
                self.hypo_anchors[anchor_num].position = swap_yz(apose[1][1])
                self.hypo_anchors[anchor_num].rotation = to_ursina_rotation(apose[1][0])
            else:
                apose = updates['anchor_pose']
                anchor_num = apose[0]
                self.anchors[anchor_num].enabled = True
                self.anchors[anchor_num].pose = apose[1]
                self.anchors[anchor_num].position = swap_yz(apose[1][1])
                self.anchors[anchor_num].rotation = to_ursina_rotation(apose[1][0])
                self.gantry.redraw_wires()
                self.redraw_walls()

        if 'preview_image' in updates:
            pili = updates['preview_image']
            anchor = self.anchors[pili['anchor_num']]
            if anchor.hasSetImage:
                anchor.camview.texture._texture.setRamImage(pili['image'])
            else:
                # we only need to do this the first time, so the allocated texture is the right size
                # even though this method of updating a texture exists, it's horribly slow.
                anchor.camview.texture = Texture(Image.fromarray(pili['image']))
                anchor.hasSetImage = True

        if 'connection_status' in updates:
            status = updates['connection_status']
            print(status)
            if status['websocket'] == 0:
                user_status_str = 'Not Detected'
            elif status['websocket'] == 1:
                user_status_str = 'Connecting...'
            elif  status['websocket'] == 2:
                user_status_str = 'Online'
            else:
                user_status_str = 'Unknown'

            if status['video']:
                vidstatus_tex = 'vid_ok.png'
                number_color = color.black
            else:
                vidstatus_tex = 'vid_out.png'
                number_color = color.white

            if 'anchor_num' in status:
                self.anchors[status['anchor_num']].setStatus(user_status_str)
                self.vid_status[status['anchor_num']].texture = vidstatus_tex
                self.vid_status[status['anchor_num']].numbertext.color = number_color
                if 'ip_address' in status:
                    self.anchors[status['anchor_num']].ip_address = status['ip_address']
            elif 'gripper' in status:
                self.gripper.setStatus(user_status_str)
                self.vid_status[4].texture = vidstatus_tex
                self.vid_status[4].numbertext.color = number_color

        if 'vid_stats' in updates:
            stats = updates['vid_stats']
            self.detections_text.text = detections_format_str.format(val=stats['detection_rate'])
            self.video_latency_text.text = video_latency_format_str.format(val=stats['video_latency'])
            self.video_framerate_text.text = video_framerate_format_str.format(val=stats['video_framerate'])

        if 'gantry_goal_marker' in updates:
            pos = updates['gantry_goal_marker']
            if pos is not None:
                self.goal_marker.position = swap_yz(pos)
                self.goal_marker.enabled = True
            else:
                self.goal_marker.enabled = False

        if 'solids' in updates:
            for key, val in updates["solids"].items():
                for mesh in val:
                    self.solids.add(partial(update_from_trimesh_with_color, solid_colors[key], mesh))

        if 'prisms' in updates:
            self.prisms.replace(updates["prisms"], update_from_trimesh)

        # if 'gripper_rvec' in updates:
        #     self.gripper.rotation = to_ursina_rotation(updates['gripper_rvec'])

        if 'origin_poses' in updates:
            o_poses = updates['origin_poses']
            # list of poses of detected origin cards from all anchors
            # (anchor_num, (rvec, tvec))
            for anum, pose in o_poses:
                self.origin_cards.add(partial(update_origin_card, anum, pose))

    def start(self):
        self.app.run()

def start_ui(to_ui_q, to_ob_q, register_input):
    """
    Entry point to be used when starting this from main.py with multiprocessing
    """
    cpui = ControlPanelUI(to_ob_q)
    register_input(cpui)

    # use simple threading here. ursina has it's own loop that conflicts with asyncio
    estimator_update_thread = threading.Thread(target=cpui.receive_updates, args=(to_ui_q, ), daemon=True)
    estimator_update_thread.start()

    rgo = threading.Thread(target=cpui.periodic_actions, daemon=True)
    rgo.start()

    def stop_other_processes():
        print("UI window closed. stopping other processes")
        cpui.run_periodic_actions = False
        to_ui_q.put({'STOP':None}) # stop our own listening thread too
        to_ob_q.put({'STOP':None})

    # ursina has no way to tell us when the window is closed. but this python module can do it.
    atexit.register(stop_other_processes)

    cpui.start()

if __name__ == "__main__":
    from multiprocessing import Queue
    to_ui_q = Queue()
    to_ob_q = Queue()
    def register_input_2(cpui):
        global input
        input = cpui.input
    start_ui(to_ui_q, to_ob_q, register_input_2)
