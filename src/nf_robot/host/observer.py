from __future__ import annotations

import signal
import sys
import threading
import time
import socket
import asyncio
import argparse
from zeroconf import IPVersion, ServiceStateChange, Zeroconf
from zeroconf.asyncio import (
    AsyncServiceBrowser,
    AsyncServiceInfo,
    AsyncZeroconf,
    AsyncZeroconfServiceTypes,
    InterfaceChoice,
)
from multiprocessing import Pool
import numpy as np
import scipy.optimize as optimize
from scipy.spatial.transform import Rotation
from random import random
import traceback
import cv2
import pickle
from collections import deque, defaultdict
import uuid
import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from functools import partial
from pathlib import Path
import json
from importlib.resources import files

from nf_robot.common.pose_functions import compose_poses
from nf_robot.common.cv_common import *
from nf_robot.common.config_loader import *
import nf_robot.common.definitions as model_constants
from nf_robot.common.util import *
from nf_robot.generated.nf import telemetry, control, common
from nf_robot.host.data_store import DataStore
from nf_robot.host.stats import StatCounter
from nf_robot.host.target_queue import TargetQueue
from nf_robot.host.calibration import optimize_anchor_poses
from nf_robot.host.anchor_client import RaspiAnchorClient, max_origin_detections
from nf_robot.host.gripper_client import RaspiGripperClient
from nf_robot.host.arp_gripper_client import ArpeggioGripperClient
from nf_robot.host.position_estimator import Positioner2

# Define the service names for network discovery
anchor_service_name = 'cranebot-anchor-service'
anchor_power_service_name = 'cranebot-anchor-power-service'
gripper_service_name = 'cranebot-gripper-service'
arp_gripper_service_name = 'cranebot-gripper-arpeggio-service'

N_ANCHORS = 4
INFO_REQUEST_TIMEOUT_MS = 3000 # milliseconds
CONTROL_PLANE_PRODUCTION = "wss://neufangled.com"
CONTROL_PLANE_STAGING = "wss://nf-site-monolith-staging-690802609278.us-east1.run.app"
CONTROL_PLANE_LOCAL = "ws://localhost:8080"
UNPROCESSED_DIR = "square_centering_data_unlabeled"
USER_TARGETS_DIR = "user_targets_data"
METADATA_PATH = os.path.join(USER_TARGETS_DIR, "metadata.jsonl")

def capture_gripper_image(ndimage, gripper_occupied=False):
    """
    Saves an image to the unprocessed directory. 
    Encodes gripper state in filename: {uuid}_g{1|0}.jpg
    """
    if not os.path.exists(UNPROCESSED_DIR):
        os.makedirs(UNPROCESSED_DIR)
    
    h, w = ndimage.shape[:2]
        
    state_str = "g1" if gripper_occupied else "g0"
    file_id = str(uuid.uuid4())
    img_filename = f"{file_id}_{state_str}.jpg"
    img_full_path = os.path.join(UNPROCESSED_DIR, img_filename)
    
    # Save (ensure RGB/BGR consistency)
    cv2.imwrite(img_full_path, ndimage)
    print(f"Captured: {img_filename} (Gripper: {gripper_occupied})")

class AsyncObserver:
    """
    Manager of multiple tasks running clients connected to each robot component
    The job of this class in a nutshell is to discover four anchors and a gripper on the network,
    connect to them, and forward data between them and the position estimator, shape tracker, and UI.

    It reads from the config file to find any components it already knows about.
    It starts zeroconf to discover any components it doesn't know about and add them to the config.
    it starts keep_robot_connected to continually reconnect to all known components.
    It starts position_estimator to continually run kalman filters on the observed variables.
    It starts run_perception to continually run inference on the camera feeds.
    It starts a websocket server to accept connections from local UIs 

    It starts a websocket server to accept connections from local UIs 
    It reads from the config file to find any components it already knows about.
    It starts zeroconf to discover any components it doesn't know about and add them to the config.
    As soon as a component in the config has a known address, it starts keep_robot_connected to continually reconnect to all known components.
    As soon as the first component websocket is connected, It starts position_estimator to continually run kalman filters on the observed variables.
    As soon as a feed from the first preferred camera is up, It starts run_perception to continually run inference on the camera feeds.

    Since this class serves as the coordination center of all the robot compnents, it also contains methods to perform
    various actions like calibration and the pick and place routine.
    """
    def __init__(self, terminate_with_ui, config_path, telemetry_env=None) -> None:
        self.terminate_with_ui = terminate_with_ui
        self.position_update_task = None
        self.aiobrowser: AsyncServiceBrowser | None = None
        self.aiozc: AsyncZeroconf | None = None
        self.run_command_loop = True
        self.datastore = DataStore()
        self.pool = None
        # all clients by server name
        self.bot_clients = {}
        # all connected anchors
        self.anchors = []
        # convenience reference to gripper client
        self.gripper_client = None
        # TODO allow a command line argument to override the config file path
        self.config_path = config_path
        self.config = load_config(config_path)
        self.telemetry_env = telemetry_env
        self.stat = StatCounter(self)
        self.enable_shape_tracking = False
        self.shape_tracker = None
        # Position Estimator. this used to be a seperate process so it's still somewhat independent.
        self.pe = Positioner2(self.datastore, self)
        self.sim_task = None
        self.locate_anchor_task = None
        # only one motion task can be active at a time
        self.motion_task = None
        # only used for integration test only to allow some code to run right after sending the gantry to a goal point
        self.test_gantry_goal_callback = None
        # event used to notify tasks that gripper is connected.
        self.gripper_client_connected = asyncio.Event()
        self.grpc_server = None
        self.last_gp_action = None
        self.last_user_move_time = time.time()
        self.episode_control_events = set()
        self.named_positions = {}
        self.target_model = None
        self.centering_model = None
        self.predicted_lateral_vector = None
        # targets
        self.target_queue = TargetQueue()
        self.last_snapshot_hash = None # to spare the UI from too many updates
        # websockets to locally connected UIs
        self.connected_local_clients = set()
        self.telemetry_buffer = deque(maxlen=100)
        self.telemetry_buffer_lock = threading.RLock()
        self.startup_complete = asyncio.Event()
        self.any_anchor_connected = asyncio.Event() # fires as soon as first anchor connects, starting pe
        self.cloud_telem_websocket = None
        self.gip_task = None
        self.cloud_telem = None
        self.passive_safety_task = None
        # last attempt to connect, keyed by service name
        self.connection_tasks: dict[str, asyncio.Task] = {}
        self.run_collect_images = False

    async def send_setup_telemetry(self):
        print('Sending setup telemetry')
        self.send_ui(new_anchor_poses=telemetry.AnchorPoses(
            poses=[a.pose for a in self.config.anchors]
        ))
        for client in self.bot_clients.values():
            client.send_conn_status()
            if client.local_udp_port is not None and client.anchor_num in [None, *self.config.preferred_cameras]:
                self.send_ui(video_ready=telemetry.VideoReady(
                    is_gripper=client.anchor_num is None,
                    anchor_num=client.anchor_num,
                    local_uri=f'udp://127.0.0.1:{client.local_udp_port}'
                ))
        r = await self.flush_tele_buffer()

    async def handle_local_client(self, websocket):
        # Called when Ursina connects to a websocket that is opened to accept control commands
        self.connected_local_clients.add(websocket)
        print('Connection received from local UI process')

        # send anything that it would need up-front
        r = await self.send_setup_telemetry()
        try:
            async for message in websocket:
                r = await self.handle_command(message) # Handle 'ControlBatchUpdate'
                # warning, any uncaught exception here will kill this websocket connection
                # but the observer would go on running, possibly in a bad state.
        except (ConnectionClosedError, ConnectionClosedOK) as e:
            pass
        finally:
            self.connected_local_clients.remove(websocket)
            if len(self.connected_local_clients) == 0 and self.terminate_with_ui:
                # The only local UI has disconnected and we were asked to shutdown when it disconnects
                self.run_command_loop = False

    async def handle_command(self, message: bytes):
        """ Decodes a binary batch of commands """
        # betterproto .parse() returns a standard python dataclass
        batch = control.ControlBatchUpdate().parse(message)
        # Safety check: Ignore commands meant for other robots
        if batch.robot_id and batch.robot_id != self.config.robot_id:
            pass
            # print(f'warning: UI is sending commands identified as being for robot {batch.robot_id} to robot {self.config.robot_id}')
            # return
        for update in batch.updates:
            r = await self._dispatch_update(update)

    async def _dispatch_update(self, item: control.ControlItem):
        # In betterproto, 'oneof' fields appear as attributes. 
        # Only one will be non-None.
        
        # Standard Commands (Stop, Calibrate, Zero)
        if item.command:
            r = await self._handle_common_command(item.command.name)

        # Movement Vector (Gamepad/AI Policy)
        elif item.move:
            r = await self._handle_movement(item.move)

        # Setting gantry goal
        elif item.gantry_goal_pos:
            r = await self._handle_gantry_goal_pos(tonp(item.gantry_goal_pos.pos))

        # Manual Spool Control
        elif item.jog_spool:
            r = await self._handle_jog_spool(item.jog_spool)

        # Lerobot Episode Control (Start/Stop Recording)
        elif item.episode_control:
            self._handle_add_episode_control_events(item.episode_control)

        elif item.scale_room:
            self._handle_scale_room(item.scale_room)

        elif item.add_cam_target:
            self._handle_add_cam_target(item.add_cam_target)

        elif item.delete_target:
            self._handle_delete_target(item.delete_target)

    def _handle_delete_target(self, item: control.DeleteTarget):
        if item.target_id is not None:
            self.target_queue.remove_target(item.target_id);

    def _handle_add_cam_target(self, item: control.AddTargetFromAnchorCam):
        # Add the target
        targets2d = [[item.img_norm_x, item.img_norm_y]]
        match_anchors = matching_items = [x for x in self.anchors if x.anchor_num == item.anchor_num]
        if len(match_anchors) != 1:
            return
        floor_points = project_pixels_to_floor(targets2d, match_anchors[0].camera_pose, self.config.camera_cal)
        print(f'adding target at floor point ({floor_points}) from image point ({targets2d[0]}) in anchor cam {item.anchor_num}')
        if (len(floor_points) == 1):
            if item.target_id is not None:
                self.target_queue.set_target_position(item.target_id, floor_points[0])
            else:   
                new_id = self.target_queue.add_user_target(floor_points[0], dropoff='hamper')
        self.send_tq_to_ui()

    def submitTargets(self):
        """snapshot any active cameras at 1920x1080 and save images in the raw dir"""
        images = []
        for anchor in self.anchors:
            if anchor.frame is not None:
                images.append(anchor.frame.copy())

        def save_data(images):
            directory_path = Path("target_heatmap_data_unlabeled")
            directory_path.mkdir(exist_ok=True, parents=True)
            
            for img in images:
                img_filename = f"{str(uuid.uuid4())}.jpg"
                # write the image
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_full_path = directory_path / img_filename
                cv2.imwrite(str(img_full_path), rgb_image)

        threading.Thread(target=save_data, args=(images,)).start()

    def _handle_scale_room(self, item: control.ScaleRoom):
        if item.scale:
            # move positions of anchors towards or away from origin
            print(f'scaling by {item.scale}')
            anchor_poses = [(client.anchor_pose[0], client.anchor_pose[1]*item.scale) for client in self.anchors]

            # update everything
            for client in self.anchors:
                self.config.anchors[client.anchor_num].pose = poseTupleToProto(anchor_poses[client.anchor_num])
                client.updatePose(anchor_poses[client.anchor_num])
            save_config(self.config, self.config_path)
            # inform UI
            self.send_ui(new_anchor_poses=telemetry.AnchorPoses(poses=[
                poseTupleToProto(p)
                for p in anchor_poses
            ]))
            # inform position estimator
            anchor_points = np.array([compose_poses([pose, model_constants.anchor_grommet])[1] for pose in anchor_poses])
            self.pe.set_anchor_points(anchor_points)

        if item.tiltcams:
            print(f'tilting cams inward by {item.tiltcams} deg')
            for client in self.anchors:
                client.extratilt += item.tiltcams
                client.updatePose(client.anchor_pose)


    async def _handle_common_command(self, cmd: control.Command):
        # betterproto Enums are IntEnums, comparable directly
        match cmd:
            case control.Command.STOP_ALL:
                r = await self.stop_all()
            case control.Command.TIGHTEN_LINES:
                r = await self.tension_lines()
            case control.Command.ZERO_WINCH:
                asyncio.create_task(self._handle_zero_winch_line())
            case control.Command.HALF_CAL:
                r = await self.invoke_motion_task(self.half_auto_calibration())
            case control.Command.FULL_CAL:
                r = await self.invoke_motion_task(self.full_auto_calibration())
            case control.Command.ENABLE_LEROBOT:
                self.training_task = asyncio.create_task(self.begin_training_mode())
            case control.Command.PICK_AND_DROP:
                r = await self.invoke_motion_task(self.pick_and_place_loop())
            case control.Command.HORIZONTAL_CHECK:
                r = await self.invoke_motion_task(self.horizontal_line_task())
            case control.Command.COLLECT_GRIPPER_IMAGES:
                self._handle_collect_images()
            case control.Command.SHUTDOWN:
                self.run_command_loop = False
            case control.Command.PARK:
                r = await self.invoke_motion_task(self.park())
            case control.Command.UNPARK:
                r = await self.invoke_motion_task(self.unpark())
            case control.Command.GRASP:
                r = await self.invoke_motion_task(self.execute_grasp())
            case control.Command.SUBMIT_TARGETS_TO_DATASET:
                self.submitTargets()

    async def _handle_jog_spool(self, jog: control.JogSpool):
        """Handles manually jogging a spool motor."""

        # identify the client we need to send the command to
        client = None
        if jog.is_gripper:
            client = self.gripper_client
        else:
            for c in self.anchors:
                if c.anchor_num == jog.anchor_num:
                    client = c

        # send the right kind of jog command
        if client is not None:
            if jog.speed is not None:
                asyncio.create_task(client.send_commands({'aim_speed': jog.speed}))
            elif jog.offset is not None:
                asyncio.create_task(client.send_commands({'jog': jog.offset}))

    async def _handle_gantry_goal_pos(self, goal_pos: np.ndarray):
        """Handles moving the gantry to a specific goal position."""
        self.gantry_goal_pos = goal_pos
        await self.invoke_motion_task(self.seek_gantry_goal())

    async def _handle_slow_stop_one(self, stop_data: dict):
        """Handles stopping a single spool motor."""
        if stop_data.get('id') == 'gripper' and self.gripper_client:
            asyncio.create_task(self.gripper_client.slow_stop_spool())
        else:
            for client in self.anchors:
                if client.anchor_num == stop_data.get('id'):
                    asyncio.create_task(client.slow_stop_spool())

    async def _handle_zero_winch_line(self):
        if self.gripper_client is not None and isinstance(self.gripper_client, RaspiGripperClient):
            await self.gripper_client.zero_winch()

    async def _handle_movement(self, move: control.CombinedMove):
        # if we have to clip these values to legal limits, save what they were clipped to
        winch, finger, wrist = await self.send_gripper_move(move.winch, move.finger, move.wrist)

        direction = np.zeros(3)
        if move.direction:
            direction = tonp(move.direction)
        commanded_vel = await self.move_direction_speed(direction, move.speed)

        # the saved values will be what we return from GetLastAction
        if winch is not None:
            self.last_gp_action = (commanded_vel, winch, finger)
        else:
            self.last_gp_action = (commanded_vel, wrist, finger)
        self.last_user_move_time = time.time()

    async def passive_safety(self):
        """
        This not only has the effect of stopping the robot if the user disappears mid-move,
        but by feeding zeros to the kalman filter for commanded velocity every second, jitter is greatly reduced.
        this task's action is suppressed whenever there is a motion task or the user is actually sending commands.
        """
        while self.run_command_loop:
            if not ((self.motion_task is not None and not self.motion_task.done()) or (self.last_user_move_time > (time.time()-1))):
                self.slow_stop_all_spools()
            await asyncio.sleep(1)

    def update_avg_named_pos(self, key: str, position: np.ndarry):
        """Update the running average of the named position"""
        if key not in self.named_positions:
            self.named_positions[key] = position
        # exponential moving average
        self.named_positions[key] = self.named_positions[key] * 0.75 + position * 0.25
        pos = self.named_positions[key]
        self.send_ui(named_position=telemetry.NamedObjectPosition(position=fromnp(pos), name=key))

    async def invoke_motion_task(self, coro):
        """
        Cancel whatever else is happening and start a new long running motion task
        Any task that can be called this way is known in this file as a "motion task"
        The defining feature of a motion task is that it could send a second motion command to any client after any amount of sleeping
        every motion task must have the follwing structure

        try:
            # do something
        except asyncio.CancelledError:
            raise
        finally:
            # perform any clean up work

        Do not call invoke_motion_task from within a motion task or it will cancel itself.
        It is ok to call a motion task from within another, just don't start it with invoke_motion_task
        Do not call stop_all from within a motion task. use slow_stop_all_spools instead

        """
        if self.motion_task is not None and not self.motion_task.done():
            print(f"Cancelling previous motion task: {self.motion_task.get_name()}")
            self.motion_task.cancel()
            try:
                # Wait briefly for the old task's cleanup to complete.
                result = await self.motion_task
            except asyncio.CancelledError:
                pass # Expected behavior

        self.motion_task = asyncio.create_task(coro)
        self.motion_task.set_name(coro.__name__)

    async def tension_lines(self):
        """Request all anchors to reel in all lines until tight.
        This is a fire and forget function"""
        for client in self.anchors:
            asyncio.create_task(client.send_commands({'tighten': None}))
        # This function does not  wait for confirmation from every anchor, as it would just hold up the processing of the ob_q
        # this is similar to sending a manual move command. it can be overridden by any subsequent command.
        # thus, it should be done while paused.

    async def wait_for_tension(self):
        """this function returns only once all anchors are reporting tight lines in their regular line record"""
        POLL_INTERVAL_S = 0.1 # seconds
        SPEED_SUM_THRESHOLD = 0.01 # m/s
        
        complete = False
        while not complete:
            await asyncio.sleep(POLL_INTERVAL_S)
            records = np.array([alr.getLast() for alr in self.datastore.anchor_line_record])
            speeds = np.array(records[:,2])
            tight = np.array(records[:,3])
            print(f'wait for tension speeds={speeds} tight={tight}')
            complete = np.all(tight) and abs(np.sum(speeds)) < SPEED_SUM_THRESHOLD
        return True

    async def tension_and_wait(self):
        """Send tightening command and wait until lines appear tight. This is not a motion task"""
        print('Tightening all lines')
        await self.tension_lines()
        await self.wait_for_tension()

    async def sendReferenceLengths(self, lengths):
        if len(lengths) != N_ANCHORS:
            print(f'Cannot send {len(lengths)} ref lengths to anchors')
            return
        # any anchor that receives this and is slack would ignore it
        # If only some anchors are connected, this would still send reference lengths to those
        for client in self.anchors:
            asyncio.create_task(client.send_commands({'reference_length': lengths[client.anchor_num]}))

        # use swing to estimate winch line length in pilot gripper
        if self.gripper_client is not None and isinstance(self.gripper_client, RaspiGripperClient):
            winch_length = self.pe.get_pendulum_length()
            if winch_length is not None:
                asyncio.create_task(self.gripper_client.send_commands({'reference_length': winch_length}))

        # reset biases on kalman filter
        data = self.datastore.gantry_pos.deepCopy()
        position = np.mean(data[:,2:], axis=0)
        print(f'reseting filter biases with assumed position of {position}')
        self.pe.kf.reset_biases(position)

    async def set_simulated_data_mode(self, mode):
        if self.sim_task is not None:
            self.sim_task.cancel()
            result = await self.sim_task
        if mode == 'circle':
            self.sim_task = asyncio.create_task(self.add_simulated_data_circle())
        elif mode == 'point2point':
            self.sim_task = asyncio.create_task(self.add_simulated_data_point2point())

    async def stop_all(self):
        # If lerobot scripts are connected this must also stop them
        self.episode_control_events.add('stop_recording')

        # Cancel any active motion task
        if self.motion_task is not None:
            # Store the handle and clear the class attribute immediately.
            # This prevents race conditions if another command comes in.
            task_to_stop = self.motion_task
            self.motion_task = None

            # Only cancel the task if it's actually still running.
            if not task_to_stop.done():
                print(f"Cancelling motion task: {task_to_stop.get_name()}")
                task_to_stop.cancel()

            # Now, await the task's completion.
            try:
                # Awaiting a task will re-raise any exception it had,
                # or raise CancelledError if we just cancelled it.
                await task_to_stop
            except asyncio.CancelledError:
                # This is the expected, non-error outcome of a clean cancellation.
                print(f"Task '{task_to_stop.get_name()}' was successfully stopped.")
            except Exception as e:
                # If any other exception occurred, print it now.
                print(f"An unhandled exception occurred in motion task '{task_to_stop.get_name()}':\n{e}")
                traceback.print_exc()

        self.slow_stop_all_spools()

    def slow_stop_all_spools(self):
        for name, client in self.bot_clients.items():
            # Slow stop all spools. gripper too
            asyncio.create_task(client.slow_stop_spool())
        self.pe.record_commanded_vel(np.zeros(3))

    async def begin_training_mode(self):
        """Begin allowing the robot to be controlled from the grpc server
        movement could occur at any time while the server is running"""

        from nf_robot.ml.control_service import start_robot_control_server
        from nf_robot.ml.stringman_record_loop import record_until_disconnected

        try:
            # begin allowing requests from self.grpc_server
            self.grpc_server = await start_robot_control_server(self)
            # Start child process to run the dataset manager?
            # dataset_process = multiprocessing.Process(target=record_until_disconnected, name='lerobot_record')
            # dataset_process.daemon = False
            # dataset_process.start()
            await self.grpc_server.wait_for_termination()
        except asyncio.CancelledError:
            raise
        finally:
            print('training server closed.')
            await self.grpc_server.stop(grace=5)
            self.grpc_server = None
            self.slow_stop_all_spools()

    def locate_anchors(self):
        """using the record of recent origin detections and cal_assist marker detections, estimate the pose of each anchor."""
        markers = ['origin', 'cal_assist_1', 'cal_assist_2', 'cal_assist_3', 'gantry']
        averages = defaultdict(lambda: [[]]*4)
        for client in self.anchors:
            # average each list of detections, but leave them in the camera's reference frame.
            for marker in markers:
                if marker == 'gantry':
                    averages[marker][client.anchor_num] = list(client.raw_gant_poses)
                else:
                    averages[marker][client.anchor_num] = list(client.origin_poses[marker])
                print(f'anchor {client.anchor_num} has {len(averages[marker][client.anchor_num])} observations of {marker}')

        # run optimization in pool
        async_result = self.pool.apply_async(optimize_anchor_poses, (dict(averages),))
        anchor_poses = async_result.get(timeout=30)
        print(f'obtained result from find_cal_params anchor_poses=\n{anchor_poses}')

        return np.array(anchor_poses)

    async def half_auto_calibration(self):
        """
        Set line lengths from observation
        tighten, wait for obs, estimate line lengths, move up slightly, estimate line lengths, move down slightly
        This is a motion task
        """
        NUM_SAMPLE_POINTS = 3
        OPTIMIZER_TIMEOUT_S = 60  # seconds
        
        try:
            if len(self.anchors) < N_ANCHORS:
                print('Cannot run half calibration until all anchors are connected')
                return

            for direction in [[0,0,-1], [0,0,1]]:
                await self.tension_and_wait()
                # wait for some new obs
                await asyncio.sleep(0.5)
                lengths = np.linalg.norm(self.pe.anchor_points - self.pe.visual_pos, axis=1)
                await self.sendReferenceLengths(lengths)
                await asyncio.sleep(0.25)
                # move in direction for short time
                await self.move_direction_speed(direction, 0.05, downward_bias=0)
                await asyncio.sleep(0.25)
                self.slow_stop_all_spools()

        except asyncio.CancelledError:
            raise

    async def full_auto_calibration(self):
        """Automatically determine anchor poses and zero angles
        This is a motion task"""
        DETECTION_WAIT_S = 1.0 # seconds
        try:
            if len(self.anchors) < N_ANCHORS:
                print('Cannot run full calibration until all anchors are connected')
                return
            elif len(self.anchors) > N_ANCHORS:
                print(f'Too many anchors, what is going on here\n{self.anchors}')
            self.anchors.sort(key=lambda x: x.anchor_num)
            # collect observations of origin card aruco marker to get initial guess of anchor poses.
            #   origin pose detections are actually always stored by all connected clients,
            #   it is only necessary to ensure enough have been collected from each client and average them.
            for a in self.anchors:
                a.save_raw = True
            num_o_dets = []
            while len(num_o_dets) == 0 or min(num_o_dets) < max_origin_detections:
                print(f'Waiting for enough origin card detections from every anchor camera {num_o_dets}')
                await asyncio.sleep(DETECTION_WAIT_S)
                num_o_dets = [len(client.origin_poses['origin']) for client in self.anchors]
            
            anchor_poses = self.locate_anchors()

            for a in self.anchors:
                a.save_raw = False

            # Use the optimization output to update anchor poses and spool params
            for client in self.anchors:
                self.config.anchors[client.anchor_num].pose = poseTupleToProto(anchor_poses[client.anchor_num])
                client.updatePose(anchor_poses[client.anchor_num])
            save_config(self.config, self.config_path)
            # inform UI
            self.send_ui(new_anchor_poses=telemetry.AnchorPoses(poses=[
                poseTupleToProto(p)
                for p in anchor_poses
            ]))
            # inform position estimator
            anchor_points = np.array([compose_poses([pose, model_constants.anchor_grommet])[1] for pose in anchor_poses])
            self.pe.set_anchor_points(anchor_points)

            await self.half_auto_calibration()

            # open grip enough that we can see an unobstructed view from the palm camera
            asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': -30}))

            # move over the origin card
            self.gantry_goal_pos = np.array([0,0,1.2])
            await self.seek_gantry_goal()

            # there should be some swing when we get there. 
            await self.half_auto_calibration()

            if self.gripper_client.last_frame_resized is not None:
                # record the z rotation of the gantry card from the perspective of the gripper camera's stabilized frame
                # when the stabilization is done without any existing z rotation term
                self.gripper_client.calibrating_room_spin = True

                if isinstance(self.gripper_client, ArpeggioGripperClient):
                    # measurement must be taken at the wrist's zero point
                    asyncio.create_task(self.gripper_client.send_commands({'set_wrist_angle': 0}))
                    # wait till within 1 degree of target or up to 10 seconds
                    actual_wrist = 100
                    end_time = time.time() + 10
                    while abs(actual_wrist) > 1.0 and time.time() < end_time:
                        await asyncio.sleep(0.2)
                        actual_wrist = self.datastore.winch_line_record.getLast()[1]

                # detect origin card
                await asyncio.sleep(0.1)
                origin_card_pose = [None]
                def special_handle_det(timestamp, detections):
                    for d in detections:
                        if d['n'] == 'origin':
                            # a pose of the origin card in the frame of reference of the stabilized gripper cam.
                            origin_card_pose[0] = d['p']
                while origin_card_pose[0] is None:
                    async_result = self.pool.apply_async(
                        locate_markers_gripper,
                        (self.gripper_client.last_frame_resized, self.config.camera_cal),
                        callback=partial(special_handle_det, time.time()))
                    detections = async_result.get(timeout=5)
                
                euler_rot = Rotation.from_rotvec(origin_card_pose[0][0]).as_euler('zyx')
                print(f'euler rotation of origin card relative to stabilized gripper camera {euler_rot}')
                roomspin = -euler_rot[0]
                self.config.gripper.frame_room_spin = roomspin
                self.config.has_been_calibrated = True
                save_config(self.config, self.config_path)
                self.gripper_client.calibrating_room_spin = False
            else:
                print('Warning, cannot calibrate the relationship between gripper IMU zero angle and camera if gripper camera is offline!')

            # TODO "Calibration complete. Would you like stringman to pick up the cards and put them in the trash? yes/no"
            self.send_ui(pop_message=telemetry.Popup(
                message='Calibration complete. Cards can be removed from the floor.'
            ))

        except asyncio.CancelledError:
            raise

    async def horizontal_line_task(self):
        """
        Attempt to move the gantry in a perfectly horizontal line. How hard could this be?
        This is a motion task
        """
        await self.tension_and_wait()
        await asyncio.sleep(1)
        range_at_start = self.datastore.range_record.getLast()[1]
        result = await self.move_direction_speed([1,0,0], 0.2, downward_bias=0)
        await asyncio.sleep(4)
        self.slow_stop_all_spools()
        await asyncio.sleep(1)
        range_at_end = self.datastore.range_record.getLast()[1]
        print(f'During attempted horizontal move, height rose by {range_at_end - range_at_start} meters')

    async def park(self):
        """
        Drop item and park on the saddle for safe power down. 
        
        This procedure is absolutely naieve and never going to work reliably with feedback at every step.
        Probably a specific network can be trained on the gripper camera to produce the needed feedback.
        """
        # open gripper
        asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': -30}))

        # move over saddle.
        # TODO Since the saddle can be high and close to the wall, we may want to slow down signifigantly before we get there.
        winch_len = self.datastore.winch_line_record.getLast()[1]
        self.gantry_goal_pos = fromnp(self.config.saddle_position) + np.array([0,0, winch_len + 0.1])
        await self.seek_gantry_goal()

        # slowly lower onto saddle. How do we know when we hit it?
        # gripper camera goes black
        r = await self.move_direction_speed(np.array([0,0,-0.05]))
        await asyncio.sleep(0.1)
        await self.stop_all()

        # close gripper enough to clamp onto saddle
        asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': 10}))


    async def unpark(self):
        """ Unpark from the saddle and move clear of it. """
        pass
        # tighten all

        # open grip

        # move up

        # move about a meter towards origin while extending winch to operating length

        # half cal

    def on_service_state_change(self, 
        zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange
    ) -> None:
        if 'cranebot' in name:
            print(f"Service {name} of type {service_type} state changed: {state_change}")
            if state_change is ServiceStateChange.Added:
                asyncio.create_task(self.add_service(zeroconf, service_type, name))
            if state_change is ServiceStateChange.Updated:
                asyncio.create_task(self.update_service(zeroconf, service_type, name))
            if state_change is ServiceStateChange.Removed:
                asyncio.create_task(self.remove_service(service_type, name))
            elif state_change is ServiceStateChange.Updated:
                pass

    async def add_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        """Records the information about a discovered service in the config"""
        info = AsyncServiceInfo(service_type, name)
        await info.async_request(zc, INFO_REQUEST_TIMEOUT_MS)
        if not info or info.server is None or info.server == '':
            return None;
        namesplit = name.split('.')
        kind = namesplit[1]
        key  = ".".join(namesplit[:3])

        print(f"Service discovered: {info}")
        address = socket.inet_ntoa(info.addresses[0])

        is_power_anchor = kind == anchor_power_service_name
        is_standard_anchor = kind == anchor_service_name
        is_standard_gripper = kind == gripper_service_name
        is_arp_gripper = kind == arp_gripper_service_name

        if is_power_anchor or is_standard_anchor:
            # the number of anchors is decided ahead of time (in main.py)
            # but they are assigned numbers as we find them on the network
            # and the chosen numbers are persisted in configuration.json
            anchor_num_map = {a.service_name: a.num for a in self.config.anchors if a.service_name is not None}
            if key in anchor_num_map:
                anchor_num = anchor_num_map[key]
            else:
                anchor_num = len(anchor_num_map)
                if anchor_num >= N_ANCHORS:
                    # Discovering more that four anchors could be a sign that another robot in the same network is turned on.
                    # We need a way to know that, but for now, you'll have to make sure only one is one at a time while discovering.
                    # After discovery, it should be ok to have more than one on at a time.
                    print(f"Discovered another anchor server on the network, but we already know of 4 {key} {address}")
                    return None
            self.config.anchors[anchor_num].num = anchor_num
            self.config.anchors[anchor_num].service_name = key
            self.config.anchors[anchor_num].address = address
            self.config.anchors[anchor_num].port = info.port
            save_config(self.config, self.config_path)

        elif is_standard_gripper or is_arp_gripper:
            # a gripper has been discovered, assume it is ours only if we have never seen one before
            if self.config.gripper.service_name is None:
                self.config.gripper.service_name = key
                self.config.gripper.address = address
                self.config.gripper.port = info.port
                save_config(self.config, self.config_path)
                print(f'Discovered gripper at "{address}" and adopted it as the gripper for this robot')
            elif address != self.config.gripper.address:
                print(f'Discovered gripper at "{address}" and ignored it because ours is at {self.config.gripper.address}')

    async def update_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        # when zerconf has detected a change in address or port
        pass

    async def remove_service(self, service_type: str, name: str) -> None:
        """
        Finds if we have a client connected to this service. if so, ends the task if it is running, and deletes the client
        """
        namesplit = name.split('.')
        kind = namesplit[1]
        key  = ".".join(namesplit[:3])
        print(f'Removing service {key} from {self.bot_clients.keys()}')

        # only in this dict if we are connected to it.
        if key in self.bot_clients:
            client = self.bot_clients[key]
            await client.shutdown()
            if kind == anchor_service_name or kind == anchor_power_service_name:
                self.anchors.remove(client)
            elif kind == gripper_service_name:
                self.gripper_client = None
            del self.bot_clients[key]

    async def keep_robot_connected(self):
        """
        Keep a connection open to every robot component known in the config
        components are keyed by their service name which is the first three components of info.name, eg
        123.cranebot-anchor-service.2ccf67bc3fc4
        """
        # sleep until there is something to do
        while not config_has_any_address(self.config) and self.run_command_loop:
            await asyncio.sleep(0.5)
        print('Config not empty, begin connecting to discovered components')

        while self.run_command_loop:
            # is everything up the way we want it to be?
            if len([b for b in self.bot_clients.values() if b.connected])==5:
                await asyncio.sleep(0.5)
                continue # All websocket connections are up.

            # make sure we have either a live connection to, or an ongoing attempt to connect to every component we know about.
            for cpt in [self.config.gripper, *self.config.anchors]:
                # assume only the common attributes between those two types
                key = cpt.service_name
                if key is None or cpt.address is None or cpt.port is None:
                    continue

                if key not in self.connection_tasks:
                    # Start a connection to this component. connect_component will also remove it when it completes regardless of success or failure.
                    self.connection_tasks[key] = asyncio.create_task(self.connect_component(key))

            await asyncio.sleep(0.5)
        for task in self.connection_tasks.values():
            task.cancel()
        result = await asyncio.gather(*self.connection_tasks.values())

    async def connect_component(self, service_name):
        """Connect to the component with the given name using the address stored in the config."""
        client = None
        try:
            name_component = service_name.split('.')[1]
        except IndexError:
            print(f'invalid service name "{service_name}"')
            return

        is_power_anchor = name_component == anchor_power_service_name
        is_standard_anchor = name_component == anchor_service_name
        is_standard_gripper = name_component == gripper_service_name
        is_arp_gripper = name_component == arp_gripper_service_name

        if is_standard_gripper:
            client = RaspiGripperClient(self.config.gripper.address, self.config.gripper.port, self.datastore, self, self.pool, self.stat, self.pe, self.telemetry_env)
            self.gripper_client_connected.clear()
            client.connection_established_event = self.gripper_client_connected
            self.gripper_client = client
            self.pe.set_gripper_type('pilot')
        if is_arp_gripper:
            client = ArpeggioGripperClient(self.config.gripper.address, self.config.gripper.port, self.datastore, self, self.pool, self.stat, self.pe, self.telemetry_env)
            self.gripper_client_connected.clear()
            client.connection_established_event = self.gripper_client_connected
            self.gripper_client = client
            self.pe.set_gripper_type('arp')
        else:
            for a in self.config.anchors:
                if a.service_name != service_name:
                    continue
                client = RaspiAnchorClient(a.address, a.port, a.num, self.datastore, self, self.pool, self.stat, self.telemetry_env)
                client.connection_established_event = self.any_anchor_connected
                self.anchors.append(client)

        if client:
            self.bot_clients[service_name] = client
            # this function runs as long as the client is connected and returns true if the client was forced to disconnect abnormally
            abnormal_close = await client.startup()
            print('observer: client.startup() has returned')
            # remove client
            r = await self.remove_service(None, service_name)
            if abnormal_close:
                self.send_ui(pop_message=telemetry.Popup(
                    message=f'lost connection to {service_name}'
                ))
                await self.stop_all()
            # delete this task from the dict as it ends, so keep_robot_connected will try agian. 
            del self.connection_tasks[service_name]

    async def connect_cloud_telemetry(self):
        ws_protocol_and_host = CONTROL_PLANE_LOCAL
        if self.telemetry_env == 'staging':
            ws_protocol_and_host = CONTROL_PLANE_STAGING
        if self.telemetry_env == 'production':
            ws_protocol_and_host = CONTROL_PLANE_PRODUCTION

        while self.run_command_loop:
            try:
                use_id = self.config.robot_id
                ws_path = f"{ws_protocol_and_host}/telemetry/{use_id}"
                print(f"Connecting to control plane at {ws_path}")
                async with websockets.connect(ws_path, max_size=None, open_timeout=10) as websocket:
                    self.cloud_telem_websocket = websocket
                    print(f'connected to control_plane {websocket}')
                    # send anything that it would need up-front
                    await self.send_setup_telemetry()
                    try:
                        async for message in websocket:
                            r = await self.handle_command(message)
                            if not self.run_command_loop:
                                r = await websocket.close()
                    except ConnectionClosedOK as e:
                        pass
                    except ConnectionClosedError as e:
                        print(e)
                    finally:
                        print(f'disconnected from control_plane {websocket}')
                        self.cloud_telem_websocket = None
            except (asyncio.exceptions.CancelledError, websockets.exceptions.ConnectionClosedOK):
                pass # normal close
            except ConnectionRefusedError:
                print(f'Connection to control plane refused')
            except websockets.exceptions.InvalidMessage:
                print('Connection to control plane ended due to invalid message')
            await asyncio.sleep(2)

    def send_ui(self, **kwargs):
        """
        Ensure that the given telemetry item is sent to every connected UI
        keyword args are passed directly to telemetry item, so you can construct one like this

        self.send_ui(pop_message=telemetry.Popup('hello'))
        """
        if len(kwargs.keys()) != 1:
            raise ValueError
        key, msg = list(kwargs.items())[0]
        # if message is equal to a default instance of itself, dont send it.
        if msg == type(msg)():
            return

        # mark certain messages with a retain key. the server will resend them to new UIs
        item = telemetry.TelemetryItem(**kwargs)
        if key == 'new_anchor_poses':
            item.retain_key = 'new_anchor_poses'
        if key == 'component_conn_status':
            item.retain_key = f'component_conn_status_{msg.anchor_num}'
        if key == 'video_ready':
            item.retain_key = f'video_ready_{msg.anchor_num}'

        # Add item to batch
        with self.telemetry_buffer_lock:
            self.telemetry_buffer.append(item)

    async def flush_tele_buffer(self):
        """
        Flush the teloperation buffer. sending all data to all UI clients.
        Normally called within position estimator's 60hz loop
        """
        with self.telemetry_buffer_lock:
            batch = telemetry.TelemetryBatchUpdate(
                robot_id="0",
                updates=list(self.telemetry_buffer)
            )
            self.telemetry_buffer.clear()
        to_send = bytes(batch)
        # copy list to prevent RuntimeError: Set changed size during iteration
        connected_clients = self.connected_local_clients.copy()
        if self.cloud_telem_websocket:
            connected_clients.add(self.cloud_telem_websocket) # will only be connected when self.telemetry_env is not None
        for ui_websocket in connected_clients:
            try:
                r = await ui_websocket.send(to_send)
            except (ConnectionClosedOK, ConnectionClosedError) as e:
                pass # stale connection

    async def start_pe_when_ready(self):
        await self.any_anchor_connected.wait()
        r = await self.pe.main()

    async def main(self) -> None:
        self.startup_complete.clear()

        # broken
        self.passive_safety_task = asyncio.create_task(self.passive_safety())

        if self.telemetry_env is not None:
            self.cloud_telem = asyncio.create_task(self.connect_cloud_telemetry())

        # statistic counter - measures things like average camera frame latency
        asyncio.create_task(self.stat.stat_main())

        # A task that continuously estimates the position of the gantry
        # remains asleep until at least one anchor connects.
        self.pe_task = asyncio.create_task(self.start_pe_when_ready())

        # main process must own pool, and there's only one. multiple subprocesses may submit work.
        with Pool(processes=5) as pool:
            self.pool = pool

            # zeroconf only discovers services and keeps their addresses and ports up to date in the config.
            # start a task to connect and reconnect to all known robot components.
            self.keeper = asyncio.create_task(self.keep_robot_connected())

            # the only reason it might not be none is if a unit test set before calling main.
            if self.aiozc is None:
                self.aiozc = AsyncZeroconf(ip_version=IPVersion.V4Only, interfaces=InterfaceChoice.All)

            try:
                print("get services list")
                services = list(
                    await AsyncZeroconfServiceTypes.async_find(aiozc=self.aiozc, ip_version=IPVersion.V4Only)
                )
                print("start service browser")
                self.aiobrowser = AsyncServiceBrowser(
                    self.aiozc.zeroconf, services, handlers=[self.on_service_state_change]
                )
            except asyncio.exceptions.CancelledError:
                await self.aiozc.async_close()
                return

            # perception model
            # task remains in a lightweight sleep until frames arrive.
            asyncio.create_task(self.run_perception())

            # start a websocket server to accept incoming connection from local ursia UI process
            async with websockets.serve(self.handle_local_client, "127.0.0.1", 4245):
                # await something that will end when the program closes to keep serving and
                # keep zeroconf alive and discovering services.
                try:
                    self.startup_complete.set()
                    result = await self.keeper
                except asyncio.exceptions.CancelledError:
                    pass
            await self.async_close()

    async def async_close(self) -> None:
        result = await self.stop_all()
        self.run_command_loop = False
        self.stat.run = False
        self.pe.run = False
        self.pe_task.cancel()
        tasks = [self.pe_task, self.keeper, self.passive_safety_task]
        if self.cloud_telem:
            self.cloud_telem.cancel()
            tasks.append(self.cloud_telem)
        if self.grpc_server is not None:
            tasks.append(self.grpc_server.stop(grace=5))
        if self.aiobrowser is not None:
            tasks.append(self.aiobrowser.async_cancel())
        if self.aiozc is not None:
            tasks.append(self.aiozc.async_close())
        if self.sim_task is not None:
            tasks.append(self.sim_task)
        if self.locate_anchor_task is not None:
            tasks.append(self.locate_anchor_task)
        if self.gip_task is not None:
            tasks.append(self.gip_task)

        tasks.extend([client.shutdown() for client in self.bot_clients.values()])
        try:
            result = await asyncio.gather(*tasks)
        except asyncio.exceptions.CancelledError:
            pass

    async def add_simulated_data_circle(self):
        """ Simulate the gantry moving in a circle"""
        TIME_DIVISOR_FOR_ANGLE = 8.0
        GANTRY_Z_HEIGHT = 1.3 # meters
        RANDOM_EVENT_CHANCE = 0.5
        RANDOM_NOISE_MAGNITUDE = 0.1 # meters
        WINCH_LINE_LENGTH = 1.0 # meters
        RANGEFINDER_OFFSET = 1.0 # meters
        LOOP_SLEEP_S = 0.05 # seconds
        
        while self.run_command_loop:
            try:
                t = time.time()
                gantry_real_pos = np.array([t, np.sin(t/TIME_DIVISOR_FOR_ANGLE), np.cos(t/TIME_DIVISOR_FOR_ANGLE), GANTRY_Z_HEIGHT])
                if random() > RANDOM_EVENT_CHANCE:
                    anum = anchor_num = np.random.randint(4)
                    dp = gantry_real_pos + np.array([t, anum, random()*RANDOM_NOISE_MAGNITUDE, random()*RANDOM_NOISE_MAGNITUDE, random()*RANDOM_NOISE_MAGNITUDE])
                    self.datastore.gantry_pos.insert(dp)
                    self.send_ui(gantry_sightings=telemetry.GantrySightings(sightings=[fromnp(dp[1:])]))
                # winch line always 1 meter
                self.datastore.winch_line_record.insert(np.array([t, WINCH_LINE_LENGTH, 0.0]))
                # range always perfect
                self.datastore.range_record.insert(np.array([t, gantry_real_pos[3]-RANGEFINDER_OFFSET]))
                # anchor lines always perfectly agree with gripper position
                for i, simanc in enumerate(self.pe.anchor_points):
                    dist = np.linalg.norm(simanc - gantry_real_pos[1:])
                    last = self.datastore.anchor_line_record[i].getLast()
                    timesince = t-last[0]
                    travel = dist-last[1]
                    speed = travel/timesince
                    self.datastore.anchor_line_record[i].insert(np.array([t, dist, speed, 1.0]))
                tt = self.datastore.anchor_line_record[0].getLast()[0]
                await asyncio.sleep(LOOP_SLEEP_S)
            except asyncio.exceptions.CancelledError:
                break

    async def add_simulated_data_point2point(self):
        """Simulate the gantry moving from random point to random point.
        The only purpose of this simulation at the moment is to test the position estimator and it's feedback
        """
        LOWER_Z_BOUND = 1.0 # meters
        UPPER_Z_OFFSET = 0.3 # meters
        MAX_SPEED_MPS = 0.25 # m/s
        GOAL_PROXIMITY_THRESHOLD = 0.03 # meters
        SOFT_SPEED_FACTOR = 0.25
        RANDOM_EVENT_CHANCE = 0.5
        CAM_BIAS_STD_DEV = 0.2 # meters
        OBSERVATION_NOISE_STD_DEV = 0.01 # meters
        WINCH_LINE_LENGTH = 1.0 # meters
        RANGEFINDER_OFFSET = 1.0 # meters
        LOOP_SLEEP_S = 0.05 # seconds
        
        # each camera produces measurements with a position bias that can be around 20x larger than the position noise from a given camera.
        cam_bias = np.random.normal(0, CAM_BIAS_STD_DEV, (4, 3))

        pending_obs = deque()

        lower = np.min(self.pe.anchor_points, axis=0)
        upper = np.max(self.pe.anchor_points, axis=0)
        lower[2] = LOWER_Z_BOUND
        upper[2] = upper[2] - UPPER_Z_OFFSET
        # starting position
        gantry_real_pos = np.random.uniform(lower, upper)
        # initial goal
        travel_goal = np.random.uniform(lower, upper)
        t = time.time()
        while self.run_command_loop:
            try:
                now = time.time()
                elapsed_time = now - t
                t = now
                # move the gantry towards the goal
                to_goal_vec = travel_goal - gantry_real_pos
                dist_to_goal = np.linalg.norm(to_goal_vec)
                if dist_to_goal < GOAL_PROXIMITY_THRESHOLD:
                    # choose new goal
                    travel_goal = np.random.uniform(lower, upper)
                else:
                    soft_speed = dist_to_goal * SOFT_SPEED_FACTOR
                    # normalize
                    to_goal_vec = to_goal_vec / dist_to_goal
                    velocity = to_goal_vec * min(soft_speed, MAX_SPEED_MPS)
                    gantry_real_pos = gantry_real_pos + velocity * elapsed_time
                if random() > RANDOM_EVENT_CHANCE:
                    anchor_num = np.random.randint(4) # which camera it was observed from.
                    observed_position = gantry_real_pos + cam_bias[anchor_num] + np.random.normal(0, OBSERVATION_NOISE_STD_DEV, (3,))
                    dp = np.concatenate([[t], [anchor_num], observed_position])
                    # simulate delayed data
                    pending_obs.appendleft(dp)
                    if len(pending_obs) > 10:
                        dp = pending_obs.pop()
                        self.datastore.gantry_pos.insert(dp)
                        self.datastore.gantry_pos_event.set()
                        self.send_ui(gantry_sightings=telemetry.GantrySightings(sightings=[fromnp(dp[2:])]))
                
                # winch line always 1 meter
                self.datastore.winch_line_record.insert(np.array([t, WINCH_LINE_LENGTH, 0.0]))
                
                # range always perfect
                self.datastore.range_record.insert(np.array([t, gantry_real_pos[2]-RANGEFINDER_OFFSET]))

                # anchor lines always perfectly agree with gripper position
                for i, simanc in enumerate(self.pe.anchor_points):
                    dist = np.linalg.norm(simanc - gantry_real_pos)
                    last = self.datastore.anchor_line_record[i].getLast()
                    timesince = t-last[0]
                    travel = dist-last[1]
                    speed = travel/timesince # referring to the specific speed of this line, not the gantry
                    self.datastore.anchor_line_record[i].insert(np.array([t, dist, speed, 1.0]))
                    self.datastore.anchor_line_record_event.set()
                tt = self.datastore.anchor_line_record[0].getLast()[0]
                await asyncio.sleep(LOOP_SLEEP_S)
            except asyncio.exceptions.CancelledError:
                break

    def collect_gant_frame_positions(self):
        result = np.zeros((4,3))
        for client in self.anchors:
            result[client.anchor_num] = client.last_gantry_frame_coords
        return result

    async def send_gripper_move(self, line_speed, finger_angle, wrist_angle):
        """Command the gripper's motors in one update."""
        update = {}
        if line_speed is not None:
            update['aim_speed'] = line_speed
        if finger_angle is not None:
            update['set_finger_angle'] = clamp(finger_angle, -90, 90)
        if wrist_angle is not None:
            update['set_wrist_angle'] = wrist_angle
        if update and self.gripper_client is not None:
            asyncio.create_task(self.gripper_client.send_commands(update))
        return line_speed, finger_angle, wrist_angle

    async def clear_gantry_goal(self):
        self.gantry_goal_pos = None
        self.send_ui(named_position=telemetry.NamedObjectPosition(name='gantry_goal_marker')) # not setting position causes it to be hidden

    async def seek_gantry_goal(self):
        """
        Move towards a goal position, using the constantly updating gantry position provided by the position estimator
        This is a motion task
        """
        GOAL_PROXIMITY_M = 0.07 # meters
        GANTRY_SPEED_MPS = 0.24 # m/s
        LOOP_SLEEP_S = 0.2 # seconds
        
        try:
            self.send_ui(named_position=telemetry.NamedObjectPosition(position=fromnp(self.gantry_goal_pos), name='gantry_goal_marker'))
            while self.gantry_goal_pos is not None:
                vector = self.gantry_goal_pos - self.pe.gant_pos
                dist = np.linalg.norm(vector)
                if dist < GOAL_PROXIMITY_M:
                    break
                vector = vector / dist
                result = await self.move_direction_speed(vector, GANTRY_SPEED_MPS, self.pe.gant_pos)
                await asyncio.sleep(LOOP_SLEEP_S)
        except asyncio.CancelledError:
            raise
        finally:
            self.slow_stop_all_spools()
            await self.clear_gantry_goal()

    async def blind_move_to_goal(self):
        """Only measures current position once, then moves in the right direction
        for the amount of time it should take to get to the goal.
        This is a motion task."""
        GANTRY_SPEED_MPS = 0.25 # m/s
        
        try:
            self.send_ui(named_position=telemetry.NamedObjectPosition(position=fromnp(self.gantry_goal_pos), name='gantry_goal_marker'))
            vector = self.gantry_goal_pos - self.pe.gant_pos
            dist = np.linalg.norm(vector)
            if dist > 0:
                result = await self.move_direction_speed(vector / dist, GANTRY_SPEED_MPS, self.pe.gant_pos)
                await asyncio.sleep(dist / GANTRY_SPEED_MPS)
        except asyncio.CancelledError:
            raise
        finally:
            self.slow_stop_all_spools()
            await self.clear_gantry_goal()

    async def move_direction_speed(self, uvec, speed=None, starting_pos=None, downward_bias=-0.04):
        """Move in the direction of the given unit vector at the given speed.
        Any move must be based on some assumed starting position. if none is provided,
        we will use the last one sent from position_estimator

        Due to inaccuaracy in the positions of the anchors and lengths of the lines,
        the speeds we command from the spools will not be perfect.
        On average, half will be too high, and half will be too low.
        Because there are four lines and the gantry only hangs stably from three,
        the actual point where the gantry ends up hanging after any move will always be higher than intended
        So a small downward bias is introduced into the requested direction to account for this.
        The size of the bias should theoretically be a function of the the magnitude of position and line errors,
        but we don't have that info. alternatively we could calibrate the bias to make horizontal movements level
        according to the laser rangefinder.

        if speed is None, uvec is assumed to be velocity and used directly with no bias
        """
        KINEMATICS_STEP_SCALE = 10.0 # Determines the size of the virtual step to calculate line speed derivatives

        if starting_pos is None:
            starting_pos = self.pe.gant_pos

        # when speed is not provided, use uvec as a velocity vector in m/s (mode used with lerobot)
        if speed is None:
            speed = np.linalg.norm(uvec)

        # Enforce a height dependent speed limit.
        # the reason being that as gantry height approaches anchor height, the line tension increases exponentially,
        # and a slower speed is need to maintain enough torque from the stepper motors.
        # The speed limit is proportional to how far the gantry hangs below a level 10cm below the average anchor.
        # This makes the behavior consistent across installations of different heights.
        hang_distance = np.mean(self.pe.anchor_points[:, 2]) - starting_pos[2]
        speed_limit = clamp(0.28 * (hang_distance - 0.1), 0.01, 0.55)
        speed = min(speed, speed_limit)

        # when a very small speed is provided, clamp it to zero.
        if speed < 0.005:
            speed = 0

        if speed == 0:
            for client in self.anchors:
                asyncio.create_task(client.send_commands({'aim_speed': 0}))
            velocity = np.zeros(3)
            self.pe.record_commanded_vel(velocity)
            return velocity

        # normalize, apply downward bias and renormalize
        uvec  = uvec / np.linalg.norm(uvec)
        uvec = uvec + np.array([0,0,downward_bias])
        uvec  = uvec / np.linalg.norm(uvec)
        velocity = uvec * speed

        anchor_positions = np.zeros((4,3))
        for a in self.anchors:
            anchor_positions[a.anchor_num] = np.array(a.anchor_pose[1])

        # line lengths at starting pos
        lengths_a = np.linalg.norm(starting_pos - self.pe.anchor_points, axis=1)
        # line lengths at new pos
        new_pos = starting_pos + (uvec / KINEMATICS_STEP_SCALE)
        # zero the speed if this would move the gantry out of the work area
        if not self.pe.point_inside_work_area(new_pos):
            speed = 0
            velocity = np.zeros(3)
        lengths_b = np.linalg.norm(new_pos - self.pe.anchor_points, axis=1)
        # length changes needed to travel a small distance in uvec direction from starting_pos
        deltas = lengths_b - lengths_a
        line_speeds = deltas * KINEMATICS_STEP_SCALE * speed

        # send move
        for client in self.anchors:
            asyncio.create_task(client.send_commands({'aim_speed': line_speeds[client.anchor_num]}))
        self.pe.record_commanded_vel(velocity)
        return velocity

    def get_last_frame(self, camera_key):
        """gets the last frame of video from the given camera if possible
        camera_key should be one of 'g' 0, 1, 2, 3
        """
        image = None
        if camera_key == 'g':
            if self.gripper_client is not None:
                image = self.gripper_client.lerobot_jpeg_bytes
        else:
            anum = int(camera_key)
            for client in self.anchors:
                if client.anchor_num == anum:
                    image = client.lerobot_jpeg_bytes
        if image is not None:
            return image
        return bytes()

    def get_episode_control_events(self):
        e = list(self.episode_control_events)
        self.episode_control_events.clear()
        return e

    def _handle_add_episode_control_events(self, data: control.EpControl):
        for k in data.events:
            self.episode_control_events.add(str(k))

    def send_tq_to_ui(self):
        snapshot = self.target_queue.get_queue_snapshot()
        # Create a deterministic hash
        current_hash = hash(bytes(snapshot))
        if current_hash != self.last_snapshot_hash:
            self.send_ui(target_list=snapshot)
            self.last_snapshot_hash = current_hash

    async def run_perception(self):
        """
        Run the target heatmap network on preferred cameras at a modest rate.
        Send heatmaps to UI.
        Store target candidates and confidence.
        """
        TARGET_HM_MODEL_PATH = files("nf_robot.ml").joinpath("models/target_heatmap.pth")
        CENTERING_MODEL_PATH = files("nf_robot.ml").joinpath("models/square_centering.pth")
        DEVICE = "cpu"
        LOOP_DELAY = 0.1
        FIND_TARGETS_EVERY = 5 # loops

        # do nothing until at least one camera from the preferred set is producing frames
        have_gripper_frames = False
        have_anchor_frames = False
        while not (have_gripper_frames or have_anchor_frames):
            await asyncio.sleep(1)
            have_gripper_frames = self.gripper_client is not None and self.gripper_client.last_frame_resized is not None
            for client in self.anchors:
                if client.anchor_num in self.config.preferred_cameras and client.last_frame_resized is not None:
                    have_anchor_frames = True

        if self.target_model is None:
            import torch
            from nf_robot.ml.target_heatmap import extract_targets_from_heatmap, TargetHeatmapNet, HM_IMAGE_RES
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading model from {TARGET_HM_MODEL_PATH}...")
            self.target_model = TargetHeatmapNet().to(DEVICE)
            self.target_model.load_state_dict(torch.load(TARGET_HM_MODEL_PATH, map_location=DEVICE))
            self.target_model.eval()

        if self.centering_model is None:
            import torch
            from nf_robot.ml.centering import CenteringNet
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading model from {CENTERING_MODEL_PATH}...")
            self.centering_model = CenteringNet().to(DEVICE)
            self.centering_model.load_state_dict(torch.load(CENTERING_MODEL_PATH, map_location=DEVICE))
            self.centering_model.eval()

        if self.telemetry_env == 'local':
            print(f'To control visit http://localhost:5173/playroom?robotid={self.config.robot_id}')
        if self.telemetry_env == 'staging':
            print(f'To control visit https://nf-site-monolith-staging-690802609278.us-east1.run.app/playroom?robotid={self.config.robot_id}')
        if self.telemetry_env == 'production':
            print(f'To control visit https://neufangled.com/playroom?robotid={self.config.robot_id}')

        counter = 0
        while self.run_command_loop:
            await asyncio.sleep(LOOP_DELAY)
            counter += 1

            if self.gripper_client is not None and self.gripper_client.last_frame_resized is not None:

                # network was trained on BGR
                bgr_image = cv2.cvtColor(self.gripper_client.last_frame_resized, cv2.COLOR_BGR2RGB)
                gripper_image_tensor = torch.from_numpy(bgr_image).permute(2, 0, 1).float() / 255.0
                if gripper_image_tensor is not None:
                    gripper_image_tensor = gripper_image_tensor.unsqueeze(0).to(DEVICE) # Add batch dimension
                    with torch.no_grad():
                        pred_vec, pred_valid, pred_grip, grip_angle = self.centering_model(gripper_image_tensor)
                        vec = pred_vec[0].cpu().numpy()
                        self.gripper_sees_target = pred_valid[0].item()
                        self.gripper_sees_holding = pred_grip[0].item()

                        # you get a normalized u,v coordinate in the [-1,1] range
                        self.predicted_lateral_vector = vec if self.gripper_sees_target > 0.5 else np.zeros(2)
                        self.send_ui(grip_cam_preditions=telemetry.GripCamPredictions(
                            move_x = self.predicted_lateral_vector[0],
                            move_y = self.predicted_lateral_vector[1],
                            prob_target_in_view = self.gripper_sees_target,
                            prob_holding = self.gripper_sees_holding,
                        ))

            if counter < FIND_TARGETS_EVERY:
                continue
            counter = 0

            # collect images from any anchors that have one
            valid_anchor_clients = []
            img_tensors = []
            for client in self.anchors:
                if client.last_frame_resized is None or client.anchor_num not in self.config.preferred_cameras:
                    continue
                # these are already assumed to be at the correct resolution 
                bgr_image = cv2.cvtColor(client.last_frame_resized, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(bgr_image).permute(2, 0, 1).float() / 255.0
                img_tensors.append(img_tensor)
                valid_anchor_clients.append(client)
            if img_tensors:
                # run batch inference on GPU and get targets
                all_floor_target_arrs = []
                batch = torch.stack(img_tensors).to(DEVICE)
                with torch.no_grad():
                    heatmaps_out = self.target_model(batch)

                # Shape: (Batch, 1, H, W) -> (Batch, H, W)
                heatmaps_np = heatmaps_out.squeeze(1).cpu().numpy() # this is the blocking call
                for i, heatmap_np in enumerate(heatmaps_np):
                    client = valid_anchor_clients[i]
                    results = extract_targets_from_heatmap(heatmap_np)

                    if len(results) > 0:
                        targets2d = results[:,:2] # the third number is confidence
                        # if this is an anchor, project points to floor using anchor's specific pose
                        floor_points = project_pixels_to_floor(targets2d, client.camera_pose, self.config.camera_cal)
                        all_floor_target_arrs.append(floor_points)
                        # TODO retain information about the original image coordinates of targets for display in UI

                if len(all_floor_target_arrs) > 0:
                    # filter out targets that are not inside the work area.
                    floor_targets = [
                        {'position': np.array([p[0], p[1], 0]), 'dropoff': 'hamper'}
                        for p in np.concatenate(all_floor_target_arrs)
                        if self.pe.point_inside_work_area_2d(p)
                    ]
                else:
                    floor_targets = []
                # add any floor targets dicovered during this batch to the target queue.
                self.target_queue.add_ai_targets(floor_targets)
                self.send_tq_to_ui()

    async def pick_and_place_loop(self):
        """
        Long running motion task that repeatedly identifies targets picks them up and drops them over the hamper
        """
        GANTRY_HEIGHT_OVER_TARGET = 1.0
        GANTRY_HEIGHT_OVER_DROPOFF = 0.9
        RELAXED_OPEN = 0 # enough to drop something
        DELAY_AFTER_DROP = 0.6 # long enough that the payload is not visible anymore in the hand
        LOOP_DELAY = 0.5

        try:
            gtask = None
            while self.run_command_loop:

                # hover over the hamper
                # await asyncio.sleep(1)
                # if 'hamper' in self.named_positions:
                #     self.gantry_goal_pos = self.named_positions['hamper'] + np.array([0,0,GANTRY_HEIGHT_OVER_DROPOFF])
                #     await self.seek_gantry_goal()
                # continue

                next_target = self.target_queue.get_best_target()
                if next_target is None:
                    print('no target was found, be still')
                    if gtask is not None:
                        gtask.cancel()
                    self.gantry_goal_pos = None
                    # TODO park on the saddle.
                    await asyncio.sleep(LOOP_DELAY)
                    continue

                print(f'selected target {next_target.id} with dropoff {next_target.dropoff}')

                self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.SELECTED)
                self.send_tq_to_ui()

                # pick Z position for gantry
                goal_pos = next_target.position + np.array([0, 0, GANTRY_HEIGHT_OVER_TARGET])
                self.gantry_goal_pos = goal_pos

                # gantry is now heading for a position over next_target
                # wait only one second for it to arrive.
                try:
                    if gtask is None or gtask.done():
                        gtask = asyncio.create_task(self.seek_gantry_goal())
                    await asyncio.wait_for(gtask, 1)
                    print('arrived at target')
                except TimeoutError:
                    # if doesn't arrive in one second, run target selection again since a better one might have appeared or the user might have put one in their queue
                    print('did not arrive yet')
                    self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.SEEN)
                    continue

                if self.gripper_client is None:
                    print('pick and place aborted because we lost the gripper connection')
                    break

                # when we reach this point we arrived over the item. commit to it unless it proves impossible to pick up.
                print('attempt grasp')
                start = time.time()
                success = await self.execute_grasp()
                print(f'grasp succeeded {success} took {time.time() - start}s')
                if not success:
                    # just pick another target, but consider downranking this object or something.
                    self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.SEEN)
                    self.send_tq_to_ui()
                    await asyncio.sleep(LOOP_DELAY)
                    continue
                else:
                    self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.PICKED_UP)
                    self.send_tq_to_ui()
                    print('object picked up')

                # tension now just in case.
                # await self.tension_and_wait()

                # If user specified drop point...
                if not isinstance(next_target.dropoff, str):
                    drop_point = next_target.dropoff
                # otherwise go to the named drop point
                if next_target.dropoff in self.named_positions:
                    drop_point = self.named_positions[next_target.dropoff]
                else:
                    # otherwise use the origin as a drop point :/
                    # TODO this is not ideal, as we will continue to pick things up from this spot most likely now that we are close to it.
                    # either need to drop it somewhere we know we won't ever see it again, or have a sign for this drop point so we don't touch things inside it.
                    print("No drop point specified, using (0,0,0) as a drop point")
                    drop_point = np.zeros(3)

                # fly to to drop point
                print(f'flying to drop point {drop_point}')
                self.gantry_goal_pos = drop_point + np.array([0,0,GANTRY_HEIGHT_OVER_DROPOFF])
                await self.seek_gantry_goal()
                # open gripper
                asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': RELAXED_OPEN}))
                # don't immediately select a new target, because there's a chance it'll be the sock you're holding.
                # TODO train network on more data containing examples of this, so it knows that only socks on the floor count.
                await asyncio.sleep(DELAY_AFTER_DROP)
                self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.DROPPED)
                self.send_tq_to_ui()
                # keep score


        except asyncio.CancelledError:
            raise
        finally:
            if gtask is not None:
                gtask.cancel()
            self.slow_stop_all_spools()
            await self.clear_gantry_goal()

    async def execute_grasp(self):
        """Try to grasp whatever is directly below the gripper"""
        OPEN = -30
        CLOSED = 85
        FINGER_LENGTH = 0.1 # length between rangefinder and floor when fingers touch in meters
        HALF_VIRTUAL_FOV = model_constants.rpi_cam_3_fov * SF_SCALE_FACTOR / 2 * (np.pi/180)
        DOWNWARD_SPEED = -0.06
        VISUAL_CONF_THRESHOLD = 0.1 # level below which we give up on the target
        COMMIT_HEIGHT = 0.3 # height below which giving up due to visual disconfidence is not allowed.
        LAT_TRAVEL_FRACTION = 0.75 # try to finish lateral travel by this fraction of the time spent travelling downwards
        LAT_SPEED_ADJUSTMENT = 5.00 # final adjustment to lateral speed
        LOOP_DELAY = 0.1
        PRESSURE_SENSE_WAIT = 2.0

        try:
            attempts = 3
            while not self.pe.holding and attempts > 0 and self.run_command_loop:
                attempts -= 1
                asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': OPEN}))

                # move laterally until target is centered
                # at the same time, move downward until tip is detected.

                nothing_seen_countdown = 15
                self.pe.tip_over.clear()
                while (self.predicted_lateral_vector is not None and not self.pe.tip_over.is_set()):
                    distance_to_floor = self.datastore.range_record.getLast()[1]
                    if distance_to_floor < FINGER_LENGTH:
                        print(f'stop going down, distance to floor is {distance_to_floor}')
                        break

                    if self.gripper_sees_target < VISUAL_CONF_THRESHOLD and distance_to_floor > COMMIT_HEIGHT:
                        nothing_seen_countdown -= 1
                        if nothing_seen_countdown == 0:
                            print('print nothing seen during centering loop')
                            break
                    else:
                        nothing_seen_countdown = 15
                    # calculate eta to the floor using laser range, we want to finish lateral travel at 0.75 of that eta
                    lat_travel_seconds = (distance_to_floor-FINGER_LENGTH)/(-DOWNWARD_SPEED)*LAT_TRAVEL_FRACTION
                    if lat_travel_seconds > 0:
                        # determine which direction we'd have to move laterally to center the object
                        # you get a normalized u,v coordinate in the [-1,1] range
                        # for now assume that the up direction in the gripper image is -Y in world space 
                        # stabilize_frame produced this direction and I think it depends on the compass.
                        # the direction in world space depends on how the user placed the origin card on the ground
                        # we need to capture a number during calibration to relate these two.
                        # +1 is the edge of the image. how far laterally that would be depends on how far from the ground the gripper is.
                        pred_vector = self.predicted_lateral_vector
                        pred_vector[1] *= -1
                        # lateral distance to object
                        lateral_vector = np.sin(pred_vector * HALF_VIRTUAL_FOV) * distance_to_floor
                        # lateral distance in meters
                        lateral_distance = np.linalg.norm(lateral_vector)
                        # speed to travel that lateral distance in lat_travel_seconds
                        lateral_speed = lateral_distance / lat_travel_seconds * LAT_SPEED_ADJUSTMENT
                    else:
                        # once we get too close, go straight down, stop relying on the camera
                        lateral_speed = 0
                    lateral_vector *= lateral_speed

                    await self.move_direction_speed([lateral_vector[0],lateral_vector[1],DOWNWARD_SPEED])

                    try:
                        # the normal sleep on this loop would be LOOP_DELAY s, but if tip is detected
                        # we want to stop immediately.
                        await asyncio.wait_for(self.pe.tip_over.wait(), LOOP_DELAY)
                        print('detected tip over, must be floor')
                        break
                    except TimeoutError:
                        pass

                self.slow_stop_all_spools()
                self.pe.tip_over.clear()

                if nothing_seen_countdown == 0:
                    print('Nothing seen')
                    continue # find new target?

                print('close gripper')
                await self.gripper_client.send_commands({'set_finger_angle': CLOSED})
                print(f'wait up to {PRESSURE_SENSE_WAIT} seconds for pad to sense object.')
                try:
                    await asyncio.wait_for(self.pe.finger_pressure_rising.wait(), PRESSURE_SENSE_WAIT)
                    self.pe.finger_pressure_rising.clear()
                except TimeoutError:
                    print('did not detect a successful hold, open and go back up high enough to get a view of the object')
                    # move up slowly at first, till fingers just touch ground and we are veritical. this keeps unwanted swinging to a minimum
                    await self.move_direction_speed([0,0,0.06])
                    await asyncio.sleep(1.0)
                    # now move up a little faster in a slightly random direction
                    direction = np.concatenate([np.random.uniform(-0.025, 0.025, (2)), [0.12]])
                    await self.move_direction_speed(direction)
                    asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': OPEN}))
                    await asyncio.sleep(2.0)
                    self.slow_stop_all_spools()
                    continue
                print('Successful grasp')
                return True
            print(f'Gave up on grasp after {attempts} attempts. self.pe.holding={self.pe.holding}')
            return False

        except asyncio.CancelledError:
            raise
        finally:
            self.slow_stop_all_spools()

    def _handle_collect_images(self):
        if self.run_collect_images:
            self.run_collect_images = False # ends the task
        else:
            self.run_collect_images = True
            self.gip_task = asyncio.create_task(self.collect_images())

    async def collect_images(self):
        """Collects data for the centering network"""
        while self.run_command_loop and self.run_collect_images:
            if self.gripper_client.last_frame_resized is not None:
                print(self.gripper_client.last_frame_resized.shape)
                rgb_image = cv2.cvtColor(self.gripper_client.last_frame_resized, cv2.COLOR_BGR2RGB)
                capture_gripper_image(rgb_image, gripper_occupied=self.pe.holding)
            else:
                print('no resized frame available from gripper')
            await asyncio.sleep(1)

def start_observation(terminate_with_ui=False, config_path='configuration.json'):
    """Entry point to be used when starting this from main.py with multiprocessing."""
    ob = AsyncObserver(terminate_with_ui, config_path, telemetry_env=None)
    asyncio.run(ob.main())
    # set ob.run_command_loop = False or kill or connect to websocket and send shutdown command

def main():
    """
    Run stringman in a headless manner

    note that connecting to a local telemetry enviroment is distinct from lan mode
    when observer.py is run directly it is always connecting to some telemetry server
    even if it is the full stack running on the local machine
    in contrast, LAN mode is enabled by running main.py which starts the Ursina UI and observer.py together
    and where observer.py sends telemetry only to the ursina UI process.
    the lan mode entrypoint is start_observation() above

    typical run command for local testing

    python3 observer.py --config=conf_bedroom.json local

    """
    parser = argparse.ArgumentParser(description="Stringman motion controller")
    parser.add_argument("--config", type=str, default='configuration.json')
    parser.add_argument(
            '--telemetry_env',
            type=str,
            choices=['local', 'staging', 'production'],
            default='production',
            help="The telemetry server to connect to (choices: local, staging, production) Used in development only"
        )
    args = parser.parse_args()

    async def run_async():
        runner = AsyncObserver(False, args.config, telemetry_env=args.telemetry_env)

        # when running as a standalone process, register signal handler
        def stop():
            # Must be idempotent, may be called multiple times
            runner.run_command_loop = False
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(getattr(signal, 'SIGINT'), stop)
        result = await runner.main()

    asyncio.run(run_async())

if __name__ == "__main__":
    main()