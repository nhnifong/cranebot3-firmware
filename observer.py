from __future__ import annotations

import signal
import sys
import threading
import time
import socket
import asyncio
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
from data_store import DataStore
from raspi_anchor_client import RaspiAnchorClient, max_origin_detections
from raspi_gripper_client import RaspiGripperClient
from random import random
from config import Config
from stats import StatCounter
from position_estimator import Positioner2
from cv_common import invert_pose, compose_poses, average_pose, project_pixels_to_floor
from calibration import optimize_anchor_poses
import model_constants
import traceback
import cv2
import pickle
from collections import deque, defaultdict
from trainer.control_service import start_robot_control_server
from trainer.stringman_record_loop import record_until_disconnected
from trainer.centering_labler import capture_gripper_image
import multiprocessing

# Define the service names for network discovery
anchor_service_name = 'cranebot-anchor-service'
anchor_power_service_name = 'cranebot-anchor-power-service'
gripper_service_name = 'cranebot-gripper-service'

N_ANCHORS = 4
INFO_REQUEST_TIMEOUT_MS = 3000 # milliseconds

def figure_8_coords(t):
    """
    Calculates the (x, y) coordinates for a figure-8.
    figure fits within a box from -2 to +2
    Args:
        t: The input parameter (angle in radians, from 0 to 2*pi).
    Returns:
        A tuple (x, y) representing the position on the figure-8.
    """
    x = 2 * np.sin(t)
    y = 2 * np.sin(2 * t)
    return (x, y)

# Manager of multiple tasks running clients connected to each robot component
# The job of this class in a nutshell is to discover four anchors and a gripper on the network,
# connect to them, and forward data between them and the position estimator, shape tracker, and UI.
class AsyncObserver:
    def __init__(self, to_ui_q, to_ob_q) -> None:
        self.position_update_task = None
        self.aiobrowser: AsyncServiceBrowser | None = None
        self.aiozc: AsyncZeroconf | None = None
        self.run_command_loop = True
        self.calmode = "pause"
        self.detection_count = 0
        self.datastore = DataStore()
        self.to_ui_q = to_ui_q
        self.to_ob_q = to_ob_q
        self.pool = None
        # all clients by server name
        self.bot_clients = {}
        # all connected anchors
        self.anchors = []
        # convenience reference to gripper client
        self.gripper_client = None
        # The index of the anchor with the power-delivering line
        self.power_spool_index = None
        # read a mapping of server names to anchor numbers from the config file
        self.config = Config()
        self.config.write()
        self.stat = StatCounter(self.to_ui_q)
        self.enable_shape_tracking = False
        self.shape_tracker = None
        # Position Estimator. this used to be a seperate process so it's still somewhat independent.
        self.pe = Positioner2(self.datastore, self.to_ui_q, self)
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
        self.episode_control_events = set()
        self.named_positions = {}
        self.dobby_model = None
        self.floor_targets = []
        self.centering_model = None
        self.predicted_lateral_vector = None

        # Command dispatcher maps command strings to handler methods
        self.command_handlers = {
            'future_anchor_lines': self._handle_future_anchor_lines,
            'future_winch_line': self._handle_future_winch_line,
            'set_run_mode': self.set_run_mode,
            'do_line_calibration': self.sendReferenceLengths,
            'tension_lines': self.tension_lines,
            'full_cal': lambda _: self.invoke_motion_task(self.full_auto_calibration()),
            'half_cal': lambda _: self.invoke_motion_task(self.half_auto_calibration()),
            'jog_spool': self._handle_jog_spool,
            'toggle_previews': self._handle_toggle_previews,
            'gantry_dir_sp': self._handle_gantry_dir_sp,
            'gantry_goal_pos': self._handle_gantry_goal_pos,
            'fig_8': lambda _: self.invoke_motion_task(self.move_in_figure_8()),
            'set_grip': self._handle_set_grip,
            'slow_stop_one': self._handle_slow_stop_one,
            'slow_stop_all': self.stop_all,
            'set_simulated_data_mode': self.set_simulated_data_mode,
            'zero_winch': self._handle_zero_winch_line,
            'horizontal_task': lambda _: self.invoke_motion_task(self.horizontal_line_task()),
            'winch_and_finger': self._handle_send_winch_finger,
            'gamepad': self._handle_gamepad_action,
            'episode_ctrl': self._handle_add_episode_control_event,
            'avg_named_pos': self._handle_avg_named_pos,
            'lost_conn': self._handle_lost_conn,
        }

    def listen_queue_updates(self, loop):
        """
        Receive any updates on our process input queue

        this thread doesn't actually have a running event loop.
        so run any coroutines back in the main thread with asyncio.run_coroutine_threadsafe
        """
        while self.run_command_loop:
            updates = self.to_ob_q.get()
            if 'STOP' in updates:
                print('Observer shutdown')
                break
            else:
                asyncio.run_coroutine_threadsafe(self.process_update(updates), loop)

    async def process_update(self, updates: dict):
        """
        Processes incoming commands from the input queue using a dispatch table.
        This iterates through all commands in the 'updates' dictionary.
        """
        try:
            for command, data in updates.items():
                handler = self.command_handlers.get(command)
                if handler:
                    # The await handles both regular functions and coroutines
                    result = handler(data)
                    if asyncio.iscoroutine(result):
                        await result
                else:
                    print(f"Warning: No handler found for command '{command}'")
        except Exception:
            traceback.print_exc(file=sys.stderr)

    async def _handle_future_anchor_lines(self, fal_data: dict):
        """Handles sending future line plans to anchors."""
        if not (fal_data.get('sender') == 'pe' and self.calmode != 'run'):
            for client in self.anchors:
                message = {'length_plan': fal_data['data'][client.anchor_num].tolist()}
                if 'host_time' in fal_data:
                    message['host_time'] = fal_data['host_time']
                asyncio.create_task(client.send_commands(message))

    async def _handle_future_winch_line(self, fwl_data: dict):
        """Handles sending future winch line plans to the gripper."""
        if not (fwl_data.get('sender') == 'pe' and self.calmode != 'run'):
            if self.gripper_client:
                message = {'length_plan': fwl_data['data']}
                await self.gripper_client.send_commands(message)

    async def _handle_jog_spool(self, jog_data: dict):
        """Handles manually jogging a spool motor."""
        if 'anchor' in jog_data:
            for client in self.anchors:
                if client.anchor_num == jog_data['anchor']:
                    asyncio.create_task(client.send_commands({'aim_speed': jog_data['speed']}))
        elif 'gripper' in jog_data and self.gripper_client:
            asyncio.create_task(self.gripper_client.send_commands({'aim_speed': jog_data['speed']}))

    async def _handle_toggle_previews(self, preview_data: dict):
        """Handles turning camera previews on or off."""
        if 'anchor' in preview_data:
            for client in self.anchors:
                if client.anchor_num == preview_data['anchor']:
                    client.sendPreviewToUi = preview_data['status']
        elif 'gripper' in preview_data and self.gripper_client:
            self.gripper_client.sendPreviewToUi = preview_data['status']

    async def _handle_gantry_dir_sp(self, dir_sp_data: dict):
        """Handles a direct directional move command, cancelling other motion tasks."""
        await self.invoke_motion_task(self.move_direction_speed(dir_sp_data['direction'], dir_sp_data['speed']))

    async def _handle_gantry_goal_pos(self, goal_pos: np.ndarray):
        """Handles moving the gantry to a specific goal position."""
        self.gantry_goal_pos = goal_pos
        await self.invoke_motion_task(self.seek_gantry_goal())

    async def _handle_set_grip(self, grip_closed: bool):
        """Handles opening or closing the gripper."""
        if self.gripper_client:
            command = 'closed' if grip_closed else 'open'
            asyncio.create_task(self.gripper_client.send_commands({'grip': command}))

    async def _handle_slow_stop_one(self, stop_data: dict):
        """Handles stopping a single spool motor."""
        if stop_data.get('id') == 'gripper' and self.gripper_client:
            asyncio.create_task(self.gripper_client.slow_stop_spool())
        else:
            for client in self.anchors:
                if client.anchor_num == stop_data.get('id'):
                    asyncio.create_task(client.slow_stop_spool())

    async def _handle_zero_winch_line(self, data):
        await self.gripper_client.zero_winch()

    async def _handle_gamepad_action(self, data):
        # if we have to clip these values to legal limits, save what they were clipped to
        winch, finger = await self.send_winch_and_finger(data['winch'], data['finger'])
        commanded_vel = await self.move_direction_speed(data['dir'], data['speed'])
        # the saved values will be what we return from GetLastAction
        self.last_gp_action = (commanded_vel, winch, finger)

    async def _handle_avg_named_pos(self, data):
        """Keep running averages of named positions"""
        (key, position) = data
        if key not in self.named_positions:
            self.named_positions[key] = position
        self.named_positions[key] = self.named_positions[key] * 0.75 + position * 0.25

        if key=='gamepad':
            # UI needs to know about this one
            p2 = self.named_positions['gamepad'][:2] # only x and y
            self.to_ui_q.put({'gp_pos': p2})

    async def _handle_lost_conn(self, data):
        anchor_num = data
        name = f'anchor {anchor_num}' if anchor_num is not None else 'gripper'
        self.to_ui_q.put({'pop_message': f'lost connection to {name}'})


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

    async def tension_lines(self, _=None):
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
        # winch line length is set to last swing based estimate
        if self.gripper_client is not None:
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

    async def stop_all(self, _=None):
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

    async def set_run_mode(self, mode):
        """Sets the robot mode."""

        # exit previous mode
        if self.calmode == "train":
            if self.grpc_server is not None:
                await self.grpc_server.stop(grace=5)

        if mode == "run":
            self.calmode = mode
            print("run mode")
            await self.invoke_motion_task(self.pick_and_place_loop())
        elif mode == "pause":
            self.calmode = mode
            await self.stop_all()
        elif mode == 'train':
            self.training_task = asyncio.create_task(self.begin_training_mode())

    async def begin_training_mode(self):
        """Begin allowing the robot to be controlled from the grpc server
        movement could occur at any time while the server is running"""
        try:
            self.calmode = 'training'
            # begin allowing requests from self.grpc_server
            self.grpc_server = await start_robot_control_server(self)
            # Start child process to run the dataset manager
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
        markers = ['origin', 'cal_assist_1', 'cal_assist_2', 'cal_assist_3']
        averages = defaultdict(lambda: [None]*4)
        for client in self.anchors:
            # average each list of detections, but leave them in the camera's reference frame.
            for marker in markers:
                averages[marker][client.anchor_num] = average_pose(list(client.origin_poses[marker]))

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
            self.anchors.sort(key=lambda x: x.anchor_num)
            # collect observations of origin card aruco marker to get initial guess of anchor poses.
            #   origin pose detections are actually always stored by all connected clients,
            #   it is only necessary to ensure enough have been collected from each client and average them.
            num_o_dets = [len(client.origin_poses['origin']) for client in self.anchors]
            while len(num_o_dets) == 0 or min(num_o_dets) < max_origin_detections:
                print(f'Waiting for enough origin card detections from every anchor camera {num_o_dets}')
                await asyncio.sleep(DETECTION_WAIT_S)
                num_o_dets = [len(client.origin_poses['origin']) for client in self.anchors]
            
            anchor_poses = self.locate_anchors()

            # Use the optimization output to update anchor poses and spool params
            config = Config()
            for i, client in enumerate(self.anchors):
                config.anchors[client.anchor_num].pose = anchor_poses[client.anchor_num]
                self.to_ui_q.put({'anchor_pose': (client.anchor_num, anchor_poses[client.anchor_num])})
                client.anchor_pose = anchor_poses[client.anchor_num]
                # await client.send_commands({'set_zero_angle': zero_angles[client.anchor_num]})
            config.write()
            # inform position estimator
            anchor_points = np.array([compose_poses([pose, model_constants.anchor_grommet])[1] for pose in anchor_poses])
            self.pe.set_anchor_points(anchor_points)

            await self.half_auto_calibration()

            # TODO raise gantry to about 1 meter from floor and induce a small swing. then perform half cal again.

            # TODO "Calibration complete. Would you like stringman to pick up the cards and put them in the trash? yes/no"
            self.to_ui_q.put({'pop_message':'Calibration complete. Cards can be removed from the floor.'})

        except asyncio.CancelledError:
            raise

    async def horizontal_line_task(self):
        """
        Attempt to move the gantry in a perfectly horizontal line. How hard could this be?
        This is a motion task
        """
        await asyncio.gather([self.gripper_client.zero_winch(), self.tension_and_wait()])
        await asyncio.sleep(1)
        range_at_start = self.datastore.range_record.getLast()[1]
        result = await self.move_direction_speed([1,0,0], 0.2, downward_bias=0)
        await asyncio.sleep(4)
        self.slow_stop_all_spools()
        await asyncio.sleep(1)
        range_at_end = self.datastore.range_record.getLast()[1]
        print(f'During attempted horizontal move, height rose by {range_at_end - range_at_start} meters')


    def async_on_service_state_change(self, 
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
        """Starts a client to connect to the indicated service"""
        info = AsyncServiceInfo(service_type, name)
        await info.async_request(zc, INFO_REQUEST_TIMEOUT_MS)
        if not info or info.server is None or info.server == '':
            return None;
        print(f"Service {name} added, service info: {info}, type: {service_type}")
        address = socket.inet_ntoa(info.addresses[0])
        name_component = name.split('.')[1]

        is_power_anchor = name_component == anchor_power_service_name
        is_standard_anchor = name_component == anchor_service_name
        is_standard_gripper = name_component == gripper_service_name

        if is_power_anchor or is_standard_anchor:
            # the number of anchors is decided ahead of time (in main.py)
            # but they are assigned numbers as we find them on the network
            # and the chosen numbers are persisted in configuration.json

            if info.server in self.config.anchor_num_map:
                anchor_num = self.config.anchor_num_map[info.server]
            else:
                anchor_num = len(self.config.anchor_num_map)
                if anchor_num >= N_ANCHORS:
                    # we do not support yet multiple crane bot assemblies on a single network
                    print(f"Discovered another anchor server on the network, but we already know of 4 {info.server} {address}")
                    print(f"existing anchors: {self.config.anchor_num_map.keys()}")
                    return None
                self.config.anchor_num_map[info.server] = anchor_num
                self.config.anchors[anchor_num].service_name = info.server
                self.config.write()
            
            # If this is the power anchor, record its index.
            if is_power_anchor:
                if self.power_spool_index is not None:
                    print(f"ERROR: Discovered a second power spool anchor ({info.server}) but one is already registered (anchor #{self.power_spool_index}). Ignoring.")
                else:
                    self.power_spool_index = anchor_num
                    print(f"Power spool anchor discovered and assigned to anchor number {anchor_num}")


            ac = RaspiAnchorClient(address, info.port, anchor_num, self.datastore, self.to_ui_q, self.to_ob_q, self.pool, self.stat)
            self.bot_clients[info.server] = ac
            self.anchors.append(ac)
            print('appending anchor client to list and starting server')
            # this function runs as long as the client is connected and returns true if the client was forced to disconnect abnormally
            abnormal_close = await ac.startup()
            if abnormal_close:
                self.to_ui_q.put({'pop_message': f'lost connection to {name}'})
                await self.remove_service(service_type, name)
                await self.stop_all()
                # TODO: if recording training data, abort current episode

        elif name_component == gripper_service_name:
            # a gripper has been discovered, connect immediately.
            print(f'Connecting to "{info.server}" as gripper')
            assert(self.gripper_client is None)
            await self.connect_to_gripper(address, info.port, info.server)

    async def connect_to_gripper(self, address, port, key):
        gc = RaspiGripperClient(address, port, self.datastore, self.to_ui_q, self.to_ob_q, self.pool, self.stat, self.pe)
        self.gripper_client_connected.clear()
        gc.connection_established_event = self.gripper_client_connected
        self.bot_clients[key] = gc
        self.gripper_client = gc
        # this function runs as long as the client is connected and returns true if the client was forced to disconnect abnormally
        abnormal_close = await gc.startup()
        if abnormal_close:
            self.to_ui_q.put({'pop_message': f'lost connection to {key}'})
            self.gripper_client = None
            del self.bot_clients[key]
            await gc.shutdown()
            await self.stop_all()
            # TODO: if recording training data, abort current episode

    async def update_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        # this occurs when a component goes down abnormally and comes back up.
        # nothing is expected to have changed, and we should already be disconnected from it, but make sure.
        namesplit = name.split('.')
        name_component = namesplit[1]
        key  = ".".join(namesplit[:3])+'.'
        # see if we already have this loaded
        if key in self.bot_clients:
            # we do not expect a service to change without having gone down first
            # however it could simply be that it went down so recently that we're not done closing the client.
            await asyncio.sleep(5)
            if key in self.bot_clients:
                return
        # reconnect
        await self.add_service(zc, service_type, name)

    async def remove_service(self, service_type: str, name: str) -> None:
        """
        Finds if we have a client connected to this service. if so, ends the task if it is running, and deletes the client
        """
        namesplit = name.split('.')
        name_component = namesplit[1]
        key  = ".".join(namesplit[:3])+'.'
        print(f'Removing service {key} from {self.bot_clients.keys()}')

        # only in this dict if we are connected to it.
        if key in self.bot_clients:
            client = self.bot_clients[key]
            await client.shutdown()
            if name_component == anchor_service_name or name_component == anchor_power_service_name:
                self.anchors.remove(client)
            elif name_component == gripper_service_name:
                self.gripper_client = None
            del self.bot_clients[key]

    async def main(self) -> None:
        self.to_ui_q.cancel_join_thread()
        self.to_ob_q.cancel_join_thread()

        POOL_PROCESSES = 8
        # main process loop
        with Pool(processes=POOL_PROCESSES) as pool:
            self.pool = pool
            if self.aiozc is None:
                self.aiozc = AsyncZeroconf(ip_version=IPVersion.All, interfaces=InterfaceChoice.All)

            try:
                print("get services list")
                services = list(
                    await AsyncZeroconfServiceTypes.async_find(aiozc=self.aiozc, ip_version=IPVersion.All)
                )
                print("start service browser")
                self.aiobrowser = AsyncServiceBrowser(
                    self.aiozc.zeroconf, services, handlers=[self.async_on_service_state_change]
                )
            except asyncio.exceptions.CancelledError:
                await self.aiozc.async_close()
                return

            # this task is blocking an runs in it's own thread, but needs a reference to the running loop to start tasks.
            print("start observer's queue listener task")
            self.ob_queue_task = asyncio.create_task(asyncio.to_thread(self.listen_queue_updates, loop=asyncio.get_running_loop()))

            # statistic counter - measures things like average camera frame latency
            asyncio.create_task(self.stat.stat_main())

            # perception model
            asyncio.create_task(self.run_perception())

            # self.sim_task = asyncio.create_task(self.add_simulated_data_circle())

            # A task that continuously estimates the position of the gantry
            self.pe_task = asyncio.create_task(self.pe.main())
            
            # await something that will end when the program closes that to keep zeroconf alive and discovering services.
            try:
                # tasks started with to_thread must use result = await or exceptions that occur within them are silenced.
                result = await self.ob_queue_task
            except asyncio.exceptions.CancelledError:
                pass
            await self.async_close()

    async def async_close(self) -> None:
        result = await self.stop_all()
        self.run_command_loop = False
        self.stat.run = False
        self.pe.run = False
        self.fig_8 = False
        self.pe_task.cancel()
        tasks = [self.pe_task]
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
                    self.to_ui_q.put({'gantry_observation': dp[1:]})
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
                        self.to_ui_q.put({'gantry_observation': dp[2:]})
                
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

    async def _handle_send_winch_finger(self, data):
        await self.send_winch_and_finger(**data)

    async def send_winch_and_finger(self, line_speed, finger_angle):
        """Command the gripper's motors in one update.
        command the winch to change line length at the given speed.
        Enforces extents based on last reported length and returns actual speed commanded.

        Commands the finger to the given angle.
        """
        last_length = self.datastore.winch_line_record.getLast()[1]
        # if last_length < 0.02 or last_length > 1.9:
        #     print('winch too short')
        #     line_speed = 0
        finger_angle = clamp(finger_angle, -90, 90)
        update = {
            'aim_speed': line_speed,
            'set_finger_angle': finger_angle,
        }
        if self.gripper_client is not None:
            asyncio.create_task(self.gripper_client.send_commands(update))
        return line_speed, finger_angle

    async def seek_gantry_goal(self):
        """
        Move towards a goal position, using the constantly updating gantry position provided by the position estimator
        This is a motion task
        """
        GOAL_PROXIMITY_M = 0.07 # meters
        GANTRY_SPEED_MPS = 0.2 # m/s
        LOOP_SLEEP_S = 0.2 # seconds
        
        try:
            self.to_ui_q.put({'gantry_goal_marker': self.gantry_goal_pos})
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
            self.gantry_goal_pos = None
            self.to_ui_q.put({'gantry_goal_marker': self.gantry_goal_pos})

    async def blind_move_to_goal(self):
        """Only measures current position once, then moves in the right direction
        for the amount of time it should take to get to the goal.
        This is a motion task."""
        GANTRY_SPEED_MPS = 0.25 # m/s
        
        try:
            self.to_ui_q.put({'gantry_goal_marker': self.gantry_goal_pos})
            vector = self.gantry_goal_pos - self.pe.gant_pos
            dist = np.linalg.norm(vector)
            if dist > 0:
                result = await self.move_direction_speed(vector / dist, GANTRY_SPEED_MPS, self.pe.gant_pos)
                await asyncio.sleep(dist / GANTRY_SPEED_MPS)
        except asyncio.CancelledError:
            raise
        finally:
            self.slow_stop_all_spools()
            self.gantry_goal_pos = None
            self.to_ui_q.put({'gantry_goal_marker': self.gantry_goal_pos})

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
        # The speed limit is proportional to how far the gantry hangs below a level 30cm below the average anchor.
        # This makes the behavior consistent across installations of different heights.
        hang_distance = np.mean(self.pe.anchor_points[:, 2]) - starting_pos[2]
        speed_limit = clamp(0.3 * (hang_distance - 0.1), 0.01, 0.55)
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

    async def move_in_figure_8(self):
        """
        Move in a figure-8 pattern until interrupted by some other input.
        This is a motion task.
        """
        PATTERN_HEIGHT = 1.5 # meters
        GANTRY_SPEED_MPS = 0.2 # m/s
        CIRCUIT_DISTANCE_M = 18.85 # meters
        PLAN_SEND_INTERVAL_S = 1.0 # seconds
        PLAN_DURATION_S = 1.3 # seconds
        PLAN_STEP_INTERVAL_S = 1/30 # seconds
        
        # move to starting position (over origin)
        try:
            self.gantry_goal_pos = np.array([0,0,PATTERN_HEIGHT])
            await self.invoke_motion_task(self.seek_gantry_goal())
            
            circuit_time = CIRCUIT_DISTANCE_M / GANTRY_SPEED_MPS
            # multiply time.time by this to get a term that increases by 2pi every circuit_time seconds.
            time_scaler = 2*np.pi / circuit_time
            start_time = time.time()

            while True:
                now = time.time()
                times = [now + PLAN_STEP_INTERVAL_S * i for i in range(int(PLAN_DURATION_S / PLAN_STEP_INTERVAL_S))]
                xy_positions = np.array([figure_8_coords((t - start_time) * time_scaler) for t in times])
                positions = np.column_stack([xy_positions, np.repeat(PATTERN_HEIGHT,len(xy_positions))])
                # calculate all line lengths in one statement
                distances = np.linalg.norm(self.pe.anchor_points[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)

                for client in self.anchors:
                    plan = np.column_stack([times, distances[client.anchor_num]])
                    message = {'length_plan' : plan.tolist()}
                    asyncio.create_task(client.send_commands(message))

                await asyncio.sleep(PLAN_SEND_INTERVAL_S)
        except asyncio.CancelledError:
            raise
        finally:
            self.slow_stop_all_spools()

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

    def _handle_add_episode_control_event(self, data):
        for k in data:
            self.episode_control_events.add(str(k))

    async def run_perception(self):
        """
        Run the dobby network on preferred cameras at a modest rate.
        Send heatmaps to UI.
        Store target candidates and confidence.
        """
        DOBBY_MODEL_PATH = "trainer/models/sock_tracker.pth"
        CENTERING_MODEL_PATH = "trainer/models/sock_gripper.pth"
        IMAGE_RES = (640, 360)
        DEVICE = "cpu"
        LOOP_DELAY = 0.5

        if self.dobby_model is None:
            import torch
            from trainer.dobby import DobbyNet
            from trainer.dobby_eval import extract_targets_from_heatmap
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading model from {DOBBY_MODEL_PATH}...")
            self.dobby_model = DobbyNet().to(DEVICE)
            self.dobby_model.load_state_dict(torch.load(DOBBY_MODEL_PATH, map_location=DEVICE))
            self.dobby_model.eval()

        if self.centering_model is None:
            import torch
            from trainer.centering import CenteringNet
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading model from {CENTERING_MODEL_PATH}...")
            self.centering_model = CenteringNet().to(DEVICE)
            self.centering_model.load_state_dict(torch.load(CENTERING_MODEL_PATH, map_location=DEVICE))
            self.centering_model.eval()

        config = Config()

        while self.run_command_loop:
            await asyncio.sleep(LOOP_DELAY)

            # collect images from any anchors that have one
            valid_anchor_clients = []
            img_tensors = []
            gripper_image_tensor = None
            for client in self.bot_clients.values():
                if client.last_frame_resized is None or client.anchor_num not in config.preferred_cameras:
                    continue
                # these are already assumed to be at the correct resolution 
                img_tensor = torch.from_numpy(client.last_frame_resized).permute(2, 0, 1).float() / 255.0
                if client.anchor_num is not None:
                    img_tensors.append(img_tensor)
                    valid_anchor_clients.append(client)
                else:
                    gripper_image_tensor = img_tensor
            
            if gripper_image_tensor is not None:
                with torch.no_grad():
                    # you get a normalized u,v coordinate in the [-1,1] range
                    self.predicted_lateral_vector = self.centering_model(gripper_image_tensor).cpu().squeeze().numpy()
                    self.to_ui_q.put({'grip_lat': self.predicted_lateral_vector})

            if not img_tensors:
                # no anchor images to process right now. cameras are probably still connecting.
                continue

            # run batch inference on GPU and get targets
            all_floor_target_arrs = []
            batch = torch.stack(img_tensors).to(DEVICE)
            with torch.no_grad():
                heatmaps_out = self.dobby_model(batch)

            # Shape: (Batch, 1, H, W) -> (Batch, H, W)
            heatmaps_np = heatmaps_out.squeeze(1).cpu().numpy() # this is the blocking call
            for i, heatmap_np in enumerate(heatmaps_np):
                client = valid_anchor_clients[i]
                results = extract_targets_from_heatmap(heatmap_np)
                if len(results) > 0:
                    targets2d = results[:,:2] # the third number is confidence
                    # if this is an anchor, project points to floor using anchor's specific pose
                    floor_points = project_pixels_to_floor(targets2d, client.camera_pose)
                    all_floor_target_arrs.append(floor_points)
                
                # create a visual diagnostic image in the same manner as dobby_eval and pass it to the UI
                heatmap_vis = (heatmap_np * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
                self.to_ui_q.put({'heatmap': {'anchor_num':valid_anchor_clients[i].anchor_num, 'image':heatmap_color}})

            if len(all_floor_target_arrs) == 0:
                continue

            # filter out targets that are not inside the work area.
            self.floor_targets = np.array([p for p in np.concatenate(all_floor_target_arrs) if self.pe.point_inside_work_area_2d(p)])

    def select_target(self):
        # TODO select the next target from the user's queue.

        # if the user's queue is empty, pick one from the targets identified by dobby.
        if len(self.floor_targets) == 0:
            return None

        # pick closest point to gantry as next target
        gantry2d = self.pe.gant_pos[:2]
        dist_sq = np.sum((self.floor_targets - gantry2d)**2, axis=1)
        closest_idx = np.argmin(dist_sq)
        next_target = self.floor_targets[closest_idx]

        # no calibration is perfect, and usually what's wrong is that the scale of the room is off, the height of the anchor is off, or the tilt of the camera is off.
        # the yaw though is always bang on. So, instead of just projecting points onto the floor, it might be better to project a line onto the floor.
        # in other words, add some small offset vertically above and below the point in pixel space and project both of these, then  draw a segment between them on the floor.
        # then find the spot where lines from multiple cameras come closest to intersecting.

        # however, this isn't actually going to produce good results unless you are doing it for the same object in both views, which you just have to guess at.

        print(f'selected object at floor position {next_target}')
        return next_target

    async def pick_and_place_loop(self):
        """
        Long running motion task that repeatedly identifies targets picks them up and drops them over the hamper
        """
        PICK_WINCH_LENGTH = 0.8
        DROP_WINCH_LENGTH = 0.4

        try:
            gtask = None
            while self.run_command_loop:
                next_target = self.select_target()
                if next_target is None:
                    # TODO park on the saddle.
                    await asyncio.sleep(0.5)
                    continue

                # pick Z position for gantry
                goal_pos = np.array([next_target[0], next_target[1], 1.2])
                self.gantry_goal_pos = goal_pos

                # set winch for ideal pickup length (0.8m?)
                await self.gripper_client.send_commands({'length_plan': [time.time(), PICK_WINCH_LENGTH]})

                # gantry is now heading for a position over next_target
                # wait only one second for it to arrive.
                try:
                    if gtask is None or gtask.done():
                        gtask = asyncio.create_task(self.seek_gantry_goal())
                    await asyncio.wait_for(gtask, 1)
                except TimeoutError:
                    # if doesn't arrive in one second, run target selection again since a better one might have appeared or the user might have put one in their queue
                    continue

                if self.gripper_client is None:
                    print('pick and place aborted because we lost the gripper connection')
                    break

                # when we reach this point we arrived over the item. commit to it unless it proves impossible to pick up.
                success = await self.execute_grasp()
                if not success:
                    # just pick another target, but consider downranking this object or something.
                    await asyncio.sleep(0.5)
                    continue

                # tension now just in case.
                # await self.tension_and_wait()

                await self.gripper_client.send_commands({'length_plan': [time.time(), DROP_WINCH_LENGTH]})

                # fly to to drop point
                self.gantry_goal_pos = self.named_positions['hamper'] + np.array([0,0,0.5])
                await self.seek_gantry_goal()
                # open gripper
                asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': -10}))

                # keep score


        except asyncio.CancelledError:
            raise
        finally:
            if gtask is not None:
                gtask.cancel()
            self.slow_stop_all_spools()
            self.gantry_goal_pos = None
            self.to_ui_q.put({'gantry_goal_marker': self.gantry_goal_pos})

    async def execute_grasp(self):
        """Try to grasp whatever is directly below the gripper"""
        OPEN = -30
        CLOSED = 85
        CENTERING_SEC = 1.5

        attempts = 4
        while not self.pe.holding and attempts > 0:
            attempts -= 1
            asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': OPEN}))
            await asyncio.sleep(0.7) # wait for fingers to get out of the way of the camera

            # determine which direction we'd have to move laterally to center the object
            if self.predicted_lateral_vector is not None:
                # you get a normalized u,v coordinate in the [-1,1] range
                pred_vector = self.predicted_lateral_vector
                # for now assume that the up direction in the gripper image is +Y in world space 
                # stabilize_frame produced this direction and I think it depends on the compass.
                # the direction in world space depends on how the user placed the origin card on the ground
                # we need to capture a number during calibration to relate these two.
                pred_vector[1] *= -1 # invert Y
                # +1 is the edge of the image. how far laterally that would be depends on how far from the ground the gripper is.
                distance_to_floor = 0.8 # TODO fix i2c bus problem with rangefinder
                # and the FOV of the simulated image, which is sf_scale_factor * (camera module 3 fov)
                half_virtual_fov = model_constants.rpi_cam_3_fov * sf_scale_factor / 2
                # lateral distance to object
                lateral_vector = np.sin(pred_vector * half_virtual_fov) * distance_to_floor
                # lateral distance in meters
                lateral_distance = np.linalg.norm(lateral_vector)
                # speed to travel that lateral distance in CENTERING_SEC
                lateral_speed = lateral_distance / CENTERING_SEC
                await self.move_direction_speed([lateral_vector[0],lateral_vector[1],0], lateral_speed)
                await asyncio.sleep(CENTERING_SEC)

            print('move gripper down until laser rangefinder shows about 5cm or IMU shows tipping or timeout elapsed')
            self.pe.tip_over.clear()
            await self.move_direction_speed([0,0,-1], 0.05, downward_bias=0)
            try:
                await asyncio.wait_for(self.pe.tip_over.wait(), 4)
            except TimeoutError:
                pass
            finally:
                self.slow_stop_all_spools()
            print('close gripper')
            asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': CLOSED}))
            print('wait up to 2 seconds for pad to sense object.')
            try:
                await asyncio.wait_for(self.pe.finger_pressure_rising.wait(), 2)
                self.pe.finger_pressure_rising.clear()
            except TimeoutError:
                print('did not detect a successful hold, open and hop.')
                direction = np.concatenate([np.random.uniform(-0.025, 0.025, (2)), [0.1]])
                await self.move_direction_speed(direction, 0.05)
                await asyncio.sleep(3.0)
                asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': OPEN}))
                self.slow_stop_all_spools()
                continue
            print('Successful grasp')
            return True
        print('Gave up on grasp after a few attempts')
        return False

def start_observation(to_ui_q, to_ob_q):
    """
    Entry point to be used when starting this from main.py with multiprocessing
    """
    ob = AsyncObserver(to_ui_q, to_ob_q)
    asyncio.run(ob.main())

if __name__ == "__main__":
    from multiprocessing import Queue
    to_ui_q = Queue()
    to_ob_q = Queue()
    to_ui_q.cancel_join_thread()
    to_ob_q.cancel_join_thread()

    # when running as a standalone process (debug only, linux only), register signal handler
    def stop():
        print("\nwait for clean observer shutdown")
        to_ob_q.put({'STOP':None})
    async def main():
        runner = AsyncObserver(datastore, to_ui_q, to_ob_q)
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(getattr(signal, 'SIGINT'), stop)
        result = await runner.main()
    asyncio.run(main())
