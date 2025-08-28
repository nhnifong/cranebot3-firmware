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
from cv_common import invert_pose, compose_poses, average_pose
from calibration import order_points_for_low_travel, find_cal_params
import model_constants
import traceback
import cv2
import pickle

# Define the service names for network discovery
cranebot_anchor_service_name = 'cranebot-anchor-service'
cranebot_anchor_power_service_name = 'cranebot-anchor-power-service'
cranebot_gripper_service_name = 'cranebot-gripper-service'

N_ANCHORS = 4

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

    #region Command Handlers
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
    #endregion

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
                await self.motion_task
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

    async def set_simulated_data_mode(self, mode):
        if self.sim_task is not None:
            self.sim_task.cancel()
            result = await self.sim_task
        if mode == 'circle':
            self.sim_task = asyncio.create_task(self.add_simulated_data_circle())
        elif mode == 'point2point':
            self.sim_task = asyncio.create_task(self.add_simulated_data_point2point())

    def stop_all(self, _=None):
        # First, cancel any high-level motion task that's running.
        # This stops new commands from being generated.
        if self.motion_task is not None and not self.motion_task.done():
            print(f"Cancelling previous motion task: {self.motion_task.get_name()}")
            self.motion_task.cancel()
            self.motion_task = None
        self.slow_stop_all_spools()

    def slow_stop_all_spools(self):
        for name, client in self.bot_clients.items():
            # Slow stop all spools. gripper too
            asyncio.create_task(client.slow_stop_spool())

    def set_run_mode(self, mode):
        """
        Sets the calibration mode of connected bots
        "run" - move autonomously to pick up objects
        "pause" - hold all motors at current position, but continue to make observations
        """
        if mode == "run":
            self.calmode = mode
            print("run mode")
        elif mode == "pause":
            self.calmode = mode
            self.stop_all()
        elif mode == 'pose':
            self.calmode = mode
            self.locate_anchor_task = asyncio.create_task(self.locate_anchors())

    async def locate_anchors(self):
        """using the record of recent origin detections, estimate the pose of each anchor."""
        MIN_ORIGIN_OBSERVATIONS = 6
        
        if len(self.anchors) != N_ANCHORS:
            print(f'anchor pose calibration should not be performed until all anchors are connected. len(anchors)={len(self.anchors)}')
            return
        anchor_poses = []
        for client in self.anchors:
            if len(client.origin_poses) < MIN_ORIGIN_OBSERVATIONS:
                print(f'Too few origin observations ({len(client.origin_poses)}) from anchor {client.anchor_num}')
                continue
            print(f'locating anchor {client.anchor_num} from {len(client.origin_poses)} detections')
            pose = np.array(invert_pose(compose_poses([model_constants.anchor_camera, average_pose(client.origin_poses)])))
            assert pose is not None
            self.to_ui_q.put({'anchor_pose': (client.anchor_num, pose)})
            anchor_poses.append(pose)
        return np.array(anchor_poses)


    async def half_auto_calibration(self):
        """Optimize zero angles from a few points
        This is a motion task"""
        NUM_SAMPLE_POINTS = 3
        OPTIMIZER_TIMEOUT_S = 60  # seconds
        
        try:
            if len(self.anchors) < N_ANCHORS:
                print('Cannot run half calibration until all anchors are connected')
                return
            anchor_poses = np.array([a.anchor_pose for a in self.anchors])
            # Estimate a reasonable height for calibration points
            h = anchor_poses[0,1,2]
            sample_points = np.array([[np.cos((i/NUM_SAMPLE_POINTS)*2*np.pi), np.sin((i/NUM_SAMPLE_POINTS)*2*np.pi), h/3] for i in range(NUM_SAMPLE_POINTS)])
            data = await self.collect_data_at_points(sample_points, anchor_poses=anchor_poses)
            print(f'Starting optimizer with sample points {sample_points}')
            async_result = self.pool.apply_async(find_cal_params, (anchor_poses, data, self.power_spool_index, 'zero_angles_only'))
            _, zero_angles = async_result.get(timeout=OPTIMIZER_TIMEOUT_S)
            print(f'zero angles obtained from optimization {zero_angles}')
            await self.tension_and_wait()
            for client in self.anchors:
                await client.send_commands({'set_zero_angle': zero_angles[client.anchor_num]})
        except asyncio.CancelledError:
            raise

    async def full_auto_calibration(self):
        """Automatically determine anchor poses and zero angles
        This is a motion task"""
        DETECTION_WAIT_S = 1.0 # seconds
        STABILIZATION_WAIT_S = 12.0 # seconds
        SET_LENGTHS_WAIT_S = 4.0 # seconds
        BOUNDING_BOX_SCALE = 0.6
        Z_SHIFT = 0.2 # meters
        FLOOR_Z_OFFSET = 0.4 # meters
        GRID_STEPS_X = 3j
        GRID_STEPS_Y = 4j
        GRID_STEPS_Z = 2j
        OPTIMIZER_TIMEOUT_S = 60 # seconds
        
        try:
            if len(self.anchors) < N_ANCHORS:
                print('Cannot run full calibration until all anchors are connected')
                return
            self.anchors.sort(key=lambda x: x.anchor_num)
            # collect observations of origin card aruco marker to get initial guess of anchor poses.
            #   origin pose detections are actually always stored by all connected clients,
            #   it is only necessary to ensure enough have been collected from each client and average them.
            num_o_dets = [len(client.origin_poses) for client in self.anchors]
            while len(num_o_dets) == 0 or min(num_o_dets) < max_origin_detections:
                print(f'Waiting for enough origin card detections from every anchor camera {num_o_dets}')
                await asyncio.sleep(DETECTION_WAIT_S)
                num_o_dets = [len(client.origin_poses) for client in self.anchors]
            # Maybe wait on input from user here to confirm the positions and ask "Are the lines clear to start moving?"
            anchor_poses = await self.locate_anchors()
            print(f'anchor poses based on origin card {anchor_poses}')

            # the true distance between anchor 0 and anchor 2 should be 5.334 meters
            a = anchor_poses[0,1,:2]
            b = anchor_poses[1,1,:2]
            print(f'distance between anchor 0 and 1 = {np.linalg.norm(a-b)}')

            for i, client in enumerate(self.anchors):
                client.anchor_pose = anchor_poses[i]
            anchor_points = np.array([compose_poses([pose, model_constants.anchor_grommet])[1] for pose in anchor_poses])
            self.pe.set_anchor_points(anchor_points)
            print(f'Provisional anchor points relative to origin card \n{anchor_points}')

            # reel in all lines until the switches indicate they are tight
            await self.tension_and_wait()
            cutoff = time.time()
            print('Collecting observations of gantry now that its still and lines are tight')
            await asyncio.sleep(STABILIZATION_WAIT_S)

            # use aruco observations of gantry to obtain initial guesses for zero angles
            # use this information to perform rough movements
            gantry_data = self.datastore.gantry_pos.deepCopy(cutoff=cutoff)
            position = np.mean(gantry_data[:,1:], axis=0)
            print(f'the visual position of the gantry is {position}')
            lengths = np.linalg.norm(anchor_points - position, axis=1)
            print(f'Line lengths from coarse calibration ={lengths}')
            await self.sendReferenceLengths(lengths)
            await asyncio.sleep(SET_LENGTHS_WAIT_S)

            # Determine the bounding box of the work area and shrink it to stay away from the walls and ceiling
            min_x, min_y, min_anchor_z = np.min(anchor_points, axis=0) * BOUNDING_BOX_SCALE
            max_x, max_y, _ = np.max(anchor_points, axis=0) * BOUNDING_BOX_SCALE
            floor_z = FLOOR_Z_OFFSET - Z_SHIFT
            max_gantry_z = min_anchor_z - Z_SHIFT
            print(f'Maximum allowed height of gantry sample point {max_gantry_z}')

            # make a polygon of the exact work area so we can test if points are inside it
            contour = anchor_points[:,0:2].astype(np.float32)

            # still evaluating whether this random sample point method was better than the orderly method, so don't delete it yet
            # ... (commented block remains)

            # use an orderly grid of sample points
            sample_points = np.mgrid[min_x:max_x:GRID_STEPS_X, min_y:max_y:GRID_STEPS_Y, floor_z:max_gantry_z:GRID_STEPS_Z].reshape(3, -1).T

            # choose an ordering of the points that results in a low travel distance
            sample_points, distance = order_points_for_low_travel(sample_points)
            print(f'Ordered points for near-optimal travel. total distance = {distance} m')

            # prepare to collect final observations
            data = await self.collect_data_at_points(sample_points, anchor_poses=anchor_poses, save_progress=True)
            print(f'Completed data collection. Performing optimization of calibration parameters.')

            # feed collected data to the optimization process in calibration.py
            with open('collected_cal_data.pickle', 'wb') as f:
                f.write(pickle.dumps((anchor_poses, data)))

            async_result = self.pool.apply_async(find_cal_params, (anchor_poses, data, self.power_spool_index))
            anchor_poses, zero_angles = async_result.get(timeout=OPTIMIZER_TIMEOUT_S)
            print(f'obtained result from find_cal_params {result_params}')

            # Use the optimization output to update anchor poses and spool params
            config = Config()
            for i, client in enumerate(self.anchors):
                config.anchors[client.anchor_num].pose = anchor_poses[client.anchor_num]
                self.to_ui_q.put({'anchor_pose': (client.anchor_num, anchor_poses[client.anchor_num])})
                client.anchor_pose = anchor_poses[client.anchor_num]
                await client.send_commands({'set_zero_angle': zero_angles[client.anchor_num]})
            config.write()
            # inform position estimator
            anchor_points = np.array([compose_poses([pose, model_constants.anchor_grommet])[1] for pose in anchor_poses])
            self.pe.set_anchor_points(anchor_points)

            # move to random locations and determine the quality of the calibration by how often all four lines are tight during and after moves.
        except asyncio.CancelledError:
            raise

    async def collect_data_at_points(self, sample_points, anchor_poses=None, save_progress=False):
        SETTLING_TIME_S = 2.0 # seconds
        OBSERVATION_COLLECTION_TIME_S = 7.0 # seconds
        MIN_TOTAL_GANTRY_OBSERVATIONS = 3
        
        # these gantry observations these need to be raw gantry aruco poses in the camera coordinate space
        # not the poses in self.datastore.gantry_pos so we set a flag in the anchor clients that cause them to save that
        for client in self.anchors:
            client.save_raw = True
            await client.send_commands({'report_raw': True})

        print('zero the winch line')
        await self.gripper_client.zero_winch()

        print('Collect data at each position')
        # data format is described in docstring of calibration_cost_fn
        # [{'encoders':[0, 0, 0, 0], 'visuals':[[pose, pose, ...], x4]}, ...]
        data = []
        for i, point in enumerate(sample_points):
            # move to a position.
            print(f'Moving to point {i+1}/{len(sample_points)} at {point}')
            self.gantry_goal_pos = point
            await self.invoke_motion_task(self.blind_move_to_goal())
            # await self.seek_gantry_goal()

            # integration-test specific behavior
            if self.test_gantry_goal_callback is not None:
                self.test_gantry_goal_callback(point)

            # reel in all lines until they are tight
            await self.tension_and_wait()
            await asyncio.sleep(SETTLING_TIME_S)
            cutoff = time.time()

            # collect several visual observations of the gantry from each camera
            print('Collecting observations of gantry')
            # clear old observations
            for client in self.anchors:
                client.raw_gant_poses = []
            await asyncio.sleep(OBSERVATION_COLLECTION_TIME_S)

            v = [client.raw_gant_poses for client in self.anchors]
            # many positions are not visible to all four cameras,
            # but to use this calibration point, the gantry must have been oberved at least three times in total.
            if sum([len(poses) for poses in v]) >= MIN_TOTAL_GANTRY_OBSERVATIONS:
                # save current raw encoder angles and visual observations
                entry = {
                    'encoders': [client.last_raw_encoder for client in self.anchors],
                    'visuals': v,
                    # we are assuming the gripper was pointed straght down when this was collected, but it is possible to verify it with the IMU
                    'laser_range': self.datastore.range_record.getLast()[1],
                }
                print(f'Collected {len(v[0])}, {len(v[1])}, {len(v[2])}, {len(v[3])} visual observations of gantry pose')
                data.append(entry)

                # save data after every collected point if requested.
                if save_progress and anchor_poses is not None:
                    print("Saving calibration progress to 'collected_cal_data.pickle'...")
                    with open('collected_cal_data.pickle', 'wb') as f:
                        f.write(pickle.dumps((anchor_poses, data)))
            else:
                print(f'Point {point} skipped because gantry could not be observed in that position')

        assert(len(data) > 0)

        for client in self.anchors:
            client.save_raw = False
            asyncio.create_task(client.send_commands({'report_raw': False}))

        return data

    def async_on_service_state_change(self, 
        zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange
    ) -> None:
        if 'cranebot' in name:
            print(f"Service {name} of type {service_type} state changed: {state_change}")
            if state_change is ServiceStateChange.Added:
                asyncio.create_task(self.add_service(zeroconf, service_type, name))
            if state_change is ServiceStateChange.Removed:
                asyncio.create_task(self.remove_service(zeroconf, service_type, name))
            elif state_change is ServiceStateChange.Updated:
                pass

    async def add_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        """
        Starts a client to connect to the indicated service
        """
        INFO_REQUEST_TIMEOUT_MS = 3000 # milliseconds
        
        info = AsyncServiceInfo(service_type, name)
        await info.async_request(zc, INFO_REQUEST_TIMEOUT_MS)
        if not info or info.server is None or info.server == '':
            return None;
        print(f"Service {name} added, service info: {info}, type: {service_type}")
        address = socket.inet_ntoa(info.addresses[0])
        name_component = name.split('.')[1]

        is_power_anchor = name_component == cranebot_anchor_power_service_name
        is_standard_anchor = name_component == cranebot_anchor_service_name

        if is_power_anchor or is_standard_anchor:
            # the number of anchors is decided ahead of time (in main.py)
            # but they are assigned numbers as we find them on the network
            # and the chosen numbers are persisted on disk

            if info.server in self.config.anchor_num_map:
                anchor_num = self.config.anchor_num_map[info.server]
            else:
                anchor_num = len(self.config.anchor_num_map)
                if anchor_num >= N_ANCHORS:
                    # we do not support yet multiple crane bot assemblies on a single network
                    print(f"Discovered another anchor server on the network, but we already know of 4 {info.server} {address}")
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


            ac = RaspiAnchorClient(address, info.port, anchor_num, self.datastore, self.to_ui_q, self.to_ob_q, self.pool, self.stat, self.shape_tracker)
            self.bot_clients[info.server] = ac
            self.anchors.append(ac)
            print('appending anchor client to list and starting server')
            asyncio.create_task(ac.startup())

        elif name_component == cranebot_gripper_service_name:
            gc = RaspiGripperClient(address, info.port, self.datastore, self.to_ui_q, self.to_ob_q, self.pool, self.stat, self.pe)
            self.bot_clients[info.server] = gc
            self.gripper_client = gc
            asyncio.create_task(gc.startup())

    async def remove_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        """
        Finds if we have a client connected to this service. if so, ends the task if it is running, and deletes the client
        """
        namesplit = name.split('.')
        name_component = namesplit[1]
        key  = ".".join(namesplit[:3])+'.'
        print(f'Removing service {key} from {self.bot_clients.keys()}')

        if key in self.bot_clients:
            client = self.bot_clients[key]
            await client.shutdown()
            if name_component == cranebot_anchor_service_name:
                self.anchors.remove(client)
            elif name_component == cranebot_gripper_service_name:
                self.gripper_client = None
            del self.bot_clients[key]

    async def main(self) -> None:
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

            # experimental object segmentation model
            if self.enable_shape_tracking:
                # FastSAM model
                from segment import ShapeTracker
                self.shape_tracker = ShapeTracker()
                asyncio.create_task(self.run_shape_tracker())

            # self.sim_task = asyncio.create_task(self.add_simulated_data_circle())

            # A task that continuously estimates the position of the gantry
            asyncio.create_task(self.pe.main())
            
            # await something that will end when the program closes that to keep zeroconf alive and discovering services.
            try:
                # tasks started with to_thread must use result = await or exceptions that occur within them are silenced.
                result = await self.ob_queue_task
            except asyncio.exceptions.CancelledError:
                pass
            await self.async_close()

    async def async_close(self) -> None:
        self.stop_all()
        self.run_command_loop = False
        self.stat.run = False
        self.pe.run = False
        tasks = []
        if self.aiobrowser is not None:
            tasks.append(self.aiobrowser.async_cancel())
        if self.aiozc is not None:
            tasks.append(self.aiozc.async_close())
        if self.sim_task is not None:
            tasks.append(self.sim_task)
        if self.locate_anchor_task is not None:

            tasks.append(self.locate_anchor_task)
        tasks.extend([client.shutdown() for client in self.bot_clients.values()])
        result = await asyncio.gather(*tasks)

    async def run_shape_tracker(self):
        MIN_SLEEP_S = 0.05 # seconds
        
        while self.run_command_loop:
            elapsed = 0
            if self.calmode == "pose":
                # send the UI some shapes representing the whole frustum of each camera
                prisms = []
                for anchor_num in range(4):
                    shps = self.shape_tracker.make_shapes(anchor_num, [[[0.0,1.0], [1.0,1.0], [1.0,0.0], [0.0,0.0]]])
                    prisms.append(shps.values()[0])
                self.to_ui_q.put({
                    'prisms': prisms,
                })

            elif len(self.anchors) > 1:
                start = time.time()
                trimesh_list = self.shape_tracker.merge_shapes()
                elapsed = time.time() - start
                print(f'Shape merging took {elapsed} sec')
                prisms = []
                for sdict in self.shape_tracker.last_shapes_by_camera:
                    prisms.extend(sdict.values())
                if not self.run_command_loop:
                    return
                self.to_ui_q.put({
                    'solids': trimesh_list,
                    'prisms': prisms,
                })

            await asyncio.sleep(max(MIN_SLEEP_S, self.shape_tracker.preferred_delay - elapsed))

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
                    dp = gantry_real_pos + np.array([0, random()*RANDOM_NOISE_MAGNITUDE, random()*RANDOM_NOISE_MAGNITUDE, random()*RANDOM_NOISE_MAGNITUDE])
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
        """ Simulate the gantry moving from random point to random point"""
        LOWER_Z_BOUND = 1.0 # meters
        UPPER_Z_OFFSET = 0.3 # meters
        MAX_SPEED_MPS = 0.2 # m/s
        GOAL_PROXIMITY_THRESHOLD = 0.03 # meters
        SOFT_SPEED_FACTOR = 0.25
        RANDOM_EVENT_CHANCE = 0.5
        OBSERVATION_NOISE_STD_DEV = 0.05 # meters
        WINCH_LINE_LENGTH = 1.0 # meters
        RANGEFINDER_OFFSET = 1.0 # meters
        LOOP_SLEEP_S = 0.05 # seconds
        
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
                    dp = np.concatenate([[t], gantry_real_pos + np.random.normal(0, OBSERVATION_NOISE_STD_DEV, (3,))])
                    self.datastore.gantry_pos.insert(dp)
                    self.to_ui_q.put({'gantry_observation': dp[1:]})
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
                tt = self.datastore.anchor_line_record[0].getLast()[0]
                await asyncio.sleep(LOOP_SLEEP_S)
            except asyncio.exceptions.CancelledError:
                break

    def collect_gant_frame_positions(self):
        result = np.zeros((4,3))
        for client in self.anchors:
            result[client.anchor_num] = client.last_gantry_frame_coords
        return result

    async def seek_gantry_goal(self):
        """
        Move towards a goal position, using the constantly updating gantry position provided by the position estimator
        This is a motion task
        """
        GOAL_PROXIMITY_M = 0.05 # meters
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
        GANTRY_SPEED_MPS = 0.2 # m/s
        
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

    async def move_direction_speed(self, uvec, speed, starting_pos=None):
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
        """
        DOWNWARD_BIAS_Z = -0.08 # meters
        KINEMATICS_STEP_SCALE = 10.0 # Determines the size of the virtual step to calculate line speed derivatives
        MAX_LINE_SPEED_MPS = 0.5 # m/s
        
        if speed == 0:
            for client in self.anchors:
                asyncio.create_task(client.send_commands({'aim_speed': 0}))
            return

        # apply downward bias and renormalize
        uvec = uvec + np.array([0,0,DOWNWARD_BIAS_Z])
        uvec  = uvec / np.linalg.norm(uvec)

        # even if the starting position is off slightly, this method should not produce jerky moves.
        # because it's not commanding any absolute length from the spool motor
        if starting_pos is None:
            starting_pos = self.pe.gant_pos

        # line lengths at starting pos
        lengths_a = np.linalg.norm(starting_pos - self.pe.anchor_points, axis=1)
        # line lengths at new pos
        starting_pos += (uvec / KINEMATICS_STEP_SCALE)
        lengths_b = np.linalg.norm(starting_pos - self.pe.anchor_points, axis=1)
        # length changes needed to travel a small distance in uvec direction from starting_pos
        deltas = lengths_b - lengths_a
        line_speeds = deltas * KINEMATICS_STEP_SCALE * speed
        
        if np.max(np.abs(line_speeds)) > MAX_LINE_SPEED_MPS:
            print('abort move because it\'s too fast.')
            return

        # send move
        for client in self.anchors:
            asyncio.create_task(client.send_commands({'aim_speed': line_speeds[client.anchor_num]}))

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
