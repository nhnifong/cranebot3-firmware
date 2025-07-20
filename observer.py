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
from math import sin,cos,pi
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
from new_calibration import order_points_for_low_travel, find_cal_params
import model_constants
import traceback

fields = ['Content-Type', 'Content-Length', 'X-Timestamp-Sec', 'X-Timestamp-Usec']
cranebot_anchor_service_name = 'cranebot-anchor-service'
cranebot_gripper_service_name = 'cranebot-gripper-service'

def anchor_pose_cost_fn(params, observations):
    """mean squared error between expected and actual observations given pose
    params = [rx,ry,rz,x,y,z]
    """
    # what is the expected observed origin pose given these parameters
    anchor_pose = params.reshape((2,3))
    expected_origin_pose = np.array(invert_pose(compose_poses([anchor_pose, model_constants.anchor_camera])))
    distances = np.linalg.norm(observations.reshape(-1,6) - expected_origin_pose.reshape(6), axis=1)
    return np.mean(distances**2)

def figure_8_coords(t):
    """
    Calculates the (x, y) coordinates for a figure-8.
    figure fits within a box from -2 to +2
    Args:
        t: The input parameter (angle in radians, typically from 0 to 2*pi).
    Returns:
        A tuple (x, y) representing the position on the figure-8.
    """
    x = 2 * math.sin(t)
    y = 2 * math.sin(2 * t)
    return (x, y)

# Manager of multiple tasks running clients connected to each robot component
# The job of this class in a nutshell is to discover four anchors and a gripper on the network,
# connect to them, and forward data between them and the position estimate, shape tracker, and UI.
class AsyncObserver:
    def __init__(self, to_ui_q, to_ob_q) -> None:
        self.position_update_task = None
        self.aiobrowser: AsyncServiceBrowser | None = None
        self.aiozc: AsyncZeroconf | None = None
        self.send_position_updates = True
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

        # read a mapping of server names to anchor numbers from the config file
        self.config = Config()
        self.config.write()

        self.stat = StatCounter(to_ui_q)

        self.enable_shape_tracking = False
        self.shape_tracker = None

        # Position Estimator. this used to be a seperate process so it's still somewhat independent.
        self.pe = Positioner2(self.datastore, self.to_ui_q, self)

        self.sim_task = None
        self.locate_anchor_task = None

    def listen_queue_updates(self, loop):
        """
        Receive any updates on our process input queue

        this thread doesn't actually have a running event loop.
        so run any coroutines back in the main thread with asyncio.run_coroutine_threadsafe
        """
        while self.send_position_updates:
            updates = self.to_ob_q.get()
            if 'STOP' in updates:
                print('Observer shutdown')
                break
            else:
                asyncio.run_coroutine_threadsafe(self.process_update(updates), loop)

    async def process_update(self, updates):
        try:
            if 'future_anchor_lines' in updates:
                fal = updates['future_anchor_lines']
                if not (fal['sender'] == 'pe' and self.calmode != 'run'):
                    for client in self.anchors:
                        message = {'length_plan' : fal['data'][client.anchor_num].tolist()}
                        if 'host_time' in fal:
                            message['host_time'] = fal['host_time']
                        asyncio.create_task(client.send_commands(message))

            if 'future_winch_line' in updates:
                fal = updates['future_winch_line']
                if not (fal['sender'] == 'pe' and self.calmode != 'run'):
                    if self.gripper_client is not None:
                        message = {'length_plan' : fal['data']}
                        await self.gripper_client.send_commands(message)
            if 'set_run_mode' in updates:
                print(f"set_run_mode to '{updates['set_run_mode']}'") 
                self.set_run_mode(updates['set_run_mode'])
            if 'do_line_calibration' in updates:
                await self.sendReferenceLengths(updates['do_line_calibration'])
            if 'tension_lines' in updates:
                self.tension_lines()
            if 'jog_spool' in updates:
                if 'anchor' in updates['jog_spool']:
                    for client in self.anchors:
                        if client.anchor_num == updates['jog_spool']['anchor']:
                            asyncio.create_task(client.send_commands({
                                'aim_speed': updates['jog_spool']['speed']
                            }))
                elif 'gripper' in updates['jog_spool']:
                    # we can also jog the gripper spool
                    print(f'gripper spool moved with update {updates}')
                    if self.gripper_client is not None:
                        asyncio.create_task(self.gripper_client.send_commands({
                            'aim_speed': updates['jog_spool']['speed']
                        }))
            if 'toggle_previews' in updates:
                tp = updates['toggle_previews']
                if 'anchor' in tp:
                    for client in self.anchors:
                        if client.anchor_num == tp['anchor']:
                            client.sendPreviewToUi = tp['status']
                elif 'gripper' in tp:
                    if self.gripper_client is not None:
                        self.gripper_client.sendPreviewToUi = tp['status']
            if 'gantry_dir_sp' in updates:
                dir_sp = updates['gantry_dir_sp']
                self.gantry_goal_pos = None
                asyncio.create_task(self.move_direction_speed(dir_sp['direction'], dir_sp['speed']))
            if 'gantry_goal_pos' in updates:
                self.gantry_goal_pos = updates['gantry_goal_pos']
                asyncio.create_task(self.seek_gantry_goal())
            if 'fig_8' in updates:
                # todo we need to keep track of a single task at a time that is able to send control inputs
                asyncio.create_task(self.move_in_figure_8())
            if 'set_grip' in updates:
                if self.gripper_client is not None:
                    asyncio.create_task(self.gripper_client.send_commands({
                        'grip' : 'closed' if updates['set_grip'] else 'open'
                    }))
            if 'slow_stop_one' in updates:
                if updates['slow_stop_one']['id'] == 'gripper':
                    if self.gripper_client is not None:
                        asyncio.create_task(self.gripper_client.slow_stop_spool())
                else:
                    for client in self.anchors:
                        if client.anchor_num == updates['slow_stop_one']['id']:
                            asyncio.create_task(client.slow_stop_spool())
            if 'slow_stop_all' in updates:
                self.slow_stop_all_spools()
            if 'set_simulated_data_mode' in updates:
                m = updates['set_simulated_data_mode']
                await self.set_simulated_data_mode(m)
            if 'confirm_anchors' in updates:
                poses = updates['confirm_anchors']
                for client in self.anchors:
                    if client.anchor_num in poses:
                        client.anchor_pose = poses[client.anchor_num]
                        self.config.anchors[client.anchor_num].pose = client.anchor_pose
                        self.pe.anchor_points[client.anchor_num] = client.anchor_pose[1]
                self.config.write()
        except Exception as e:
            traceback.print_exc(file=sys.stderr)

    async def tension_lines(self):  
        """Request all anchors to reel in all lines until tight"""
        for client in self.anchors:
            asyncio.create_task(client.send_commands({'tighten':None}))
        # This function does not  wait for confirmation from every anchor, as it would just hold up the processing of the ob_q
        # this is similar to sending a manual move command. it can be overridden by any subsequent command.
        # thus, it should be done while paused.

    async def wait_for_tension()
        # this function returns only once all anchors are reporting tight lines in their regular line record, and are not moving
        complete = False
        while not complete:
            await asyncio.sleep(0.1)
            records = np.array([alr.getLast() for alr in self.datastore.anchor_line_record])
            speeds = np.array(records[:,2])
            tight = np.array(records[:,3])
            complete = np.all(tight) and np.sum(speeds) == 0:
        return True


    async def sendReferenceLengths(self, lengths):
        if len(lengths) != 4:
            print(f'Cannot send {len(lengths)} ref lengths to anchors')
            return
        # any anchor that receives this and is slack would ignore it
        # any anchor which is tight would calculate a zero angle and average it in
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

    def slow_stop_all_spools(self):
        self.fig_8 = False
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
            self.slow_stop_all_spools()
        elif mode == 'pose':
            self.calmode = mode
            self.locate_anchor_task = asyncio.create_task(self.locate_anchors())

    async def locate_anchors(self):
        """using the record of recent origin detections, estimate the pose of each anchor."""
        if len(self.anchors) != 4:
            print(f'anchor pose calibration should not be performed until all anchors are connected. len(anchors)={len(self.anchors)}')
            return
        anchor_poses = []
        for client in self.anchors:
            if len(client.origin_poses) < 6:
                print(f'Too few origin observations ({len(client.origin_poses)}) from anchor {client.anchor_num}')
                continue
            print(f'locating anchor {client.anchor_num} from {len(client.origin_poses)} detections')
            pose = np.array(invert_pose(compose_poses([model_constants.anchor_camera, average_pose(client.origin_poses)])))
            assert pose is not None:
            self.to_ui_q.put({'anchor_pose': (client.anchor_num, pose)})
            anchor_poses.append(pose)
        return np.array(anchor_points)


    async def full_auto_calibration(self):
        self.anchors.sort(key=lambda x: x.anchor_num)
        # collect observations of origin card aruco marker to get initial guess of anchor poses.
        #   origin pose detections are actually always stored by all connected clients,
        #   it is only necessary to ensure enough have been collected from each client and average them.
        while min([len(client.origin_poses) for client in self.anchors]) < max_origin_detections:
            print('Waiting for enough origin card detections from every anchor camera')
            await asyncio.sleep(1)
        # Maybe wait on input from user here to confirm the positions and ask "Are the lines clear to start moving?"
        anchor_poses = await self.locate_anchors()
        anchor_points = np.array([compose_poses([pose, model_constants.anchor_grommet])[1] for pose in anchor_poses])

        # reel in all lines until the switches indicate they are tight
        print('Tightening all lines')
        await self.tension_lines()
        await self.wait_for_tension()
        cutoff = time.time()
        print('Collecting observations of gantry now that its still and lines are tight')
        await asyncio.sleep(6)

        # use aruco observations of gantry to obtain initial guesses for zero angles
        # use this information to perform rough movements
        gantry_data = self.datastore.gantry_pos.deepCopy(cutoff=cutoff)
        position = np.mean(data[:,1:])
        lengths = np.linalg.norm(anchor_points - position, axis=1)
        print(f'Line lengths from coarse calibration ={lengths}')
        await self.ob.sendReferenceLengths(lengths)

        # Determine the bounding box of the work are
        min_x, min_y, min_anchor_z = np.min(anchor_points, axis=0)
        max_x, max_y, _ = np.max(anchor_points, axis=0)
        floor_z = 0
        max_gantry_z = min_anchor_z - 0.7

        # make a polygon of the exact work area so we can test if points are inside it
        polygon_path = mpath.Path(anchor_points[:,0:2])

        # generate some sample positions within the work area
        n_points = 15
        random_x = np.random.uniform(min_x, max_x, n_points*2)
        random_y = np.random.uniform(min_y, max_y, n_points*2)
        candidate_2d_points = np.column_stack((random_x, random_y))
        mask = polygon_path.contains_points(candidate_2d_points)
        sample_2d_points = candidate_2d_points[mask] # not every point in the bounding box will be in the polygon.

        # truncate to 15 points
        sample_2d_points = sample_2d_points[:n_points]

        # Generate random z-coordinates for the valid 2D points
        sample_points = np.column_stack((sample_2d_points, np.random.uniform(floor_z, max_gantry_z, len(smaple_2d_points))))

        # choose an ordering of the points that results in a low travel distance
        sample_points = order_points_for_low_travel(sample_points)

        # prepare to collect final observations
        # these gantry observations these need to be raw gantry aruco poses in the camera coordinate space
        # not the poses in self.datastore.gantry_pos so we set a flag in the anchor clients that cause them to save that
        for client in self.anchors:
            self.save_raw = True

        # Collect data at each position
        # data format is described in docstring of calibration_cost_fn
        # [{'encoders':[0, 0, 0, 0], 'visuals':[[pose, pose, ...], x4]}, ...]
        data = []
        for i, point in enumerate(sample_points):
            # move to a position.
            print(f'Moving to point {i+1}/{len(sample_points)} at {point}')
            self.gantry_goal_pos = point
            await self.seek_gantry_goal()

            # reel in all lines until they are tight
            print('Tightening all lines')
            await self.tension_lines()
            await self.wait_for_tension()
            cutoff = time.time()

            # save current raw encoder angles
            entry = {'encoders': [client.last_raw_encoder for client in self.anchors]}

            # collect several visual observations of the gantry from each camera
            print('Collecting observations of gantry')
            # clear old observations
            for client in self.anchors:
                client.raw_gant_poses = []
            await asyncio.sleep(6)

            v = [client.raw_gant_poses for client in self.anchors]
            entry['visuals'] = v
            print(f'Collected {len(v[0])}, {len(v[1])}, {len(v[2])}, {len(v[3])} visual observations of gantry pose')
            data.append(entry)

        for client in self.anchors:
            self.save_raw = False

        print(f'Completed data collection. Performing optimization of calibration parameters.')

        # feed collected data to the optimization process in new_calibration.py
        result_params = find_cal_params(anchor_poses, data)

        # Use the optimization output to update anchor poses and spool params
        for i, client in enumerate(self.anchors):
            pose = result_params[i][0:1]
            self.to_ui_q.put({'anchor_pose': (client.anchor_num, pose)})

        # move to random locations and determine the quality of the calibration by how often all four lines are tight during and after moves.

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
        info = AsyncServiceInfo(service_type, name)
        await info.async_request(zc, 3000)
        if not info or info.server is None or info.server == '':
            return None;
        print(f"Service {name} added, service info: {info}, type: {service_type}")
        address = socket.inet_ntoa(info.addresses[0])
        name_component = name.split('.')[1]

        if name_component == cranebot_anchor_service_name:
            # the number of anchors is decided ahead of time (in main.py)
            # but they are assigned numbers as we find them on the network
            # and the chosen numbers are persisted on disk

            if info.server in self.config.anchor_num_map:
                anchor_num = self.config.anchor_num_map[info.server]
            else:
                anchor_num = len(self.config.anchor_num_map)
                if anchor_num >= 4:
                    # we do not support yet multiple crane bot assemblies on a single network
                    print(f"Discovered another anchor server on the network, but we already know of 4 {info.server} {address}")
                    return None
                self.config.anchor_num_map[info.server] = anchor_num
                self.config.anchors[anchor_num].service_name = info.server
                self.config.write()

            ac = RaspiAnchorClient(address, anchor_num, self.datastore, self.to_ui_q, self.to_ob_q, self.pool, self.stat, self.shape_tracker)
            self.bot_clients[info.server] = ac
            self.anchors.append(ac)
            print('appending anchor client to list and starting server')
            asyncio.create_task(ac.startup())

        elif name_component == cranebot_gripper_service_name:
            gc = RaspiGripperClient(address, self.datastore, self.to_ui_q, self.to_ob_q, self.pool, self.stat, self.pe)
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

    async def main(self, interfaces=InterfaceChoice.All) -> None:
        # main process loop
        with Pool(processes=8) as pool:
            self.pool = pool
            self.aiozc = AsyncZeroconf(ip_version=IPVersion.All, interfaces=interfaces)

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

            asyncio.create_task(self.stat.stat_main())

            if self.enable_shape_tracking:
                # FastSAM model
                from segment import ShapeTracker
                self.shape_tracker = ShapeTracker()
                asyncio.create_task(self.run_shape_tracker())

            # self.sim_task = asyncio.create_task(self.add_simulated_data_circle())
            asyncio.create_task(self.pe.main())
            
            # await something that will end when the program closes that to keep zeroconf alive and discovering services.
            try:
                # tasks started with to_thread must used result = await or exceptions that occur within them are silenced.
                result = await self.ob_queue_task
            except asyncio.exceptions.CancelledError:
                pass
            await self.async_close()

    async def async_close(self) -> None:
        self.send_position_updates = False
        self.stat.run = False
        self.pe.run = False
        self.fig_8 = False
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
        while self.send_position_updates:
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
                if not self.send_position_updates:
                    return
                self.to_ui_q.put({
                    'solids': trimesh_list,
                    'prisms': prisms,
                })

            await asyncio.sleep(max(0.05, self.shape_tracker.preferred_delay - elapsed))

    async def add_simulated_data_circle(self):
        """ Simulate the gantry moving in a circle"""
        while self.send_position_updates:
            try:
                t = time.time()
                gantry_real_pos = np.array([t, sin(t/8), cos(t/8), 1.3])
                if random()>0.5:
                    dp = gantry_real_pos + np.array([0, random()*0.1, random()*0.1, random()*0.1])
                    self.datastore.gantry_pos.insert(dp)
                    self.to_ui_q.put({'gantry_observation': dp[1:]})
                # winch line always 1 meter
                self.datastore.winch_line_record.insert(np.array([t, 1.0, 0.0]))
                # range always perfect
                self.datastore.range_record.insert(np.array([t, gantry_real_pos[3]-1]))
                # anchor lines always perfectly agree with gripper position
                for i, simanc in enumerate(self.pe.anchor_points):
                    dist = np.linalg.norm(simanc - gantry_real_pos[1:])
                    last = self.datastore.anchor_line_record[i].getLast()
                    timesince = t-last[0]
                    travel = dist-last[1]
                    speed = travel/timesince
                    self.datastore.anchor_line_record[i].insert(np.array([t, dist, speed, 1.0]))
                tt = self.datastore.anchor_line_record[0].getLast()[0]
                await asyncio.sleep(0.05)
            except asyncio.exceptions.CancelledError:
                break

    async def add_simulated_data_point2point(self):
        """ Simulate the gantry moving from random point to random point"""
        lower = np.min(self.pe.anchor_points, axis=0)
        upper = np.max(self.pe.anchor_points, axis=0)
        lower[2]=1
        upper[2] = upper[2]-0.3
        # starting position
        gantry_real_pos = np.random.uniform(lower, upper)
        # initial goal
        travel_goal = np.random.uniform(lower, upper)
        max_speed = 0.2 # m/s
        t = time.time()
        while self.send_position_updates:
            try:
                now = time.time()
                elapsed_time = now - t
                t = now
                # move the gantry towards the goal
                to_goal_vec = travel_goal - gantry_real_pos
                dist_to_goal = np.linalg.norm(to_goal_vec)
                if dist_to_goal < 0.03:
                    # choose new goal
                    travel_goal = np.random.uniform(lower, upper)
                else:
                    soft_speed = dist_to_goal * 0.25
                    # normalize
                    to_goal_vec = to_goal_vec / dist_to_goal
                    velocity = to_goal_vec * min(soft_speed, max_speed)
                    gantry_real_pos = gantry_real_pos + velocity * elapsed_time
                if random()>0.5:
                    dp = np.concatenate([[t], gantry_real_pos + np.random.normal(0, 0.05, (3,))])
                    self.datastore.gantry_pos.insert(dp)
                    self.to_ui_q.put({'gantry_observation': dp[1:]})
                # winch line always 1 meter
                self.datastore.winch_line_record.insert(np.array([t, 1.0, 0.0]))
                # range always perfect
                self.datastore.range_record.insert(np.array([t, gantry_real_pos[2]-1]))
                # anchor lines always perfectly agree with gripper position
                for i, simanc in enumerate(self.pe.anchor_points):
                    dist = np.linalg.norm(simanc - gantry_real_pos)
                    last = self.datastore.anchor_line_record[i].getLast()
                    timesince = t-last[0]
                    travel = dist-last[1]
                    speed = travel/timesince # referring to the specific speed of this line, not the gantry
                    self.datastore.anchor_line_record[i].insert(np.array([t, dist, speed, 1.0]))
                tt = self.datastore.anchor_line_record[0].getLast()[0]
                await asyncio.sleep(0.05)
            except asyncio.exceptions.CancelledError:
                break

    def collect_gant_frame_positions(self):
        result = np.zeros((4,3))
        for client in self.anchors:
            result[client.anchor_num] = client.last_gantry_frame_coords
        return result

    async def seek_gantry_goal(self):
        self.fig_8 = False
        self.to_ui_q.put({'gantry_goal_marker': self.gantry_goal_pos})
        while self.gantry_goal_pos is not None:
            vector = self.gantry_goal_pos - self.pe.gant_pos
            dist = np.linalg.norm(vector)
            if dist < 0.05:
                break
            vector = vector / dist
            result = await self.move_direction_speed(vector, 0.2, self.pe.gant_pos)
            await asyncio.sleep(0.2)
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
        self.fig_8 = False
        if speed == 0:
            for client in self.anchors:
                asyncio.create_task(client.send_commands({'aim_speed': 0}))
            return

        # apply downward bias and renormalize
        uvec = uvec + np.array([0,0,-0.08])
        uvec  = uvec / np.linalg.norm(uvec)

        anchor_positions = np.zeros((4,3))
        for a in self.anchors:
            anchor_positions[a.anchor_num] = np.array(a.anchor_pose[1])
        print(f'move direction speed {uvec} {speed}')

        # even if the starting position is off slightly, this method should not produce jerky moves.
        # because it's not commanding any absolute length from the spool motor
        if starting_pos is None:
            starting_pos = self.pe.gant_pos

        # line lengths at starting pos
        lengths_a = np.linalg.norm(starting_pos - anchor_positions, axis=1)
        # line lengths at new pos
        s = 10 # don't add a whole meter, curvature might matter at that distance.
        starting_pos += (uvec / s)
        lengths_b = np.linalg.norm(starting_pos - anchor_positions, axis=1)
        # length changes needed to travel 0.1 meters in uvec direction from starting_pos
        deltas = lengths_b - lengths_a
        line_speeds = deltas * s * speed
        print(f'computed line speeds {line_speeds}')
        
        if np.max(np.abs(line_speeds)) > 0.6:
            print('abort move because it\'s too fast')
            return

        # send move
        for client in self.anchors:
            asyncio.create_task(client.send_commands({'aim_speed': line_speeds[client.anchor_num]}))

    async def move_in_figure_8(self):
        """
        Move in a figure-8 pattern until interrupted by some other input.
        """
        # move to starting position (over origin)
        height = 1.5
        self.gantry_goal_pos = np.array([0,0,height])
        await self.seek_gantry_goal()
        # desired speed in meters per second
        speed = 0.2
        circuit_distance = 18.85
        circuit_time = circuit_distance / speed
        # multiply time.time by this to get a term that increases by 2pi every circuit_time seconds.
        slow = 2*math.pi / circuit_time
        start_time = time.time()

        # every 1 second, send a length plan to every anchor covering the next 1.3 seconds at an interval of 1/30 second
        send_interval = 1.0
        plan_duration = 1.3
        step_interval = 1/30
        self.fig_8 = True
        while self.fig_8:
            now = time.time()
            times = [now + step_interval * i for i in range(int(plan_duration / step_interval))]
            xy_positions = np.array([figure_8_coords((t - start_time) * slow) for t in times])
            positions = np.column_stack([xy_positions, np.repeat(height,len(xy_positions))])
            # calculate all line lengths in one statement
            distances = np.linalg.norm(self.pe.anchor_points[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)

            for client in self.anchors:
                plan = np.column_stack([times, distances[client.anchor_num]])
                message = {'length_plan' : plan.tolist()}
                asyncio.create_task(client.send_commands(message))

            await asyncio.sleep(send_interval)

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
