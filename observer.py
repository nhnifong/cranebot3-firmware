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
from raspi_anchor_client import RaspiAnchorClient
from raspi_gripper_client import RaspiGripperClient
from random import random
from config import Config
from stats import StatCounter
from position_estimator import Positioner2
from cv_common import invert_pose, compose_poses, average_pose
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
        self.pe = Positioner2(self.datastore, self.to_ui_q, self.to_ob_q)

        self.last_gant_pos = np.zeros(3)
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
            if 'last_gant_pos' in updates:
                self.last_gant_pos = updates['last_gant_pos']
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
                lengths = updates['do_line_calibration']
                print(f'do_line_calibration lengths={lengths}')
                for client in self.anchors:
                    asyncio.create_task(client.send_commands({'reference_length': lengths[client.anchor_num]}))
            if 'jog_spool' in updates:
                if 'anchor' in updates['jog_spool']:
                    for client in self.anchors:
                        if client.anchor_num == updates['jog_spool']['anchor']:
                            asyncio.create_task(client.send_commands({
                                'aim_speed': updates['jog_spool']['speed']
                            }))
                elif 'gripper' in updates['jog_spool']:
                    # we can also jog the gripper spool
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
                asyncio.create_task(self.move_direction_speed(dir_sp['direction'], dir_sp['speed']))
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


    async def set_simulated_data_mode(self, mode):
        if self.sim_task is not None:
            self.sim_task.cancel()
            result = await self.sim_task
        if mode == 'circle':
            self.sim_task = asyncio.create_task(self.add_simulated_data_circle())
        elif mode == 'point2point':
            self.sim_task = asyncio.create_task(self.add_simulated_data_point2point())

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
            self.slow_stop_all_spools()
        elif mode == 'pose':
            self.calmode = mode
            self.locate_anchor_task = asyncio.create_task(self.locate_anchors())

    async def locate_anchors(self):
        """ find location of all anchors every half second until mode changes. """
        print('locate anchor task running')
        while self.calmode == 'pose' and self.send_position_updates:
            await asyncio.sleep(0.5)
            # using the record of recent origin detections, estimate the actual pose of each anchor.
            if len(self.anchors) != 4:
                print(f'anchor pose calibration should not be performed until all anchors are connected. len(anchors)={len(self.anchors)}')
                return
            for client in self.anchors:
                if len(client.origin_poses) < 6:
                    print(f'Too few origin observations ({len(client.origin_poses)}) from anchor {client.anchor_num}')
                    continue
                print(f'locating anchor {client.anchor_num} from {len(client.origin_poses)} detections')
                pose = self.optimize_single_anchor_pose(np.array(client.origin_poses))
                if pose is not None:
                    self.to_ui_q.put({'anchor_pose': (client.anchor_num, pose)})
        print(f'locate anchor task loop finished self.calmode={self.calmode}')

    def optimize_single_anchor_pose(self, observations):
        apose = np.array(invert_pose(compose_poses([model_constants.anchor_camera, average_pose(observations)])))
        # depending on which quadrant the average anchor pose falls in, constrain the XY rotation,
        # while still allowing very minor deviation because of crooked mounting and misalignment of the foam shock absorber on the camera.
        xsign = 1 if apose[1,0]>0 else -1
        ysign = 1 if apose[1,1]>0 else -1

        initial_guess = np.array(apose).reshape(6)
        initial_guess[0] = 0 # no x component in rotation axis
        initial_guess[1] = 0 # no x component in rotation axis
        initial_guess[2] = -xsign*(2-ysign)*pi/4 # one of four diagonals. points -Y towards middle of work area
        print(f'signs = ({xsign},{ysign}) initial guess {initial_guess}')

        bounds = np.array([
            (initial_guess[0] - 0.2, initial_guess[0] + 0.2), # x component of rotation vector
            (initial_guess[1] - 0.2, initial_guess[1] + 0.2), # y component of rotation vector
            (initial_guess[2] - 0.2, initial_guess[2] + 0.2), # z component of rotation vector
            (-8, 8), # x component of position
            (-8, 8), # y component of position
            ( 1, 6), # z component of position
        ])
        result = optimize.minimize(anchor_pose_cost_fn, initial_guess, args=(observations), method='SLSQP', bounds=bounds,
            options={'disp': False,'maxiter': 100}) # if it's not really fast we're not interested
        if result.success:
            pose = result.x.reshape(2,3)
            return pose
        return None

    def async_on_service_state_change(self, 
        zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange
    ) -> None:
        if 'cranebot' in name:
            print(f"Service {name} of type {service_type} state changed: {state_change}")
            if state_change is ServiceStateChange.Added:
                task = asyncio.create_task(self.add_service(zeroconf, service_type, name))
            elif state_change is ServiceStateChange.Updated:
                # it will already have been disconnectd and be in an exponential backoff retry loop trying to talk to the old address
                pass

    async def add_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        info = AsyncServiceInfo(service_type, name)
        await info.async_request(zc, 3000)
        if info:
            if info.server is None or info.server == '':
                return;
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
                        return
                    self.config.anchor_num_map[info.server] = anchor_num
                    self.config.anchors[anchor_num].service_name = info.server
                    self.config.write()

                ac = RaspiAnchorClient(address, anchor_num, self.datastore, self.to_ui_q, self.to_ob_q, self.pool, self.stat, self.shape_tracker)
                self.bot_clients[info.server] = ac
                self.anchors.append(ac)
                result = await ac.startup()
            elif name_component == cranebot_gripper_service_name:
                gc = RaspiGripperClient(address, self.datastore, self.to_ui_q, self.to_ob_q, self.pool, self.stat, self.pe)
                self.bot_clients[info.server] = gc
                self.gripper_client = gc
                result = await gc.startup()

    async def main(self, interfaces=InterfaceChoice.All) -> None:
        # main process loop
        with Pool(processes=10) as pool:
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

    async def lines_stable(self, threshold=0.05, timeout=10):
        """return once all lines have stopped moving"""
        last_lengths = None
        lengths = np.array([alr.getLast()[1] for alr in enumerate(self.datastore.anchor_line_record)])
        changes = [1,0,0,0]
        start = time.time()
        while sum(changes) > threshold and time.time()-start < timeout:
            if last_length is not None:
                changes = lengths - last_lengths
            last_lengths = lengths
            asyncio.sleep(0.3)

    async def move_direction_speed(self, uvec, speed, starting_pos=None):
        """Move in the direction of the given unit vector at the given speed.
        Any move must be based on some assumed starting position. if none is provided,
        we will use the last one sent from position_estimator
        """
        if speed == 0:
            for client in self.anchors:
                asyncio.create_task(client.send_commands({'aim_speed': 0}))
            return

        anchor_positions = np.zeros((4,3))
        for a in self.anchors:
            anchor_positions[a.anchor_num] = np.array(a.anchor_pose[1])
        print(f'move direction speed {uvec} {speed}')

        # even if the starting position is off slightly, this method should not produce jerky moves.
        # because it's not commanding any absolute length from the spool motor
        if starting_pos is None:
            starting_pos = self.last_gant_pos

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
