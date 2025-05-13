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
)
from multiprocessing import Pool
from math import sin,cos
import numpy as np
from data_store import DataStore
from raspi_anchor_client import RaspiAnchorClient
from raspi_gripper_client import RaspiGripperClient
from random import random
from segment import ShapeTracker
from config import Config
from stats import StatCounter
from position_estimator import Positioner2

fields = ['Content-Type', 'Content-Length', 'X-Timestamp-Sec', 'X-Timestamp-Usec']
cranebot_anchor_service_name = 'cranebot-anchor-service'
cranebot_gripper_service_name = 'cranebot-gripper-service'

# Manager of multiple tasks running clients connected to each robot component
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

        # FastSAM model
        self.shape_tracker = ShapeTracker()

        # Position Estimator. this used to be a seperate process so it's still somewhat independent.
        self.pe = Positioner2(self.datastore, self.to_ui_q, self.to_ob_q)

        self.last_gant_pos = np.zeros(3)
        self.sim_task = None

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
            if 'last_gant_pos' in updates:
                self.last_gant_pos = updates['last_gant_pos']
            if 'future_anchor_lines' in updates:
                fal = updates['future_anchor_lines']
                if not (fal['sender'] == 'pe' and self.calmode != 'run'):
                    for client in self.anchors:
                        message = {'length_plan' : fal['data'][client.anchor_num].tolist()}
                        if 'host_time' in fal:
                            message['host_time'] = fal['host_time']
                        asyncio.run_coroutine_threadsafe(client.send_commands(message), loop)

            if 'future_winch_line' in updates:
                fal = updates['future_winch_line']
                if not (fal['sender'] == 'pe' and self.calmode != 'run'):
                    if self.gripper_client is not None:
                        message = {'length_plan' : fal['data']}
                        asyncio.run_coroutine_threadsafe(self.gripper_client.send_commands(message, loop))
            if 'set_run_mode' in updates:
                print("set_run_mode") 
                self.set_run_mode(updates['set_run_mode'], loop)
            if 'do_line_calibration' in updates:
                lengths = updates['do_line_calibration']
                print(f'do_line_calibration lengths={lengths}')
                for client in self.anchors:
                    asyncio.run_coroutine_threadsafe(client.send_commands({'reference_length': lengths[client.anchor_num]}), loop)
            if 'equalize_line_tension' in updates:
                print('equalize line tension')
                asyncio.run_coroutine_threadsafe(self.equalize_tension(), loop)
            if 'jog_spool' in updates:
                if 'anchor' in updates['jog_spool']:
                    for client in self.anchors:
                        if client.anchor_num == updates['jog_spool']['anchor']:
                            asyncio.run_coroutine_threadsafe(client.send_commands({
                                'aim_speed': updates['jog_spool']['speed']
                            }), loop)
                elif 'gripper' in updates['jog_spool']:
                    # we can also jog the gripper spool
                    if self.gripper_client is not None:
                        asyncio.run_coroutine_threadsafe(self.gripper_client.send_commands({
                            'aim_speed': updates['jog_spool']['speed']
                        }), loop)
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
                asyncio.run_coroutine_threadsafe(self.move_direction_speed(dir_sp['direction'], dir_sp['speed']), loop)
            if 'set_grip' in updates:
                if self.gripper_client is not None:
                    asyncio.run_coroutine_threadsafe(self.gripper_client.send_commands({
                        'grip' : 'closed' if updates['set_grip'] else 'open'
                    }), loop)
            if 'slow_stop_one' in updates:
                if updates['slow_stop_one']['id'] == 'gripper':
                    if self.gripper_client is not None:
                        asyncio.run_coroutine_threadsafe(self.gripper_client.slow_stop_spool(), loop)
                else:
                    for client in self.anchors:
                        if client.anchor_num == updates['slow_stop_one']['id']:
                            asyncio.run_coroutine_threadsafe(client.slow_stop_spool(), loop)
            if 'slow_stop_all' in updates:
                self.slow_stop_all_spools(loop)
            if 'stop_if_not_slack' in updates:
                # a command to stop any spools that were reeling in during tension equaliztion
                for client in self.anchors:
                    asyncio.run_coroutine_threadsafe(client.send_commands({'equalize_tension': {'action': 'stop_if_not_slack'}}), loop)
            if 'measure_ref_load' in updates:
                for client in self.anchors:
                    if client.anchor_num == updates['measure_ref_load']['anchor_num']:
                        asyncio.run_coroutine_threadsafe(client.send_commands({
                            'measure_ref_load': updates['measure_ref_load']['load']
                            }), loop)
            if 'set_simulated_data_mode' in updates:
                m = updates['set_simulated_data_mode']
                asyncio.run_coroutine_threadsafe(self.set_simulated_data_mode(m), loop)

    async def set_simulated_data_mode(self, mode):
        if self.sim_task is not None:
            self.sim_task.cancel()
            result = await self.sim_task
        if mode == 'circle':
            self.sim_task = asyncio.create_task(self.add_simulated_data_circle())
        elif mode == 'point2point':
            self.sim_task = asyncio.create_task(self.add_simulated_data_point2point())

    def slow_stop_all_spools(self, loop):
        for name, client in self.bot_clients.items():
            # Slow stop all spools. gripper too
            asyncio.run_coroutine_threadsafe(client.slow_stop_spool(), loop)

    def set_run_mode(self, mode, loop):
        """
        Sets the calibration mode of connected bots
        "run" - not in a calibration mode
        "pose" - observe the origin board
        "pause" - hold all motors at current position, but continue to make observations
        """
        if mode == "run":
            if self.calmode == "pose":
                config = Config()
                for client in self.anchors:
                    client.calibration_mode = False
                    config.anchors[client.anchor_num].pose = client.anchor_pose
                config.write()
                print('Wrote new anchor poses to configuration.json')
            self.calmode = mode
            print("run mode")
        elif mode == "pose":
            self.calmode = mode
            for name, client in self.bot_clients.items():
                client.calibration_mode = True
                print(f'setting {name} to pose calibration mode')
        elif mode == "pause":
            if self.calmode == "pose":
                # call calibrate_pose on all anchors when exiting pose calibration mode
                for client in self.anchors:
                    client.calibrate_pose()
                    client.calibration_mode = False
            self.calmode = mode
            self.slow_stop_all_spools(loop)

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

    async def main(self) -> None:
        # main process loop
        with Pool(processes=6) as pool:
            self.pool = pool
            self.aiozc = AsyncZeroconf(ip_version=IPVersion.All)

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
            # asyncio.create_task(self.monitor_tension())
            # asyncio.create_task(self.run_shape_tracker())
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
        if self.aiobrowser is not None:
            await self.aiobrowser.async_cancel()
        if self.aiozc is not None:
            await self.aiozc.async_close()
        if self.sim_task is not None:
            result = await self.sim_task
        result = await asyncio.gather(*[client.shutdown() for client in self.bot_clients.values()])

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


    async def run_tension_based_line_calibration(self):
        """
        Tension based line calibration process is as follows

        Set the reference length on the lines the original way (aruco observation of gantry)
        Starting with taut lines
        do while disparity in line tension after a move is large,
            Record frame position of gantry in each cam
            Move to a new position, maybe 20cm away.
            Measure disparity in line tension
            Equalize the tension on the lines, as estimated by the motor shaft error.
            Change reference length by whatever amount of line was spooled or unspooled.
            Record frame position of gantry in each cam. Store frame positions (start and finish) along with line deltas in dataset. 
        """
        for move_n in range(4):
            starts = collect_gant_frame_positions()
            vector = np.random.normal(0, 0.1, (3))
            vector = vector / np.linalg.norm(vector)
            self.move_direction_speed(vector, 0.1)
            await asyncio.sleep(2)
            self.move_direction_speed(None, 0)
            await self.lines_stable()
            await self.equalize_tension()
            finishes = collect_gant_frame_positions()

    async def equalize_tension(self):
        """Inner loop of run_tension_based_line_calibration"""
        print('equalize_tension 1')
        for client in self.anchors:
            client.tension_seek_running = True
            print('send_commands 1')
            asyncio.create_task(client.send_commands({'equalize_tension': {
                'action': 'start',
            }}))

        print('equalize_tension 2')
        # wait for all clients to finish
        while any([a.tension_seek_running for a in self.anchors]):
            await asyncio.sleep(1/10)
        asyncio.create_task(client.send_commands({'equalize_tension': {'action': 'complete'}}))
        print('tension equalization finished.')


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
