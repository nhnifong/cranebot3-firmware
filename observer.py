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
from raspi_anchor_client import RaspiAnchorClient
from raspi_gripper_client import RaspiGripperClient
from random import random
from segment import ShapeTracker
from config import Config

TENSION_SLACK_THRESH = 0.4

fields = ['Content-Type', 'Content-Length', 'X-Timestamp-Sec', 'X-Timestamp-Usec']
cranebot_anchor_service_name = 'cranebot-anchor-service'
cranebot_gripper_service_name = 'cranebot-gripper-service'

class StatCounter:
    def __init__(self, to_ui_q):
        self.to_ui_q = to_ui_q
        self.detection_count = 0
        self.latency = []
        self.framerate = []
        self.last_update = time.time()
        self.run = True

    async def stat_main(self):
        while self.run:
            now = time.time()
            elapsed = now-self.last_update
            mean_latency = np.mean(np.array(self.latency))
            mean_framerate = np.mean(np.array(self.framerate))
            detection_rate = self.detection_count / elapsed
            self.last_update = now
            self.latency = []
            self.framerate = []
            self.detection_count = 0
            self.to_ui_q.put({'vid_stats':{
                'detection_rate':detection_rate,
                'video_latency':mean_latency,
                'video_framerate':mean_framerate,
                }})
            await asyncio.sleep(0.5)

# Manager of multiple tasks running clients connected to each robot component
class AsyncObserver:
    def __init__(self, datastore, to_ui_q, to_pe_q, to_ob_q) -> None:
        self.position_update_task = None
        self.aiobrowser: AsyncServiceBrowser | None = None
        self.aiozc: AsyncZeroconf | None = None
        self.send_position_updates = True
        self.calmode = "pause"
        self.detection_count = 0

        self.datastore = datastore
        self.to_ui_q = to_ui_q
        self.to_pe_q = to_pe_q
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

        self.stat = StatCounter(to_ui_q)

        # FastSAM model
        self.shape_tracker = ShapeTracker()

    def listen_position_updates(self, loop):
        """
        Receive any updates on our process input queue

        this thread doesn't actually have a running event loop.
        so run any coroutines back in the main thread with asyncio.run_coroutine_threadsafe
        """
        while self.send_position_updates:
            updates = self.to_ob_q.get()
            if 'STOP' in updates:
                print('stopping listen_position_updates thread due to STOP message in queue')
                break
            if 'future_anchor_lines' in updates:
                fal = updates['future_anchor_lines']
                if not (fal['sender'] == 'pe' and self.calmode != 'run'):
                    for client in self.anchors:
                        asyncio.run_coroutine_threadsafe(client.send_commands({
                            'length_plan' : fal['data'][client.anchor_num].tolist()
                        }), loop)

            if 'future_winch_line' in updates:
                fal = updates['future_anchor_lines']
                if not (fal['sender'] == 'pe' and self.calmode != 'run'):
                    if self.gripper_client is not None:
                        asyncio.run_coroutine_threadsafe(self.gripper_client.send_commands({
                            'length_plan' : fal['data']
                        }), loop)
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
                            # send an anchor the command 'jog' with a relative length change in meters.
                            asyncio.run_coroutine_threadsafe(client.send_commands({
                                'jog' : updates['jog_spool']['rel']
                            }), loop)
                elif 'gripper' in updates['jog_spool']:
                    print(f"jog gripper spool {updates['jog_spool']['rel']}")
                    # we can also jog the gripper spool
                    if self.gripper_client is not None:
                        asyncio.run_coroutine_threadsafe(self.gripper_client.send_commands({
                            'jog' : updates['jog_spool']['rel']
                        }), loop)
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
            if 'measure_no_load' in updates:
                for client in self.anchors:
                    asyncio.run_coroutine_threadsafe(client.send_commands({'measure_no_load': None}), loop)

    def slow_stop_all_spools(self, loop):
        for name, client in self.bot_clients.items():
            # Slow stop all spools. gripper too
            asyncio.run_coroutine_threadsafe(client.slow_stop_spool(), loop)

    def set_run_mode(self, mode, loop):
        """
        Sets the calibration mode of connected bots
        "run" - not in a calibration mode
        "cam" - calibrate distortion parameters of cameras
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

                ac = RaspiAnchorClient(address, anchor_num, self.datastore, self.to_ui_q, self.to_pe_q, self.to_ob_q, self.pool, self.stat, self.shape_tracker)
                self.bot_clients[info.server] = ac
                self.anchors.append(ac)
                await ac.startup()
            elif name_component == cranebot_gripper_service_name:
                gc = RaspiGripperClient(address, self.datastore, self.to_ui_q, self.to_pe_q, self.to_ob_q, self.pool, self.stat)
                self.bot_clients[info.server] = gc
                self.gripper_client = gc
                await gc.startup()

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

            print("start position listener")
            self.position_update_task = asyncio.create_task(asyncio.to_thread(self.listen_position_updates, loop=asyncio.get_running_loop()))

            asyncio.create_task(self.stat.stat_main())
            # asyncio.create_task(self.monitor_tension())
            # asyncio.create_task(self.run_shape_tracker())
            # asyncio.create_task(self.add_simulated_data())
            

            # await something that will end when the program closes that to keep zeroconf alive and discovering services.
            try:
                result = await self.position_update_task
            except asyncio.exceptions.CancelledError:
                pass
            await self.async_close()

    async def async_close(self) -> None:
        self.send_position_updates = False
        self.stat.run = False
        if self.aiobrowser is not None:
            await self.aiobrowser.async_cancel()
        if self.aiozc is not None:
            await self.aiozc.async_close()
        for client in self.bot_clients.values():
            client.shutdown()

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

    async def add_simulated_data(self):
        sim_anchors = np.array([
            [-2,2.6, 3],
            [ 2,2.6, 3],
            [ -1,2.6,-2],
            [ -2,2.6,-2]])
        while self.send_position_updates:
            t = time.time()
            # move the gantry in a circle
            dp = np.array([t, 0,0,0, sin(t/8) + random()*0.2, cos(t/8) + random()*0.2, 1.8 + random()*0.2])
            self.datastore.gantry_pose.insert(dp)
            # winch line always 1 meter
            self.datastore.winch_line_record.insert(np.array([t, 1.0]))
            # grippers always directly below gantry
            dp[6] = 1 + random()*0.2
            self.datastore.gripper_pose.insert(dp)
            # anchor lines always perfectly agree with gripper position
            for i, simanc in enumerate(sim_anchors):
                dist = np.linalg.norm(simanc - dp[4:])
                self.datastore.anchor_line_record[i].insert(np.array([t, dist]))
            await asyncio.sleep(0.15)

    async def run_tension_based_line_calibration(self):
        """
        Tension based line calibration process is as follows

        Set the reference length on the lines the original way (aruco observation of gantry)
            do while disparity in line tension is large,
                Move to a new position, maybe 20cm away.
                Measure disparity in line tension
                Equalize the tension on the lines, as estimated by the motor shaft error.
                Change reference length by whatever amount of line was spooled or unspooled.
        """
        pass

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


def start_observation(datastore, to_ui_q, to_pe_q, to_ob_q):
    """
    Entry point to be used when starting this from main.py with multiprocessing
    """
    ob = AsyncObserver(datastore, to_ui_q, to_pe_q, to_ob_q)
    asyncio.run(ob.main())

if __name__ == "__main__":
    from multiprocessing import Queue
    from data_store import DataStore
    datastore = DataStore(horizon_s=10, n_cables=4)
    to_ui_q = Queue()
    to_pe_q = Queue()
    to_ob_q = Queue()
    to_ui_q.cancel_join_thread()
    to_pe_q.cancel_join_thread()
    to_ob_q.cancel_join_thread()

    # when running as a standalone process (debug only, linux only), register signal handler
    def stop():
        print("\nwait for clean observer shutdown")
        to_ob_q.put({'STOP':None})
    async def main():
        runner = AsyncObserver(datastore, to_ui_q, to_pe_q, to_ob_q)
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(getattr(signal, 'SIGINT'), stop)
        await runner.main()
    asyncio.run(main())
