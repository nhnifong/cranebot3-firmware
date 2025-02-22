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
from raspi_anchor_client import RaspiAnchorClient

fields = ['Content-Type', 'Content-Length', 'X-Timestamp-Sec', 'X-Timestamp-Usec']
cranebot_anchor_service_name = 'cranebot-anchor-service'
cranebot_gripper_service_name = 'cranebot-gripper-service'

# Manager of multiple tasks running clients connected to each robot component
class AsyncObserver:
    def __init__(self, datastore, to_ui_q, to_pe_q, to_ob_q, pool) -> None:
        self.position_update_task = None
        self.aiobrowser: AsyncServiceBrowser | None = None
        self.aiozc: AsyncZeroconf | None = None
        self.send_position_updates = True
        self.calmode = "run"

        self.datastore = datastore
        self.to_ui_q = to_ui_q
        self.to_pe_q = to_pe_q
        self.to_ob_q = to_ob_q
        self.pool = pool

        # keyed by server name
        self.bot_clients = {}

        # read a mapping of server names to anchor numbers from a file
        self.next_available_anchor_num = 0;
        self.anchor_num_map = {}
        self.load_anchor_num_map()

    def load_anchor_num_map(self):
        try:
            with open('anchor_mapping.txt', 'r') as f:
                for line in f:
                    s = line.split(':')
                    self.anchor_num_map[s[0]] = int(s[1])
            if len(self.anchor_num_map) == 0:
                return
            self.next_available_anchor_num = max(self.anchor_num_map.values())+1
        except FileNotFoundError:
            pass

    def save_anchor_num_map(self):
        with open('anchor_mapping.txt', 'w') as f:
            for k,v in self.anchor_num_map.items():
                f.write(f'{k}:{v}\n')

    def listen_position_updates(self, loop):
        """
        Receive any updates on our process input queue
        """
        while self.send_position_updates:
            updates = self.to_ob_q.get()
            if 'STOP' in updates:
                print('stopping listen_position_updates thread due to STOP message in queue')
                break
            if 'future_anchor_lines' in updates and self.set_calibration_mode == 'run':
                # this should have one column for each anchor
                for client in self.bot_clients.values():
                    # this thread doesn't actually have a running event loop.
                    # so run this back in the main thread.
                    asyncio.run_coroutine_threadsafe(client.send_anchor_commands({
                        'length_plan' : updates['future_anchor_lines'][client.anchor_num]
                    }), loop)
            if 'future_winch_line' in updates and self.set_calibration_mode == 'run':
                pass
            if 'set_calibration_mode' in updates:
                print("set_calibration_mode") 
                self.set_calibration_mode(updates['set_calibration_mode'])

    def set_calibration_mode(self, mode):
        """
        Sets the calibration mode of connected bots
        "run" - not in a calibration mode
        "cam" - calibrate distortion parameters of cameras
        "pose" - observe the origin board
        """
        if mode == "run":
            if self.calmode == "pose":
                # call calibrate_pose on all anchors when exiting pose calibration mode
                for name, client in self.bot_clients.items():
                    client.calibrate_pose()
                    client.calibration_mode = False
            self.calmode = mode
            print("run mode")
        elif mode == "cam":
            pass
        elif mode == "pose":
            self.calmode = mode
            for name, client in self.bot_clients.items():
                client.calibration_mode = True
                print(f'setting {name} to pose calibration mode')

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

            if name.split(".")[1] == cranebot_anchor_service_name:
                # the number of anchors is decided ahead of time (in main.py)
                # but they are assigned numbers as we find them on the network
                # and the chosen numbers are persisted on disk
                if info.server in self.anchor_num_map:
                    anchor_num = self.anchor_num_map[info.server]
                else:
                    anchor_num = self.next_available_anchor_num
                    self.anchor_num_map[info.server] = anchor_num
                    self.next_available_anchor_num += 1
                    self.save_anchor_num_map()
                ac = RaspiAnchorClient(address, anchor_num, self.datastore, self.to_ui_q, self.to_pe_q, self.pool)
                self.bot_clients[info.server] = ac
                await ac.startup()

    async def main(self) -> None:
        # main process loop
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


        # await something that will end when the program closes that to keep zeroconf alive and discovering services.
        try:
            await self.position_update_task
        except asyncio.exceptions.CancelledError:
            pass
        await self.async_close()

    async def async_close(self) -> None:
        self.send_position_updates = False
        if self.aiobrowser is not None:
            await self.aiobrowser.async_cancel()
        if self.aiozc is not None:
            await self.aiozc.async_close()
        for client in self.bot_clients.values():
            client.shutdown()
        # if self.position_update_task:
        #     await self.position_update_task

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