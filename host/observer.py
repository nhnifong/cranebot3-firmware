from __future__ import annotations

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

# global that will point to a DataStore passed to this process.
datastore = None
# queue for sending info to user interface
to_ui_q = None
# queue for sending info to position estimator
to_pe_q = None
# queue for receiving info meant for this process
to_ob_q = None
# global calibration mode
calibration_mode = True


cranebot_anchor_service_name = 'cranebot-anchor-service'
cranebot_gripper_service_name = 'cranebot-gripper-service'

# Manager of multiple tasks running clients connected to each robot component
class AsyncDiscovery:
    def __init__(self) -> None:
        self.aiobrowser: AsyncServiceBrowser | None = None
        self.aiozc: AsyncZeroconf | None = None
        self.send_position_updates = True

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

    def listen_position_updates(self):
        while self.send_position_updates:     
            updates = to_ob_q.get()
            if 'future_anchor_lines' in updates:
                # this should have one column for each anchor
                for name, client in self.bot_clients:
                    client.send_anchor_commands({'length_plan' : updates['future_anchor_lines'][client.anchor_num]})
            if 'future_winch_line' in updates:
                pass

    def async_on_service_state_change(self, 
        zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange
    ) -> None:
        if 'cranebot' in name:
            print(f"Service {name} of type {service_type} state changed: {state_change}")
            if state_change is ServiceStateChange.Added:
                task = asyncio.create_task(self.add_service(zeroconf, service_type, name))

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
                    self.save_anchor_num_map()
                ac = RaspiAnchorClient(address, anchor_num, datastore, to_ui_q, to_pe_q)
                self.bot_clients[info.server] = ac
                await ac.connect_all()

    async def async_run(self) -> None:
        self.position_update_task = asyncio.to_thread(self.listen_position_updates)
        self.aiozc = AsyncZeroconf(ip_version=IPVersion.All)

        services = list(
            await AsyncZeroconfServiceTypes.async_find(aiozc=self.aiozc, ip_version=IPVersion.All)
        )
        self.aiobrowser = AsyncServiceBrowser(
            self.aiozc.zeroconf, services, handlers=[self.async_on_service_state_change]
        )

        while True:
            try:
                await asyncio.sleep(1)
            except asyncio.exceptions.CancelledError:
                await self.async_close()

    async def async_close(self) -> None:
        assert self.aiozc is not None
        assert self.aiobrowser is not None
        await self.aiobrowser.async_cancel()
        await self.aiozc.async_close()
        await asyncio.gather(*[client.close_all() for name,client in self.bot_clients.items()])
        self.send_position_updates = False
        await self.position_update_task

def start_observation(shared_array, to_ui_q, to_pe_q, to_ob_q):
    # set the global 
    datastore = shared_array
    to_ui_q = to_ui_q # for sending to the UI
    to_pe_q = to_pe_q # for sending to the position estimator
    to_ob_q = to_ob_q # queue where other processes send to us

    async def main():
        runner = AsyncDiscovery()
        try:
            await runner.async_run()
        except KeyboardInterrupt:
            await runner.async_close()
    asyncio.run(main())

if __name__ == "__main__":
    from multiprocessing import Queue
    from data_store import DataStore
    datastore = DataStore(horizon_s=10, n_cables=4)
    to_ui_q = Queue()
    to_pe_q = Queue()
    to_ob_q = Queue()
    start_observation(datastore, to_ui_q, to_pe_q, to_ob_q)