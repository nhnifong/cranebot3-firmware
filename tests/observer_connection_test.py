import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import unittest
from unittest.mock import patch, Mock, ANY, MagicMock, AsyncMock
from multiprocessing import Pool, Queue
import numpy as np
from observer import AsyncObserver
from position_estimator import Positioner2
from raspi_anchor_client import RaspiAnchorClient
from raspi_gripper_client import RaspiGripperClient
from math import pi
from cv_common import invert_pose, compose_poses
import model_constants
import time
from zeroconf import IPVersion, ServiceStateChange
from zeroconf.asyncio import AsyncZeroconf

# Todo, automate the following tests
# with a blank config, start observer, allow it to discover the bots, assert it wrote the addresses in the config.
# the remainining tests are to be run when the observer has a config that already tells it the ip addresses of the bots

# start observer, turn on bot (mock clients change from refusing connections to accepting them), wait for all good, close observer
# start observer, turn on bot, before all connections succeed, close observer.
# start observer, wait for it to fail to connect to known bots at least once, then turn the bot on, and let it connect. assert self.anchors is only four clients. close observer
# turn on bot first (mock clients connect ok), start observer, wait for all good, close observer
# turn on bot first, start observer, wait for all good, kill bot, close observer
# start observer, turn on bot, wait for all good, kill bot, power up bot, wait for all good, close observer.
# turn on bot, start observer, wait for all good, kill bot, power up bot, wait for all good, close observer.

class TestObserver(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.to_ui_q = Queue()
        self.to_ob_q = Queue()
        self.to_ui_q.cancel_join_thread()
        self.to_ob_q.cancel_join_thread()

        self.anchor_points = np.array([
            [-2,  3, 2],
            [ 2,  3, 2],
            [ -1,-2, 2],
            [ -2,-2, 2],
        ], dtype=float)

        self.patchers = []
        self.watchable_startup_event = asyncio.Event()
        self.watchable_shutdown_event = asyncio.Event()

        self.mock_pe_class = Mock(spec=Positioner2)
        self.mock_pe = self.mock_pe_class.return_value
        async def mock_pe_main():
            await asyncio.sleep(0.01)
        self.mock_pe.main = mock_pe_main
        self.mock_pe.anchor_points = self.anchor_points
        self.mock_pe.gant_pos = np.array([0.4, 0.5, 0.6])
        self.patchers.append(patch('observer.Positioner2', self.mock_pe_class))

        self.mock_gripper_client_class = Mock(spec=RaspiGripperClient)
        self.mock_gripper_client = self.mock_gripper_client_class.return_value
        self.mock_gripper_client.startup = self.event_startup
        self.mock_gripper_client.shutdown = self.event_shutdown
        self.mock_gripper_client.slow_stop_spool = self.instant_nothing
        self.mock_gripper_client.connection_established_event = asyncio.Event()
        self.mock_gripper_client.last_frame_resized = None
        self.patchers.append(patch('observer.RaspiGripperClient', self.mock_gripper_client_class))

        # Create four mock anchor clients
        self.mock_anchor_clients = [Mock(spec=RaspiAnchorClient) for _ in range(4)]
        for i, client in enumerate(self.mock_anchor_clients):
            client.startup = self.event_startup
            client.shutdown = self.event_shutdown
            client.slow_stop_spool = self.instant_nothing
            client.anchor_num = i

        self.async_service_browser_mock = MagicMock()
        d = MagicMock()
        d.async_cancel = self.instant_nothing
        self.async_service_browser_mock.return_value = d
        self.mock_zc_service_browser_patch = patch('observer.AsyncServiceBrowser', self.async_service_browser_mock)
        self.patchers.append(self.mock_zc_service_browser_patch)

        self.mock_zc_types_patch = patch('zeroconf.asyncio.AsyncZeroconfServiceTypes.async_find', new_callable=AsyncMock)
        self.mock_zc_types_find = self.mock_zc_types_patch.start()
        self.mock_zc_types_find.return_value = [] # Return empty list immediately
        self.patchers.append(self.mock_zc_types_patch)


        # The side_effect makes it so each call to the constructor returns the next mock
        self.patchers.append(patch('observer.RaspiAnchorClient', side_effect=self.mock_anchor_clients))

        self.mock_pool = MagicMock()
        self.mock_pool.__enter__.return_value = self.mock_pool
        self.patchers.append(patch('observer.Pool', return_value=self.mock_pool))

        for p in self.patchers:
            p.start()

        # Zeroconf will attempt to discover services .
        self.ob = AsyncObserver(False)
        # before starting, set it's zeroconf instance to a special version that searches on localhost only
        # self.zc = AsyncZeroconf(ip_version=IPVersion.All)
        self.zc = MagicMock()
        self.zc.async_close = self.instant_nothing
        self.ob.aiozc = self.zc
        self.ob_task = asyncio.create_task(self.ob.main())
        await asyncio.wait_for(self.ob.startup_complete.wait(), 20)

    async def find_services(self):
        print('mock mock_zc_types.async_find called')
        return []

    async def asyncTearDown(self):
        # this should have the effect of making the update listener task stop, which the main task is awaiting.
        # it should then run it's own async_close method, which runs the async shutdown methods on zeroconf
        # and any connected clients.
        # print('test teardown is unregistering all services')
        # await self.ob.aiozc.async_unregister_all_services()
        self.to_ob_q.put({'STOP':None})
        # allow up to 10 seconds for shutdown.
        await asyncio.wait_for(self.ob_task, 10)
        for p in self.patchers:
            p.stop()

    async def event_startup(self):
        """Mock client startup function"""
        print('client startup called')
        self.mock_gripper_client.connection_established_event.set()
        self.watchable_startup_event.set()
        return False

    async def event_shutdown(self):
        """Mock client shutdown function"""
        print('client shutdown called')
        self.watchable_shutdown_event.set()

    async def instant_nothing(self):
        pass

    async def _setup_mock_anchors(self):
        """Helper method to populate the observer's anchor list with our mocks."""
        self.ob.anchors = self.mock_anchor_clients
        for i, client in enumerate(self.ob.anchors):
            # Give each mock an anchor_pose attribute, as used in the method under test
            client.anchor_pose = (np.identity(3), self.anchor_points[i])

    async def test_startup_shutdown(self):
        # this test is only confirming that that asyncSetUp, which calls ob.main(), works correctly,
        # and that asyncTearDown, which just sends a STOP command to the ob queue, ultimately results
        # in it shutting down after calling it's own async_close.
        # If this test fails, you'll know it's a startup/shutdown issue even if all the others fail to.
        # you can then run this one in isolation.
        self.assertFalse(self.ob_task.done())

    async def advertise_service(self, name, service_type, port, properties={}):
        """
        Manually triggers the observer's on_service_state_change callback.
        Patches AsyncServiceInfo so that when observer calls add_service, 
        it receives the mock data we define here.
        """
        full_name = f"{name}.{service_type}"
        
        # Mock the AsyncServiceInfo that observer.add_service will instantiate
        with patch('observer.AsyncServiceInfo') as mock_service_info_cls:
            mock_instance = mock_service_info_cls.return_value
            mock_instance.async_request = AsyncMock()
            
            # Setup the data that observer.py expects to find on the info object
            mock_instance.server = name
            mock_instance.port = port
            # addresses is a list of bytes
            mock_instance.addresses = [b'\x7f\x00\x00\x01'] # 127.0.0.1
            mock_instance.properties = properties
            
            # Trigger the callback on the observer
            state = ServiceStateChange.Added
            
            # Call the handler directly
            self.ob.on_service_state_change(self.zc, service_type, full_name, state)
            
            # Yield to event loop so the task created by on_service_state_change can run
            await asyncio.sleep(0.05)
            
            return mock_instance

    async def test_discover_gripper(self):
        # advertise a service on localhost that matches a gripper.
        # it does not matter that we are running no such service.
        # when the observer creates a gripper client, it will create a mock object
        # gripper client to server communication is tested in gripper_client_test. 
        await self.advertise_service(f"123.cranebot-gripper-service.test", "_http._tcp.local.", 8765)
        # by default zeroconf checks for advertized services every 10 seconds
        # so if the timeout is raised, then it means observer didn't discover the service.
        await asyncio.wait_for(self.watchable_startup_event.wait(), 10)
        self.assertFalse(self.ob_task.done())
        self.assertTrue(self.ob.gripper_client is not None)

    async def test_anchor_connect_familiar(self):
        """Confirm that we can connnect to an anchor that advertises a name we recognize from our configuration"""
        await self.advertise_service(f"123.cranebot-anchor-service.test_0", "_http._tcp.local.", 8765)
        await asyncio.wait_for(self.watchable_startup_event.wait(), 10)
        self.assertFalse(self.ob_task.done())
        self.assertEqual(len(self.ob.anchors), 1)

    async def test_anchor_reconnect(self):
        """Confirm that if an anchor server shuts down cleanly and restarts, that we reconnect to it.
        In this test, we are neither running a real websocket server, or client. the client is a mock and the server doesn't exist.
        all we are confirming here is that if the MDNS advertisement for the services goes down and back up, that we start the client task again.
        """
        info = await self.advertise_service(f"123.cranebot-anchor-service.test_0", "_http._tcp.local.", 8765)
        await asyncio.wait_for(self.watchable_startup_event.wait(), 10)
        self.assertFalse(self.ob_task.done())
        self.assertEqual(len(self.ob.anchors), 1)

        await self.ob.aiozc.async_unregister_service(info)
        await asyncio.sleep(2)
        self.assertEqual(len(self.ob.anchors), 0)
        print('observer removed anchor client correctly, re-advertising service')

        self.watchable_startup_event.clear()
        await asyncio.sleep(0.01) # it is necessary to reliquish control of the event loop after calling clear or it may not have the intended effect

        info = await self.advertise_service(f"123.cranebot-anchor-service.test_0", "_http._tcp.local.", 8765)
        await asyncio.wait_for(self.watchable_startup_event.wait(), 20)
        self.assertEqual(len(self.ob.anchors), 1)
