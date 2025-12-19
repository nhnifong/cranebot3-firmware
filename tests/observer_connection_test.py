import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import unittest
from unittest.mock import patch, Mock, ANY, MagicMock, AsyncMock
from multiprocessing import Pool, Queue
import numpy as np
from functools import partial
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
from config_loader import create_default_config, config_has_any_address

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
        self.anchor_points = np.array([
            [-2,  3, 2],
            [ 2,  3, 2],
            [ -1,-2, 2],
            [ -2,-2, 2],
        ], dtype=float)

        self.patchers = []
        self.watchable_startup_events = [asyncio.Event() for i in range(5)] # index=4 is gripper
        self.watchable_shutdown_event = asyncio.Event()

        self.mock_pe_class = Mock(spec=Positioner2)
        self.mock_pe = self.mock_pe_class.return_value
        async def mock_pe_main():
            await asyncio.sleep(0.01)
        self.mock_pe.main = mock_pe_main
        self.mock_pe.anchor_points = self.anchor_points
        self.mock_pe.gant_pos = np.array([0.4, 0.5, 0.6])
        self.patchers.append(patch('observer.Positioner2', self.mock_pe_class))

        self.clients_refuse_connections = False
        self.clients_disconnect_abnormally = False
        self.kill_clients = asyncio.Event()

        self.mock_gripper_client_class = Mock(spec=RaspiGripperClient)
        self.mock_gripper_client = self.mock_gripper_client_class.return_value
        self.mock_gripper_client.startup = partial(self.event_startup, 4)
        self.mock_gripper_client.shutdown = self.event_shutdown
        self.mock_gripper_client.slow_stop_spool = self.instant_nothing
        self.mock_gripper_client.connection_established_event = asyncio.Event()
        self.mock_gripper_client.last_frame_resized = None
        self.mock_gripper_client.connected = False

        self.patchers.append(patch('observer.RaspiGripperClient', self.mock_gripper_client_class))

        # Create four mock anchor clients
        self.mock_anchor_clients = [Mock(spec=RaspiAnchorClient) for _ in range(4)]
        for i, client in enumerate(self.mock_anchor_clients):
            client.startup = partial(self.event_startup, i)
            client.shutdown = self.event_shutdown
            client.slow_stop_spool = self.instant_nothing
            client.anchor_num = i
            client.last_frame_resized = None
            client.connected = False

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


        # The side_effect makes it so that the correct mock anchor client is returned
        self.patchers.append(patch('observer.RaspiAnchorClient', side_effect=lambda a, b, num, d, e, f, g, : self.mock_anchor_clients[num]))

        self.mock_pool = MagicMock()
        self.mock_pool.__enter__.return_value = self.mock_pool
        self.patchers.append(patch('observer.Pool', return_value=self.mock_pool))

        for p in self.patchers:
            p.start()

        # Create observer with test default config (no components are known)
        cfg = create_default_config()
        self.ob = AsyncObserver(False, cfg)
        # before running main, set it's zeroconf instance to a mock
        self.zc = MagicMock()
        self.zc.async_close = self.instant_nothing
        self.ob.aiozc = self.zc
        self.ob_task = asyncio.create_task(self.ob.main())
        await asyncio.wait_for(self.ob.startup_complete.wait(), 20)

    async def find_services(self):
        print('mock mock_zc_types.async_find called')
        return []

    async def asyncTearDown(self):
        self.kill_clients.set()
        self.ob.run_command_loop = False
        # allow up to 2 seconds for shutdown.
        await asyncio.wait_for(self.ob_task, 2)
        for p in self.patchers:
            p.stop()

    async def event_startup(self, i):
        """Mock client startup function"""
        # the event our unit test uses
        self.watchable_startup_events[i].set()
        print(f'client {i} startup called')
        if self.clients_refuse_connections:
            return
        if i<4:
            self.mock_anchor_clients[i].connected = True
        elif i==4:
            self.mock_gripper_client.connected = True
        # the event the observer is watching
        self.mock_gripper_client.connection_established_event.set()
        await self.kill_clients.wait()
        return self.clients_disconnect_abnormally

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

    async def advertise_service(self, name, properties={}):
        """
        Manually triggers the observer's on_service_state_change callback.
        Patches AsyncServiceInfo so that when observer calls add_service, 
        it receives the mock data we define here.
        """
        service_type = "_http._tcp.local."
        full_name = f"{name}.{service_type}"
        
        # Mock the AsyncServiceInfo that observer.add_service will instantiate
        with patch('observer.AsyncServiceInfo') as mock_service_info_cls:
            mock_instance = mock_service_info_cls.return_value
            mock_instance.async_request = AsyncMock()
            
            # Setup the data that observer.py expects to find on the info object
            mock_instance.server = name
            mock_instance.port = 8765
            # addresses is a list of bytes
            mock_instance.addresses = [b'\x7f\x00\x00\x01'] # 127.0.0.1
            mock_instance.properties = properties
            
            # Trigger the callback on the observer
            state = ServiceStateChange.Added
            
            # Call the handler directly
            self.ob.on_service_state_change(self.zc, service_type, full_name, state)
            
            # Yield to event loop so the task created by on_service_state_change can run
            await asyncio.sleep(0.01)
            
            return mock_instance

    async def observer_accepts_local_ui_connection_test(self):
        """observer should listen for local UI at localhost:4245"""
        async with websockets.connect("ws://127.0.0.1:4245") as ws:
            await asyncio.sleep(0.01)
            self.assertFalse(self.ob_task.done(), "Server should still be running after getting a connection")
            await ws.close()
            # the observer in this test was specifically initialized with terminate_with_ui=False
            self.assertFalse(self.ob_task.done(), "Server should still be running after losing a connection")

    async def test_keep_robot_connected_waits(self):
        """With a blank config, the observer should sleep and just wait for zeroconf"""
        self.assertFalse(config_has_any_address(self.ob.config))
        await asyncio.sleep(1.01) # one tick from keep_robot_connected
        self.assertIsNone(self.ob.gripper_client)
        self.assertEqual(0, len(self.ob.anchors))
        self.assertEqual(0, len(self.ob.connection_tasks))
        self.assertEqual(0, len(self.ob.bot_clients))

    async def test_discover_gripper(self):
        # advertise a service that matches a gripper.
        # it does not matter that we are running no such service.
        # when the observer creates a gripper client, it will create a mock object
        name = "123.cranebot-gripper-service.test"
        await self.advertise_service(name)
        # since we are immediately calling the observer's add_service callback, it should add this to the config immediately
        self.assertTrue(config_has_any_address(self.ob.config))
        self.assertEqual(name, self.ob.config.gripper.service_name)
        self.assertEqual('127.0.0.1', self.ob.config.gripper.address)
        self.assertEqual(8765, self.ob.config.gripper.port)

        # After half second, keep_robot_connected should wake up and start a client to connect to the gripper
        await asyncio.wait_for(self.watchable_startup_events[4].wait(), 0.51)
        self.assertTrue(self.ob.gripper_client is not None)
        self.assertEqual(1, len(self.ob.connection_tasks))
        # assert the client task is running.
        self.assertFalse(self.ob.connection_tasks[name].done())

    async def test_discover_anchor(self):
        # advertise a service that matches an anchor.
        name = "123.cranebot-anchor-service.test"
        await self.advertise_service(name)
        # since we are immediately calling the observer's add_service callback, it should add this to the config immediately
        self.assertTrue(config_has_any_address(self.ob.config))
        # the observer should assign the first anchor it discovers num=0
        self.assertEqual(name, self.ob.config.anchors[0].service_name)
        self.assertEqual('127.0.0.1', self.ob.config.anchors[0].address)
        self.assertEqual(8765, self.ob.config.anchors[0].port)

        # After half second, keep_robot_connected should wake up and start a client to connect to the gripper
        await asyncio.wait_for(self.watchable_startup_events[0].wait(), 0.51)
        self.assertTrue(self.ob.anchors[0] is not None)
        self.assertEqual(1, len(self.ob.connection_tasks))
        # assert the client task is running.
        self.assertFalse(self.ob.connection_tasks[name].done())

    async def test_anchor_connect_familiar(self):
        """
        Confirm that if observer is started with a non_blank configuration, it will connect to an anchor in that config
        In this test, the connection immediately succeeds
        """
        name = "123.cranebot-anchor-service.test"
        # name, address, and port must be set for observer to attempt a connection
        self.ob.config.anchors[0].service_name = name
        self.ob.config.anchors[0].address = '127.0.0.1'
        self.ob.config.anchors[0].port = 8765

        # After half second, keep_robot_connected should wake up and start a client to connect to the anchor
        await asyncio.wait_for(self.watchable_startup_events[0].wait(), 0.51)
        self.assertTrue(self.ob.anchors[0] is not None)
        self.assertEqual(1, len(self.ob.connection_tasks))
        # assert the client task is running.
        self.assertFalse(self.ob.connection_tasks[name].done())

    async def test_anchor_connect_familiar_late(self):
        """
        Confirm that if observer is started with a non_blank configuration,
        and attempts to connect to an anchor in that config
        and the connection is initially refused, but later works,
        The observer will still have only one connection task, one anchor, and it will be connected.
        """
        self.clients_refuse_connections = True

        name = "123.cranebot-anchor-service.test"
        # name, address, and port must be set for observer to attempt a connection
        self.ob.config.anchors[0].service_name = name
        self.ob.config.anchors[0].address = '127.0.0.1'
        self.ob.config.anchors[0].port = 8765
        self.assertTrue(config_has_any_address(self.ob.config))

        # After half second, keep_robot_connected should wake up and start a client to connect to the gripper
        # but it will return immediately, which is how the real clients behave when a connection is refused.
        await asyncio.wait_for(self.watchable_startup_events[0].wait(), 0.51)
        # client appeared to startup and refuse the connection. now make sure it was removed
        self.assertEqual(0, len(self.ob.anchors))
        self.assertEqual(0, len(self.ob.connection_tasks))

        # clients will now accept connections (startup() task will run indefinitely)
        self.watchable_startup_events[0].clear()
        self.clients_refuse_connections = False

        await asyncio.wait_for(self.watchable_startup_events[0].wait(), 0.51)
        self.assertTrue(self.ob.anchors[0] is not None)
        self.assertEqual(1, len(self.ob.connection_tasks))
        # assert the client task is running.
        self.assertFalse(self.ob.connection_tasks[name].done())

    async def test_anchor_connection_lost_reconnect(self):
        """Confirm that if observer loses a connection to an anchor that it reconnects as soon as possible."""

    async def test_anchor_connection_lost_abnormal_reconnect(self):
        """Confirm that if observer has a connection to an anchor abnormally,
        that it sends an appropriate message to a connected UI,
        and  that it reconnects to the anchor as soon as possible.
        """

    async def test_connect_all_components(self):
        """Confirm that when the observer has a properly filled out config, it connects to all components.
        Confirm that the ob.all_components_connected event is set within 1 second of the config being populated.
        """
        self.clients_refuse_connections = False
        names = []
        for i in range(4):
            name = f"123.cranebot-anchor-service.{i}"
            names.append(name)
            self.ob.config.anchors[i].service_name = name
            self.ob.config.anchors[i].address = '127.0.0.1'
            self.ob.config.anchors[i].port = 8765

        gripper_name = "123.cranebot-gripper-service.test"
        names.append(gripper_name)
        self.ob.config.gripper.service_name = gripper_name
        self.ob.config.gripper.address = '127.0.0.1'
        self.ob.config.gripper.port = 8765

        # After half second, keep_robot_connected should wake up and start a client to connect to all these components.
        # all mock components are pointing a the same startup event, and we're not going to both distinguishing between them.
        await asyncio.wait_for(asyncio.gather(*[e.wait() for e in self.watchable_startup_events]), 0.55)
        for i in range(4):
            self.assertTrue(self.ob.anchors[i] is not None)
        self.assertTrue(self.ob.gripper_client is not None)
        self.assertEqual(5, len(self.ob.connection_tasks))
        # assert the client task is running.
        for name in names:
            self.assertFalse(self.ob.connection_tasks[name].done())