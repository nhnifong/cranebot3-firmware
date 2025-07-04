import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import unittest
from unittest.mock import patch, Mock
from multiprocessing import Pool, Queue
import numpy as np
from observer import AsyncObserver
from position_estimator import Positioner2
from raspi_anchor_client import RaspiAnchorClient
from raspi_gripper_client import RaspiGripperClient
import zeroconf
from math import pi
from cv_common import invert_pose, compose_poses
import model_constants
import time

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
        self.watchable_event = asyncio.Event()

        self.mock_pe_class = Mock(spec=Positioner2)
        self.mock_pe = self.mock_pe_class.return_value
        async def mock_pe_main():
            await asyncio.sleep(0.01)
        self.mock_pe.main = mock_pe_main
        self.mock_pe.anchor_points = self.anchor_points
        self.patchers.append(patch('observer.Positioner2', self.mock_pe_class))

        self.mock_gripper_client_class = Mock(spec=RaspiGripperClient)
        self.mock_gripper_client = self.mock_gripper_client_class.return_value
        self.mock_gripper_client.startup = self.watchable_routine
        self.mock_gripper_client.shutdown = self.watchable_routine
        self.patchers.append(patch('observer.RaspiGripperClient', self.mock_gripper_client_class))

        self.mock_anchor_client_class = Mock(spec=RaspiAnchorClient)
        self.mock_anchor_client = self.mock_anchor_client_class.return_value
        self.mock_anchor_client.startup = self.watchable_routine
        self.mock_anchor_client.shutdown = self.watchable_routine
        self.patchers.append(patch('observer.RaspiAnchorClient', self.mock_anchor_client_class))

        for p in self.patchers:
            p.start()

        # Zeroconf will attempt to discover services on localhost only.
        self.ob = AsyncObserver(self.to_ui_q, self.to_ob_q)
        self.ob_task = asyncio.create_task(self.ob.main(interfaces=["127.0.0.1"]))
        await asyncio.sleep(0.1)

    async def asyncTearDown(self):
        # this should have the effect of making the update listener task stop, which the main task is awaiting.
        # it should then run it's own async_close method, which runs the async shutdown methods on zeroconf
        # and any connected clients.
        await self.ob.aiozc.async_unregister_all_services()
        self.to_ob_q.put({'STOP':None})
        await self.ob_task
        for p in self.patchers:
            p.stop()

    async def watchable_routine(self):
        self.watchable_event.set()

    async def test_startup_shutdown(self):
        self.assertFalse(self.ob_task.done())

    async def advertise_service(self, name, service_type, port, properties={}):
        info = zeroconf.ServiceInfo(
            service_type,
            name + "." + service_type,
            port=port,
            properties=properties,
            addresses=["127.0.0.1"],
            server=name,
        )
        # i'm pretty sure it doesn't matter whether we use the observer's instance of AsyncZeroConf or make another one
        # use it to advertize the existence of our test gripper server
        await self.ob.aiozc.async_register_service(info)
        return info

    async def test_discover_gripper(self):
        # advertise a service on localhost that matches a gripper.
        # it does not matter that we are running no such service.
        # when the observer creates a gripper client, it will create a mock object
        # gripper client to server communication is tested in gripper_client_test. 
        await self.advertise_service(f"123.cranebot-gripper-service.test", "_http._tcp.local.", 8765)
        # by default zeroconf checks for advertized services every 10 seconds
        # so if the timeout is raised, then it means observer didn't discover the service.
        await asyncio.wait_for(self.watchable_event.wait(), 10)
        self.assertFalse(self.ob_task.done())
        self.assertTrue(self.ob.gripper_client is not None)

        # all the things observer may do with the gripper client:
        # toggle sendPreviewToUi attribute
        # run sendCommands()
        # run slow_stop_spool()

    async def test_optimize_single_anchor_pose(self):
        """
        Run the optimizer once on origin card poses
        """
        expected_pose = np.array([[0,0,-pi/4], [2.3,2.4,2.5]])
        simulated_observations = []
        for i in range(25):
            op = invert_pose(compose_poses([expected_pose, model_constants.anchor_camera]))
            simulated_observations.append(op)
        simulated_observations = np.array(simulated_observations)
        simulated_observations += np.random.normal(0,1e-4,simulated_observations.shape)
        pose = self.ob.optimize_single_anchor_pose(simulated_observations)
        np.testing.assert_array_almost_equal(expected_pose, pose, 3)

    async def test_anchor_connect_familiar(self):
        """Confirm that we can connnect to an anchor that advertises a name we recognizes from our configuration"""
        await self.advertise_service(f"123.cranebot-anchor-service.test_0", "_http._tcp.local.", 8765)
        await asyncio.wait_for(self.watchable_event.wait(), 10)
        self.assertFalse(self.ob_task.done())
        self.assertEqual(len(self.ob.anchors), 1)

    async def test_anchor_reconnect(self):
        """Confirm that if an anchor server goes down and restarts, that we reconnect to it.
        In this test, we are neither running a real websocket server, or client. the client is a mock and the server doesn't exist.
        all we are confirming here is that if the MDNS advertisement for the services goes down and back up, that we start the client task again.
        """
        info = await self.advertise_service(f"123.cranebot-anchor-service.test_0", "_http._tcp.local.", 8765)
        await asyncio.wait_for(self.watchable_event.wait(), 10)
        self.assertFalse(self.ob_task.done())
        self.assertEqual(len(self.ob.anchors), 1)

        await self.ob.aiozc.async_unregister_service(info)
        await asyncio.sleep(2)
        self.assertEqual(len(self.ob.anchors), 0)
        print('observer removed anchor client correctly, re-advertising service')

        self.watchable_event.clear()
        await asyncio.sleep(0.01) # it is necessary to reliquish control of the event loop after calling clear or it may not have the intended effect

        info = await self.advertise_service(f"123.cranebot-anchor-service.test_0", "_http._tcp.local.", 8765)
        await asyncio.wait_for(self.watchable_event.wait(), 10)
        self.assertEqual(len(self.ob.anchors), 1)