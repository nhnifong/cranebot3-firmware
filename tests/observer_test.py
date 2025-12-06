import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import unittest
from unittest.mock import patch, Mock, ANY
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
from zeroconf import IPVersion
from zeroconf.asyncio import AsyncZeroconf

# Todo, automate the following tests
# 
# start observer, turn on bot, wait for all good, close observer
#     pass
# start observer, turn on bot, before all connections succeed, close observer
#     pass
# turn on bot, start observer, wait for all good, close observer
#     pass
# start observer, turn on bot, wait for all good, kill bot, close observer
#     observer hangs
# turn on bot, start observer, wait for all good, kill bot, close observer
#     observer hangs
# start observer, turn on bot, wait for all good, kill bot, power up bot, wait for all good, close observer.
#     observer recconnects like it should
# turn on bot, start observer, wait for all good, kill bot, power up bot, wait for all good, close observer.
#     observer recconnects like it should

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
        self.patchers.append(patch('observer.RaspiGripperClient', self.mock_gripper_client_class))

        # Create four mock anchor clients
        self.mock_anchor_clients = [Mock(spec=RaspiAnchorClient) for _ in range(4)]
        for i, client in enumerate(self.mock_anchor_clients):
            client.startup = self.event_startup
            client.shutdown = self.event_shutdown
            client.slow_stop_spool = self.instant_nothing
            client.anchor_num = i

        # The side_effect makes it so each call to the constructor returns the next mock
        self.patchers.append(patch('observer.RaspiAnchorClient', side_effect=self.mock_anchor_clients))

        for p in self.patchers:
            p.start()

        # Zeroconf will attempt to discover services .
        self.ob = AsyncObserver(self.to_ui_q, self.to_ob_q)
        # before starting, set it's zeroconf instance to a special version that searches on localhost only
        self.zc = AsyncZeroconf(ip_version=IPVersion.All)
        self.ob.aiozc = self.zc
        self.ob_task = asyncio.create_task(self.ob.main())
        await asyncio.sleep(0.1)

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
        print('client startup called')
        self.mock_gripper_client.connection_established_event.set()
        self.watchable_startup_event.set()
        return False

    async def abnormal_event_startup(self):
        print('abnormal client startup called')
        self.mock_gripper_client.connection_established_event.set()
        self.watchable_startup_event.set()
        await asyncio.sleep(0.1)
        return True

    async def event_shutdown(self):
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

    async def advertise_service(self, name, service_type, port, properties={}, update=False):
        info = zeroconf.ServiceInfo(
            service_type,
            name + "." + service_type,
            port=port,
            properties=properties,
            addresses=["127.0.0.1"],
            server=name,
        )
        # We can't make a second zeroconf instance so we use the observer's isntance to advertize the existence of our test gripper server
        if update:
            await self.zc.async_update_service(info)
        else:
            await self.zc.async_register_service(info)
        return info

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

    async def test_anchor_abnormal_reconnect(self):
        """ Confirm that if an anchor server goes offline abruptly, then comes back up and advertises itself that we reconnect to it.
        In real installations, this looks like observer's websocket handler throwing a ConnectionClosedError exception,
        the service never being unregisterd, but zerconf issuing a service update when it comes back up where nothing has actually changed about the address or port
        """
        # in order to immitate this scenario with a mock client, it's startup() method only needs to return True
        self.mock_anchor_clients[0].startup = self.abnormal_event_startup

        # advertise the service
        info = await self.advertise_service(f"123.cranebot-anchor-service.test_0", "_http._tcp.local.", 8765)
        await asyncio.wait_for(self.watchable_startup_event.wait(), 10)
        self.assertFalse(self.ob_task.done())
        self.assertEqual(len(self.ob.anchors), 1)

        # 0.1 second later, the client's startup() function returns True to indicate an abnormal disconnect
        await asyncio.wait_for(self.watchable_shutdown_event.wait(), 0.2)
        # assert this resulted in it's removal as a listed client
        self.assertEqual(len(self.ob.anchors), 0)
        self.assertEqual(len(self.ob.bot_clients), 0)

        # reset events
        self.watchable_startup_event.clear()
        self.watchable_shutdown_event.clear()

        # back to normal anchor behavior
        self.mock_anchor_clients[0].startup = self.event_startup

        # advertise it again
        print('advertise it again')
        info = await self.advertise_service(f"123.cranebot-anchor-service.test_0", "_http._tcp.local.", 8766, update=True)

        # confirm reconnection
        await asyncio.wait_for(self.watchable_startup_event.wait(), 10)
        self.assertEqual(len(self.ob.anchors), 1)

    async def test_move_direction_speed_normal_case(self):
        """Tests the standard execution path with valid inputs."""
        await self._setup_mock_anchors()
        self.mock_pe.point_inside_work_area.return_value = True

        uvec = np.array([1.0, 0.0, 0.0])
        speed = 0.1
        downward_bias = -0.04
        
        # Manually calculate expected results
        biased_uvec = uvec + np.array([0, 0, downward_bias])
        normalized_biased_uvec = biased_uvec / np.linalg.norm(biased_uvec)
        expected_velocity = normalized_biased_uvec * speed
        
        # Reproduce the kinematics calculation that is done in the code under test
        KINEMATICS_STEP_SCALE = 10.0
        starting_pos = self.mock_pe.gant_pos
        lengths_a = np.linalg.norm(starting_pos - self.anchor_points, axis=1)
        new_pos = starting_pos + (normalized_biased_uvec / KINEMATICS_STEP_SCALE)
        lengths_b = np.linalg.norm(new_pos - self.anchor_points, axis=1)
        deltas = lengths_b - lengths_a
        expected_line_speeds = deltas * KINEMATICS_STEP_SCALE * speed

        # Call the method
        returned_velocity = await self.ob.move_direction_speed(uvec, speed)
        await asyncio.sleep(0) # Allow created tasks to run

        # Assertions
        np.testing.assert_allclose(returned_velocity, expected_velocity, atol=1e-6)
        self.mock_pe.record_commanded_vel.assert_called_once_with(ANY)
        np.testing.assert_allclose(self.mock_pe.record_commanded_vel.call_args[0][0], expected_velocity, atol=1e-6)

        for i, client in enumerate(self.ob.anchors):
            client.send_commands.assert_called_once_with({'aim_speed': expected_line_speeds[i]})

    async def test_move_direction_speed_no_speed_provided(self):
        """Tests the path where speed is derived from the uvec magnitude (lerobot mode)."""
        await self._setup_mock_anchors()
        self.mock_pe.point_inside_work_area.return_value = True

        # uvec magnitude is the speed
        velocity_vec = np.array([0.0, 0.2, 0.0])
        
        returned_velocity = await self.ob.move_direction_speed(velocity_vec)
        await asyncio.sleep(0)

        # Expected speed is the norm of the input vector
        expected_speed = np.linalg.norm(velocity_vec)
        self.assertAlmostEqual(expected_speed, 0.2)

        # Ensure that send_commands was called on all anchors with non-zero speeds
        for client in self.ob.anchors:
            client.send_commands.assert_called_once()
            sent_speed = client.send_commands.call_args[0][0]['aim_speed']
            self.assertNotEqual(sent_speed, 0)
        self.mock_pe.record_commanded_vel.assert_called_once()

    async def test_move_direction_speed_limited_by_height(self):
        """Tests the path where the requested speed is clamped by the height-based speed limit."""
        await self._setup_mock_anchors()
        self.mock_pe.point_inside_work_area.return_value = True
        
        # Set a high z-position to get a low speed limit
        self.mock_pe.gant_pos = np.array([0.0, 0.0, 1.9])
        expected_speed = 0.01
        
        # Request a speed much higher than the limit
        uvec = np.array([0.0, 1.0, 0.0])
        speed = 0.5 # much higher than ~0.055
        
        returned_velocity = await self.ob.move_direction_speed(uvec, speed)
        await asyncio.sleep(0)

        # The magnitude of the returned velocity should be the speed limit, not the requested speed
        returned_speed = np.linalg.norm(returned_velocity)
        self.assertAlmostEqual(returned_speed, expected_speed, places=5)
        self.mock_pe.record_commanded_vel.assert_called_once()
        for client in self.ob.anchors:
            client.send_commands.assert_called_once()

    async def test_move_direction_speed_clamps_to_zero(self):
        """Tests the path where a very low speed is clamped to zero, causing an early return."""
        await self._setup_mock_anchors()
        
        uvec = np.array([0.0, 1.0, 0.0])
        speed = 0.004 # Below the 0.005 threshold
        
        returned_velocity = await self.ob.move_direction_speed(uvec, speed)
        await asyncio.sleep(0)

        # Assertions for the speed == 0 path
        np.testing.assert_array_equal(returned_velocity, np.zeros(3))
        self.mock_pe.record_commanded_vel.assert_called_once_with(ANY)
        np.testing.assert_array_equal(self.mock_pe.record_commanded_vel.call_args[0][0], np.zeros(3))
        
        for client in self.ob.anchors:
            client.send_commands.assert_called_once_with({'aim_speed': 0})
            
    async def test_move_direction_speed_outside_work_area(self):
        """Tests the safety feature where a move outside the work area results in zero speed commands."""
        await self._setup_mock_anchors()
        # Mock the safety check to return False
        self.mock_pe.point_inside_work_area.return_value = False

        uvec = np.array([0.0, 0.0, 1.0])
        speed = 0.2

        returned_velocity = await self.ob.move_direction_speed(uvec, speed)
        await asyncio.sleep(0)

        # The returned velocity should be zero
        np.testing.assert_allclose(returned_velocity, np.zeros(3), atol=1e-6)
        self.mock_pe.record_commanded_vel.assert_called_once_with(ANY)
        np.testing.assert_allclose(self.mock_pe.record_commanded_vel.call_args[0][0], np.zeros(3), atol=1e-6)
        
        # The line speeds should be zero because the internal speed variable was zeroed out
        for client in self.ob.anchors:
            client.send_commands.assert_called_once_with({'aim_speed': 0.0})

    async def test_invoke_motion_task_cancels_previous(self):
        """
        Verifies that calling invoke_motion_task while another motion task is
        running will cancel the first task before starting the second.
        """
        # Events to signal the state of our mock tasks
        task1_started = asyncio.Event()
        task1_cleanup_finished = asyncio.Event()
        task2_started = asyncio.Event()

        # Define a mock motion task that follows the required structure
        async def mock_motion_task_1():
            try:
                task1_started.set()
                # Keep the task alive until it's cancelled
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                raise
            finally:
                # Signal that cleanup has run
                task1_cleanup_finished.set()

        async def mock_motion_task_2():
            task2_started.set()
            await asyncio.sleep(0.1) # a short-lived task

        # Invoke the first task
        await self.ob.invoke_motion_task(mock_motion_task_1())
        # Store the task object for later inspection
        first_task_handle = self.ob.motion_task
        
        # Wait until the first task is confirmed to be running
        await asyncio.wait_for(task1_started.wait(), timeout=1)
        self.assertFalse(first_task_handle.done(), "First task should be running.")

        # Invoke the second task, which should cancel the first
        await self.ob.invoke_motion_task(mock_motion_task_2())
        second_task_handle = self.ob.motion_task

        # Verify the cancellation and cleanup of the first task
        await asyncio.wait_for(task1_cleanup_finished.wait(), timeout=1)
        self.assertTrue(first_task_handle.cancelled(), "First task should be marked as cancelled.")
        
        # Verify the second task started successfully
        await asyncio.wait_for(task2_started.wait(), timeout=1)
        self.assertNotEqual(first_task_handle, second_task_handle, "A new task object should have been created.")
        self.assertFalse(second_task_handle.done(), "Second task should be running initially.")
        
        # Allow the second task to finish
        await asyncio.sleep(0.2)
        self.assertTrue(second_task_handle.done(), "Second task should have completed by now.")

# other functions to test:
# stop_all
# _handle_jog_spool
# sendReferenceLengths
# _handle_zero_winch_line
# tension_and_wait
# locate_anchors