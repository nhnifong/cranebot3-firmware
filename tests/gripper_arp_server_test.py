"""
Unit tests for GripperArpServer.
Mocks all hardware interfaces to validate internal state machine and safety timeout logic without physical hardware.
"""
import pytest
pytestmark = pytest.mark.pi
pytest.importorskip("gpiodevice")

import unittest
from unittest.mock import patch, Mock
import asyncio
import websockets
import json
import time

from nf_robot.robot.gripper_arp_server import GripperArpServer

async def recv_until(ws, key, timeout=1.0, max_messages=20):
    """Receive websocket messages until one contains the given key, and return it.
    Tests must look for messages by key rather than assuming ordering, since the server
    sends other messages too (e.g. nf_robot_v right after the client connects)."""
    for _ in range(max_messages):
        data = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if key in data:
            return data
    raise AssertionError(f'did not receive a message containing key {key!r} within {max_messages} messages')

class TestGripperArpServer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Establish a clean environment by patching out all hardware-dependent libraries before the server instantiates.
        # This prevents I2C bus errors and missing device faults.
        self.patchers = [
            patch('nf_robot.robot.gripper_arp_server.board', create=True),
            patch('nf_robot.robot.gripper_arp_server.busio', create=True),
            patch('nf_robot.robot.gripper_arp_server.get_mac_address', return_value='00:11:22:33:44:55'),
            patch('nf_robot.robot.gripper_arp_server.VL53L1X'),
            patch('nf_robot.robot.gripper_arp_server.ADS1015'),
            patch('nf_robot.robot.gripper_arp_server.AnalogIn'),
            patch('nf_robot.robot.gripper_arp_server.MPU6050'),
            patch('nf_robot.robot.gripper_arp_server.SimpleSTS3215'),
            # Avoid writing a real arp_gripper_state.json into the test working directory.
            patch.object(GripperArpServer, 'save_state'),
        ]

        # Start all patches and unpack the resulting mock objects
        (self.mock_board, self.mock_busio, self.mock_mac,
         self.mock_vl53l1x_class, self.mock_ads_class, self.mock_analog_class,
         self.mock_mpu_class, self.mock_sts_class,
         self.mock_save_state) = [p.start() for p in self.patchers]

        # Configure Rangefinder Mock
        self.mock_range = self.mock_vl53l1x_class.return_value
        self.mock_range.model_info = (1, 2, 3)
        self.mock_range.data_ready = True
        self.mock_range.distance = 30.0

        # Configure Pressure Sensor Mock
        self.mock_pressure = self.mock_analog_class.return_value
        self.mock_pressure.voltage = 3.3  # Represents an untouched state

        # Configure IMU Mock
        self.mock_imu = self.mock_mpu_class.return_value
        self.mock_imu.gyro = [0.0, 0.0, 0.0]

        # Configure STS3215 Motors Mock
        self.mock_motors = self.mock_sts_class.return_value
        self.mock_motors.get_feedback.return_value = {
            'position': 2048, 'speed': 0, 'load': 0, 'voltage': 7.4, 'temp': 30, 'moving': 0
        }

        # Force a clean boot state by hiding any existing state files the test runner might accidentally see
        self.file_patcher = patch('os.path.exists', return_value=False)
        self.file_patcher.start()

        self.server = GripperArpServer()
        self.server_task = asyncio.create_task(self.server.main())
        
        # Yield execution momentarily to allow the server's internal asyncio loops to spin up
        await asyncio.sleep(0.1)

    async def asyncTearDown(self):
        self.server.shutdown()
        # Wait for the server task to cleanly terminate to avoid dangling coroutine warnings
        await asyncio.wait_for(self.server_task, timeout=1.0)
        
        for p in self.patchers:
            p.stop()
        self.file_patcher.stop()

    async def test_boot_health_no_state_file(self):
        """
        Verify the server safely initializes default state when the JSON state file is missing.

        With no saved wrist angle (defaults to 0) and the mocked motor reporting a
        raw position of 2048 (180 degrees), getWristAngle's boot reconciliation finds
        a one-revolution discrepancy: it re-anchors unrolled_wrist_angle to 180 (within
        [0, 1080], no motion), shifts saved_unrolled_wrist_angle by the same revolution,
        and records that revolution in total_wrist_turns.
        """
        self.assertFalse(self.server_task.done(), "Server crashed during startup")
        self.assertEqual(self.server.saved_unrolled_wrist_angle, 360)
        self.assertEqual(self.server.saved_finger_angle, 0)
        self.assertEqual(self.server.unrolled_wrist_angle, 180)
        self.assertEqual(self.server.wrist_step_offset, 0.0)
        self.assertEqual(self.server.total_wrist_turns, -1)
        self.mock_save_state.assert_called()

    async def test_websocket_connection(self):
        """
        Ensure the server does not fault when a client establishes and drops a connection.
        Exceptions are deliberately unhandled to surface native tracebacks on failure.
        """
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            await asyncio.sleep(0.1)
            self.assertFalse(self.server_task.done(), "Server halted after client connection")

    async def test_telemetry_payload_format(self):
        """
        Validate the structure and types of the telemetry dictionary flushed to the websocket.
        Avoids hardcoded assertions to remain robust against tuning tweaks.
        """
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            data = await recv_until(ws, 'grip_sensors')

            self.assertIn('grip_sensors', data, "Telemetry missing critical root key")
            sensors = data['grip_sensors']
            
            self.assertIsInstance(sensors.get('time'), float)
            self.assertIn('fing_v', sensors)  # Now holding filtered force as preferred
            self.assertIn('fing_a', sensors)
            self.assertIn('wrist_a', sensors)
            self.assertIsInstance(sensors.get('range'), float)

    async def test_wrist_speed_timeout_safety(self):
        """
        Verify that a commanded motion safely expires and resets to zero after ACTION_TIMEOUT.
        """
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            await ws.send(json.dumps({'set_wrist_speed': 50}))
            
            # Allow enough time for the message to queue and updateMotors to consume it
            await asyncio.sleep(0.05)
            self.assertEqual(self.server.desired_wrist_speed, 50)
            self.mock_motors.set_position.assert_called()
            
            # Wait just past the ACTION_TIMEOUT window
            await asyncio.sleep(self.server.conf['ACTION_TIMEOUT'] + 0.05)
            
            self.assertEqual(self.server.desired_wrist_speed, 0, "Wrist speed failed to time out")

    async def test_dynamic_force_mode_transition(self):
        """
        Simulate an object strike during closure to confirm the state machine jumps into force mode.
        """
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            # Command fingers to close (positive speed)
            await ws.send(json.dumps({'set_finger_speed': 20}))
            await asyncio.sleep(0.05)
            self.assertFalse(self.server.in_force_mode, "Should not be in force mode while traversing open air")
            
            # Simulate a pressure pad strike by drastically dropping the voltage representation
            self.mock_pressure.voltage = 2.0 
            
            # Yield to the update loop so it can run get_current_grip_force and flip the state machine
            await asyncio.sleep(0.1)
            
            self.assertTrue(self.server.in_force_mode, "Failed to enter force mode after detecting pressure drop")
            self.assertGreater(self.server.desired_force, 0)

    async def test_boot_reanchor_broadcasts_total_wrist_turns(self):
        """
        The one-revolution re-anchor performed during boot (see test_boot_health_no_state_file)
        must be reported to clients via the 'total_wrist_turns' update key.
        """
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            data = await recv_until(ws, 'total_wrist_turns')
            self.assertIn('total_wrist_turns', data)
            self.assertEqual(data['total_wrist_turns'], -1)

    async def test_untwist_command(self):
        """
        Verify the 'untwist' command rotates the wrist by whole revolutions using
        encoder midpoint resets, updates and persists total_wrist_turns, blocks other
        wrist commands while in progress, and sends a completion confirmation.
        """
        self.server.total_wrist_turns = 1
        self.mock_save_state.reset_mock()

        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            await ws.send(json.dumps({'untwist': 1}))

            # Give the untwist task a moment to start and claim the busy flag.
            await asyncio.sleep(0.2)
            self.assertTrue(self.server.wrist_busy, "untwistWrist should mark the wrist busy")

            # Other wrist commands must be ignored while untwisting.
            stale_angle = self.server.desired_wrist_angle
            await ws.send(json.dumps({'set_wrist_angle': 999}))
            await asyncio.sleep(0.1)
            self.assertNotEqual(self.server.desired_wrist_angle, 999)
            self.assertEqual(self.server.desired_wrist_angle, stale_angle)

            # Wait for the operation to complete (two encoder resets, ~4.4s of sleeps).
            # Poll the instance attribute directly rather than counting broadcasts,
            # since self.update is flushed (and cleared) every ~40ms by
            # stream_measurements and could easily be missed between polls.
            for _ in range(80):
                if not self.server.wrist_busy:
                    break
                await asyncio.sleep(0.1)
            self.assertFalse(self.server.wrist_busy, "untwistWrist did not clear the busy flag")

        self.assertEqual(self.server.total_wrist_turns, 0)
        self.assertEqual(self.server.wrist_step_offset, 0.0)
        self.mock_motors.reset_encoder_to_midpoint.assert_called_with(2)  # WRIST
        self.assertEqual(self.mock_motors.reset_encoder_to_midpoint.call_count, 2)
        self.mock_save_state.assert_called()

    async def test_untwist_default_uses_total_wrist_turns(self):
        """
        Sending 'untwist': null should default to fully untwisting (turns = total_wrist_turns).
        With total_wrist_turns already at 0, this should be a no-op that still confirms.
        """
        self.server.total_wrist_turns = 0
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            await ws.send(json.dumps({'untwist': None}))

            completion = None
            for _ in range(20):
                data = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
                if 'untwist_complete' in data:
                    completion = data['untwist_complete']
                    break
            self.assertIsNotNone(completion, "Never received untwist_complete confirmation")
            self.assertEqual(completion, {'turns_done': 0, 'total_wrist_turns': 0})

        self.assertFalse(self.server.wrist_busy)