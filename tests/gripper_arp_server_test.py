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
            patch('nf_robot.robot.gripper_arp_server.SimpleSTS3215')
        ]
        
        # Start all patches and unpack the resulting mock objects
        (self.mock_board, self.mock_busio, self.mock_mac, 
         self.mock_vl53l1x_class, self.mock_ads_class, self.mock_analog_class, 
         self.mock_mpu_class, self.mock_sts_class) = [p.start() for p in self.patchers]

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
        """
        self.assertFalse(self.server_task.done(), "Server crashed during startup")
        self.assertEqual(self.server.saved_unrolled_wrist_angle, 0)
        self.assertEqual(self.server.saved_finger_angle, 0)

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
            data = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
            
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