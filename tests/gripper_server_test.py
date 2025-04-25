"""
Test that starts up greipper server and connects to it with a websocket.
"""
import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch, Mock, MagicMock, ANY
import asyncio
from gripper_server import RaspiGripperServer
from debug_motor import DebugMotor
import websockets
import json
from config import Config
from spools import SpoolController
from inventorhatmini import InventorHATMini, SERVO_1, SERVO_2
from adafruit_bno08x.i2c import BNO08X_I2C
import time

class TestGripperServer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patchers = []

        self.mock_spool_class = Mock(spec=SpoolController)
        self.patchers.append(patch('gripper_server.SpoolController', self.mock_spool_class))
        self.mock_spooler = self.mock_spool_class.return_value

        #replace certain functions in the mocked spooler

        # trackingLoop should be a blocking call. the server runs only as long as this does.
        self.runLoop = True
        def mock_tracking_loop():
            while self.runLoop:
                time.sleep(0.05)
        self.mock_spooler.trackingLoop = mock_tracking_loop

        # measureRefLoad is an async function
        self.lastRefLoad = -1
        async def mock_mes_ref_load(val):
            self.lastRefLoad = val
        self.mock_spooler.measureRefLoad = mock_mes_ref_load

        # must be a list
        self.mock_spooler.popMeasurements.return_value = [123.4, 5.6]

        # mock inventor hat mini
        self.mock_hat_class = Mock(spec=InventorHATMini)
        self.patchers.append(patch('gripper_server.InventorHATMini', self.mock_hat_class))
        self.mock_hat = self.mock_hat_class.return_value

        # gpio pins
        self.mock_hat.gpio_pin_value.return_value = 1.0

        self.servos = {SERVO_1: MagicMock(), SERVO_2: MagicMock()}
        self.mock_hat.servos = MagicMock()
        self.mock_hat.servos.__getitem__.side_effect = lambda key: self.servos[key]

        self.encoders = {0: MagicMock(), 1: MagicMock()}
        self.mock_hat.encoders = MagicMock()
        self.mock_hat.encoders.__getitem__.side_effect = lambda key: self.encoders[key]

        # prevent this call from failing         i2c = busio.I2C(board.SCL, board.SDA)
        self.patchers.append(patch('gripper_server.board.SCL', None, create=True))
        self.patchers.append(patch('gripper_server.board.SDA', None, create=True))
        self.patchers.append(patch('gripper_server.busio.I2C', lambda a, b: None,))

        # mock IMU
        self.mock_imu_class = Mock(spec=BNO08X_I2C)
        self.patchers.append(patch('gripper_server.BNO08X_I2C', self.mock_imu_class))
        self.mock_imu = self.mock_imu_class.return_value

        # mock motor
        self.debug_motor = DebugMotor()

        for p in self.patchers:
            p.start()
        self.server = RaspiGripperServer(self.debug_motor)
        self.server_task = asyncio.create_task(self.server.main())
        await asyncio.sleep(0.1)  # Give the server a moment to start

    async def asyncTearDown(self):
        self.runLoop = False
        self.server.shutdown()
        await asyncio.wait_for(self.server_task, 1)
        for p in self.patchers:
            p.stop()

    async def test_startup_shutdown(self):
        # Startup and shutdown are handled in setUp and tearDown
        self.mock_spool_class.assert_called_once_with(
            ANY, empty_diameter=20, full_diameter=36, full_length=2, conf=self.server.conf, tension_support=False)
        self.assertTrue(self.server_task.done() is False, "Server should be running")

    async def test_connect(self):
        """
        Assert that the server is still running after a client connects and disconnects.
        """
        try:
            async with websockets.connect("ws://127.0.0.1:8765") as ws:
                await asyncio.sleep(0.1)
                self.assertFalse(self.server_task.done(), "Server should still be running after getting a connection")
                await ws.close()
        except Exception as e:
            self.fail(f"Connection failed: {e}")

    async def command_and_check(self, command, check, timeout, sleep=0.1):
        """
        Send a command dict to the server, assert it is still running.
        run check(resp), which can assert anything about self.server or the resp.
        resp contains anything the websocket sent within timeout seconds.
        """
        try:
            async with websockets.connect("ws://127.0.0.1:8765") as ws:
                await ws.send(json.dumps(command))
                await asyncio.sleep(sleep)
                self.assertFalse(self.server_task.done(), "Server should still be running")
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout)
                    check(resp)
                except asyncio.TimeoutError:
                    check(None) # Allow check function to handle timeouts if needed
                await ws.close()
        except Exception as e:
            self.fail(f"Command execution failed: {e}")

    async def test_send_zero_winch_line(self):
        self.mock_spooler.setReferenceLength.reset_mock()
        self.mock_hat.gpio_pin_value.return_value = 0
        try:
            async with websockets.connect("ws://127.0.0.1:8765") as ws:
                await ws.send(json.dumps({'zero_winch_line': None}))
                await asyncio.sleep(0.1)
                self.assertFalse(self.server_task.done(), "Server should still be running")
                self.assertEqual(-1, self.debug_motor.speed)
                self.mock_hat.gpio_pin_value.return_value = 1
                await asyncio.sleep(0.1)
                self.assertEqual(0, self.debug_motor.speed)
                self.mock_spooler.setReferenceLength.assert_called_once_with(0.01)
                await ws.close()
        except Exception as e:
            self.fail(f"Command execution failed: {e}")

    async def test_send_grip(self):
        # There is actually a lot of complex behavior that happens as a result of setting this,
        # but it's just not that useful to unit test. the physical tests are the only thing that
        # really cover it. 
        def check(resp):
            self.assertEqual(False, self.server.tryHold)
        await self.command_and_check({'grip': 'open'}, check, 0.1)
        def check(resp):
            self.assertEqual(True, self.server.tryHold)
        await self.command_and_check({'grip': 'closed'}, check, 0.1)
