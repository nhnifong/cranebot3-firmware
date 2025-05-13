"""
Test that starts up anchor server and connects to it with a websocket.

Commands sent to the anchor server generally all just call a corresponding method in SpoolController
So we just mock that. 

Any method common to anchor and gripper server is tested here in anchor
"""
import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch, Mock
import asyncio
from anchor_server import RaspiAnchorServer
import websockets
import json
from config import Config
from debug_motor import DebugMotor
from spools import SpoolController  # Import the class to be mocked
import time

class TestAnchorServer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.debug_motor = DebugMotor()
        self.mock_spool_class = Mock(spec=SpoolController)
        self.patcher = patch('anchor_server.SpoolController', self.mock_spool_class)
        self.patcher.start()  # This is the mocked class
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


        self.server = RaspiAnchorServer(power_anchor=False, mock_motor=self.debug_motor)
        self.server_task = asyncio.create_task(self.server.main())
        await asyncio.sleep(0.1)  # Give the server a moment to start

    async def asyncTearDown(self):
        self.runLoop = False
        self.server.shutdown()
        await asyncio.wait_for(self.server_task, 1)
        self.patcher.stop()

    async def test_startup_shutdown(self):
        # Startup and shutdown are handled in setUp and tearDown
        self.mock_spool_class.assert_called_once_with(self.debug_motor, empty_diameter=25, full_diameter=27, full_length=10, conf=self.server.conf, gear_ratio=20/51)
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

    async def test_abnormal(self):
        """
        Assert that the server is still running after a client connects and then the client crashes and sends code 1011.
        """
        try:
            async with websockets.connect("ws://127.0.0.1:8765") as ws:
                await asyncio.sleep(0.1)
                self.assertFalse(self.server_task.done(), "Server should still be running after getting a connection")
                await ws.close(code=1011, reason=f'Test client pretends to crash')
        except Exception as e:
            self.fail(f"Connection failed: {e}")

    async def command_and_check(self, command, check, timeout, sleep=0.1):
        """
        Send a command dict to the server, assert it is still running.
        run check(resp), which can assert anything about self.server or the resp.
        resp contains anything the websocket sent within timeout seconds.
        """
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

    async def test_send_config(self):
        config = Config()
        anchor_config_vars = config.vars_for_anchor(0)
        anchor_config_vars['SPECIAL'] = 'NAT'

        def check(resp):
            self.assertEqual('NAT', self.server.conf['SPECIAL'])
        await self.command_and_check({'set_config_vars': anchor_config_vars}, check, 0.1)

    async def test_send_reference_length(self):
        self.mock_spooler.setReferenceLength.reset_mock()
        reference_length = 0.3
        def check(resp):
            self.mock_spooler.setReferenceLength.assert_called_once_with(reference_length)
        await self.command_and_check({'reference_length': reference_length}, check, 0.1)

    async def test_send_jog(self):
        jog_value = 0.3
        def check(resp):
            self.mock_spooler.jogRelativeLen.assert_called_once_with(jog_value)
        await self.command_and_check({'jog': jog_value}, check, 0.1)

    async def test_send_length_plan(self):
        length_plan = [
            [1745592964.2, 0.4],
            [1745592965.2, 0.45],
            [1745592966.2, 0.5],
        ]
        def check(resp):
            pass
            self.mock_spooler.setPlan.assert_called_once_with(length_plan)
        await self.command_and_check({'length_plan': length_plan}, check, 0.1)

    async def test_send_measure_ref_load(self):
        load = 0.3
        def check(resp):
            self.assertEqual(self.lastRefLoad, load)
        await self.command_and_check({'measure_ref_load': load}, check, 0.1)

    async def test_send_measure_ref_load_0(self):
        load = 0
        def check(resp):
            self.assertEqual(self.lastRefLoad, load)
        await self.command_and_check({'measure_ref_load': load}, check, 0.1)
