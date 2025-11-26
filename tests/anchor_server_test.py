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
from unittest.mock import patch, Mock, ANY
import asyncio
from anchor_server import RaspiAnchorServer
import websockets
import json
from config import Config
from debug_motor import DebugMotor
from spools import SpoolController  # Import the class to be mocked
import time
import subprocess

class TestAnchorServer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.debug_motor = DebugMotor()
        self.mock_spool_class = Mock(spec=SpoolController)
        self.patcher = patch('anchor_server.SpoolController', self.mock_spool_class)
        self.patcher.start()  # This is the mocked class
        self.mock_spooler = self.mock_spool_class.return_value
        self.patcher2 = patch('anchor_server.stream_command', ['sleep', 'infinity'])
        self.patcher2.start()

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
        self.patcher2.stop()

        # make sure we didn't leave and subprocesses running
        command = 'ps aux | grep "sleep infinity" | grep -v grep'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # stdout should be empty if no 'sleep infinity' processes are left
        self.assertTrue(
            result.stdout.strip() == '',
            f"Found orphaned processes running after test teardown: {repr(result.stdout)}"
        )

    def assertLastAimSpeed(self, speed):
        # Assert that the last call to setAimSpeed had the argument speed
        self.assertEqual(self.mock_spooler.setAimSpeed.call_args[0][0], speed)

    async def test_startup_shutdown(self):
        # Startup and shutdown are handled in setUp and tearDown
        self.mock_spool_class.assert_called_once_with(
            self.debug_motor,
            empty_diameter=ANY,
            full_diameter=ANY,
            full_length=ANY,
            conf=self.server.conf,
            gear_ratio=ANY,
            tight_check_fn=ANY
        )
        self.assertTrue(self.server_task.done() is False, "Server should be running")


    async def test_connect(self):
        """
        Assert that the server is still running after a client connects and disconnects.
        """
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            await asyncio.sleep(0.1)
            self.assertFalse(self.server_task.done(), "Server should still be running after getting a connection")
            await ws.close()

    async def test_abnormal(self):
        """
        Assert that the server is still running after a client connects and then the client crashes and sends code 1011.
        """
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            await asyncio.sleep(0.1)
            self.assertFalse(self.server_task.done(), "Server should still be running after getting a connection")
            await ws.close(code=1011, reason=f'Test client pretends to crash')


    async def test_subprocess_cleanup(self):
        """
        Assert that the rpicam-vid subprocess is killed when the client disconnects.
        """
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            await asyncio.sleep(0.1)
            await ws.close()

        # in this test the server is on the same event loop and needs a chance to run
        await asyncio.sleep(1)

        self.assertIsNotNone(self.server.rpicam_process.returncode)
        self.assertLess(self.server.rpicam_process.returncode, 0) # Should have a negative return code if killed by signal

    async def test_subprocess_cleanup_client_has_error(self):
        """
        Assert that the rpicam-vid subprocess is killed when the client disconnects with an internal error
        """
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            await asyncio.sleep(0.1)
            await ws.close(1011)
        await asyncio.sleep(1)

        self.assertIsNotNone(self.server.rpicam_process.returncode)
        self.assertLess(self.server.rpicam_process.returncode, 0)

    async def test_subprocess_cleanup_server_stopped(self):
        """
        Assert that the rpicam-vid subprocess is killed when the server is stopped while a client is connected
        """
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            await asyncio.sleep(0.1)
            # crash spool tracking loop
            self.runLoop = False
            await asyncio.sleep(0.1)
        await asyncio.sleep(1)

        self.assertIsNotNone(self.server.rpicam_process.returncode)
        self.assertLess(self.server.rpicam_process.returncode, 0)

    async def test_subprocess_cleanup_line_timeout(self):
        """
        Assert that the rpicam-vid subprocess is killed when it has been allowed to time out at least once and be restarted,
        before a normal client disconnect.

        TODO find a way to check for orphaned process after server closes
        """
        self.server.line_timeout = 1
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            print(dir(ws))
            await asyncio.sleep(6.1) # one second for it to timeout, 5 more for it to wait before restarting 
            await ws.close()
            await asyncio.sleep(0.1)
        await asyncio.sleep(1)

        self.assertIsNotNone(self.server.rpicam_process.returncode)
        self.assertLess(self.server.rpicam_process.returncode, 0)


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
        length = 0.4
        def check(resp):
            self.mock_spooler.setTargetLength.assert_called_once_with(length)
        await self.command_and_check({'length_set': length}, check, 0.1)

    async def test_tighten(self):
        # make a local variable to contain a tight or not tight bool
        local_var = {'tight': False}
        # make a function that captures it
        def local_t():
            return local_var['tight']
        # set this function to be the server's way of checking whether the line is tight, so we can alter it at will
        self.server.tight_check = local_t
        
        # connect to the server
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            # send command under test
            await ws.send(json.dumps({'tighten':None}))
            # sleep at least one ws_delay (0.03) and at least 0.05
            await asyncio.sleep(0.1)
            # spool should still be reeling in
            self.assertLastAimSpeed(self.server.conf['tightening_speed'])
            # click the switch
            local_var['tight'] = True
            print('set tight var')
            await asyncio.sleep(0.1)
            self.assertLastAimSpeed(0)
            self.assertFalse(self.server_task.done(), "Server should still be running")
            # close websocket. tighten command should not prevent this.
            await asyncio.wait_for(ws.close(), 2)

    async def test_tighten_disconnect(self):
        """ Confirm disconnecting while the tighten command is running doesn't leave the spool reeling in."""
        # make a local variable to contain a tight or not tight bool
        local_var = {'tight': False}
        # make a function that captures it
        def local_t():
            return local_var['tight']
        # set this function to be the server's way of checking whether the line is tight, so we can alter it at will
        self.server.tight_check = local_t
        
        # connect to the server
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            # send command under test
            await ws.send(json.dumps({'tighten':None}))
            # sleep at least one ws_delay (0.03) and at least 0.05
            await asyncio.sleep(0.1)
            # spool should still be reeling in
            self.assertLastAimSpeed(self.server.conf['tightening_speed'])
            # close websocket. tighten command should not prevent this.
            await asyncio.wait_for(ws.close(), 2)

        await asyncio.sleep(0.1)
        self.assertLastAimSpeed(0)
        self.assertFalse(self.server_task.done(), "Server should still be running")