"""
Test that starts up anchor server and connects to it with a websocket.
"""

import unittest
import asyncio
from anchor_server import RaspiAnchorServer
from debug_motor import DebugMotor
import websockets
import json
from config import Config

class TestAnchorServerClientConnection(unittest.TestCase):
    def setUp(self):
        self.motor = DebugMotor()
        self.server = RaspiAnchorServer(power_anchor=False, mock_motor=self.motor)

    def test_startup_shutdown(self):
        async def m():
            task = asyncio.create_task(self.server.main())
            await asyncio.sleep(0.1)
            self.server.shutdown()
            await asyncio.wait_for(task, 1)
        asyncio.run(m())

    def test_connect(self):
        """
        Assert that the server is still running after a client connects and disconnects.
        then shut it down and assert it stops cleanly.
        """
        async def m():
            server_task = asyncio.create_task(self.server.main())
            await asyncio.sleep(0.1)
            try:
                async with websockets.connect("ws://127.0.0.1:8765") as ws:
                    self.assertFalse(server_task.done(), "Server should still be running after getting a connection")
                    await ws.close()
            except Exception as e:
                self.server.shutdown()
                raise e
            self.server.shutdown()
            await asyncio.wait_for(server_task, 1)

        asyncio.run(m())

    def command_and_check(self, command, check, timeout):
        """
        Start a server, send a command dict to it, asset it is still running.
        run check(resp), which can assert anything about self.server or the resp.
        resp contains anything the websocket sent within timeout seconds.
        """
        async def m():
            server_task = asyncio.create_task(self.server.main())
            await asyncio.sleep(0.1)
            try:
                async with websockets.connect("ws://127.0.0.1:8765") as ws:
                    await ws.send(json.dumps(command))
                    await asyncio.sleep(0.1)
                    self.assertFalse(server_task.done(), "Server should still be running")
                    resp = await asyncio.wait_for(ws.recv(), timeout)
                    check(resp)
                    await ws.close()
            except Exception as e:
                self.server.shutdown()
                raise e
            self.server.shutdown()
            await asyncio.wait_for(server_task, 1)
        asyncio.run(m())

    def test_send_config(self):
        config = Config()
        anchor_config_vars = config.vars_for_anchor(0)
        anchor_config_vars['SPECIAL'] = 'NAT'

        def check(resp):
            self.assertEqual('NAT', self.server.conf['SPECIAL'])
            self.assertEqual('NAT', self.server.spooler.conf['SPECIAL'])

        self.command_and_check({'set_config_vars': anchor_config_vars}, check, 0.1)

    def test_send_reference_length(self):
        def check(resp):
            pass

        self.command_and_check({'reference_length': 0.3}, check, 0.1)