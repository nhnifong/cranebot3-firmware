"""
Tests for ComponentClient's shared connect/reconnect/shutdown lifecycle, exercised
through ArpeggioAnchorClient since it's a thin, easily-instantiated subclass.

This is the direct (non-mocked-away) unit coverage of ComponentClient — ported from the
now-deleted pilot tests/anchor_client_test.py, which exercised the same base-class behavior
through RaspiAnchorClient before the pilot anchor was removed.
"""

import unittest
from unittest.mock import Mock, MagicMock, call
import asyncio
import numpy as np
import websockets
from websockets.exceptions import (
    ConnectionClosedOK,
    ConnectionClosedError,
)

from multiprocessing import Pool
import json

from nf_robot.host.data_store import DataStore
from nf_robot.host.arp_anchor_client import ArpeggioAnchorClient
from nf_robot.host.stats import StatCounter
from nf_robot.generated.nf import telemetry, common
from nf_robot.common.config_loader import create_default_config

ws_port = 8765

class TestArpAnchorClient(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.datastore = DataStore()
        self.ob_mock = MagicMock()
        self.ob_mock.config = create_default_config()
        self.mock_pool_class = Mock(spec=Pool)
        self.pool = self.mock_pool_class.return_value

        self.stat = StatCounter(self.ob_mock)

        self.server_ws = None
        self.receiver = MagicMock()

        self.got_connection = asyncio.Event()
        self.close_test_server = asyncio.Event()
        self.server_task = asyncio.create_task(self.runTestServer())
        await asyncio.sleep(0.5)

    async def asyncTearDown(self):
        if not self.close_test_server:
            self.close_test_server.set()
            await self.server_task

    async def runTestServer(self):
        async with websockets.serve(self.serverHandler, "127.0.0.1", 8765):
            result = await self.close_test_server.wait()

    async def serverHandler(self, ws):
        print('test serverHandler got a connected and started listening')
        self.got_connection.set()
        self.server_ws = ws
        hb_task = asyncio.create_task(self.send_heartbeat())
        while True:
            try:
                message = await ws.recv()
                update = json.loads(message)
                print(update)
                self.receiver.update(update)
            except ConnectionClosedOK:
                print("Client disconnected")
                break
            except ConnectionClosedError as e:
                print(f"Client disconnected with {e}")
                break
        self.server_ws = None
        r = await hb_task

    async def send_heartbeat(self):
        while self.server_ws is not None:
            await self.server_ws.send(json.dumps({'spool0': [], 'spool1': []}))
            await asyncio.sleep(0.9)

    async def test_shutdown_before_connect(self):
        # 1 is the anchor number
        ac = ArpeggioAnchorClient("127.0.0.1", ws_port, 1, self.datastore, self.ob_mock, self.pool, self.stat, None)
        self.assertFalse(ac.connected)
        await ac.shutdown()

    async def clientSetup(self):
        self.ac = ArpeggioAnchorClient("127.0.0.1", ws_port, 1, self.datastore, self.ob_mock, self.pool, self.stat, None)
        self.client_task = asyncio.create_task(self.ac.startup())
        result = await asyncio.wait_for(self.got_connection.wait(), 2)
        await asyncio.sleep(0.1) # client_task needs a chance to act

    async def clientTearDown(self):
        await self.ac.shutdown()
        result = await self.client_task

    async def test_connect_and_shutdown(self):
        await self.clientSetup()
        self.assertTrue(self.ac.connected)

        # should report to UI: Connecting, connected
        self.ob_mock.assert_has_calls([
            call.send_ui(component_conn_status=telemetry.ComponentConnStatus(
                anchor_num=1, websocket_status=telemetry.ConnStatus.CONNECTING, ip_address='127.0.0.1', gripper_model=telemetry.GripperModel.ARPEGGIO)),
            call.send_ui(component_conn_status=telemetry.ComponentConnStatus(
                anchor_num=1, websocket_status=telemetry.ConnStatus.CONNECTED, ip_address='127.0.0.1', gripper_model=telemetry.GripperModel.ARPEGGIO)),
        ])
        self.ob_mock.reset_mock()
        await self.clientTearDown()

    async def test_server_closes(self):
        """
        The client task should not attempt to reconnect if the server closes normally.
        """
        await self.clientSetup()
        self.assertTrue(self.ac.connected)
        # trigger normal shutdown by stopping server task
        self.close_test_server.set()
        await asyncio.sleep(0.1)
        self.assertFalse(self.ac.connected)
        result = await asyncio.wait_for(self.client_task, 1)
        self.assertFalse(result) # false means normal shutdown

    async def test_server_closes_abnormally(self):
        """
        The client task should not attempt to reconnect if the server closes abnormally, but it should return true.
        """
        pass # TODO stop heartbeat signal. confirm client closes itself

    async def test_line_record(self):
        """
        When an arp anchor client receives "spool0"/"spool1" it should put the data in the
        anchor line array in datastore corresponding to that anchor's two lines.
        """
        await self.clientSetup()
        # expected (time, length, speed, torque)
        linerecord = [(88.0, 2.0, 0.0, 1.0), (89.0, 2.1, 0.1, 1.0)]
        asyncio.create_task(self.server_ws.send(json.dumps({'spool0': linerecord})))
        await asyncio.sleep(0.1)
        # anchor_num=1, spool 0 -> line_number = 1*2+0 = 2
        np.testing.assert_array_almost_equal(self.datastore.anchor_line_record[2].getLast(), linerecord[-1])
        await self.clientTearDown()

    async def test_abnormal_disconnect(self):
        """Confirm the client's startup() task would return True if the mock server disconnected with ConnectionClosedError"""
