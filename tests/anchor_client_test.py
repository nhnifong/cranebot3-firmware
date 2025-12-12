"""
Test that starts up gripper server and connects to it with a gripper client
"""
import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch, Mock, MagicMock, ANY, AsyncMock, call
import asyncio
import numpy as np
import websockets
from websockets.exceptions import (
    ConnectionClosedOK,
    ConnectionClosedError,
)
from websockets.frames import Close

from multiprocessing import Pool, Queue
import json

from data_store import DataStore
from raspi_anchor_client import RaspiAnchorClient
from stats import StatCounter
from generated.nf import telemetry, control, common

ws_port = 8765

class TestAnchorClient(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.datastore = DataStore()
        self.ob_mock = MagicMock()
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
            await self.server_ws.send(json.dumps({'line_record': [1,2,3,4]}))
            await asyncio.sleep(0.9)

    async def test_shutdown_before_connect(self):
        # 1 is the anchor number
        ac = RaspiAnchorClient("127.0.0.1", ws_port, 1, self.datastore, self.ob_mock, self.pool, self.stat)
        self.assertFalse(ac.connected)
        await ac.shutdown()

    async def clientSetup(self):
        self.ac = RaspiAnchorClient("127.0.0.1", ws_port, 1, self.datastore, self.ob_mock, self.pool, self.stat)
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
                anchor_num=1, websocket_status=telemetry.ConnStatus.CONNECTING, ip_address='127.0.0.1')),
            call.send_ui(component_conn_status=telemetry.ComponentConnStatus(
                anchor_num=1, websocket_status=telemetry.ConnStatus.CONNECTED, ip_address='127.0.0.1')),
        ])
        self.ob_mock.reset_mock()

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
        When an achor client receives "line_record" it should put it in the anchor line array in datastore corresponding to it's anchor number
        """
        await self.clientSetup()
        # expected (time, length, speed)
        linerecord = [(88.0, 2.0, 0.0, True), (89.0, 2.1, 0.1, True)]
        asyncio.create_task(self.server_ws.send(json.dumps({'line_record': linerecord})))
        await asyncio.sleep(0.1)
        np.testing.assert_array_almost_equal(self.datastore.anchor_line_record[1].getLast(), linerecord[-1])
        await self.clientTearDown()

    async def test_handle_detections_gantry(self):
        """
        When the anchor client gets an aruco marker detection from the pool
        if it is a gantry or gripper aruco, and the client is not in pose cal mode
        it should use that info to add an entry to the gantry or gripper pose arrays in the datastore.
        """
        detection_list = [
            {
                'n': 'gantry',
                'r': [0.1,0.2,0.3],
                't': [0.4,0.5,0.6]
            },
        ]
        timestamp = 123456789.0
        await self.clientSetup()
        self.ob_mock.reset_mock() # discard connection status telemetry for this test
        self.ac.handle_detections(detection_list, timestamp)
        last_position = self.datastore.gantry_pos.getLast()
        self.assertEqual(timestamp, last_position[0])

        # assert it recorded the gantry sighting correctly
        # note that the detection was in camera coordinate space, and the gantry position is in world space
        # handle_detections did the solvepnp math but that's tested in cv_common_test not here
        np.testing.assert_array_almost_equal(self.ac.gantry_pos_sightings, [last_position[2:]])

        # assert it sends the gantry sightings to the UI as a telemetry update the next time it receives a message on the websocket.
        await asyncio.sleep(1.0) # at least one heartbeat will be receievd
        self.ob_mock.send_ui.assert_called_once_with(gantry_sightings=telemetry.GantrySightings(
            sightings=[common.Vec3(x=np.float64(last_position[2]), y=np.float64(last_position[3]), z=np.float64(last_position[4]))]
        ))


        await self.clientTearDown()

    async def test_abnormal_disconnect(self):
        """Confirm the client's startup() task would return True if the mock server disconnected with ConnectionClosedError"""
