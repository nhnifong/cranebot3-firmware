"""
Test that starts up gripper server and connects to it with a gripper client
"""

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

from nf_robot.host.data_store import DataStore
from nf_robot.host.anchor_client import RaspiAnchorClient
from nf_robot.host.stats import StatCounter
from nf_robot.generated.nf import telemetry, control, common
from nf_robot.common.config_loader import create_default_config

ws_port = 8765

class TestAnchorClient(unittest.IsolatedAsyncioTestCase):

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
            await self.server_ws.send(json.dumps({'line_record': [1,2,3,4]}))
            await asyncio.sleep(0.9)

    async def test_shutdown_before_connect(self):
        # 1 is the anchor number
        ac = RaspiAnchorClient("127.0.0.1", ws_port, 1, self.datastore, self.ob_mock, self.pool, self.stat, None)
        self.assertFalse(ac.connected)
        await ac.shutdown()

    async def clientSetup(self):
        self.ac = RaspiAnchorClient("127.0.0.1", ws_port, 1, self.datastore, self.ob_mock, self.pool, self.stat, None)
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
                anchor_num=1, websocket_status=telemetry.ConnStatus.CONNECTING, ip_address='127.0.0.1', gripper_model=telemetry.GripperModel.PILOT)),
            call.send_ui(component_conn_status=telemetry.ComponentConnStatus(
                anchor_num=1, websocket_status=telemetry.ConnStatus.CONNECTED, ip_address='127.0.0.1', gripper_model=telemetry.GripperModel.PILOT)),
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
                'p': np.array([[0.1,0.2,0.3], [0.4,0.5,0.6]]), # p for pose, (rvec,tvec)
                'center': (50,50),
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


class TestStreamVideoLoopCompressedPassthrough(unittest.TestCase):
    """stream_video_loop should only stand up the CompressedStreamer (LAN raw-passthrough
    broadcast) when stringman-headless was started with a non-default --bind_address --
    with the default loopback bind, nothing off-machine could reach it anyway. Mocks
    NfVideoStreamer so this runs without binding any real sockets or threads; sets
    self.connected = False (ComponentClient's default) so stream_video_loop's `while
    self.connected:` loop body never executes and the method returns right after
    construction/start(), which is all this test needs to observe.
    """

    def _make_client(self, bind_address):
        datastore = DataStore()
        ob_mock = MagicMock()
        ob_mock.config = create_default_config()
        ob_mock.bind_address = bind_address
        pool = Mock(spec=Pool).return_value
        stat = StatCounter(ob_mock)
        return RaspiAnchorClient("127.0.0.1", ws_port, 1, datastore, ob_mock, pool, stat, None), ob_mock

    def test_default_bind_address_does_not_start_compressed_streamer(self):
        client, ob_mock = self._make_client(bind_address="127.0.0.1")
        with patch("nf_robot.host.anchor_client.NfVideoStreamer") as mock_nfvs:
            client.stream_video_loop(feed_number=1)
        self.assertIsNone(mock_nfvs.call_args.kwargs["compressed_port"])

    def test_nondefault_bind_address_starts_compressed_streamer_on_expected_port(self):
        client, ob_mock = self._make_client(bind_address="0.0.0.0")
        with patch("nf_robot.host.anchor_client.NfVideoStreamer") as mock_nfvs:
            client.stream_video_loop(feed_number=1)
        # anchor_num=1 -> mjpegport = 4247 + 1 = 4248 -> compressedport = mjpegport + 100
        self.assertEqual(mock_nfvs.call_args.kwargs["compressed_port"], 4348)

    def test_video_ready_advertises_compressed_uri_from_the_streamer(self):
        """on_ready builds the VideoReady message; its compressed_uri should reflect
        whatever NfVideoStreamer actually ended up with, not be recomputed separately."""
        client, ob_mock = self._make_client(bind_address="0.0.0.0")
        with patch("nf_robot.host.anchor_client.NfVideoStreamer") as mock_nfvs:
            mock_vs_instance = mock_nfvs.return_value
            mock_vs_instance.compressed_uri = "tcp://192.168.1.50:4348"
            client.stream_video_loop(feed_number=1)
            on_ready = mock_nfvs.call_args.kwargs["on_ready"]
            on_ready("http://192.168.1.50:4248/stream.mjpeg", "stringman/lan/1")

        sent = ob_mock.send_ui.call_args.kwargs["video_ready"]
        self.assertEqual(sent.compressed_uri, "tcp://192.168.1.50:4348")
