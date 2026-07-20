"""
Tests for ComponentClient's shared connect/reconnect/shutdown lifecycle, exercised
through ArpeggioAnchorClient since it's a thin, easily-instantiated subclass.

This is the direct (non-mocked-away) unit coverage of ComponentClient — ported from the
now-deleted pilot tests/anchor_client_test.py, which exercised the same base-class behavior
through RaspiAnchorClient before the pilot anchor was removed.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
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
        return ArpeggioAnchorClient("127.0.0.1", ws_port, 1, datastore, ob_mock, pool, stat, None), ob_mock

    def test_default_bind_address_does_not_start_compressed_streamer(self):
        client, ob_mock = self._make_client(bind_address="127.0.0.1")
        with patch("nf_robot.host.component_client.NfVideoStreamer") as mock_nfvs:
            client.stream_video_loop(feed_number=1)
        self.assertIsNone(mock_nfvs.call_args.kwargs["compressed_port"])

    def test_nondefault_bind_address_starts_compressed_streamer_on_expected_port(self):
        client, ob_mock = self._make_client(bind_address="0.0.0.0")
        with patch("nf_robot.host.component_client.NfVideoStreamer") as mock_nfvs:
            client.stream_video_loop(feed_number=1)
        # anchor_num=1 -> mjpegport = 4247 + 1 = 4248 -> compressedport = mjpegport + 100
        self.assertEqual(mock_nfvs.call_args.kwargs["compressed_port"], 4348)

    def test_video_ready_advertises_compressed_uri_from_the_streamer(self):
        """on_ready builds the VideoReady message; its compressed_uri should reflect
        whatever NfVideoStreamer actually ended up with, not be recomputed separately."""
        client, ob_mock = self._make_client(bind_address="0.0.0.0")
        with patch("nf_robot.host.component_client.NfVideoStreamer") as mock_nfvs:
            mock_vs_instance = mock_nfvs.return_value
            mock_vs_instance.compressed_uri = "tcp://192.168.1.50:4348"
            client.stream_video_loop(feed_number=1)
            on_ready = mock_nfvs.call_args.kwargs["on_ready"]
            on_ready("http://192.168.1.50:4248/stream.mjpeg", "stringman/lan/1")

        sent = ob_mock.send_ui.call_args.kwargs["video_ready"]
        self.assertEqual(sent.compressed_uri, "tcp://192.168.1.50:4348")
