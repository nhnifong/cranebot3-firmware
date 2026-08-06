import asyncio
import unittest
from unittest.mock import patch

import websockets

from nf_robot.generated.nf import common, telemetry
from nf_robot.host.telemetry_manager import (
    CLOUD,
    CONTROL_PLANE_LOCAL,
    CONTROL_PLANE_PRODUCTION,
    CONTROL_PLANE_STAGING,
    LOCAL,
    TelemetryManager,
)

from port_utils import free_port


class FakeConfig:
    """Just the two fields the manager reads out of the real config."""

    def __init__(self):
        self.robot_id = 'test-robot'
        self.relay_credentials = {}


class FakeSocket:
    """Stands in for a destination websocket so flush can be tested without a peer."""

    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(data)

    def batches(self):
        return [telemetry.TelemetryBatchUpdate().parse(b) for b in self.sent]


class TelemetryManagerTestBase(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.config = FakeConfig()
        self.control_messages = []
        self.connects = []
        self.disconnects = []
        self.tm = self.make_manager()

    def make_manager(self, telemetry_env=None):
        async def on_control_message(message):
            self.control_messages.append(message)

        async def on_peer_connected(peer):
            self.connects.append(peer)

        async def on_peer_disconnected(peer, local_remaining):
            self.disconnects.append((peer, local_remaining))

        return TelemetryManager(
            config=self.config,
            telemetry_env=telemetry_env,
            bind_address='127.0.0.1',
            port=free_port(),
            on_control_message=on_control_message,
            on_peer_connected=on_peer_connected,
            on_peer_disconnected=on_peer_disconnected,
        )

    async def wait_for(self, predicate, timeout=2.0):
        """Poll until predicate() is true. Returns whether it became true."""
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if predicate():
                return True
            await asyncio.sleep(0.01)
        return predicate()


class TestBuffering(TelemetryManagerTestBase):
    """send() only buffers; flush() is what reaches the sockets."""

    async def test_send_requires_exactly_one_item(self):
        with self.assertRaises(ValueError):
            self.tm.send()
        with self.assertRaises(ValueError):
            self.tm.send(task_status=telemetry.TaskStatus(), logs=telemetry.Logs(line=['a']))

    async def test_nothing_is_sent_until_flush(self):
        sock = FakeSocket()
        self.tm.connected_local_clients.add(sock)
        self.tm.send(logs=telemetry.Logs(line=['before']))
        self.assertEqual(sock.sent, [])

        await self.tm.flush()
        batch, = sock.batches()
        self.assertEqual(batch.robot_id, 'test-robot')
        self.assertEqual([u.logs.line[0] for u in batch.updates], ['before'])

    async def test_flush_clears_the_buffer(self):
        sock = FakeSocket()
        self.tm.connected_local_clients.add(sock)
        self.tm.send(logs=telemetry.Logs(line=['once']))
        await self.tm.flush()
        await self.tm.flush()
        # the second batch is empty rather than a repeat of the first
        self.assertEqual(len(sock.batches()[1].updates), 0)

    async def test_buffer_is_bounded(self):
        # the deque drops the oldest items rather than growing without limit when
        # nothing is flushing, e.g. before any destination has connected
        for i in range(150):
            self.tm.send(logs=telemetry.Logs(line=[str(i)]))
        sock = FakeSocket()
        self.tm.connected_local_clients.add(sock)
        await self.tm.flush()
        batch, = sock.batches()
        self.assertEqual(len(batch.updates), 100)
        self.assertEqual(batch.updates[0].logs.line[0], '50')

    async def test_send_is_thread_safe(self):
        """Camera and logging threads call send() directly."""
        import threading

        def worker(start):
            for i in range(200):
                self.tm.send(logs=telemetry.Logs(line=[str(start + i)]))

        threads = [threading.Thread(target=worker, args=(t * 1000,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # maxlen bounds it; the point is that no append raced into a corrupt state
        self.assertEqual(len(self.tm._buffer), 100)

    async def test_flush_survives_a_closed_destination(self):
        class ClosedSocket:
            async def send(self, data):
                raise websockets.exceptions.ConnectionClosedError(None, None)

        good = FakeSocket()
        self.tm.connected_local_clients.add(ClosedSocket())
        self.tm.connected_local_clients.add(good)
        self.tm.send(logs=telemetry.Logs(line=['x']))
        await self.tm.flush()  # must not raise
        self.assertEqual(len(good.sent), 1)


class TestRetainKeys(TelemetryManagerTestBase):
    """The relay replays the newest item per retain_key to UIs that connect late."""

    def retain_key_for(self, **kwargs):
        self.tm.send(**kwargs)
        return self.tm._buffer[-1].retain_key

    async def test_constant_retain_keys(self):
        self.assertEqual(self.retain_key_for(task_status=telemetry.TaskStatus()), 'task_status')
        self.assertEqual(
            self.retain_key_for(new_anchor_poses=telemetry.AnchorPoses()), 'new_anchor_poses')
        self.assertEqual(
            self.retain_key_for(auto_targeting_state=telemetry.AutoTargetingState(enabled=True, present=True)),
            'auto_targeting_state')

    async def test_per_component_retain_keys(self):
        self.assertEqual(
            self.retain_key_for(component_conn_status=telemetry.ComponentConnStatus(is_gripper=True)),
            'component_conn_status_g')
        self.assertEqual(
            self.retain_key_for(component_conn_status=telemetry.ComponentConnStatus(is_gripper=False, anchor_num=2)),
            'component_conn_status_2')
        self.assertEqual(
            self.retain_key_for(video_ready=telemetry.VideoReady(feed_number=3)), 'video_ready_3')

    async def test_episode_control_retained_only_with_a_status(self):
        self.assertEqual(
            self.retain_key_for(episode_control=common.EpisodeControl(
                status=common.LerobotSessionStatus(status=common.LerobotStatus.RECORDING))),
            'lerobot_status')
        # a bare command is a one-shot, not state to replay
        self.assertIsNone(self.retain_key_for(
            episode_control=common.EpisodeControl(command=common.EpCommand.PING)))

    async def test_unretained_items_have_no_key(self):
        self.assertIsNone(self.retain_key_for(logs=telemetry.Logs(line=['hi'])))


class TestLocalServer(TelemetryManagerTestBase):

    async def test_connect_exchange_and_disconnect(self):
        async with self.tm.serving():
            async with websockets.connect(f'ws://127.0.0.1:{self.tm.port}') as ws:
                # the owner is told before any message is read, so it can push setup telemetry
                self.assertTrue(await self.wait_for(lambda: self.connects == [LOCAL]))
                self.assertEqual(len(self.tm.connected_local_clients), 1)

                self.tm.send(logs=telemetry.Logs(line=['hello']))
                await self.tm.flush()
                batch = telemetry.TelemetryBatchUpdate().parse(await asyncio.wait_for(ws.recv(), 2))
                self.assertEqual(batch.updates[0].logs.line, ['hello'])

                await ws.send(b'\x08\x01')
                self.assertTrue(await self.wait_for(lambda: self.control_messages == [b'\x08\x01']))

            self.assertTrue(await self.wait_for(lambda: self.disconnects == [(LOCAL, 0)]))
            self.assertEqual(len(self.tm.connected_local_clients), 0)

    async def test_batches_go_to_every_local_client(self):
        async with self.tm.serving():
            async with websockets.connect(f'ws://127.0.0.1:{self.tm.port}') as a, \
                    websockets.connect(f'ws://127.0.0.1:{self.tm.port}') as b:
                self.assertTrue(await self.wait_for(lambda: len(self.tm.connected_local_clients) == 2))
                self.tm.send(logs=telemetry.Logs(line=['both']))
                await self.tm.flush()
                for ws in (a, b):
                    batch = telemetry.TelemetryBatchUpdate().parse(await asyncio.wait_for(ws.recv(), 2))
                    self.assertEqual(batch.updates[0].logs.line, ['both'])

    async def test_disconnect_reports_remaining_local_clients(self):
        """The owner shuts down on the *last* local client leaving, so the count matters."""
        async with self.tm.serving():
            async with websockets.connect(f'ws://127.0.0.1:{self.tm.port}') as keeper:
                async with websockets.connect(f'ws://127.0.0.1:{self.tm.port}'):
                    self.assertTrue(await self.wait_for(lambda: len(self.tm.connected_local_clients) == 2))
                self.assertTrue(await self.wait_for(lambda: self.disconnects == [(LOCAL, 1)]))
            self.assertTrue(await self.wait_for(lambda: len(self.disconnects) == 2))
            self.assertEqual(self.disconnects[1], (LOCAL, 0))

    async def test_server_stops_when_the_block_exits(self):
        async with self.tm.serving():
            pass
        with self.assertRaises((ConnectionRefusedError, OSError)):
            await websockets.connect(f'ws://127.0.0.1:{self.tm.port}')


class TestControlPlaneSelection(TelemetryManagerTestBase):

    async def test_host_per_environment(self):
        self.assertEqual(self.make_manager(None).control_plane_host, CONTROL_PLANE_LOCAL)
        self.assertEqual(self.make_manager('local').control_plane_host, CONTROL_PLANE_LOCAL)
        self.assertEqual(self.make_manager('staging').control_plane_host, CONTROL_PLANE_STAGING)
        self.assertEqual(self.make_manager('production').control_plane_host, CONTROL_PLANE_PRODUCTION)

    async def test_cloud_robot_id_is_blank_until_bound(self):
        tm = self.make_manager('production')
        self.assertEqual(tm.cloud_robot_id, '')
        self.config.relay_credentials[CONTROL_PLANE_PRODUCTION] = common.RelayCreds(robot_id='r9', key='k')
        self.assertEqual(tm.cloud_robot_id, 'r9')

    async def test_lan_mode_never_starts_a_cloud_link(self):
        self.config.relay_credentials[CONTROL_PLANE_LOCAL] = common.RelayCreds(robot_id='r1', key='k')
        self.tm.start_cloud_link()  # telemetry_env is None
        self.assertIsNone(self.tm._cloud_task)


class TestCloudLink(TelemetryManagerTestBase):
    """The relay link is driven against a stand-in control plane on localhost. That's the
    address telemetry_env='local' points at, but on a free port rather than the real 8080,
    so CONTROL_PLANE_LOCAL is patched to match for the duration of the test."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.relay_paths = []
        self.relay_sockets = []

        relay_port = free_port()
        # The same string is both the connect target and the key into relay_credentials, so
        # patch it in the module under test and use self.control_plane when binding creds.
        self.control_plane = f'ws://localhost:{relay_port}'
        self.control_plane_patcher = patch(
            'nf_robot.host.telemetry_manager.CONTROL_PLANE_LOCAL', self.control_plane)
        self.control_plane_patcher.start()
        self.addCleanup(self.control_plane_patcher.stop)

        async def relay(websocket):
            self.relay_paths.append(websocket.request.path)
            self.relay_sockets.append(websocket)
            try:
                async for message in websocket:
                    await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                pass

        self.relay_server = await websockets.serve(relay, 'localhost', relay_port)
        self.tm = self.make_manager('local')
        self.addAsyncCleanup(self.stop)

    async def stop(self):
        await self.tm.aclose()
        self.relay_server.close()
        await self.relay_server.wait_closed()

    def bind(self, robot_id='r1', key='k'):
        self.config.relay_credentials[self.control_plane] = common.RelayCreds(robot_id=robot_id, key=key)

    async def test_connects_with_the_bound_robot_id(self):
        self.bind(robot_id='r42')
        self.tm.start_cloud_link()
        self.assertTrue(await self.wait_for(lambda: self.relay_paths == ['/telemetry_v2/r42']))
        self.assertTrue(await self.wait_for(lambda: self.connects == [CLOUD]))
        self.assertIsNotNone(self.tm.cloud_websocket)

    async def test_unbound_link_waits_and_connects_when_credentials_arrive(self):
        self.tm.start_cloud_link()
        await asyncio.sleep(0.2)
        self.assertEqual(self.relay_paths, [], 'must not connect without credentials')

        self.bind(robot_id='r7')
        self.tm.credentials_updated()
        self.assertTrue(await self.wait_for(lambda: self.relay_paths == ['/telemetry_v2/r7'], timeout=1.0),
                        'credentials_updated must not wait out a retry interval')

    async def test_rebinding_reconnects_with_the_new_credentials(self):
        self.bind(robot_id='r7')
        self.tm.start_cloud_link()
        self.assertTrue(await self.wait_for(lambda: self.relay_paths == ['/telemetry_v2/r7']))

        self.bind(robot_id='r8', key='k2')
        self.tm.credentials_updated()
        self.assertTrue(
            await self.wait_for(lambda: self.relay_paths == ['/telemetry_v2/r7', '/telemetry_v2/r8'], timeout=2.0))
        self.assertTrue(await self.wait_for(lambda: self.disconnects == [(CLOUD, 0)]))

    async def test_cloud_receives_batches_and_delivers_control(self):
        self.bind()
        self.tm.start_cloud_link()
        self.assertTrue(await self.wait_for(lambda: self.tm.cloud_websocket is not None))

        # the stand-in relay echoes, so one flush proves both directions
        self.tm.send(logs=telemetry.Logs(line=['cloud']))
        await self.tm.flush()
        self.assertTrue(await self.wait_for(lambda: len(self.control_messages) == 1))
        batch = telemetry.TelemetryBatchUpdate().parse(self.control_messages[0])
        self.assertEqual(batch.updates[0].logs.line, ['cloud'])

    async def test_dropped_link_reconnects(self):
        self.bind()
        self.tm.start_cloud_link()
        self.assertTrue(await self.wait_for(lambda: len(self.relay_sockets) == 1))

        await self.relay_sockets[0].close()
        self.assertTrue(await self.wait_for(lambda: self.disconnects == [(CLOUD, 0)]))
        self.assertTrue(await self.wait_for(lambda: len(self.relay_paths) == 2, timeout=5.0))

    async def test_aclose_stops_the_link(self):
        self.bind()
        self.tm.start_cloud_link()
        self.assertTrue(await self.wait_for(lambda: self.tm.cloud_websocket is not None))

        await self.tm.aclose()
        self.assertFalse(self.tm.run)
        self.assertIsNone(self.tm._cloud_task)
        await self.tm.aclose()  # idempotent

        # and it stays down
        before = len(self.relay_paths)
        await asyncio.sleep(0.3)
        self.assertEqual(len(self.relay_paths), before)


if __name__ == '__main__':
    unittest.main()
