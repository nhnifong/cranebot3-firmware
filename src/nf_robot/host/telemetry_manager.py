from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from nf_robot.generated.nf import telemetry

logger = logging.getLogger(__name__)

CONTROL_PLANE_PRODUCTION = "wss://neufangled.com"
CONTROL_PLANE_STAGING = "wss://nf-site-monolith-staging-690802609278.us-east1.run.app"
CONTROL_PLANE_LOCAL = "ws://localhost:8080"

# peer kinds handed to the connect/disconnect callbacks
LOCAL = 'local'
CLOUD = 'cloud'


def normalize_control_plane_host(host):
    """Reduce a control plane URL to the "ws(s)://host[:port]" form used as the key into
    config.relay_credentials, or return None if it is not a usable URL.

    The binding UI reports the control plane that minted a set of credentials (see
    RelayCreds.control_plane_host), and whatever it sends has to end up byte-identical to
    the constants above or the creds are filed where no run will look for them. So an
    http(s) scheme is folded to its websocket equivalent and any path, query or trailing
    slash is dropped.
    """
    if not host:
        return None
    parsed = urlparse(host.strip())
    scheme = {'http': 'ws', 'https': 'wss'}.get(parsed.scheme, parsed.scheme)
    if scheme not in ('ws', 'wss') or not parsed.netloc:
        return None
    return f'{scheme}://{parsed.netloc}'


# telemetry items whose retain_key is a constant. The relay resends the most recent item
# per retain_key to UIs that connect late.
CONSTANT_RETAIN_KEYS = (
    'new_anchor_poses',
    'swing_cancellation_state',
    'tension_regulation_state',
    'torque_state',
    'auto_targeting_state',
    'task_status',
)


class TelemetryManager:
    """
    Owns every telemetry destination and the sockets that reach them.

    Two transports carry the same batches: a local websocket server (a UI or a lerobot
    session on this machine or the LAN connects to it) and, when telemetry_env is set, one
    outbound connection to the cloud relay. Both are bidirectional, so inbound control
    bytes and connection lifecycle events go back to the owner through the three callbacks.

    send() is thread safe and only buffers; nothing reaches a socket until flush(), which
    is driven by the position estimator's 60hz loop.
    """

    def __init__(self, config, telemetry_env, bind_address, port,
                 on_control_message, on_peer_connected, on_peer_disconnected):
        """
        config          -- live configuration object; robot_id and relay_credentials are
                           read at send/connect time so updates to it take effect.
        telemetry_env   -- None (LAN only), 'local', 'staging' or 'production'.
        on_control_message(message: bytes)      -- awaited once per inbound message.
        on_peer_connected(peer)                 -- awaited after a peer connects, before
                                                   its messages are read, so the owner can
                                                   push setup telemetry.
        on_peer_disconnected(peer, local_remaining) -- awaited after a peer goes away.
        """
        self.config = config
        self.telemetry_env = telemetry_env
        self.bind_address = bind_address
        self.port = port
        self._on_control_message = on_control_message
        self._on_peer_connected = on_peer_connected
        self._on_peer_disconnected = on_peer_disconnected

        self.run = True
        # websockets to locally connected UIs
        self.connected_local_clients = set()
        self.cloud_websocket = None
        self._cloud_task = None
        self._server = None
        self._buffer = deque(maxlen=100)
        self._buffer_lock = threading.RLock()
        # set when relay credentials change, so the cloud loop can retry immediately instead
        # of polling for them
        self._creds_updated = asyncio.Event()

    @property
    def control_plane_host(self):
        """The telemetry ws_protocol_and_host of the control plane this robot belongs to,
        derived from telemetry_env. This same string is the key into config.relay_credentials
        (e.g. "wss://neufangled.com"), so binding and connecting agree on which creds to use.

        Production is the default because it is the only control plane a robot outside a dev
        enviroment is ever bound to. this decides how a key is stored when binding a robot with
        a user id on a server"""
        if self.telemetry_env == 'staging':
            return CONTROL_PLANE_STAGING
        if self.telemetry_env == 'local':
            return CONTROL_PLANE_LOCAL
        return CONTROL_PLANE_PRODUCTION

    @property
    def cloud_robot_id(self):
        """The id this robot is bound to on the control plane, or '' if it isn't bound."""
        creds = self.config.relay_credentials.get(self.control_plane_host)
        return creds.robot_id if creds else ''

    def send(self, **kwargs):
        """
        Ensure that the given telemetry item is sent to every connected destination.
        keyword args are passed directly to telemetry item, so you can construct one like this

        manager.send(pop_message=telemetry.Popup('hello'))

        Thread safe: callers on camera and logging threads reach this too.
        """
        if len(kwargs.keys()) != 1:
            raise ValueError
        key, msg = list(kwargs.items())[0]

        # mark certain messages with a retain key. the server will resend them to new UIs
        item = telemetry.TelemetryItem(**kwargs)
        if key in CONSTANT_RETAIN_KEYS:
            item.retain_key = key
        if key == 'component_conn_status':
            if msg.is_gripper:
                item.retain_key = f'component_conn_status_g'
            else:
                item.retain_key = f'component_conn_status_{msg.anchor_num}'
        if key == 'video_ready':
            item.retain_key = f'video_ready_{msg.feed_number}'
        if key == 'episode_control' and item.episode_control.status is not None:
            item.retain_key = f'lerobot_status'

        # Add item to batch
        with self._buffer_lock:
            self._buffer.append(item)

    async def flush(self):
        """
        Flush the teleoperation buffer, sending all data to every destination.
        Normally called within position estimator's 60hz loop
        """
        with self._buffer_lock:
            batch = telemetry.TelemetryBatchUpdate(
                robot_id=self.cloud_robot_id,
                updates=list(self._buffer)
            )
            self._buffer.clear()
        to_send = bytes(batch)
        # copy list to prevent RuntimeError: Set changed size during iteration
        destinations = self.connected_local_clients.copy()
        if self.cloud_websocket:
            destinations.add(self.cloud_websocket) # will only be connected when telemetry_env is not None
        for websocket in destinations:
            try:
                r = await websocket.send(to_send)
            except (ConnectionClosedOK, ConnectionClosedError) as e:
                pass # stale connection

    def start_cloud_link(self):
        """Begin connecting (and reconnecting) to the cloud relay. No-op in LAN mode."""
        if self.telemetry_env is None or self._cloud_task is not None:
            return
        self._cloud_task = asyncio.create_task(self._cloud_main())

    def credentials_updated(self):
        """Call after config.relay_credentials changes so the cloud link picks the new creds
        up now: it wakes a link waiting to be bound, and drops a link running on stale creds
        so the reconnect uses the new ones."""
        self._creds_updated.set()
        if self.cloud_websocket is not None:
            asyncio.create_task(self.cloud_websocket.close())

    @asynccontextmanager
    async def serving(self):
        """Serve the local telemetry websocket for the duration of the block."""
        async with websockets.serve(self._handle_local_client, self.bind_address, self.port) as server:
            self._server = server
            try:
                yield self
            finally:
                self._server = None

    def abort_cloud_socket(self):
        """Drop the relay connection without waiting for a close handshake. Used by the
        signal handler, which runs on the loop thread and must not block."""
        if self.cloud_websocket is not None:
            self.cloud_websocket.transport.abort()

    async def aclose(self):
        """Stop the cloud link. Idempotent."""
        self.run = False
        if self._cloud_task is not None:
            task, self._cloud_task = self._cloud_task, None
            task.cancel()
            try:
                await task
            except asyncio.exceptions.CancelledError:
                pass

    async def _handle_local_client(self, websocket):
        # Called when UI connects to a websocket that is opened to accept control commands
        self.connected_local_clients.add(websocket)
        logger.info('Connection received from local UI process')

        # send anything that it would need up-front
        r = await self._on_peer_connected(LOCAL)
        try:
            async for message in websocket:
                r = await self._on_control_message(message) # Handle 'ControlBatchUpdate'
                # warning, any uncaught exception here will kill this websocket connection
                # but the observer would go on running, possibly in a bad state.
        except (ConnectionClosedError, ConnectionClosedOK) as e:
            pass
        finally:
            self.connected_local_clients.discard(websocket)
            await self._on_peer_disconnected(LOCAL, len(self.connected_local_clients))

    async def _cloud_main(self):
        ws_protocol_and_host = self.control_plane_host

        while self.run:
            creds = self.config.relay_credentials.get(ws_protocol_and_host)
            if creds is None or not creds.key:
                logger.warning(
                    f'No relay credentials for {ws_protocol_and_host}; not connecting to the cloud '
                    f'telemetry relay. Bind this robot to an account to obtain a key.'
                )
                # park until credentials_updated() says there is something new to try
                self._creds_updated.clear()
                await self._creds_updated.wait()
                continue

            self._creds_updated.clear()
            try:
                ws_path = f"{ws_protocol_and_host}/telemetry_v2/{creds.robot_id}"
                async with websockets.connect(
                    ws_path,
                    max_size=None,
                    open_timeout=10,
                    additional_headers={"Authorization": f"Bearer {creds.key}"},
                ) as websocket:
                    self.cloud_websocket = websocket
                    if self._creds_updated.is_set():
                        # The credentials changed while this connection was being
                        # established, so credentials_updated() ran while cloud_websocket
                        # was still None and had nothing to close. Drop this link before
                        # announcing it; the next iteration connects with the new
                        # credentials. Without this the robot stays bound to the stale ones
                        # until something else happens to drop the link, because the read
                        # loop below never checks for new credentials.
                        logger.info('Credentials changed while connecting; reconnecting')
                        self.cloud_websocket = None
                        continue
                    logger.info(f'Connected to control plane {ws_path}')
                    # send anything that it would need up-front
                    await self._on_peer_connected(CLOUD)
                    try:
                        async for message in websocket:
                            r = await self._on_control_message(message)
                            if not self.run:
                                r = await websocket.close()
                    except ConnectionClosedOK as e:
                        logger.info(f'ConnectionClosedOK from {ws_path}')
                    except ConnectionClosedError as e:
                        logger.error(e)
                    finally:
                        logger.info(f'Disconnected from control plane {ws_path}')
                        self.cloud_websocket = None
                        await self._on_peer_disconnected(CLOUD, len(self.connected_local_clients))
            except (asyncio.exceptions.CancelledError, websockets.exceptions.ConnectionClosedOK):
                pass # normal close
            except websockets.exceptions.InvalidStatus as e:
                if e.response.status_code == 409:
                    logger.warning(
                        f'Control plane rejected connection (HTTP 409): another robot is '
                        f'already connected with id "{creds.robot_id}".'
                    )
                else:
                    logger.warning(
                        f'Control plane rejected connection: HTTP {e.response.status_code}'
                    )
                await asyncio.sleep(10) # still could be considered a transient error, but probably not.
            except ConnectionRefusedError:
                logger.warning(f'Connection to control plane refused')
            except websockets.exceptions.InvalidMessage:
                logger.warning('Connection to control plane ended due to invalid message')
            if not self.run:
                # cancellation during shutdown lands in the CancelledError arm above; leave
                # now rather than sitting out the backoff first
                break
            # a reconnect prompted by new credentials shouldn't wait out the backoff
            if not self._creds_updated.is_set():
                await asyncio.sleep(2)
