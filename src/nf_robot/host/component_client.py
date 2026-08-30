
import asyncio
import logging
import os
import signal
import websockets
import time
import json
import cv2
import av
import numpy as np
from functools import partial
import threading
import os
from collections import defaultdict, deque
from websockets.exceptions import ConnectionClosedError, InvalidURI, InvalidHandshake, ConnectionClosedOK
import copy

from nf_robot.common.cv_common import *
from nf_robot.common.pose_functions  import *
from nf_robot.common.util import *
from nf_robot.generated.nf import telemetry
from nf_robot.host.video_streamer import NfVideoStreamer

logger = logging.getLogger(__name__)

# Connecting to a component's video stream can be refused for a moment after it
# announces itself, especially when rpicam-vid has just been relaunched at a new
# resolution. Retry rather than losing video until the next video_ready.
VIDEO_OPEN_ATTEMPTS = 5
VIDEO_OPEN_RETRY_S = 1.5
# How long a new video session waits for the previous one's streaming thread to release
# the mjpeg port. The old thread notices its stop event within its 1s condition wait.
VIDEO_HANDOVER_TIMEOUT_S = 5.0

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'fast;1|fflags;nobuffer|flags;low_delay'

# number of origin detections to average
max_origin_detections = 12

# fastSAM parameters
# seconds between processing frames with fastSAM. there is no need need to run it on every frame, since 
# we are looking at a relatively static image.
sam_rate = 1.0 # per second
sam_confidence_cutoff = 0.75

# the genertic client for a raspberri pi based robot component
# How many recent frames the per-camera latency estimate keeps. At the gripper's 60fps
# this is about two seconds, long enough to be steady and short enough to follow a stream
# that genuinely changes - a resolution switch, or a pi that has started struggling.
VIDEO_LATENCY_SAMPLES = 120


class ComponentClient:
    def __init__(self, address, port, datastore, ob, pool, stat, telemetry_env):
        self.address = address
        self.port = port
        self.origin_poses = defaultdict(lambda: deque(maxlen=max_origin_detections))
        self.datastore = datastore
        self.ob = ob # instance of observer. mocks only need the update_avg_named_pos and send_ui methods
        self.websocket = None
        self.connected = False  # status of connection to websocket
        self.receive_task = None  # Task for receiving messages from websocket
        self.video_task = None  # Task for streaming video
        self.stream_start_ts = None
        self.pool = pool
        self.stat = stat
        self.last_gantry_frame_coords = None
        self.ct = None # task to connect to websocket
        self.save_raw = False
        self.connection_established_event = None
        self.frame = None # last frame of video seen
        self.last_frame_cap_time = None
        # This camera's own capture-to-here latency, per frame. The shared StatCounter
        # mixes every camera into one mean for the UI, which is the right number to show a
        # person and the wrong one to control on: the gripper's 60fps control stream and an
        # anchor's stream do not have the same lag, and a servo loop correcting for video
        # delay needs the lag of the feed it is actually looking at.
        self.latency_samples = deque(maxlen=VIDEO_LATENCY_SAMPLES)
        self.heartbeat_receipt = asyncio.Event()
        self.safety_task = None
        self.telemetry_env = telemetry_env
        self.firmware_update_success = None
        self.firmware_update_pending = False
        # set after a successful firmware update; the component restarts to apply it,
        # so the next dropped connection is expected rather than an abnormal shutdown.
        self.expect_disconnect_from_update = False
        # version of the nf_robot module running on the component server, reported by the
        # server right after connecting. None until reported (older firmware never sends it).
        self.nf_robot_v = None
        # set by pull_logs() while waiting for a response, then hold the log text
        # once received (see receive_loop()'s handling of the 'logs'/'thermal' keys).
        self.pulled_logs = None
        self.pulled_thermal = None
        # Error state reported by this component, or None if it has not reported one.
        # The server repeats it once a second for as long as it holds, so this tracks
        # the current fault rather than a one-shot. Stored and logged for now; nothing
        # else consumes it yet. Note a component only sends this while it is faulted,
        # so this never returns to None on its own once set.
        self.error_state = None

        # saved for setup telemetry
        self.local_video_uri = None
        self.feed_number = None
        self.remote_stream_path = None
        # the active NfVideoStreamer, set by stream_video_loop while it's running (None
        # otherwise); receive_video()'s demux loop forwards raw packet bytes through it.
        self.video_streamer = None
        # A component announces video_ready again whenever it relaunches rpicam-vid, e.g.
        # on a resolution change, so more than one set of video threads can be alive at
        # once. These hand the old set off to the new one: the event ends the previous
        # session, and the thread handle is what the new session waits on before binding
        # the mjpeg port the old one still holds.
        self.video_session_stop = None
        self.streaming_thread = None

        # things used by the video streaming thread
        self.frame_lock = threading.Lock()
        # This condition variable signals the worker when a new frame is ready
        self.new_frame_condition = threading.Condition(self.frame_lock)
        self.last_output_frame = None
        # Set to a path to remux the incoming stream to a file; the demux loop owns the
        # container. Recording keeps the compressed packets exactly as the pi encoded
        # them, so nothing is decoded or re-encoded on the way to disk.
        self.recording_path = None
        self._recording = None
        self.recorded_packets = 0
        self.recording_stream_start_ts = None
        # packets dropped while waiting for the keyframe a recording has to start on
        self._recording_skipped = 0
        # The final, encoded bytes for lerobot. Atomic write, so no lock needed.
        self.lerobot_jpeg_bytes = None
        self.lerobot_mode = False # when false disables constant encoded to improve performance.
        self.calibrating_room_spin = False # set to true momentarily during auto calibration

        self.config = ob.config

        self.conn_status = None # subclass needs to set this in init
        self.last_known_centers = {}
        self.last_known_half_extents = {}  # tag name -> apparent half-size, sizes the search crop

    def send_conn_status(self):
        self.ob.send_ui(component_conn_status=copy.deepcopy(self.conn_status))

    def receive_video(self, port, stop=None):
        """Demux one video session; `stop` ends it when a later session replaces it."""
        if stop is None:
            stop = threading.Event()
        video_uri = f'tcp://{self.address}:{port}'
        # print(f'Connecting to {video_uri}')
        self.conn_status.video_status = telemetry.ConnStatus.CONNECTING
        # cannot send here, not in event loop
        self.notify_video = True
        if self.anchor_num is None: # gripper
            camera_cal = self.config.camera_cal_wide
        else:
            camera_cal = self.config.camera_cal

        options = {
            'rtsp_transport': 'tcp',
            'fflags': 'nobuffer',
            'flags': 'low_delay',
            'fast': '1',
        }

        try:
            # The component announces video_ready when rpicam-vid prints its header line,
            # but the listening socket is not always accepting by the time we get here -
            # most visibly after a resolution change, where the process has just been
            # killed and relaunched and has more to set up before it listens. A refusal
            # here is not a dead camera, and giving up on the first one leaves the client
            # with no video until something else happens to trigger another video_ready.
            container = None
            for attempt in range(VIDEO_OPEN_ATTEMPTS):
                try:
                    container = av.open(video_uri, options=options, mode='r')
                    break
                except (av.error.ConnectionRefusedError, av.error.TimeoutError):
                    if attempt == VIDEO_OPEN_ATTEMPTS - 1:
                        raise
                    logger.info(f'Video stream at {video_uri} not accepting yet, retrying '
                                f'({attempt + 1}/{VIDEO_OPEN_ATTEMPTS})')
                    time.sleep(VIDEO_OPEN_RETRY_S)

            stream = next(s for s in container.streams if s.type == 'video')
            stream.thread_type = "SLICE"

            # start thread for streaming and forwarding frames
            streaming_thread = None
            components_to_stream = [None, *self.config.preferred_cameras]
            if self.anchor_num in components_to_stream:
                # The previous session's streamer still owns the mjpeg port until its
                # thread returns, and binding it while it does raises EADDRINUSE.
                previous = self.streaming_thread
                if previous is not None and previous.is_alive():
                    previous.join(timeout=VIDEO_HANDOVER_TIMEOUT_S)
                    if previous.is_alive():
                        logger.warning('Previous video streaming thread did not exit; '
                                       'starting the new one anyway')
                streaming_thread = threading.Thread(
                    target=self.stream_video_loop,
                    kwargs={"feed_number": components_to_stream.index(self.anchor_num),
                            "stop": stop},
                    daemon=True)
                self.streaming_thread = streaming_thread
                streaming_thread.start()

            self.conn_status.video_status = telemetry.ConnStatus.CONNECTED
            self.notify_video = True
            lastSam = time.time()
            last_time = time.time()

            next_full_scan = time.time()

            def error_callback_func(error):
                logger.error(f"Error in pool worker: {error}")

            # Demux (not decode) so we can grab each packet's original compressed bytes
            # before decoding it. bytes(packet) is the same H264 video that was hardware
            # encoded on the pi. re-using it where possible saves resources and reduces latency.
            # mjpeg streamer is still around for the UI, for which mjpeg is still the only low
            # latency option.
            for packet in container.demux(stream):
                if not self.connected or stop.is_set():
                    break
                if packet.dts is None:
                    continue  # flush packet at stream end, not real data

                self._service_recording(stream, packet)

                if self.video_streamer is not None:
                    self.video_streamer.send_packet(bytes(packet), packet.is_keyframe)

                pool_stopped = False
                for av_frame in packet.decode():
                    # determine the wall time when the frame was captured
                    timestamp = self.stream_start_ts + av_frame.time
                    self.last_frame_cap_time = timestamp

                    fr = av_frame.to_ndarray(format='rgb24')
                    with self.new_frame_condition:
                        self.frame = fr
                        self.new_frame_condition.notify()

                    # save information about stream latency and framerate
                    now = time.time()
                    self.stat.latency.append(now - timestamp)
                    self.latency_samples.append(now - timestamp)
                    fr = 1/(now - last_time)
                    self.stat.framerate.append(fr)
                    last_time = now

                    # send frame to apriltag detector
                    try:
                        if self.stat.pending_frames_in_pool < 60:

                            # perform a full scan 1/s
                            if time.time() > next_full_scan:
                                next_full_scan = time.time() + 1
                                self.stat.pending_frames_in_pool += 1

                                self.pool.apply_async(
                                    locate_markers,
                                    (self.frame, camera_cal, None),
                                    callback=partial(self.handle_detections, timestamp=timestamp),
                                    error_callback=error_callback_func
                                )
                            else:
                                # otherwise, send only small cropped areas to the pool for detection
                                crops_data = []
                                for tag_name, center in self.last_known_centers.items():
                                    # window scales with the tag's apparent size, so a tag that
                                    # grows as the camera closes in stays inside its own crop
                                    x1, y1, x2, y2 = crop_window(
                                        center,
                                        self.last_known_half_extents.get(tag_name),
                                        self.frame.shape,
                                    )

                                    # Calling .copy() severs the slice from the base array memory,
                                    # guaranteeing that pickle only sends the few kilobytes of the crop over IPC.
                                    crops_data.append({
                                        'crop': self.frame[y1:y2, x1:x2].copy(),
                                        'x1': x1,
                                        'y1': y1,
                                        'name': tag_name
                                    })

                                self.stat.pending_frames_in_pool += 1
                                self.pool.apply_async(
                                    locate_markers,
                                    (None, camera_cal, crops_data),
                                    callback=partial(self.handle_detections, timestamp=timestamp),
                                    error_callback=error_callback_func
                                )

                        else:
                            pass
                            # print(f'Dropping frame because there are already too many pending.')
                            # TODO record fraction of frames which are dropped in stat collector
                    except ValueError:
                        pool_stopped = True
                        break # the pool is not running

                    # sleep is mandatory or this thread could prevent self.handle_detections from running and fill up the pool with work.
                    # handle_detections runs in this process, but in a thread managed by the pool.
                    time.sleep(0.005)

                if pool_stopped:
                    break

            if streaming_thread is not None:
                streaming_thread.join()

        except (av.error.TimeoutError, av.error.ConnectionRefusedError):
            logger.warning('No video stream available')
            self.conn_status.video_status = telemetry.ConnStatus.NOT_DETECTED
            self.notify_video = True
            return

        finally:
            if 'container' in locals():
                container.close()

    def video_latency(self, default=None):
        """Median capture-to-here latency of this camera's recent frames, in seconds.

        Median rather than mean: a stalled decode or a burst of dropped packets produces a
        few very late frames, and a mean is dragged along by them. `default` comes back
        while the sample window is still empty, so a caller can name what to assume before
        the stream has said anything.

        One caveat on the absolute value. Capture is stamped by the component's own clock
        (stream_start_ts comes from the bot, plus the frame's time in the container) and
        "here" by ours, so any offset between the two machines is inside this number. It is
        a latency plus a clock skew. A control loop that needs to know *when* a frame was
        taken should prefer comparing capture times to other bot-stamped times - the grip
        sensor records are stamped on the gripper too, so that comparison has no skew in it
        at all - and use this for reporting, for sanity checks, and for reaching across to
        quantities that only exist on this side.
        """
        if not self.latency_samples:
            return default
        return float(np.median(self.latency_samples))

    def stream_video_loop(self, feed_number, stop=None):
        """
        This runs in a dedicated thread. It waits for a signal that a new frame is
        available, runs it through process_frame (a hardware-specific hook subclasses may
        override; most hardware just passes the frame through untouched now that components
        capture video at exactly the resolution each consumer needs), and forwards the
        result to the local MJPEG stream, the RTMP
        remote, and the LAN compressed-passthrough broadcast, whichever are configured.

        Numpy/cv2 functions release the GIL, which is why this is a thread rather than a
        task (the main loop can keep running while this one works).

        feed_number identifies which of the preferred cameras this is. 0 is the gripper, 1 and 2 are the two overhead cams.

        `stop` ends this session, which is how the ports get released for the session that
        replaces it when the component relaunches its camera.
        """
        if stop is None:
            stop = threading.Event()
        # The media server authorizes a publish by the robot id in this path (it looks up
        # whether that id has a live telemetry uplink), so it has to be the id the control
        # plane minted for us, not anything local.
        path = f'stringman/{self.ob.telemetry.cloud_robot_id}/{feed_number}'
        mjpegport = 4246 if self.anchor_num is None else 4247 + self.anchor_num

        bind_address = getattr(self.ob, 'bind_address', '127.0.0.1')
        # The compressed-passthrough broadcast is only useful to consumers off this
        # machine (e.g. a lerobot recorder elsewhere on the LAN); with the default loopback
        # bind, nothing outside this machine could reach it anyway, so don't bother running
        # it. --bind_address must be explicitly set to something else to opt in.
        if bind_address != '127.0.0.1':
            # A separate port range for the broadcast (see CompressedStreamer): same offset
            # from mjpegport for every feed, so it's easy to find the pair for a given stream.
            compressedport = mjpegport + 100
        else:
            compressedport = None

        def on_ready(local_uri, stream_path):
            self.local_video_uri = local_uri
            self.feed_number = feed_number
            self.remote_stream_path = stream_path
            t = telemetry.VideoReady(
                is_gripper=self.anchor_num is None,
                anchor_num=self.anchor_num,
                local_uri=local_uri,
                stream_path=stream_path,
                feed_number=feed_number,
                compressed_uri=vs.compressed_uri,
            )
            logger.info(f'Sending video ready {t}')
            self.ob.send_ui(video_ready=t)

        vs = NfVideoStreamer(
            # width/height/fps only matter for NfVideoStreamer's non-passthrough (encode)
            # mode; this stream is always passthrough=True, so they're unused here.
            width=0, height=0, fps=0,
            mjpeg_port=mjpegport, stream_path=path,
            telemetry_env=self.telemetry_env, on_ready=on_ready,
            bind_address=bind_address,
            # This stream is backed by the component's own hardware H264 encoder (see
            # receive_video()'s demux loop), so the RTMP remote and the LAN compressed
            # broadcast can both be fed the original compressed bytes directly instead of
            # decoding and re-encoding them.
            passthrough=True,
            compressed_port=compressedport,
        )
        vs.start()
        # Exposed so receive_video()'s demux loop can forward each packet's original
        # compressed bytes straight to vs.send_packet() (RTMP + LAN passthrough) without
        # waiting on this thread.
        self.video_streamer = vs
        logger.info(f'Streaming video locally at {vs.local_uri}, '
                    f'compressed passthrough at {vs.compressed_uri}')
        self.streaming_active = True

        while self.connected and not stop.is_set():
            with self.new_frame_condition:
                # Wait until the main receive_video loop signals us.
                # The 'wait' call will timeout after 1 second to re-check
                # the exit flags, allowing the thread to exit gracefully.
                signaled = self.new_frame_condition.wait(timeout=1.0)
                if not signaled:
                    continue
                # We were woken up, so copy the frame pointer while we have the lock
                frame_to_encode = self.frame

            if frame_to_encode is None:
                logger.debug(f'No frame to encode {self}')
                continue

            # Do the actual work outside the lock
            # This lets the receive_video loop add the next frame without waiting for the encode.
            self.last_output_frame = self.process_frame(frame_to_encode)
            ortho_event = getattr(self.ob, 'ortho_event', None)
            if ortho_event is not None:
                ortho_event.set()
            rgb = cv2.cvtColor(self.last_output_frame, cv2.COLOR_BGR2RGB)
            vs.send_frame(rgb)

        # Only tear down what is still ours: if a later session got going before this
        # thread noticed its stop event, the recording and these fields are already that
        # session's, and closing them here would take video out from under it.
        if self.video_streamer is vs:
            self._close_recording()
            self.remote_stream_path = None
            self.local_video_uri = None
            self.video_streamer = None
        vs.stop()

    def _service_recording(self, stream, packet):
        """Open, write to, or close the stream recording the demux loop owns."""
        if self.recording_path is not None and self._recording is None:
            # Start at a keyframe. Recording begins the moment a caller sets
            # recording_path, which lands mid-GOP: those leading slices reference an
            # SPS/PPS that went past before the file existed, so anything decoding the
            # result reports "non-existing PPS 0 referenced" and throws the frames away as
            # far as the next keyframe - which is the same place starting here begins. The
            # wait costs one GOP, before the sweep it is recording has started moving.
            if not packet.is_keyframe:
                self._recording_skipped += 1
                return
            container = av.open(str(self.recording_path), 'w', format='mpegts')
            self._recording = (container, container.add_stream_from_template(stream))
            self.recording_stream_start_ts = self.stream_start_ts
            self.recorded_packets = 0
            logger.info(f'Recording video to {self.recording_path} '
                        f'(waited {self._recording_skipped} packet(s) for a keyframe)')
            self._recording_skipped = 0
        elif self.recording_path is None and self._recording is not None:
            self._close_recording()

        if self._recording is not None:
            container, out_stream = self._recording
            # a fresh packet from the same bytes, so muxing does not disturb the
            # original on its way to the decoder
            copy = av.Packet(bytes(packet))
            copy.pts, copy.dts, copy.time_base = packet.pts, packet.dts, packet.time_base
            copy.stream = out_stream
            container.mux(copy)
            self.recorded_packets += 1

    def _close_recording(self):
        if self._recording is not None:
            try:
                self._recording[0].close()
            except Exception:
                logger.exception('closing video recording')
            logger.info(f'Recorded {self.recorded_packets} packets to {self.recording_path}')
            self._recording = None
        self._recording_skipped = 0

    def process_frame(self, frame_to_encode):
        """
        Identity by default: components now capture video at exactly the resolution each
        consumer needs, so no generic resize belongs here. Subclasses may still override
        this for genuine hardware-specific per-frame processing that isn't just resizing.
        The returned frame is what is used for inference and sent to any teleoperation
        pipelines.
        Runs in a separate thread from the main client.
        """
        return frame_to_encode

    async def connect_websocket(self):
        # main client loop
        self.conn_status.websocket_status = telemetry.ConnStatus.CONNECTING
        self.conn_status.video_status = telemetry.ConnStatus.NOT_DETECTED
        self.conn_status.ip_address = self.address
        self.send_conn_status()

        self.abnormal_shutdown = False # indicating we had a connection and then lost it unexpectedly
        self.failed_to_connect = False # indicating we failed to ever make a connection
        ws_uri = f"ws://{self.address}:{self.port}"
        # print(f"Connecting to {ws_uri}...")
        try:
            async with websockets.connect(ws_uri, max_size=None, open_timeout=10) as websocket:
                self.connected = True
                logger.info(f"Connected to {ws_uri}.")
                # Set an event that the observer is waiting on.
                if self.connection_established_event is not None:
                    self.connection_established_event.set()
                await self.receive_loop(websocket)
        except (asyncio.exceptions.CancelledError, websockets.exceptions.ConnectionClosedOK):
            pass # normal close
        except websockets.exceptions.ConnectionClosedError as e:
            if self.expect_disconnect_from_update:
                logger.info(f"Component server anum={self.anchor_num} disconnected to restart after a firmware update")
                self.expect_disconnect_from_update = False
            else:
                logger.warning(f"Component server anum={self.anchor_num} disconnected abnormally: {e}")
                self.abnormal_shutdown = True
        except (OSError, InvalidURI, TimeoutError, InvalidHandshake) as e:
            # normal answer when waiting for component to come online
            self.failed_to_connect = True
        finally:
            self.connected = False
        self.conn_status.websocket_status = telemetry.ConnStatus.NOT_DETECTED
        self.conn_status.video_status = telemetry.ConnStatus.NOT_DETECTED
        self.send_conn_status()
        return self.abnormal_shutdown

    async def firmware_update(self):
        # once complete, will be set to True or False
        logger.info(f'Starting firmware update on {self.address}')
        self.firmware_update_success = None
        self.firmware_update_pending = False
        await self.send_commands({'run_update': None})
        started = time.time()
        while self.firmware_update_success is None and self.connected:
            if time.time() > started+3 and not self.firmware_update_pending:
                logger.warning(f'Component does not yet support self update. run \nssh pi@{self.address} "/opt/robot/env/bin/pip install --upgrade \\"nf_robot[pi]\\""\npassword Fo0bar!!')
                return None
            await asyncio.sleep(0.5)
        return self.firmware_update_success

    async def pull_logs(self, timeout=10):
        """Request this component's recent log lines and wait for the response.
        Returns (log_text, thermal_log_text); either is None if disconnected, the
        component didn't respond in time, or the firmware doesn't send that key."""
        self.pulled_logs = None
        self.pulled_thermal = None
        await self.send_commands({'get_logs': None})
        started = time.time()
        while self.pulled_logs is None and self.connected:
            if time.time() - started > timeout:
                logger.warning(f'Timed out waiting for logs from {self.address}')
                return None, None
            await asyncio.sleep(0.2)
        return self.pulled_logs, self.pulled_thermal

    def _handle_firmware_update_complete(self, upd):
        """Handle a 'firmware_update_complete' update from the component (see run_update()
        on the server side). Called from receive_loop()."""
        if type(upd) != dict:
            return
        if 'pending' in upd:
            # this component supports updates
            self.firmware_update_pending = True
        if 'returncode' in upd:
            logger.info(f'pip install result on {self.address} = {upd["returncode"] == 0}')
            self.firmware_update_success = upd['returncode'] == 0
            if self.firmware_update_success:
                # the component will restart to apply the update, dropping
                # the connection. treat that next drop as a normal shutdown.
                self.expect_disconnect_from_update = True
            elif 'error' in upd:
                # pip's own output, so this shows up on the host's console
                # immediately instead of only in the component's own log file.
                logger.error(f'Self update failed on {self.address}:\n{upd["error"]}')

    async def receive_loop(self, websocket):
        self.conn_status.websocket_status = telemetry.ConnStatus.CONNECTED
        self.send_conn_status()
        # loop of a single websocket connection.
        # save a reference to this for send_commands
        self.websocket = websocket
        self.notify_video = False
        # send configuration to robot component to override default.
        r = await self.send_config()
        # start task to watch heartbeat event
        self.safety_task = asyncio.create_task(self.safety_monitor())
        vid_thread = None
        # Loop until disconnected
        while self.connected:
            try:
                message = await websocket.recv()
                # print(f'received message of length {len(message)}')
                update = json.loads(message)
                if 'video_ready' in update:
                    port = int(update['video_ready'][0])
                    self.stream_start_ts = float(update['video_ready'][1])
                    logger.debug(f'stream_start_ts={self.stream_start_ts} ({time.time()-self.stream_start_ts:.2f}s ago)')
                    # A component announces this again every time it relaunches its camera,
                    # e.g. after a resolution change. The threads of the previous session
                    # are demuxing a stream that is already gone and holding ports the new
                    # session needs, so retire them here rather than leaving both running.
                    if self.video_session_stop is not None:
                        self.video_session_stop.set()
                    self.video_session_stop = threading.Event()
                    vid_thread = threading.Thread(
                        target=self.receive_video,
                        kwargs={"port": port, "stop": self.video_session_stop},
                        daemon=True)
                    vid_thread.start()
                if 'firmware_update_complete' in update:
                    self._handle_firmware_update_complete(update['firmware_update_complete'])
                if 'nf_robot_v' in update:
                    self.nf_robot_v = update['nf_robot_v']
                if 'temp' in update:
                    self.conn_status.temp = update['temp']
                if 'torque' in update:
                    if update['torque']:
                        self.conn_status.motor_enabled = telemetry.MotorTorque.ENABLED
                    else:
                        self.conn_status.motor_enabled = telemetry.MotorTorque.DISABLED
                    # keep the robot-wide torque state the UI toggle reads in sync
                    self.ob.publish_torque_state()
                if 'error_state' in update:
                    # repeated by the component every second while it holds, so log the
                    # transitions rather than every repeat
                    if update['error_state'] != self.error_state:
                        logger.error(f'component {self.address} reports error state: {update["error_state"]}')
                    self.error_state = update['error_state']
                if 'logs' in update:
                    self.pulled_logs = update['logs']
                if 'thermal' in update:
                    self.pulled_thermal = update['thermal']
                # this event is used to detect an un-responsive state.
                self.heartbeat_receipt.set()
                await self.handle_update_from_ws(update)

                # do this here because we seemingly can't do it in receive_video
                if self.notify_video:
                    self.send_conn_status()
                    self.notify_video = False

            except Exception as e:
                # don't catch websockets.exceptions.ConnectionClosedOK here because we want it to trip the infinite generator in websockets.connect
                # so it will stop retrying. after it has the intended effect, websockets.connect will raise it again, so we catch it in 
                # connect_websocket
                logger.warning(f"Connection to {self.address} closed. {e}")
                self.connected = False
                self.websocket = None
                # self.conn_status.websocket_status = telemetry.ConnStatus.NOT_DETECTED
                # self.conn_status.video_status = telemetry.ConnStatus.NOT_DETECTED
                # self.send_conn_status()
                raise e # TODO figure out if this causes the abnormal shutdown return value in connect_websocket like it should
                break
        if vid_thread is not None:
            # vid_thread should stop because self.connected is False
            vid_thread.join()

    async def send_commands(self, update):
        if self.connected:
            x = json.dumps(update)
            # by trying to get the result out of the future, you force any exception in the task to be raised
            # since this could be a websockets.exceptions.ConnectionClosedError it's important not to let it disappar
            result = await self.websocket.send(x)

    async def slow_stop_spool(self):
        # spool will decelerate at the rate allowed by the config file.
        # tracking mode will switch to 'speed'
        try:
            result = await self.send_commands({'aim_speed': 0})
        except websockets.exceptions.ConnectionClosedOK:
            pass

    async def startup(self):
        self.ct = asyncio.create_task(self.connect_websocket())
        return await self.ct

    async def shutdown(self):
        if self.safety_task is not None:
            self.safety_task.cancel()
            result = await self.safety_task
        if self.connected:
            self.connected = False
            if not self.abnormal_shutdown and self.websocket:
                result = await self.websocket.close()
        elif self.ct:
            self.ct.cancel()

    def shutdown_sync(self):
        # this might get called twice
        if self.connected:
            self.connected = False
            if self.websocket:
                asyncio.create_task(self.websocket.close())
        elif self.ct:
            self.ct.cancel()

    async def safety_monitor(self):
        """Notifies observer if this anchor stops sending line record updates for some time"""
        TIMEOUT=4 # seconds
        last_update = time.time()
        while self.connected:
            try:
                result = await asyncio.wait_for(self.heartbeat_receipt.wait(), TIMEOUT)
                # if you see the event within the timeout, all is well, clear it and wait again
                self.heartbeat_receipt.clear()
                last_update = time.time()
            except TimeoutError:
                # print(f'No update sent from {self.anchor_num} in {TIMEOUT} seconds. it may have gone offline. sending ping')
                try:
                    pong_future = await self.websocket.ping()
                    latency = await asyncio.wait_for(pong_future, TIMEOUT)
                    # some hiccup on the server raspi made it unable to send anything for some time but it's not down.
                    # print(f'Pong received in {latency}s, must have been my imagination.')
                    continue
                except (ConnectionClosedError, TimeoutError):
                    # it's no longer running, either because it lost power, or the server crashed.
                    if self.anchor_num is None:
                        name = 'Gripper'
                    else:
                        name = "Anchor {self.anchor_num}"
                    logger.warning(f"{name} confirmed down. hasn't been seen in {time.time() - last_update:.1f} seconds.")
                    self.connected = False
                    # immediately trigger the "abnormal shutdown" return from the connect_websocket task
                    # this is how the observer is actually notified. follow the control flow by looking at `if abnormal_close:` in observer.py
                    if self.websocket and self.websocket.transport:
                        self.websocket.transport.close()
                except ConnectionClosedOK:
                    return
            except asyncio.exceptions.CancelledError:
                return

