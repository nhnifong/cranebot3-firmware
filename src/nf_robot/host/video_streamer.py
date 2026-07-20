import subprocess
import time
import logging
import atexit
import threading
import socket
import queue
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import io
import cv2
from urllib.parse import urlparse

from nf_robot.common.util import get_local_ip

logger = logging.getLogger(__name__)

class StreamingHandler(BaseHTTPRequestHandler):
    """Handles the HTTP requests for the MJPEG stream and the demo index page."""

    def log_message(self, format, *args):
        pass # suppress internal logging
    
    def do_GET(self):
        # Parse URL to ignore query parameters (like timestamps used for cache busting)
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            content = """
            <html>
            <head>
                <title>Stringman Local Stream</title>
                <style>
                    body { background: #111; color: #eee; font-family: sans-serif; text-align: center; padding-top: 50px; }
                    img { border: 2px solid #444; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
                </style>
            </head>
            <body>
                <h1>Local MJPEG Stream</h1>
                <img src="/stream.mjpeg" />
                <p>Generated via OpenCV + Python HTTP Server</p>
            </body>
            </html>
            """
            self.wfile.write(content.encode('utf-8'))
            
        elif path == '/stream.mjpeg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            streamer = self.server.streamer
            with streamer._client_lock:
                streamer.client_count += 1
            try:
                # Loop to send frames to the client
                while True:
                    # Wait for a new frame from the streamer
                    with streamer.frame_condition:
                        streamer.frame_condition.wait()
                        frame = streamer.latest_frame

                    if frame:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
            except Exception:
                # Client disconnected
                pass
            finally:
                with streamer._client_lock:
                    streamer.client_count -= 1
        else:
            self.send_error(404)

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True 
    def server_close(self):
        super().server_close()

class MjpegStreamer:
    """
    Streams JPEG encoded frames via HTTP. 
    Intended for low-latency local LAN streaming directly to browsers.
    Does NOT use FFmpeg. Browser must display the stream with an Img tag.
    """
    def __init__(self, width, height, port=8000, bind_address="127.0.0.1"):
        self.width = width
        self.height = height
        self.port = port
        self.bind_address = bind_address
        self.http_server = None
        self.latest_frame = None
        self.frame_condition = threading.Condition()
        self.client_count = 0
        self._client_lock = threading.Lock()
        atexit.register(self.stop)

    def start(self):
        self.http_server = ThreadingHTTPServer((self.bind_address, self.port), StreamingHandler)
        # Inject self into server so handler can access frame buffer
        self.http_server.streamer = self
        
        thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        thread.start()

    def send_frame(self, frame):
        """
        Encodes the frame as JPEG and notifies waiting HTTP clients.
        Expects BGR frame (standard OpenCV format).
        """
        with self._client_lock:
            if self.client_count == 0:
                return

        # Encode frame to JPEG directly in memory
        success, buffer = cv2.imencode('.jpg', frame) #, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        if success:
            with self.frame_condition:
                self.latest_frame = buffer.tobytes()
                self.frame_condition.notify_all()
        else:
            logger.warning("Failed to encode frame to JPEG")

    def stop(self):
        h = self.http_server
        if h is not None:
            h.shutdown()
            h.server_close()
            self.http_server = None

class CompressedStreamer:
    """
    Broadcasts the original compressed video packets over a raw TCP socket to any number of
    simultaneous LAN consumers (e.g. a lerobot recording process on the same machine or LAN),
    without decoding or re-encoding them. Same broadcast intent as MjpegStreamer, but MJPEG
    frames are each independently viewable so a new client can just start receiving whatever
    frame comes next; compressed video packets depend on prior packets via GOP structure, so
    a client that connects mid-stream needs to start at a keyframe. We keep the current GOP
    (everything since, and including, the last keyframe) buffered and replay it to each newly
    connected client before switching them over to the live broadcast.

    Each client gets its own bounded queue and dedicated writer thread, so one slow or
    stalled consumer can't block delivery to any other client, and can never block
    send_packet() -- which is called from the hot receive_video() decode loop.

    A client just needs to open this as a raw byte stream (e.g. PyAV's
    av.open(f'tcp://{host}:{port}')); the bytes are the same Annex-B H264 the component's
    hardware encoder produced, so standard format auto-detection recognizes it with no
    special handling, the same way receive_video() already opens the component's own stream
    without specifying a format.
    """
    QUEUE_MAXSIZE = 200  # packets; generous relative to one GOP so a brief stall doesn't drop a client

    def __init__(self, port, bind_address="127.0.0.1"):
        self.port = port
        self.bind_address = bind_address
        self.server_socket = None
        self._accept_thread = None
        self._running = False
        # Guards _gop_buffer and _clients together: a client must be registered and given
        # its GOP-backlog snapshot as one atomic step, or a packet sent by send_packet() in
        # between could be silently missed (never in the backlog, never broadcast to it
        # because it wasn't registered yet).
        self._lock = threading.Lock()
        self._gop_buffer = []  # packets since (and including) the last keyframe
        self._clients = {}     # socket -> queue.Queue
        atexit.register(self.stop)

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.bind_address, self.port))
        self.server_socket.listen(8)
        self.server_socket.settimeout(1.0)  # let _accept_loop notice self._running go False
        self._running = True
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self):
        while self._running:
            try:
                client_sock, _addr = self.server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed out from under us by stop()
            client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client_queue = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
            with self._lock:
                for data in self._gop_buffer:
                    client_queue.put_nowait(data)
                self._clients[client_sock] = client_queue
            writer_thread = threading.Thread(
                target=self._client_writer, args=(client_sock, client_queue), daemon=True
            )
            writer_thread.start()

    def _client_writer(self, client_sock, client_queue):
        try:
            while True:
                data = client_queue.get()
                if data is None:  # sentinel: stop() or a dropped client
                    break
                client_sock.sendall(data)
        except (BrokenPipeError, OSError):
            pass
        finally:
            with self._lock:
                self._clients.pop(client_sock, None)
            try:
                client_sock.close()
            except OSError:
                pass

    def send_packet(self, data, is_keyframe):
        """Broadcast one packet of already-compressed bytes to every connected client, and
        fold it into the GOP backlog used to bootstrap the next client that connects."""
        with self._lock:
            if is_keyframe:
                self._gop_buffer.clear()
            self._gop_buffer.append(data)
            queues = list(self._clients.values())

        for client_queue in queues:
            try:
                client_queue.put_nowait(data)
            except queue.Full:
                # This client can't keep up. Ask its writer thread to stop rather than let
                # the backlog grow unboundedly or block us; it can reconnect and resync at
                # the next keyframe.
                try:
                    client_queue.put_nowait(None)
                except queue.Full:
                    pass

    def stop(self):
        self._running = False
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except OSError:
                pass
            self.server_socket = None
        with self._lock:
            clients = list(self._clients.items())
            self._clients.clear()
            self._gop_buffer.clear()
        for client_sock, client_queue in clients:
            try:
                client_queue.put_nowait(None)
            except queue.Full:
                pass
            try:
                client_sock.close()
            except OSError:
                pass

class RTMPStreamer:
    """
    Forwards video to an RTMP server (e.g., MediaMTX) using FFmpeg.
    Ideal for cloud streaming or when a centralized media server is used.

    Two modes, chosen once at construction:
    - passthrough=True: pure stream-copy remux of already-compressed H264 Annex-B bytes fed
      via send_packet() (see anchor_client.py's receive_video(), which demuxes the incoming
      rpicam-vid stream and hands us bytes(packet) per packet). No decode, no re-encode --
      the component's camera already hardware-encoded this content once, so decoding it on
      the host just to re-encode it again in software was a pure loss of quality and CPU for
      no benefit.
    - passthrough=False (default): the original behavior. Encodes raw decoded/synthesized
      frames fed via send_frame() with software libx264. Required for streams with no
      pre-existing compressed representation -- e.g. the orthographic floor projection,
      which is computed fresh from multiple camera views (generate_orthographic_floor_maps)
      and was never itself hardware-encoded, so there are no original bytes to pass through.
    """
    def __init__(self, width, height, fps=30, rtmp_url=None, passthrough=False):
        if not rtmp_url:
            raise ValueError("RTMPStreamer requires an rtmp_url")

        self.rtmp_url = rtmp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.passthrough = passthrough
        self.process = None
        self.connection_status = 'ok'
        atexit.register(self.stop)

    def _calculate_bitrate(self):
        # Bitrate estimation based on a 0.5 bits-per-pixel heuristic. Only used in encode
        # mode; passthrough mode forwards whatever bitrate the source was already encoded at.
        raw_bitrate = int(self.width * self.height * self.fps * 0.5)
        target_bitrate = max(200000, min(raw_bitrate, 2500000))
        return f"{target_bitrate // 1000}k"

    def start(self):
        if self.process:
            return

        if self.passthrough:
            command = [
                'ffmpeg',
                '-y',
                '-use_wallclock_as_timestamps', '1',
                '-f', 'h264',
                '-i', '-',
                '-c:v', 'copy',
                '-f', 'flv', self.rtmp_url
            ]
        else:
            gop_size = max(1, int(self.fps * 2))
            bitrate = self._calculate_bitrate()
            command = [
                'ffmpeg',
                '-y',
                '-use_wallclock_as_timestamps', '1',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24', # Standard OpenCV format
                '-s', f'{self.width}x{self.height}',
                '-i', '-',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-g', str(gop_size),
                '-b:v', bitrate,
                '-f', 'flv', self.rtmp_url
            ]

        # stderr must be piped to monitor for connection losses.
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**7
        )

        self.monitor_thread = threading.Thread(target=self._monitor_stderr, daemon=True)
        self.monitor_thread.start()
        logger.info(f"FFmpeg streamer started to {self.rtmp_url} (passthrough={self.passthrough})")

    def _monitor_stderr(self):
        if not self.process or not self.process.stderr:
            return

        for line_bytes in iter(self.process.stderr.readline, b''):
            line = line_bytes.decode('utf-8', errors='ignore').strip()
            if "Error" in line or "failed" in line:
                logger.error(f"FFmpeg: {line}")

            # Check for disconnects
            if "Broken pipe" in line or "Connection reset" in line:
                logger.warning("Media server disconnected.")
                self.connection_status = 'error'
                self.stop()
                break

    def send_frame(self, frame):
        """Encode and send one raw decoded/synthesized frame. Only meaningful when passthrough=False."""
        if not self.process:
            return

        try:
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self._handle_crash(e)

    def send_packet(self, data):
        """Forward one packet of already-compressed bytes (bytes(av.Packet)) as-is. Only
        meaningful when passthrough=True."""
        if not self.process:
            return

        try:
            self.process.stdin.write(data)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self._handle_crash(e)

    def _handle_crash(self, exception):
        logger.error(f"FFmpeg pipe broken: {exception}")
        self.connection_status = 'error'
        self.stop()

    def stop(self):
        if self.process:
            try:
                if self.process.stdin: self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                pass
            finally:
                self.process = None

class NfVideoStreamer:
    """
    Combines a local MjpegStreamer with an optional RTMP RTMPStreamer and an optional
    CompressedStreamer. Calls on_ready(local_uri, stream_path) once after a warmup frame
    count (2 frames when local-only, 20 when streaming to RTMP).

    passthrough=True feeds the RTMP remote (if any) and the CompressedStreamer (if any) via
    send_packet() with already-compressed bytes instead of send_frame() with raw frames --
    see RTMPStreamer/CompressedStreamer for why. Only camera-sourced streams (backed by a
    component's hardware encoder) can use this; anything that synthesizes its own frames
    (e.g. the orthographic floor projection) must stick with the default passthrough=False
    and feed send_frame().

    compressed_port, if given (only meaningful alongside passthrough=True), starts a
    CompressedStreamer broadcasting the original compressed bytes over raw TCP -- for LAN/
    same-machine consumers like a lerobot recording process, which need the original video
    quality without the bandwidth and CPU cost of a JPEG re-encode. This is independent of
    telemetry_env/RTMP: unlike the cloud relay, it needs no external media server, so it's
    available in LAN mode too.
    """
    def __init__(self, width, height, fps, mjpeg_port, stream_path, telemetry_env, on_ready=None, bind_address="127.0.0.1", passthrough=False, compressed_port=None):
        # The advertised host must match what the bind actually accepts: when bound to a
        # specific address, advertise that address; when bound to all interfaces, advertise
        # this host's LAN IP so off-machine clients can reach it.
        if bind_address in ("0.0.0.0", "::", ""):
            advertised_host = get_local_ip() or "localhost"
        else:
            advertised_host = bind_address
        self._local_uri = f'http://{advertised_host}:{mjpeg_port}/stream.mjpeg'
        self._compressed_uri = f'tcp://{advertised_host}:{compressed_port}' if compressed_port else None
        self._stream_path = stream_path
        self._on_ready = on_ready
        self._ready_sent = False
        self._frames_sent = 0
        self._passthrough = passthrough

        self._local = MjpegStreamer(width=width, height=height, port=mjpeg_port, bind_address=bind_address)

        rtmp = None
        if telemetry_env == 'local':
            rtmp = f'rtmp://localhost:1935/{stream_path}'
        elif telemetry_env in ('staging', 'production'):
            rtmp = f'rtmp://media.neufangled.com:1935/{stream_path}'
        elif telemetry_env is not None:
            raise ValueError(f"unusable value for telemetry_env {telemetry_env}")

        self._remote = RTMPStreamer(width=width, height=height, fps=fps, rtmp_url=rtmp, passthrough=passthrough) if rtmp else None
        self._compressed = CompressedStreamer(port=compressed_port, bind_address=bind_address) if (passthrough and compressed_port) else None
        self._frames_before_ready = 2 if rtmp is None else 20

    @property
    def local_uri(self):
        return self._local_uri

    @property
    def compressed_uri(self):
        return self._compressed_uri

    @property
    def stream_path(self):
        return self._stream_path

    def start(self):
        self._local.start()
        if self._remote:
            self._remote.start()
        if self._compressed:
            self._compressed.start()
        logger.info(f'Streaming locally at {self._local_uri}')

    def send_frame(self, frame):
        """Send one decoded/synthesized (and possibly resized) frame to the local MJPEG
        stream, and to the RTMP remote too if this stream is not in passthrough mode."""
        self._local.send_frame(frame)
        if self._remote and not self._passthrough:
            self._remote.send_frame(frame)
        self._frames_sent += 1
        if not self._ready_sent and self._frames_sent >= self._frames_before_ready:
            self._ready_sent = True
            if self._on_ready:
                self._on_ready(self._local_uri, self._stream_path)

    def send_packet(self, data, is_keyframe=False):
        """Forward one packet of the original compressed bytes straight to the RTMP remote
        (a stream-copy remux) and/or the CompressedStreamer (a raw TCP broadcast), whichever
        are configured -- no decode/re-encode either way. No-op unless this stream was
        constructed with passthrough=True. Independent of send_frame(): the local MJPEG
        path always needs decoded/resized RGB frames, fed separately from wherever the raw
        packets come from upstream."""
        if not self._passthrough:
            return
        if self._remote:
            self._remote.send_packet(data)
        if self._compressed:
            self._compressed.send_packet(data, is_keyframe)

    def stop(self):
        self._local.stop()
        if self._remote:
            self._remote.stop()
        if self._compressed:
            self._compressed.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # DEMO MODE: MjpegStreamer
    WIDTH, HEIGHT = 640, 480
    streamer = MjpegStreamer(WIDTH, HEIGHT, port=8000)
    streamer.start()

    logger.info("Generating demo video... Press Ctrl+C to stop.")
    
    try:
        x_pos = 0
        direction = 1
        
        while True:
            start_t = time.time()
            
            # Create a frame (Gradient background + Moving Box)
            # 1. Gradient
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            # Simple color gradient based on x_pos
            # Use reshape to allow broadcasting the vertical gradient across the width
            frame[:, :, 0] = np.linspace(0, 255, HEIGHT).reshape(-1, 1).astype(np.uint8) # Blue channel
            frame[:, :, 1] = (x_pos % 255) # Green channel changes
            
            # 2. Moving Box
            box_size = 50
            if x_pos + box_size >= WIDTH or x_pos < 0:
                direction *= -1
            x_pos += (5 * direction)
            
            # Draw box (Red) - Input is assumed BGR
            cv2.rectangle(frame, (x_pos, 200), (x_pos + box_size, 250), (0, 0, 255), -1)
            
            # Add timestamp text
            cv2.putText(frame, f"Time: {time.time():.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Send to streamer
            streamer.send_frame(frame)
            
            # Maintain 30 FPS
            elapsed = time.time() - start_t
            time.sleep(max(0, 1/30 - elapsed))
            
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        streamer.stop()