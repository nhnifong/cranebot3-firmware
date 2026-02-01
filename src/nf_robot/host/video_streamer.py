import subprocess
import time
import logging
import atexit
import socket
import threading

logger = logging.getLogger(__name__)

class VideoStreamer:
    def __init__(self, width, height, fps=30, rtmp_url=None):
        self.rtmp_url = rtmp_url
        self.width = width
        self.height = height
        self.fps = fps # an estimate, nothing bad happens if you fail to call send_frame() exactly at this rate
        self.process = None
        self.connection_status = 'ok'
        self.local_udp_port = None
        
        # If no RTMP URL is provided, broadcast on a random free local UDP port.
        if rtmp_url is None:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.bind(('127.0.0.1', 0))
                self.local_udp_port = s.getsockname()[1]

        atexit.register(self.stop)

    def _calculate_bitrate(self):
        # Bitrate estimation based on a 0.5 bits-per-pixel heuristic for H.264.
        raw_bitrate = int(self.width * self.height * self.fps * 0.5)
        target_bitrate = max(200000, min(raw_bitrate, 2500000))
        return f"{target_bitrate // 1000}k"

    def start(self):
        """Starts the FFMPEG process and the stderr monitoring thread."""
        if self.process:
            return

        # A 2-second GOP (Group of Pictures) ensures that clients joining a live 
        # stream don't have to wait more than 2 seconds for a keyframe.
        gop_size = max(1, int(self.fps * 2))
        bitrate = self._calculate_bitrate()

        command = [
            'ffmpeg',
            '-y',
            '-use_wallclock_as_timestamps', '1',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}',
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-g', str(gop_size),
            '-b:v', bitrate,
        ]
            
        if self.rtmp_url:
            command.extend(['-f', 'flv', self.rtmp_url])

        if self.local_udp_port:
            # mpegts over UDP for low-latency local previews.
            command.extend([
                '-f', 'mpegts', 
                f'udp://127.0.0.1:{self.local_udp_port}?pkt_size=1316'
            ])

        # stderr must be piped to monitor for connection losses.
        self.process = subprocess.Popen(
            command, 
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0 # Use unbuffered I/O for immediate error detection
        )

        # Launching a daemon thread ensures the stream monitoring doesn't block 
        # the main process and shuts down automatically when the main thread exits.
        self.monitor_thread = threading.Thread(target=self._monitor_stderr, daemon=True)
        self.monitor_thread.start()

        logger.info(f"FFmpeg streamer started to {self.rtmp_url or 'local UDP'}")

    def _monitor_stderr(self):
        """
        Continuously reads FFmpeg's stderr to detect crashes or connection issues.
        """
        if not self.process or not self.process.stderr:
            return

        # Iterating over readline blocks until a line is available or the pipe closes.
        for line_bytes in iter(self.process.stderr.readline, b''):
            line = line_bytes.decode('utf-8', errors='ignore').strip()

            # Skip noise, but log actual errors.
            if "Error" in line or "failed" in line or "Connection" in line:
                logger.error(f"FFmpeg Error: {line}")

            # Specific triggers for reconnection logic or auth failure.
            # "Broken pipe" usually happens when the remote server stops accepting data.
            if "Broken pipe" in line or "Connection reset" in line:
                logger.warning("Media server disconnected. Updating status to error.")
                self.connection_status = 'error'
                self.stop()
                break
            
            if "not authorized" in line.lower() or "Authentication failed" in line:
                logger.error("Streaming unauthorized. Video can only be published from robots that are sending telemetry.")
                self.connection_status = 'unauthorized'
                self.stop()
                break

    def send_frame(self, frame):
        """
        Encodes and pushes a single frame to the stream.
        """
        if not self.process or self.connection_status != 'ok':
            return

        try:
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            # Write failures occur immediately if the background process died.
            self._handle_crash(e)

    def _handle_crash(self, exception):
        """Cleanup and update status when a write failure occurs."""
        logger.error(f"FFmpeg pipe broken during write: {exception}")
        self.connection_status = 'error'
        self.stop()

    def stop(self):
        """Gracefully shuts down the FFmpeg process."""
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception:
                # If the process is already gone, don't raise during cleanup.
                pass
            finally:
                self.process = None

        if self.connection_status not in ['unauthorized', 'error']:
            self.connection_status = 'disconnected'