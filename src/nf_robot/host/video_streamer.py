import subprocess
import cv2
import time
import logging
import atexit
import numpy as np
import socket

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
        
        # if no rtmp url, only broadcast locally on UDP
        # find a free local port for that.
        if rtmp_url == None:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.bind(('127.0.0.1', 0))
                self.local_udp_port = s.getsockname()[1]

        atexit.register(self.stop)

    def _calculate_bitrate(self):
        # Estimate bitrate based on resolution and fps
        raw_bitrate = int(self.width * self.height * self.fps * 0.5)
        target_bitrate = max(200000, min(raw_bitrate, 2500000))
        return f"{target_bitrate // 1000}k"

    def start(self):
        """
        Starts the FFMPEG process. 
        pipe raw video into stdin, and FFMPEG sends FLV/RTMP to the server.
        """
        if self.process:
            return

        # Calculate a keyframe interval (GOP) that ensures a keyframe every 2 seconds.
        # For 30fps -> GOP 60. For 2fps -> GOP 4.
        # This keeps stream join latency low (~2s) regardless of framerate.
        gop_size = max(1, int(self.fps * 2))
        bitrate = self._calculate_bitrate()

        command = [
            'ffmpeg',
            '-y', # Overwrite output files
            
            # Use wallclock time for input timestamps.
            # This handles variable frame rates correctly for live streaming.
            '-use_wallclock_as_timestamps', '1',

            '-f', 'rawvideo', # Input format
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}', # Input resolution
            '-i', '-', # Read from STDIN
            
            # Encoding settings (tune for latency)
            '-c:v', 'libx264', # h264 encoding, usually faster than anythign else.
            '-pix_fmt', 'yuv420p', # Required for compatibility
            '-preset', 'ultrafast', # Prioritize speed over compression ratio
            '-tune', 'zerolatency',
            '-g', str(gop_size), # Force keyframe every 2 seconds
            # '-b:v', "1200k", # Calculated bitrate
        ]
            
        # If streaming to a remote server over RTMP is requested, add that output
        if self.rtmp_url:
            command.extend([
                '-f', 'flv', self.rtmp_url
            ])
        

        # If a local port is requested, add the second output
        if self.local_udp_port:
            command.extend([
                '-f', 'mpegts', f'udp://127.0.0.1:{self.local_udp_port}?pkt_size=1316'
            ])

        #  redirect stderr to PIPE so we can log errors if it crashes,
        self.process = subprocess.Popen(
            command, 
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # TODO read stderr and look for this line. it usually means the media server hung up on you (not authorized)
        # ERROR:video_streamer:FFmpeg pipe broken: [Errno 32] Broken pipe
        # if seen, call
        # self.stop()
        # self.connection_status = 'error'

        logger.info(f"FFmpeg streamer started to {self.rtmp_url}")
        if self.local_udp_port:
            logger.info(f"Also streaming locally to udp://127.0.0.1:{self.local_udp_port}")

    def send_frame(self, frame):
        """
        Encodes and pushes a single frame to the stream.
        Frame must be a numpy array (OpenCV image) of size (width, height).
        """
        if not self.process:
            return

        try:
            # Write raw bytes to ffmpeg's stdin
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
        except Exception as e:
            self._handle_crash(e)


    def _handle_crash(self, exception):
        """Analyze why ffmpeg died, update status, and cleanup."""
        logger.error(f"FFmpeg pipe broken: {exception}")
        
        # Capture stderr if available for post-mortem debugging
        if self.process and self.process.stderr:
            try:
                # Non-blocking read might fail if process is already dead/closed, so we wrap it
                # We don't analyze it, just print it for the developer
                err_out = self.process.stderr.read()
                if err_out:
                    logger.debug(f"FFmpeg stderr content: {err_out.decode('utf-8', errors='ignore')}")
            except Exception:
                pass

        self.stop()
        self.connection_status = 'error'

    def stop(self):
        # Unregister to prevent memory leaks if called manually multiple times
        atexit.unregister(self.stop)
        
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=1)
            except Exception:
                # Force kill if it hangs
                self.process.kill()
            finally:
                self.process

        if self.connection_status not in ['unauthorized', 'error']:
            self.connection_status = 'disconnected'
