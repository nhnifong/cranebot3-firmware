import subprocess
import cv2
import time
import logging
import atexit
import numpy as np

logger = logging.getLogger(__name__)

class VideoStreamer:
    def __init__(self, rtmp_url, width=640, height=480, fps=30, local_udp_port=None):
        self.rtmp_url = rtmp_url
        self.width = width
        self.height = height
        self.fps = fps # an estimate, nothing bad happens if you fail to call send_frame() exactly at this rate
        self.local_udp_port = local_udp_port
        self.process = None
        self.connection_status = 'ok'
        
        # find a free port
        if local_udp_port == None:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.bind(('127.0.0.1', 0))
                self.local_udp_port = s.getsockname()[1]
        else:
            self.local_udp_port = local_udp_port

        atexit.register(self.stop)

    def _calculate_bitrate(self):
        # Estimate bitrate based on resolution and fps
        raw_bitrate = int(self.width * self.height * self.fps * 0.1)
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
            '-pix_fmt', 'bgr24', # OpenCV uses BGR
            '-s', f'{self.width}x{self.height}', # Input resolution
            '-i', '-', # Read from STDIN
            
            # Encoding settings (tune for latency)
            '-c:v', 'libx264', # h264 encoding, usually faster than anythign else.
            '-pix_fmt', 'yuv420p', # Required for compatibility
            '-preset', 'ultrafast', # Prioritize speed over compression ratio
            '-tune', 'zerolatency',
            '-g', str(gop_size), # Force keyframe every 2 seconds
            '-b:v', bitrate, # Calculated bitrate
            
            # Output format 1: Cloud RTMP
            '-f', 'flv', 
            self.rtmp_url
        ]

        # If a local port is requested, add the second output
        if self.local_udp_port:
            command.extend([
                '-f', 'mpegts',
                f'udp://127.0.0.1:{self.local_udp_port}?pkt_size=1316'
            ])

        #  redirect stderr to PIPE so we can log errors if it crashes,
        self.process = subprocess.Popen(
            command, 
            stdin=subprocess.PIPE,
            # stderr=subprocess.PIPE
        )
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

        # Ensure the frame is the correct size
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

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

if __name__ == "__main__":
    import sys
    
    # 1. Configuration
    # Replace this IP with your actual MediaMTX server IP if running remotely
    # Usage: python -m src.nf_robot.networking.video_streamer <OPTIONAL_IP>
    target_ip = "34.134.108.13"
    if len(sys.argv) > 1:
        target_ip = sys.argv[1]
        
    # Standard RTMP URL format
    rtmp_endpoint = f"rtmp://{target_ip}:1935/robot_0"
    
    width = 1920
    height = 1080
    fps = 30
    
    print(f"Starting broadcast to {rtmp_endpoint} at {width}x{height} @ {fps}fps")
    
    # 3. Start Streamer
    # We pass local_udp_port=0 to test the auto-selection logic. 
    # Check stdout to see which port it picked.
    streamer = VideoStreamer(rtmp_endpoint, width=width, height=height, fps=fps, local_udp_port=0)
    streamer.start()

    frame_count = 0
    
    try:
        while True:
            # Stop if error state is reached (unauthorized OR broken pipe)
            if streamer.connection_status in ['unauthorized', 'error']:
                print(f"Stopping stream due to fatal status: {streamer.connection_status}")
                break

            fill_value = (frame_count % 250) + 5
            frame = np.full((width, height, 3), fill_value, dtype=np.uint8)
            frame_count += 1
            
            # Send to RTMP (and local UDP)
            streamer.send_frame(frame)
            
            # Local preview window
            cv2.imshow("Broadcasting", frame)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping stream...")
        streamer.stop()
        cv2.destroyAllWindows()