import subprocess
import time
import logging
import atexit
import socket
import threading
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import io
import cv2

logger = logging.getLogger(__name__)

class StreamingHandler(BaseHTTPRequestHandler):
    """Handles the HTTP requests for the MJPEG stream and the demo index page."""
    
    def do_GET(self):
        if self.path == '/':
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
            
        elif self.path == '/stream.mjpeg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            try:
                # Loop to send frames to the client
                while True:
                    # Wait for a new frame from the streamer
                    with self.server.streamer.frame_condition:
                        self.server.streamer.frame_condition.wait()
                        frame = self.server.streamer.latest_frame
                    
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
        else:
            self.send_error(404)

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    pass

class MjpegStreamer:
    """
    Streams JPEG encoded frames via HTTP. 
    Intended for low-latency local LAN streaming directly to browsers.
    Does NOT use FFmpeg. Browser must display the stream with an Img tag.
    """
    def __init__(self, width, height, port=8000):
        self.width = width
        self.height = height
        self.port = port
        self.http_server = None
        self.latest_frame = None
        self.frame_condition = threading.Condition()
        atexit.register(self.stop)

    def start(self):
        self.http_server = ThreadingHTTPServer(('0.0.0.0', self.port), StreamingHandler)
        # Inject self into server so handler can access frame buffer
        self.http_server.streamer = self
        
        thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"MJPEG Stream accessible at http://localhost:{self.port}")

    def send_frame(self, frame):
        """
        Encodes the frame as JPEG and notifies waiting HTTP clients.
        Expects BGR frame (standard OpenCV format).
        """
        # Encode frame to JPEG directly in memory
        success, buffer = cv2.imencode('.jpg', frame) #, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        if success:
            with self.frame_condition:
                self.latest_frame = buffer.tobytes()
                self.frame_condition.notify_all()
        else:
            logger.warning("Failed to encode frame to JPEG")

    def stop(self):
        if self.http_server:
            self.http_server.shutdown()
            self.http_server.server_close()
            self.http_server = None

class VideoStreamer:
    """
    Streams video to an RTMP server (e.g., MediaMTX) using FFmpeg.
    Ideal for cloud streaming or when a centralized media server is used.
    """
    def __init__(self, width, height, fps=30, rtmp_url=None):
        if not rtmp_url:
            raise ValueError("VideoStreamer requires an rtmp_url")
            
        self.rtmp_url = rtmp_url
        self.width = width
        self.height = height
        self.fps = fps 
        self.process = None
        self.connection_status = 'ok'
        atexit.register(self.stop)

    def _calculate_bitrate(self):
        # Bitrate estimation based on a 0.5 bits-per-pixel heuristic.
        raw_bitrate = int(self.width * self.height * self.fps * 0.5)
        target_bitrate = max(200000, min(raw_bitrate, 2500000))
        return f"{target_bitrate // 1000}k"

    def start(self):
        if self.process:
            return

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
        logger.info(f"FFmpeg streamer started to {self.rtmp_url}")

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
        if not self.process:
            return

        try:
            self.process.stdin.write(frame.tobytes())
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