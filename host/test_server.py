"""
Test server

Immitates the ESP32-S3 based robot components.
streams images from a directory and fake acceleration data

Advertises itself with MDNS as cranebot-test-server

"""

import http.server
import socketserver
import argparse
import os
import json
import time
import io
import uuid
import zeroconf
import threading

include_accel_data = False
keep_advertising_mdns = True

class StreamingHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--myboundary')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            image_dir = "images"  # Directory containing sample jpeg images that simulate what the device camera would see
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_files.sort() # Ensure consistent order

            if not image_files:
                print("No images found in the 'images' directory.")
                return

            frame_count = 0
            while True:
                try:
                    image_path = os.path.join(image_dir, image_files[frame_count % len(image_files)])
                    with open(image_path, "rb") as image_file:
                        image_data = image_file.read()

                    # Simulate acceleration data
                    acceleration = {
                        "x": frame_count * 0.1,  # Example changing acceleration
                        "y": frame_count * 0.2,
                        "z": frame_count * 0.3,
                        "timestamp": time.time()
                    }
                    accel_json = json.dumps(acceleration).encode('utf-8')

                    # Send image data
                    self.wfile.write(b"--myboundary\r\n")
                    self.wfile.write(f"Content-Type: image/jpeg\r\nContent-Length: {len(image_data)}\r\nX-Timestamp-Sec: {int(time.time())}\r\nX-Timestamp-Usec: {int(time.time()*1000000)%1000000}\r\n\r\n".encode('utf-8'))
                    self.wfile.write(image_data)
                    self.wfile.write(b"\r\n")

                    if include_accel_data:
                        # Send acceleration data
                        self.wfile.write(b"--myboundary\r\n")
                        self.wfile.write(f"Content-Type: application/json\r\nContent-Length: {len(accel_json)}\r\nX-Timestamp-Sec: {int(time.time())}\r\nX-Timestamp-Usec: {int(time.time()*1000000)%1000000}\r\n\r\n".encode('utf-8'))
                        self.wfile.write(accel_json)
                        self.wfile.write(b"\r\n")

                    frame_count += 1
                    time.sleep(1/30) # Simulate frame rate

                except BrokenPipeError:
                    print("Client disconnected.")
                    break
                except Exception as e: # Catch file errors
                    print(f"Error: {e}")
                    break
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

def register_mdns_service(name, service_type, port, properties={}):
    """Registers an mDNS service on the network."""

    zc = zeroconf.Zeroconf()
    info = zeroconf.ServiceInfo(
        service_type,
        name + "." + service_type,
        port=port,
        properties=properties,
        addresses=['127.0.0.1'],
        server=f'test-server-{uuid.uuid4()}',
    )

    zc.register_service(info)
    print(f"Registered service: {name} ({service_type}) on port {port}")

    while keep_advertising_mdns:
        time.sleep(1)
    zc.unregister_service(info)
    print("Service unregistered")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTTP test server for streaming images and acceleration data.")
    parser.add_argument("-p", "--port", type=int, default=8000, help="Port number to listen on.")
    parser.add_argument("-a", "--accel", type=bool, default=False, help="Include a chunk of acceleration data for every image frame")
    parser.add_argument("-m", "--mdns", type=bool, default=True, help="Advertise the service with MDNS (Bonjour)")
    args = parser.parse_args()

    PORT = args.port
    include_accel_data = args.accel

    if args.mdns:
        # Start mdns advertisement in a separate thread
        mdns_thread = threading.Thread(target=register_mdns_service, args=("123.cranebot-service", "_http._tcp.local.", PORT), daemon=True)
        mdns_thread.start()

    with socketserver.TCPServer(("", PORT), StreamingHandler) as httpd:
        print(f"Serving at port {PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("stopping HTTP.")
            httpd.server_close()
            if args.mdns:
                keep_advertising_mdns = False
                mdns_thread.join()