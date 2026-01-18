import asyncio
import subprocess
import re
import numpy as np

# import for Picamera2
# sudo apt install python3-picamera2
from picamera2 import Picamera2

import zxingcpp

async def ensure_connection():
    """
    Checks for wifi connection.
    Returns True once a wifi connection is confirmed.
    As long as there is no connection, looks for connection data in QR codes
    Choose "share connection" from your phones wifi settings and hold it in view of the camera.
    """
    connected_event = asyncio.Event()

    # Task 1: Monitor for connection
    monitor_task = asyncio.create_task(monitor_wifi_status(connected_event))
    print("Checking for existing wifi connection...")

    try:
        # Wait up to 10 seconds for the monitor task to find a connection
        await asyncio.wait_for(connected_event.wait(), timeout=10.0)
        print("Existing connection confirmed.")
        return True

    except asyncio.TimeoutError:
        print("No connection found after 10 seconds. Starting QR Scanner...")
    
    # Task 2: Scan for QR codes
    scanner_task = asyncio.create_task(scan_and_configure_wifi(connected_event))

    print("Starting WiFi connection tasks...")

    # Wait until the monitor task signals that we are connected
    await connected_event.wait()
    
    # Cancel the scanner immediately
    scanner_task.cancel()
    
    try:
        await scanner_task
    except asyncio.CancelledError:
        print("Scanner task stopped.")

    print("WiFi connection established.")
    return True

async def monitor_wifi_status(connected_event):
    """
    Task 1: Checks for wifi connection every 5 seconds using nmcli.
    """
    while not connected_event.is_set():
        # Check connection status using nmcli
        result = subprocess.run(
            ['nmcli', '-f', 'GENERAL', 'device', 'show', 'wlan0'],
            capture_output=True,
            text=True
        )
        testmode=False
        
        # Regex to match 'GENERAL.STATE: 100 (connected)'
        if not testmode and re.search(r'GENERAL\.STATE:\s*100 \(connected\)', result.stdout):
            conn_match = re.search(r'GENERAL\.CONNECTION:\s*(.+)', result.stdout)
            conn_name = conn_match.group(1) if conn_match else "Unknown"
            
            print(f"Monitor detected active connection: {conn_name}")
            connected_event.set()
            return
        
        await asyncio.sleep(5)

async def scan_and_configure_wifi(connected_event):
    """
    Task 2: Continuously takes raw video frames to look for a WiFi QR code.
    """
    # Initialize Picamera2
    picam2 = Picamera2()
    
    # Configure camera for a moderate resolution RGB output
    config = picam2.create_preview_configuration(main={"size": (1920, 1080), "format": "RGB888"})
    picam2.configure(config)
    picam2.start(controls={"AfMode": 2}) # enable continuous autofocus

    print("Camera scanning for QR codes (Picamera2 / Direct Numpy)...")

    try:
        while not connected_event.is_set():
            # Capture raw RGB array. Shape is (480, 640, 3)
            frame = picam2.capture_array()
            
            try:
                # read_barcodes returns a list of results
                results = zxingcpp.read_barcodes(frame)
                
                for result in results:
                    qr_data = result.text
                    if qr_data.startswith("WIFI:"):
                        print(f"WiFi QR code detected: {qr_data[:20]}...")
                        
                        # Attempt to connect
                        success = await connect_via_nmcli(qr_data)
                        
                        if success:
                            print("Network configuration applied. Waiting for connection...")
                            # Pause here to let the monitor task verify the connection
                            await asyncio.sleep(10)

            except Exception:
                # Print full traceback for debugging, do not swallow the error
                traceback.print_exc()

            # Yield to event loop
            await asyncio.sleep(1)
            
    finally:
        picam2.stop()
        picam2.close()

async def connect_via_nmcli(qr_string):
    """
    Parses the QR string and invokes nmcli to connect.
    """
    # Regex to extract SSID and Password
    # Format: WIFI:S:MySSID;T:WPA;P:MyPassword;;
    ssid_match = re.search(r'S:([^;]+)', qr_string)
    pass_match = re.search(r'P:([^;]+)', qr_string)
    
    if not ssid_match or not pass_match:
        print("Invalid WiFi QR format. Missing SSID or Password.")
        return False

    ssid = ssid_match.group(1)
    password = pass_match.group(1)

    print(f"Attempting connection to '{ssid}'...")
    
    # nmcli connects and saves the profile for future reboots automatically
    cmd = ['nmcli', 'device', 'wifi', 'connect', ssid, 'password', password]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        return True
    else:
        print(f"Failed to connect: {result.stderr}")
        return False

if __name__ == "__main__":
    try:
        asyncio.run(ensure_connection())
    except KeyboardInterrupt:
        print("Interrupted.")