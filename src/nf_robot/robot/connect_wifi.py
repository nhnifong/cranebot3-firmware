import asyncio
import subprocess
import re
import numpy as np
import logging

# import for Picamera2
# sudo apt install python3-picamera2
from picamera2 import Picamera2
from libcamera import controls

# Pyzbar for decoding
# sudo apt-get install libzbar0
from pyzbar.pyzbar import decode

# seconds between attempts to reopen a QR scanner camera that failed
CAMERA_RETRY_S = 10

async def ensure_connection(feedback=None):
    """
    Checks for wifi connection.
    Returns once a wifi connection is confirmed.
    As long as there is no connection, looks for connection data in QR codes
    Choose "share connection" from your phones wifi settings and hold it in view of the camera.

    Returns True only if the connection was made by decoding a QR code.
    Returns False if an existing wifi connnection was used, even if it took longer than
    the initial check to come up and the scanner was started in the meantime.
    if no connection is achieved, doesn't return.

    feedback, if given, is the component server, whose play_feedback() signals when a
    code is read and when joining its network fails.
    """
    connected_event = asyncio.Event()
    # set by the scanner only when nmcli successfully connects using a decoded QR code
    qr_connected_event = asyncio.Event()

    # Task 1: Monitor for connection
    monitor_task = asyncio.create_task(monitor_wifi_status(connected_event))
    logging.info("Checking for existing wifi connection...")

    try:
        # Wait up to 10 seconds for the monitor task to find a connection
        await asyncio.wait_for(connected_event.wait(), timeout=10.0)
        logging.info("Existing connection confirmed.")
        return False

    except asyncio.TimeoutError:
        logging.info("No connection found after 10 seconds. Starting QR Scanner...")
    
    # Task 2: Scan for QR codes
    scanner_task = asyncio.create_task(scan_and_configure_wifi(connected_event, qr_connected_event, feedback))

    logging.info("Starting WiFi connection tasks...")

    # Wait until the monitor task signals that we are connected
    await connected_event.wait()
    
    # Cancel the scanner immediately
    scanner_task.cancel()
    
    try:
        await scanner_task
    except asyncio.CancelledError:
        logging.info("Scanner task stopped.")

    if qr_connected_event.is_set():
        logging.info("WiFi connection established from QR code.")
        return True
    logging.info("WiFi connection established using a saved network.")
    return False

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
        testmode = False
        # Regex to match 'GENERAL.STATE: 100 (connected)'
        if not testmode and re.search(r'GENERAL\.STATE:\s*100 \(connected\)', result.stdout):
            conn_match = re.search(r'GENERAL\.CONNECTION:\s*(.+)', result.stdout)
            conn_name = conn_match.group(1) if conn_match else "Unknown"
            
            logging.info(f"Monitor detected active connection: {conn_name}")
            connected_event.set()
            return
        
        await asyncio.sleep(5)

async def scan_and_configure_wifi(connected_event, qr_connected_event, feedback=None):
    """
    Task 2: Continuously takes raw video frames to look for a WiFi QR code.

    The scanner is the only way to onboard a unit, and nothing awaits this task until
    a connection is made, so a camera failure is logged here and the camera reopened
    rather than letting the task die unseen.
    """
    failures = 0
    while not connected_event.is_set():
        try:
            await scan_with_camera(connected_event, qr_connected_event, feedback)
        except Exception as e:
            # Full traceback once; repeats every CAMERA_RETRY_S would flood cranebot.log.
            if failures == 0:
                logging.exception(f"QR scanner camera failed. Retrying every {CAMERA_RETRY_S}s.")
            else:
                logging.error(f"QR scanner camera failed again ({failures + 1} times): {e!r}")
            failures += 1
            await asyncio.sleep(CAMERA_RETRY_S)

async def scan_with_camera(connected_event, qr_connected_event, feedback):
    picam2 = Picamera2()
    try:
        # Configure camera for a moderate resolution RGB output
        config = picam2.create_preview_configuration(main={"size": (1920, 1080), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        # enable continuous autofocus
        picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

        logging.info("Camera scanning for QR codes (Picamera2 / Direct Numpy)...")

        while not connected_event.is_set():
            # Capture raw RGB array
            # shape is (480, 640, 3)
            frame = picam2.capture_array()
            
            # Extract the Green channel only.
            # frame[:, :, 1] takes the 2nd channel (Green) from every pixel.
            # This creates a 2D array (480, 640) that acts as 'Grayscale'.
            # pyzbar handles 2D numpy arrays as strictly grayscale (Y800).
            gray_view = frame[:, :, 1]

            try:
                # Pass the numpy slice directly to pyzbar
                decoded_objects = decode(gray_view)
                
                for obj in decoded_objects:
                    qr_data = obj.data.decode('utf-8')
                    if qr_data.startswith("WIFI:"):
                        logging.info("WiFi QR code detected.")
                        # Played to the end before connecting: nmcli blocks the event loop,
                        # which would stall a signal running alongside it.
                        if feedback is not None:
                            await feedback.play_feedback('see_code')

                        # Attempt to connect
                        success = await connect_via_nmcli(qr_data)
                        
                        if success:
                            qr_connected_event.set()
                            logging.info("Network configuration applied. Waiting for connection...")
                            # Pause to allow the monitor task to verify the connection
                            await asyncio.sleep(10)
                        else:
                            logging.info("Could not join the network from the QR code.")
                            if feedback is not None:
                                await feedback.play_feedback('failed')

            except Exception as e:
                logging.info(f"Error during scan loop: {e}")

            # Yield to event loop
            await asyncio.sleep(1)
            
    finally:
        picam2.stop()
        picam2.close()

async def connect_via_nmcli(qr_string):
    """
    Parses the QR string and invokes nmcli to connect.
    Forces a rescan before connecting to ensure Hotspots are visible.
    """
    ssid_match = re.search(r'S:([^;]+)', qr_string)
    pass_match = re.search(r'P:([^;]+)', qr_string)
    
    if not ssid_match or not pass_match:
        logging.info("Invalid WiFi QR format. Missing SSID or Password.")
        return False

    ssid = ssid_match.group(1)
    password = pass_match.group(1)

    # Force a Rescan
    logging.info("Forcing Wi-Fi rescan to find new networks...")
    subprocess.run(['nmcli', 'device', 'wifi', 'rescan'], capture_output=True)
    # Give the wifi card a moment to populate the list
    await asyncio.sleep(3)

    # Retry Loop
    for attempt in range(1, 4):
        logging.info(f"Attempting connection to '{ssid}' (Try {attempt}/3)...")
        
        cmd = ['nmcli', 'device', 'wifi', 'connect', ssid, 'password', password]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info("Successfully associated!")
            return True
        else:
            logging.info(f"Connection failed: {result.stderr.strip()}")
            # If the error is specific to "No network found", wait a bit longer and retry
            if "No network" in result.stderr:
                await asyncio.sleep(3)
            else:
                await asyncio.sleep(1)
                
    return False

if __name__ == "__main__":
    try:
        asyncio.run(ensure_connection())
    except KeyboardInterrupt:
        logging.info("Interrupted.")