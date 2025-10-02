"""
Python script for a Raspberry Pi to connect to the ESP32-S3 BLE Game Controller.

This script uses the 'bleak' library to scan for the controller, connect to it,
and receive notifications containing the controller's state.
"""
import asyncio
from bleak import BleakScanner, BleakClient

# The name of the BLE device we want to connect to
TARGET_DEVICE_NAME = "Stringman Training Controller"

# target device address
DEVICE_ADDRESS = "34:85:18:92:1D:05"

# The UUIDs for the UART service and its transmit characteristic
# These must match the UUIDs on the ESP32-S3 firmware
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

def notification_handler(sender, data):
    """
    This function is called whenever a notification is received from the controller.
    It decodes, parses, and prints the controller's state.
    """
    # Decode the byte array into a string
    message = data.decode('utf-8')
    
    # The data is expected to be a comma-separated string: "btn1,btn2,btn3,analog"
    parts = message.split(',')
    if len(parts) == 4:
        button1 = "Pressed" if parts[0] == '1' else "Released"
        button2 = "Pressed" if parts[1] == '1' else "Released"
        button3 = "Pressed" if parts[2] == '1' else "Released"
        analog_value = float(parts[3])
        
        # Print the formatted controller state
        print(f"B1: {button1}, B2: {button2}, B3: {button3}, Trigger: {analog_value}")
    else:
        print(f"Received unexpected data format: {message}")

async def main():
    """
    The main asynchronous function that handles device connection.
    """
    print(f"Attempting to connect to {DEVICE_ADDRESS}...")

    # This context manager ensures the client is properly disconnected
    async with BleakClient(DEVICE_ADDRESS) as client:
        if client.is_connected:
            print(f"Connected to controller!")
            
            # Start listening for notifications on the TX characteristic
            await client.start_notify(UART_TX_CHAR_UUID, notification_handler)
            
            print("Receiving controller data... Press Ctrl+C to stop.")
            
            # Keep the script running to receive notifications
            while client.is_connected:
                await asyncio.sleep(1)
        else:
            print(f"Failed to connect to {DEVICE_ADDRESS}")

if __name__ == "__main__":
    try:
        # Run the main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript stopped by user.")
