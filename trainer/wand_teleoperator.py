"""
A Teleoperator based on the stringman training wand.
The wand is a lot like a game controller, in the sense that it's just a trigger and some buttons that can be connected to
over bluetooth, but it's position is measured by the robot's anchor cameras seeing april tags on the wand, in the same
way the robot locates the gantry.
So for full operation this teleoperator needs to connect to both the AsynObserver, and the wand.
"""

from functools import cached_property
from typing import Any

import numpy as np
from lerobot.robots import Robot
from lerobot.teleoperators.teleoperator import Teleoperator
import asyncio
from bleak import BleakScanner, BleakClient
import time
from .wand_teleoperator_config import WandConfig

# The UUIDs for the UART service and its transmit characteristic
# These must match the UUIDs on the ESP32-S3 firmware
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
WAND_DEVICE_NAME = 'Stringman Training Controller'

class StringmanTrainingWand(Teleoperator):

	# TODO change config class
    config_class = WandConfig
    name = "stringman_training_wand"

    def __init__(self, config: WandConfig):
        super().__init__(config)
        self.config = config
        self.channel_address = 'localhost:50051'

        # state variables for Bluetooth
        self._bt_thread: threading.Thread | None = None
        self._bt_connected: bool = False
        self._stop_event = threading.Event()
        self.last_wand_state: dict[str, Any] = {"trigger": 0.0, "buttons": [False] * 3}


    @property
    def action_features(self) -> dict[str, type]:
        return { 
            "gantry_pos_x": float,
            "gantry_pos_y": float,
            "gantry_pos_z": float,
            "winch_length": float,
            "finger_angle": float,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        # TODO: Add check for AsyncObserver GRPC connection as well
        return self._bt_connected

    def _run_in_thread(self, coro):
        """Helper to run an async coroutine in a new thread with its own event loop."""
        def run_loop():
            try:
                # Set a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                # Run the coroutine until it completes (i.e., until _bt_connect_loop finishes)
                loop.run_until_complete(coro)
            finally:
                self._bt_connected = False
                # Clean up the loop
                asyncio.get_event_loop().close()


        self._bt_thread = threading.Thread(target=run_loop, daemon=True)
        self._bt_thread.start()

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            # DeviceAlreadyConnectedError needs to be imported/defined
            raise DeviceAlreadyConnectedError(f"{self} already connected") 

        logging.info("Starting Bluetooth connection thread...")
        self._run_in_thread(self._bt_connect_loop())
        
        print(f"Establishing gRPC connection to {self.channel_address}...")
        self.channel = grpc.insecure_channel(self.channel_address)
        self.stub = RobotControlServiceStub(self.channel)
        print("gRPC channel established and stub created.")

        # wait up to 10 seconds to confirm a connection to blutooth has been made
        start = time.time()
       	while not self._bt_connected and time.time() - start < 10:
       		time.sleep(0.1)

    def disconnect(self) -> None:
        if not self.is_connected:
            # DeviceNotConnectedError needs to be imported/defined
            raise ConnectionError(f"{self} is not connected.")

        logging.info("Stopping bluetooth wand connection...")
        self._stop_event.set() 

        if self.channel:
            print("Closing gRPC channel...")
            self.channel.close()
            self.channel = None
            self.stub = None
            print("gRPC channel closed.")

        # Wait for the thread to finish
        if self._bt_thread and self._bt_thread.is_alive():
            self._bt_thread.join(timeout=5) # Wait up to 5 seconds for cleanup
            if self._bt_thread.is_alive():
                logging.warning("Bluetooth thread did not terminate cleanly.")

        self._bt_thread = None
        self._bt_connected = False
        self._stop_event.clear()

    async def _bt_connect_loop(self):
        device = await BleakScanner.find_device_by_name(WAND_DEVICE_NAME)
        if device is None:
            logging.error(f"Could not find device: {WAND_DEVICE_NAME}")
            return

        try:
            client = BleakClient(device)
            await client.connect()
            
            if self.client.is_connected:
                logging.info(f"Connected to wand: {device.address}")
                self._bt_connected = True
                # Start listening for notifications
                await self.client.start_notify(UART_TX_CHAR_UUID, self._bt_notification_handler)
                # Keep the connection alive and listen for a stop signal
                while self._client.is_connected and not self._stop_event.is_set():
                    await asyncio.sleep(0.1) # Sleep briefly to yield to the event loop
                await self.client.stop_notify(UART_TX_CHAR_UUID)
                await self.client.disconnect()
            else:
                logging.error(f"Failed to connect to {repr(device)}")
        finally:
            self._bt_connected = False
            self._client = None
            logging.info("Bluetooth loop finished.")

    def _bt_notification_handler(self, sender, data):
        """
        This function is called whenever a notification is received from the training wand.
        """
        # The data is expected to be a comma-separated string: "btn1,btn2,btn3,analog,raw"
        message = data.decode('utf-8')
        parts = message.split(',')
        if len(parts) != 5:
            logging.error(f"Received unexpected data format: {message}")
            return

        buttons = [False, False, False]
        for i in range(3):
            buttons[i] = parts[i] == '1'
        analog_value = float(parts[3])
        logging.debug(f"B1: {buttons[0]}, B2: {buttons[0]}, B3: {buttons[2]}, Trigger: {analog_value}, Raw: {parts[4]}")

        # Store last trigger and button state so it can be used in get_action
        self.last_wand_state["buttons"] = buttons
        self.last_wand_state["trigger"] = analog_value

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
    	# the position the real gantry should move to is a certain transformation of the observed position of the trainer wand.
    	# the exact offset is something the operator should be able to reset as needed.

    	# the exact winch length the teleoperator is commanding is always whatever you currently are at plus an offset.
    	# the offset is 0 unles the up or down button is pressed, in which case the offset is plus or minus some constant.

    	# the finger angle the teleoperator is commanding is a pure function of the trigger value.
        return {
            "gantry_pos_x": gant_pos[0],
            "gantry_pos_y": gant_pos[1],
            "gantry_pos_z": gant_pos[2],
            "winch_length": float,
            "finger_angle": float,
        }

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

