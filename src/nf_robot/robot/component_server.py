import sys
import logging
import time
import board
import busio

from nf_robot.robot.connect_wifi import ensure_connection

# todo maybe there is a better solution to this but systemctl starts us too early and some zeroconf things dont work
time.sleep(3)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='cranebot.log'
)

handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)

def handle_exception(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

import asyncio

async def asyncmain():
    connected = await ensure_connection()
    if not connected:
        logging.error('Wifi connection script failed to find a network')
        quit()

    # determine component type
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        addrs = set(i2c.scan())
    except ValueError:
        addrs = set([])

    if addrs == set([0x29, 0x4b, 0x17]): # gripper
        from nf_robot.robot.gripper_server import RaspiGripperServer
        gs = RaspiGripperServer()
        r = await gs.main()

    elif set([0x48, 0x29]).issubset(addrs): # arpeggio_gripper with or without IMU
        from nf_robot.robot.gripper_arp_server import GripperArpServer
        gs = GripperArpServer()
        r = await gs.main()

    elif len(addrs) == 0:
        from nf_robot.robot.anchor_server import RaspiAnchorServer

        # to differentiate power anchor, look for file written by anchor_eval.py
        component_type = 'anchor'
        try:
            with open('server.conf', 'r') as file:
                for line in file:
                    line = line.strip()  # Remove leading/trailing whitespace
                    if not line.startswith('#') and line:  # Check if line is not a comment and is not empty
                        component_type = line
        except FileNotFoundError:
            component_type = 'anchor'

        powerline = component_type == 'power anchor' # TODO differentiate in some automatic way
        ras = RaspiAnchorServer(powerline)
        r = await ras.main()

def main():
    asyncio.run(asyncmain())

if __name__ == "__main__":
    main()