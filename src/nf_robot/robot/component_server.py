import sys
import logging
import time
import argparse

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

async def main():
    parser = argparse.ArgumentParser(description="Stringman Component")
    parser.add_argument(
            'component_type',
            type=str,
            choices=['anchor', 'power_anchor', 'gripper', 'arpeggio_gripper'],
            help="The type of component server to run (choices: anchor, power_anchor, gripper, arpeggio_gripper)"
        )
    args = parser.parse_args()

    connected = await ensure_connection()
    if not connected:
        logging.error('Wifi connection script failed to find a network')
        quit()

    if args.component_type == 'anchor':
        from anchor_server import RaspiAnchorServer
        ras = RaspiAnchorServer(False)
        r = await ras.main()

    elif args.component_type == 'power_anchor':
        from anchor_server import RaspiAnchorServer
        ras = RaspiAnchorServer(True)
        r = await ras.main()

    elif args.component_type == 'gripper':
        from gripper_server import RaspiGripperServer
        gs = RaspiGripperServer()
        r = await gs.main()

    elif args.component_type == 'arpeggio_gripper':
        from gripper_arp_server import GripperArpServer
        gs = GripperArpServer()
        r = await gs.main()

asyncio.run(main())