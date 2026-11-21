import sys
import logging
import time
import board
import busio

from nf_robot.robot.connect_wifi import ensure_connection
from nf_robot.robot.gripper_i2c import ADS1015_ADDR, VL53L1X_ADDR, IMU_ADDRS

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

def make_server():
    """Build the server for whichever component this is, without starting it."""
    # determine component type
    # TODO: first boot of a new image has been observed to not have a valid I2C bus. if the image cannot be
    # fixed, then detect that problem here and reboot, because it seems to work the second time.
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        addrs = set(i2c.scan())
    except ValueError:
        addrs = set([])

    # arpeggio_gripper: pressure ADC, rangefinder and an IMU (MPU6050, or LSM6DS3TR-C on newer
    # hats). Two of the three is enough to know this is a gripper; the server reports
    # whichever one is missing as its error state.
    gripper_parts = [ADS1015_ADDR in addrs, VL53L1X_ADDR in addrs, bool(IMU_ADDRS & addrs)]
    if sum(gripper_parts) >= 2:
        from nf_robot.robot.gripper_arp_server import GripperArpServer
        return GripperArpServer(addrs)

    if len(addrs) == 0:
        # to differentiate power anchor, look for file written by anchor_arp_eval. The winding
        # field in the same file is read by AnchorArpServer itself.
        from nf_robot.robot.server_conf import read_server_conf
        component_type, _ = read_server_conf()

        if component_type == 'arpeggio anchor':
            from nf_robot.robot.anchor_arp_server import AnchorArpServer
            return AnchorArpServer(False)

        if component_type == 'arpeggio power anchor':
            from nf_robot.robot.anchor_arp_server import AnchorArpServer
            return AnchorArpServer(True)

        raise ValueError(f'Invalid type in server.conf "{component_type}"')

    raise ValueError(f'Unrecognized set of I2C devices: {sorted(hex(a) for a in addrs)}')

async def asyncmain():
    # Built before WiFi so its motors can signal how QR code onboarding is going. A unit
    # that cannot build one still joins WiFi, so it can be reached to find out why.
    try:
        server = make_server()
    except Exception:
        logger.exception('could not build the component server; joining WiFi without feedback')
        server = None

    new_connection_configured = await ensure_connection(feedback=server)

    if server is None:
        server = make_server()
    if new_connection_configured:
        await server.play_feedback('success')
    await server.main()

def main():
    asyncio.run(asyncmain())

if __name__ == "__main__":
    main()