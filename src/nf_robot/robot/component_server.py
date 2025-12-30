import sys
import logging
import time

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

with open('server.conf', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if not line.startswith('#') and line:  # Check if line is not a comment and is not empty
            component_type = line
            logging.info(f'Starting cranebot server of type {component_type}')
            break

if component_type == 'anchor':
    from anchor_server import RaspiAnchorServer
    ras = RaspiAnchorServer(False)
    asyncio.run(ras.main())

elif component_type == 'power anchor':
    from anchor_server import RaspiAnchorServer
    ras = RaspiAnchorServer(True)
    asyncio.run(ras.main())

elif component_type == 'gripper':
    from gripper_server import RaspiGripperServer
    gs = RaspiGripperServer()
    asyncio.run(gs.main())