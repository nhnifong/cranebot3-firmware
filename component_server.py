from anchor_server import RaspiAnchorServer
from gripper_server import RaspiGripperServer
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='cranebot.log'
)

with open('server.conf', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if not line.startswith('#') and line:  # Check if line is not a comment and is not empty
            component_type = line
            logging.info(f'Starting cranebot server of type {component_type}')
            break

if component_type == 'anchor':
    ras = RaspiAnchorServer(False)
    asyncio.run(ras.main())
elif component_type == 'power anchor':
    ras = RaspiAnchorServer(True)
    asyncio.run(ras.main())
elif component_type == 'gripper':
    gs = RaspiGripperServer()
    asyncio.run(gs.main())
