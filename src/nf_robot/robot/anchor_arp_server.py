import asyncio
from getmac import get_mac_address
import json
import threading
import time
import logging
import argparse

from damiao_motor import DaMiaoController

import nf_robot.common.definitions as model_constants
from nf_robot.robot.spools import SpoolController

""" Server for Arpeggio Anchor

A double anchor containing two damiao hub motors and a custom hat that provides a CAN bus interface.

"""

default_anchor_conf = {
}

class AnchorArpServer(RobotComponentServer):
    def __init__(self):
        super().__init__()
        self.conf.update(default_anchor_conf)

        unique = ''.join(get_mac_address().split(':'))
        self.service_name = 'cranebot-anchor-arpeggio-service.' + unique

        # TODO during qa and winding script, the ids of the motors need to be set, and recorded in a file
        # a flag needs to indicate whether a motor has power line or fishing line wound on it

        # https://jia-xie.github.io/python-damiao-driver/dev/package-usage/python-api/
        self.controller = DaMiaoController(channel="can0", bustype="socketcan")
        # h6220 is probaly the closest to DM-H6215 but they all seem the same to me.
        self.motor1 = controller.add_motor(motor_id=0x01, feedback_id=0x01, motor_type="h6220")
        self.motor2 = controller.add_motor(motor_id=0x02, feedback_id=0x02, motor_type="h6220")

        self.motor1.enable()
        self.motor2.enable()

        self.motor1.ensure_control_mode("VEL")
        self.motor2.ensure_control_mode("VEL")

        # TODO create a spool controller for each spool
        # spoolcontroller needs to be able to make use of the extra torque data to keep the lines tight.
        self.spooler = SpoolController(
            motor,
            empty_diameter=model_constants.empty_spool_diameter,
            full_diameter=model_constants.full_spool_diameter_fishing_line,
            full_length=model_constants.assumed_full_line_length,
            conf=self.conf, gear_ratio=ratio, tight_check_fn=self.tight_check)

    async def processOtherUpdates(self, updates, tg):
        if 'tighten' in updates:
            tg.create_task(self.tighten())

    def readOtherSensors(self):
        m1 = motor1.get_states()
        m2 = motor2.get_states()

    def startOtherTasks(self):
        return []

    async def tighten(self):
        """Pulls in the line until tight."""
        pass

    async def shutdown(self):
        """TODO, nothing calls this because there is no system yet in place to cleanly shutdown component servers."""
        self.motor1.disable()
        self.motor2.disable()
        self.controller.shutdown()
        
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--power", action="store_true",
                        help="Configures this anchor as the one which has the power line")
    args = parser.parse_args()

    ras = AnchorArpServer(args.power)
    asyncio.run(ras.main())
