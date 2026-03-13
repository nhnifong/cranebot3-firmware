import asyncio
from getmac import get_mac_address
import json
import threading
import time
import logging
import argparse

from damiao_motor import DaMiaoController

import nf_robot.common.definitions as model_constants
from nf_robot.robot.anchor_server import RobotComponentServer
from nf_robot.robot.spool_dm import DamiaoSpoolController

""" Server for Arpeggio Anchor

A double anchor containing two damiao hub motors and a custom hat that provides a CAN bus interface.

"""

default_anchor_conf = {
}

class AnchorArpServer(RobotComponentServer):
    def __init__(self, power):
        super().__init__()
        self.conf.update(default_anchor_conf)

        self.has_power_line = power

        unique = ''.join(get_mac_address().split(':'))
        self.service_name = 'cranebot-anchor-arpeggio-service.' + unique

        # https://jia-xie.github.io/python-damiao-driver/dev/package-usage/python-api/
        self.controller = DaMiaoController(channel="can0", bustype="socketcan")
        # h6220 is probaly the closest to DM-H6215 but they all seem the same to me.
        self.motor1 = self.controller.add_motor(motor_id=0x01, feedback_id=0x01, motor_type="H6220")
        self.motor2 = self.controller.add_motor(motor_id=0x02, feedback_id=0x02, motor_type="H6220")

        # Create a spool controller for each spool
        spooler1 = DamiaoSpoolController(
            self.motor1,
            empty_diameter=model_constants.damiao_empty_spool_diameter,
            full_diameter=model_constants.damiao_full_spool_diameter_fishing_line,
            full_length=model_constants.assumed_full_line_length,
            config=self.conf, direction=1)

        # the power line, if present is always on the second spool
        fulld = model_constants.damiao_full_spool_diameter_power_line if self.has_power_line else model_constants.damiao_full_spool_diameter_fishing_line
        spooler2 = DamiaoSpoolController(
            self.motor2,
            empty_diameter=model_constants.damiao_empty_spool_diameter,
            full_diameter=fulld,
            full_length=model_constants.assumed_full_line_length,
            config=self.conf, direction=-1)

        # parent class would use this to send line updates. setting it to None supresses that. we send our own.
        self.spooler = None
        self.spools = [spooler1, spooler2]

    async def processOtherUpdates(self, updates, tg):
        if 'tighten' in updates:
            tg.create_task(self.tighten())

    def readOtherSensors(self):
        """ Sends updates about both spools with the form
        {
            'spool1' : [
                (time, line_length, line_speed, torque),
                ...
            ],
            'spool2': [...]
        }
        """
        for i, spool in enumerate(self.spools):
            meas = spool.popMeasurements()
            if len(meas) > 0:
                meas = meas[:50]
            self.update[f'spool{i+2}'] = meas
        logging.info(self.update)

    def startOtherTasks(self):
        return list([
            asyncio.create_task(asyncio.to_thread(spool.trackingLoop))
            for spool in self.spools
        ])

    async def tighten(self):
        """Pulls in the line until tight."""
        pass

    def shutdown(self):
        """must be a synchronous call. triggered by signal handler"""
        super().shutdown()
        for spool in self.spools:
            spool.fastStop()
        time.sleep(0.1)
        self.controller.shutdown()
        
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--power", action="store_true",
                        help="Configures this anchor as the one which has the power line")
    args = parser.parse_args()

    ras = AnchorArpServer(args.power)
    asyncio.run(ras.main())