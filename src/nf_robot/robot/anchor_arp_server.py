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
    # speed to reel in when the 'tighten' command is received. Meters of line per second
    'TIGHTENING_SPEED': -0.12,
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
            spool_no = updates['tighten']
            tg.create_task(self.tighten(spool_no))
        if 'relax' in updates:
            spool_no = updates['relax']
            tg.create_task(self.relax(spool_no))
        if 'identify' in updates:
            self.identify()
        if 'two_reference_lengths' in updates:
            ref0, ref1 = updates['two_reference_lengths']
            self.spooler[0].setReferenceLength(float(ref0))
            self.spooler[1].setReferenceLength(float(ref1))
        if 'aim_speed' in updates:
            if updates['aim_speed'] == 0:
                self.spools[0].setAimSpeed(0)
                self.spools[1].setAimSpeed(0)
            else:
                try:
                    speed, spool_no = updates['aim_speed']
                    speed = float(speed)
                    spool_no = int(spool_no)
                    assert spool_no in [0,1]
                    self.spools[spool_no].setAimSpeed(speed)
                except (TypeError, ValueError, AssertionError):
                    logging.warning(f'invalid aim_speed command. expected (speed, spool_no). got {updates["aim_speed"]}')

    def readOtherSensors(self):
        """ Sends updates about both spools with the form
        {
            'spool0' : [
                (time, line_length, line_speed, torque),
                ...
            ],
            'spool1': [...]
        }
        """
        for i, spool in enumerate(self.spools):
            meas = spool.popMeasurements()
            if len(meas) > 0:
                meas = meas[:50]
            self.update[f'spool{i}'] = meas

    def startOtherTasks(self):
        return list([
            asyncio.create_task(asyncio.to_thread(spool.trackingLoop))
            for spool in self.spools
        ])

    async def tighten(self, spool_no):
        """
        Pulls in the line until tight. If the line slips within 3 seconds,
        it reduces the speed by 30% and retries, up to 5 times.
        """
        if spool_no not in (0, 1):
            return
        max_retries = 5
        monitoring_duration_s = 3
        check_interval_s = 0.05
        desired_tension = 1.38 # Newtons
        
        current_speed = self.conf['TIGHTENING_SPEED']

        def slack():
            return self.spools[spool_no].last_tension < desired_tension

        for attempt in range(1, max_retries + 1):
            # Pull in the line until target torque is reached
            while slack():
                self.spools[spool_no].setAimSpeed(current_speed)
                await asyncio.sleep(check_interval_s)
            self.spools[spool_no].setAimSpeed(0)

            # Monitor for re-loosening over the next 3 seconds
            loosened = False
            end_time = time.monotonic() + monitoring_duration_s
            while time.monotonic() < end_time:
                if slack():
                    loosened = True
                    break  # Exit monitoring loop immediately on slip
                await asyncio.sleep(check_interval_s)

            # Check the outcome
            if not loosened:
                return # Success!

            # If it slipped, reduce speed and the loop will try again
            current_speed *= 0.7

        # If the loop finishes, all retries have failed
        self.spools[spool_no].setAimSpeed(0)
        logging.error(f"Failed to tighten line after {max_retries} attempts.")

    async def relax(self, spool_no):
        """ Lets out line until not tight """
        if spool_no not in (0, 1):
            return
        pass

    def identify(self):
        """ make a noise """
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