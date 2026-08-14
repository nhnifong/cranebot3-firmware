import asyncio
from getmac import get_mac_address
import json
import threading
import time
import logging
import argparse

from damiao_motor import DaMiaoController

import nf_robot.common.definitions as model_constants
from nf_robot.robot.component_server import RobotComponentServer
from nf_robot.robot.spool_dm import DamiaoSpoolController
from nf_robot.robot.server_conf import read_winding

"""Server for an Arpeggio anchor: two damiao hub motors and a custom CAN bus hat."""

default_anchor_conf = {
    # meters of line per second to reel in on a 'tighten' command
    'TIGHTENING_SPEED': -0.12,

    # Which component_server.stream_modes entry the camera runs. Changing resolution or
    # framerate means adding a named, measured mode to that table, not sending a number.
    'STREAM_MODE': 'anchor_control',
}

# Modes an anchor accepts, its normal one first. A pi 3a+ has the cpu and thermal headroom
# to hold 1080p30, which the zero 2w does not, so it starts in the faster mode.
anchor_stream_modes = ('anchor_control',)
anchor_stream_modes_3a_plus = ('anchor_fast', 'anchor_control')


def board_revision_code():
    """This Raspberry Pi's board revision code, or None if it can't be read."""
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('Revision'):
                    return int(line.split(':')[1].strip(), 16)
    except (OSError, ValueError):
        pass
    return None


def is_pi_3a_plus():
    """Whether this is running on a Raspberry Pi 3 Model A+.

    From the revision code, not the model string: the code is a fixed numeric field, while
    the strings are prose whose spelling varies across boards ('Model A Plus' here, 'Zero
    2 W' elsewhere). The kernel fills both in from the board itself, so both are
    trustworthy even on our image, whose boot partition carries no 3a+ dtb.
    """
    rev = board_revision_code()
    if rev is None:
        return False
    # new-style codes (bit 23) put the board type in bits 4-11, 0x0e being the 3a+.
    # old-style codes predate the pi 3, so they are never one.
    return bool(rev & (1 << 23)) and (rev >> 4) & 0xff == 0x0e


class AnchorArpServer(RobotComponentServer):
    def __init__(self, power, winding=None):
        super().__init__()
        self.conf.update(default_anchor_conf)
        if is_pi_3a_plus():
            logging.info('running on a pi 3a+; streaming in anchor_fast')
            self.stream_modes = anchor_stream_modes_3a_plus
        else:
            self.stream_modes = anchor_stream_modes
        self.conf['STREAM_MODE'] = self.stream_modes[0]

        self.has_power_line = power

        # How much line this anchor was wound with, from server.conf. Read here rather than
        # sent by the host so lengths come out right without anything upstream knowing how
        # this anchor was assembled.
        self.winding = winding if winding is not None else read_winding()

        unique = ''.join(get_mac_address().split(':'))
        self.service_name = 'cranebot-anchor-arpeggio-service.' + unique

        # https://jia-xie.github.io/python-damiao-driver/dev/package-usage/python-api/
        self.controller = DaMiaoController(channel="can0", bustype="socketcan")
        self.motor1 = self.controller.add_motor(motor_id=0x02, feedback_id=0x02, motor_type="G6215") # high motor
        self.motor2 = self.controller.add_motor(motor_id=0x01, feedback_id=0x01, motor_type="G6215") # lower motor
        self.motors = [self.motor1, self.motor2]

        # spool 0 is the direct line (high), spool 1 the indirect (low). A power line, if
        # this anchor has one, is always on the high spool.
        high_length, high_diameter = model_constants.damiao_spool_geometry[
            (self.winding, 'high', 'power' if self.has_power_line else 'fishing')]
        low_length, low_diameter = model_constants.damiao_spool_geometry[
            (self.winding, 'low', 'fishing')]
        logging.info(f'{self.winding} winding: high spool {high_length} m to {high_diameter} mm, '
                     f'low spool {low_length} m to {low_diameter} mm')

        spooler1 = DamiaoSpoolController(
            self.motor1,
            empty_diameter=model_constants.damiao_empty_spool_diameter,
            full_diameter=high_diameter,
            full_length=high_length,
            config=self.conf, direction=-1,
            # the stiffer powerline needs extra tension to stay taut
            extra_tension_n=0.4 if self.has_power_line else 0.0)

        spooler2 = DamiaoSpoolController(
            self.motor2,
            empty_diameter=model_constants.damiao_empty_spool_diameter,
            full_diameter=low_diameter,
            full_length=low_length,
            config=self.conf, direction=1)

        # None suppresses the parent's line updates; readOtherSensors sends both spools
        self.spooler = None
        self.spools = [spooler1, spooler2]

    async def processOtherUpdates(self, updates, tg):
        if 'tighten' in updates:
            spool_no = updates['tighten']
            tg.create_task(self.tighten(spool_no))
        if 'stow' in updates:
            spool_no = updates['stow']
            tg.create_task(self.stow(spool_no))
        if 'relax' in updates:
            spool_no = updates['relax']
            tg.create_task(self.relax(spool_no))
        if 'identify' in updates:
            self.identify()
        if 'two_reference_lengths' in updates:
            ref0, ref1 = updates['two_reference_lengths']
            self.spools[0].setReferenceLength(float(ref0))
            self.spools[1].setReferenceLength(float(ref1))
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
        if 'jog' in updates:
            try:
                delta, spool_no = updates['jog']
                self.spools[int(spool_no)].jog(float(delta))
            except (TypeError, ValueError, IndexError):
                logging.warning(f'invalid jog command: {updates["jog"]}')
        if 'disable_torque' in updates:
            for spool in self.spools:
                spool.pauseTrackingLoop(disable_torque=True)
                self.update['torque'] = False
        if 'enable_torque' in updates:
            for spool in self.spools:
                spool.resumeTrackingLoop()
                self.update['torque'] = True
        if 'set_tension_reg' in updates:
            val, spool_no = updates['set_tension_reg']
            spool_no = int(spool_no)
            assert spool_no in [0,1]
            self.spools[spool_no].setTensionRegEnabled(bool(val))
        if 'set_tension_target' in updates:
            val, spool_no = updates['set_tension_target']
            spool_no = int(spool_no)
            assert spool_no in [0,1]
            self.spools[spool_no].setTensionTarget(None if val is None else float(val))

    def readOtherSensors(self):
        """Queue both spools' records: {'spool0': [(time, length, speed, torque), ...],
        'spool1': [...]}"""
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
        """Reel in until the line holds tension for 3 seconds.

        A line that goes slack again was pulled in too fast to seat, so each retry backs
        the speed off 30%, up to 5 attempts.
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
            while slack():
                self.spools[spool_no].setAimSpeed(current_speed)
                await asyncio.sleep(check_interval_s)
            self.spools[spool_no].setAimSpeed(0)

            loosened = False
            end_time = time.monotonic() + monitoring_duration_s
            while time.monotonic() < end_time:
                if slack():
                    loosened = True
                    break
                await asyncio.sleep(check_interval_s)

            if not loosened:
                return
            current_speed *= 0.7

        self.spools[spool_no].setAimSpeed(0)
        logging.error(f"Failed to tighten line after {max_retries} attempts.")

    async def stow(self, spool_no):
        """Pull the line tight, then disable the motor for storage."""
        if spool_no not in (0, 1):
            return
        check_interval_s = 0.05
        desired_tension = 1.38 # Newtons
        current_speed = self.conf['TIGHTENING_SPEED']
        def slack():
            return self.spools[spool_no].last_tension < desired_tension
        while slack():
            self.spools[spool_no].setAimSpeed(current_speed)
            await asyncio.sleep(check_interval_s)
        self.spools[spool_no].setAimSpeed(0)
        self.spools[spool_no].pauseTrackingLoop()
        self.motors[spool_no].disable()

    async def relax(self, spool_no):
        """Let out line until it is no longer tight."""
        if spool_no not in (0, 1):
            return
        pass

    def identify(self, spool_no=1):
        """Buzz one motor, so an operator can tell which anchor this is."""
        self.spools[spool_no].pauseTrackingLoop()
        m = self.motors[spool_no]

        m.send_cmd_vel(target_velocity=0.0)
        for i in range(20):
            time.sleep(0.005)
            m.send_cmd_vel(target_velocity=0.2 * (i%2-0.5))
        m.send_cmd_vel(target_velocity=0.0)
        
        self.spools[spool_no].resumeTrackingLoop()

    async def process_imu(self, ws):
        """Enable the motors when a client connects.
        TODO don't just piggyback off this, organize it"""
        for m in self.motors:
            m.enable()
        for s in self.spools:
            s.resumeTrackingLoop()

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