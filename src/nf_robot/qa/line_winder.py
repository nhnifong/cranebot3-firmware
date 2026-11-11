"""Wind a requested length of line onto the one DaMiao spool connected to this anchor hat.

For the dedicated winding machine (a Pi Zero 2 W with an anchor hat and a single spool motor).
Uses the same length-to-revolutions calculation and speed ramp as anchor_arp_eval.py, which is the
same for fishing line and power line. The wind stops early if the line tension goes over
TENSION_LIMIT_N or on ctrl-c, and in every case reports how much line actually went on, measured
from the motor's encoder.

    python -m nf_robot.qa.line_winder 15
    python -m nf_robot.qa.line_winder 7.5 --reverse
"""

import argparse
import subprocess
import time
from math import pi

from damiao_motor import DaMiaoController

from nf_robot.qa.anchor_arp_eval import (
    EMPTY_SPOOL_RADIUS_M, MOTOR_TYPE, POS_WRAP_RAD, WIND_RADIUS_M, line_revs, ramp_profile,
    scan_motors, set_motor_power)

TENSION_LIMIT_N = 18.0
MAX_REV_PER_S = 4.0  # same as anchor_arp_eval.py
ACCEL_REV_PER_S2 = 2.0
DT = 0.02
STALE_COMMANDS = 20  # commands in a row with no feedback before giving up
SETTLE_S = 0.5  # hold zero speed this long after stopping so the coast is counted


class WindStop(Exception):
    pass


class Winder:
    """Commands the spool and tracks its unwrapped position and line tension from feedback."""

    def __init__(self, motor, direction):
        self.motor = motor
        self.direction = direction
        self.raw = None
        self.pos = 0.0  # unwrapped rad since the first feedback
        self.last_frame = None
        self.stale = 0
        # the empty spool radius gives the most tension per N.m, so the limit trips early, not late
        self.torque_limit = TENSION_LIMIT_N * EMPTY_SPOOL_RADIUS_M

    def wound_m(self):
        """Line wound so far, by the same radius used to plan the wind."""
        return self.direction * self.pos * WIND_RADIUS_M

    def tick(self, vel_rad_s, check_tension=True):
        t0 = time.time()
        self.motor.send_cmd_vel(target_velocity=vel_rad_s)
        time.sleep(max(0.0, DT - (time.time() - t0)))
        # the driver replaces its state dict with every feedback frame it decodes
        frame = self.motor.state
        if not frame or frame is self.last_frame:
            self.stale += 1
            if self.stale >= STALE_COMMANDS:
                raise WindStop(f'no feedback from the motor for {self.stale} commands')
            return
        self.stale = 0
        self.last_frame = frame
        raw = frame.get('pos', 0.0)
        if self.raw is not None:
            d = raw - self.raw
            self.pos += d - POS_WRAP_RAD * round(d / POS_WRAP_RAD)
        self.raw = raw
        torq = frame.get('torq', 0.0)
        if check_tension and abs(torq) > self.torque_limit:
            raise WindStop(f'line tension about {abs(torq) / EMPTY_SPOOL_RADIUS_M:.1f} N '
                           f'(torque {torq:+.3f} N.m), over the {TENSION_LIMIT_N:.0f} N limit')

    def settle(self):
        """Command zero for SETTLE_S, still tracking position, so a coast after stopping is counted."""
        deadline = time.time() + SETTLE_S
        while time.time() < deadline:
            try:
                self.tick(0.0, check_tension=False)
            except WindStop:
                return


def find_spool_motor(controller):
    found = scan_motors(controller, motor_type=MOTOR_TYPE)
    if len(found) != 1:
        raise SystemExit(f'Expected exactly one DaMiao motor on the bus, found {len(found)} {sorted(found)}.')
    motor_id, feedback_id = next(iter(found.items()))
    if feedback_id is None:
        raise SystemExit(f'Motor {motor_id} answered but its feedback id could not be read.')
    print(f'Found spool motor: motor_id={motor_id}, feedback_id={feedback_id}')
    return controller.add_motor(motor_id=motor_id, feedback_id=feedback_id, motor_type=MOTOR_TYPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('length', type=float, help='meters of line to wind')
    parser.add_argument('--reverse', action='store_true',
                        help='spin the spool the other way (anchor_arp_eval winds the lower spool this way)')
    args = parser.parse_args()
    if args.length <= 0:
        parser.error('length must be positive')

    # In case the image runs the robot service, which holds the can bus.
    subprocess.run(['sudo', 'systemctl', 'stop', 'cranebot.service'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    set_motor_power(True)  # no-op on hats without the motor power MOSFET
    time.sleep(0.5)

    controller = DaMiaoController(channel='can0', bustype='socketcan')
    motor = find_spool_motor(controller)
    motor.disable()

    direction = -1 if args.reverse else 1
    revs = line_revs(args.length)
    print(f'Winding {args.length:.2f} m of line ({revs:.1f} spool revolutions).')
    print(f'The wind stops if line tension goes over {TENSION_LIMIT_N:.0f} N. Press ctrl-c to stop it early.')
    input('Tie the line to the spool, hold the source spool so it can spin, and press Enter to wind...')

    winder = Winder(motor, direction)
    reason = None
    try:
        motor.enable()
        time.sleep(0.1)
        motor.ensure_control_mode('VEL')
        for vel_rev_per_s in ramp_profile(revs, MAX_REV_PER_S, ACCEL_REV_PER_S2, DT):
            winder.tick(direction * vel_rev_per_s * 2 * pi)
    except WindStop as e:
        reason = str(e)
    except KeyboardInterrupt:
        reason = 'stopped with ctrl-c'
    finally:
        try:
            motor.send_cmd_vel(target_velocity=0)
            winder.settle()
        except KeyboardInterrupt:
            pass
        finally:
            motor.send_cmd_vel(target_velocity=0)
            motor.disable()

    wound = winder.wound_m()
    if reason:
        print(f'Wind stopped early: {reason}.')
    else:
        print('Wind complete.')
    print(f'Wound {wound:.2f} m of the {args.length:.2f} m requested.')


if __name__ == '__main__':
    main()
