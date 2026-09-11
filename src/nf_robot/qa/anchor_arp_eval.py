# script for Arpeggio Anchor setup
# Set motor IDs and wind up the correct length of line on each spool
# test camera

import argparse
import random
import statistics
import time
import socket
import subprocess
from damiao_motor import DaMiaoController
from math import pi, sqrt

from nf_robot.qa.set_hostname import set_component_hostname
from nf_robot.robot import server_conf

MOTOR_TYPE = "G6215"
FEEDBACK_ID_REGISTER = 7  # MST_ID
MOTOR_ID_REGISTER = 8  # ESC_ID
MOTOR_ID_SCAN_RANGE = range(0x01, 0x11)  # motor id 0x00 is reserved and bricks the motor if set
ANCHOR_MOTOR_TARGETS = [
    # (label, target motor_id, target feedback_id)
    ("lower", 1, 1),
    ("upper", 2, 2),
]

# Holding torque measurement. See server_conf.DEFAULT_HOLD_TORQUE_NM for what is measured and why.
HOLD_SLACK_M = 0.5  # slack asked for on each spool. a run reels in about 0.25 m of it
HOLD_TENSION_LIMIT_N = 10.0  # abort if the torque implies more line tension than this
HOLD_MAX_TRAVEL_RAD = 8.0  # abort past this far from the start, about 0.29 m of line
HOLD_STATIONS = 12  # stops spread over one revolution
HOLD_STROKE_RAD = 0.3  # the back and forth at each stop
HOLD_SPEED = 3.0  # rad/s
HOLD_ACCEL = 0.01  # ACC/DEC register, the value the spool loop runs at
HOLD_RAD_S2_PER_ACCEL = 1000.0  # that register value measured as a 10 rad/s^2 ramp
HOLD_STILL_S = 0.15  # still this long before reading the torque
HOLD_STILL_VEL = 0.03  # rad/s. a stopped motor reads -0.011
HOLD_MAX_SD_NM = 0.012  # per-stop scatter about 3x normal, seen when a spool rubs on something
HOLD_LOOP_HZ = 100
HOLD_STALE_COMMANDS = 20  # commands in a row with no feedback before aborting
EMPTY_SPOOL_RADIUS_M = 0.036  # the most tension per N.m, so the tension limit trips early, not late
POS_WRAP_RAD = 25.0  # reported position wraps over [-12.5, 12.5] rad


def scan_motors(controller, motor_type=MOTOR_TYPE, ids=MOTOR_ID_SCAN_RANGE, duration=0.5):
    """
    Probe a range of candidate motor IDs with a zero command and listen for responses.
    Returns {motor_id: feedback_id} for every motor that responded. Self-contained:
    leaves the controller's motor map empty afterward.
    """
    controller.motors.clear()
    controller._motors_by_feedback.clear()
    controller.flush_bus()
    for motor_id in ids:
        motor = controller.add_motor(motor_id=motor_id, feedback_id=0x00, motor_type=motor_type)
        motor.send_cmd_mit(0.0, 0.0, 0.0, 0.0, 0.0)

    deadline = time.time() + duration
    while time.time() < deadline:
        controller.poll_feedback()
        time.sleep(0.01)

    found = {}
    for motor in controller.all_motors():
        if motor.state and motor.state.get("can_id") is not None:
            try:
                feedback_id = int(motor.get_register(FEEDBACK_ID_REGISTER, timeout=0.5))
            except TimeoutError:
                feedback_id = None
            found[motor.motor_id] = feedback_id

    controller.motors.clear()
    controller._motors_by_feedback.clear()

    return found


def configure_one_motor(controller, label, target_motor_id, target_feedback_id, motor_type=MOTOR_TYPE, attempts=3):
    """
    Prompt for a single motor to be connected, then write its feedback_id (MST_ID)
    and motor_id (ESC_ID) to the requested targets, verifying the writes actually
    persisted to flash. Returns True if the motor_id (ESC_ID) was changed.

    Important: when ESC_ID (register 8) is written, the motor starts listening on
    the new id immediately. The flash store therefore has to be addressed to the
    NEW id, not the old one. The CLI (and an earlier version of this script) store
    against the old id, which is why their writes silently fail to persist. This
    mirrors the web GUI's working sequence: write a register, point the motor object
    at its new id, store; then a final explicit store like the GUI's button.

    Equally important: the motor has to be DISABLED before any of this. A store sent
    to an enabled motor is ACKed and the register writes still land in RAM, so
    everything (including this function's verification, which re-reads registers)
    reports success -- and then the values revert on the next power cycle.
    """
    if target_motor_id == 0:
        raise ValueError("target_motor_id cannot be 0 (motor_id 0 bricks the motor)")

    # Wait until exactly one motor is on the bus.
    while True:
        input(f"Plug in ONLY the {label} motor, then press Enter...")
        found = scan_motors(controller, motor_type=motor_type)
        if len(found) == 0:
            print("  No motor detected, check the connection and try again.")
        elif len(found) > 1:
            print(f"  Found more than one motor ({sorted(found)}). Unplug the other motor and try again.")
        else:
            current_motor_id, current_feedback_id = next(iter(found.items()))
            break

    if current_motor_id == target_motor_id and current_feedback_id == target_feedback_id:
        print(f"  {label} motor already has motor_id={target_motor_id}, feedback_id={target_feedback_id}.")
        return False

    motor_id_changed = current_motor_id != target_motor_id

    for attempt in range(1, attempts + 1):
        controller.motors.clear()
        controller._motors_by_feedback.clear()
        controller.flush_bus()
        # Address the motor at whatever id it currently answers on.
        motor = controller.add_motor(motor_id=current_motor_id, feedback_id=0x00, motor_type=motor_type)
        time.sleep(0.1)
        controller.poll_feedback()

        # The motor must be disabled for store_parameters() to commit to flash.
        # An enabled motor still ACKs the 0xAA store frame and still applies the
        # register writes to RAM, so the write looks like it succeeded and only
        # reverts on the next power cycle. Motors arrive here already enabled if a
        # previous run or cranebot.service left them that way, and it has to be
        # re-done on every attempt because each one re-adds the motor.
        motor.set_zero_command()
        motor.disable()
        time.sleep(0.2)

        # Feedback_id (MST_ID) first, while the motor is still at its current id.
        if current_feedback_id != target_feedback_id:
            motor.write_register(FEEDBACK_ID_REGISTER, target_feedback_id)
            time.sleep(0.1)
            motor.store_parameters()
            time.sleep(0.3)

        # Motor_id (ESC_ID) next. After the write the motor listens on the new id,
        # so repoint the motor object there before storing.
        if motor_id_changed:
            motor.write_register(MOTOR_ID_REGISTER, target_motor_id)
            motor.motor_id = target_motor_id
            time.sleep(0.1)
            motor.store_parameters()
            time.sleep(0.3)

        # Final explicit store, like the GUI's "store parameters" button.
        motor.store_parameters()
        time.sleep(0.5)

        controller.motors.clear()
        controller._motors_by_feedback.clear()

        # Verify the write took effect. The motor may now answer at its old OR new
        # id, so match on the feedback value rather than the id.
        #
        # Caveat: this reads back live registers, i.e. RAM, so it confirms the write
        # was applied but CANNOT prove store_parameters() committed it to flash -- an
        # enabled motor applies writes to RAM and ACKs the store, and only reverts on
        # the next power cycle. The disable above is what makes the commit reliable;
        # the power cycle prompt in ensure_motor_ids is what actually proves it.
        verify = scan_motors(controller, motor_type=motor_type)
        if len(verify) == 1 and target_feedback_id in verify.values():
            answering_id = next(iter(verify))
            print(f"  Set {label} motor to motor_id={target_motor_id}, feedback_id={target_feedback_id} "
                  f"(currently answering at id {answering_id}).")
            return motor_id_changed

        if len(verify) == 1:
            current_motor_id, current_feedback_id = next(iter(verify.items()))
        print(f"  Write did not take effect (attempt {attempt}/{attempts}), retrying...")

    raise RuntimeError(f"Failed to configure {label} motor after {attempts} attempts (register writes not taking effect).")


def configure_feedback_in_place(controller, label, target_motor_id, target_feedback_id, motor_type=MOTOR_TYPE, attempts=3):
    """
    Set a motor's feedback_id (MST_ID) in place, while both motors are connected,
    without changing its motor_id (ESC_ID). Because the motor keeps the id it
    already answers on, addressing target_motor_id reaches only that motor even
    with the other motor present on the bus. Used for the lower motor, which stays
    on the factory motor_id (1), so it never needs to be unplugged on its own.

    As in configure_one_motor, the motor must be disabled before the write/store or
    the store never reaches flash, silently and with a successful-looking readback.
    """
    for attempt in range(1, attempts + 1):
        controller.motors.clear()
        controller._motors_by_feedback.clear()
        controller.flush_bus()
        motor = controller.add_motor(motor_id=target_motor_id, feedback_id=0x00, motor_type=motor_type)
        time.sleep(0.1)
        controller.poll_feedback()

        # Must be disabled or store_parameters() silently fails to reach flash.
        motor.set_zero_command()
        motor.disable()
        time.sleep(0.2)

        motor.write_register(FEEDBACK_ID_REGISTER, target_feedback_id)
        time.sleep(0.1)
        motor.store_parameters()
        time.sleep(0.3)
        # Final explicit store, like the GUI's "store parameters" button.
        motor.store_parameters()
        time.sleep(0.5)

        controller.motors.clear()
        controller._motors_by_feedback.clear()

        verify = scan_motors(controller, motor_type=motor_type)
        if verify.get(target_motor_id) == target_feedback_id:
            print(f"  Set {label} motor to motor_id={target_motor_id}, feedback_id={target_feedback_id}.")
            return
        print(f"  Write did not take effect (attempt {attempt}/{attempts}), retrying...")

    raise RuntimeError(f"Failed to configure {label} motor after {attempts} attempts (register writes not taking effect).")


def ensure_motor_ids(controller, motor_type=MOTOR_TYPE, targets=ANCHOR_MOTOR_TARGETS):
    """
    Make sure the anchor's two motors are set to their expected motor_id/feedback_id.
    If they're already correct, returns immediately. Otherwise configures the upper
    motor on its own first (moving it off the factory id 1 to id 2). Once the upper
    motor is on id 2, the lower motor is the only one still answering on the factory
    id 1, so it can be configured in place with both motors connected, no further
    unplugging required.
    """
    expected = {motor_id: feedback_id for _, motor_id, feedback_id in targets}
    targets_by_label = {label: (motor_id, feedback_id) for label, motor_id, feedback_id in targets}
    upper_motor_id, upper_feedback_id = targets_by_label["upper"]
    lower_motor_id, lower_feedback_id = targets_by_label["lower"]

    print("Scanning for connected motors...")
    if scan_motors(controller, motor_type=motor_type) == expected:
        print("Motor IDs already correct.")
        return

    print("Motor IDs need to be configured.")

    # Configure the upper motor by itself, moving it off the factory id 1 to id 2.
    # Its return value (whether ESC_ID changed) is deliberately unused: both id
    # registers take effect the instant they are written, so nothing below needs a
    # power cycle to apply them.
    configure_one_motor(
        controller, "upper", upper_motor_id, upper_feedback_id, motor_type=motor_type)

    input("Plug in both motors, then press Enter...")

    # With the upper motor now on id 2, the lower motor is the only one still on the
    # factory id 1, so set its feedback_id in place without unplugging anything.
    # This runs BEFORE the power cycle prompt on purpose: every flash write then
    # happens up front, so one cycle verifies both motors at once, and a cycle that
    # takes the pi down with it (the anchor commonly powers both from the same AC
    # plug) cannot strand this step and leave the lower motor unconfigured.
    configure_feedback_in_place(
        controller, "lower", lower_motor_id, lower_feedback_id, motor_type=motor_type)

    
    input("Unplug and re-plug the upper motor to power cycle it, then press Enter...")

    # This re-reads live registers, i.e. RAM, so it confirms the ids are in effect but
    # cannot prove store_parameters() committed them to flash. This only happens when the motor power cycles.
    # This means we can't actually tell if the user followed the previous step.
    print("Confirming final motor IDs...")
    found = {}
    for _ in range(3):  # motors may need a moment to answer after the writes
        found = scan_motors(controller, motor_type=motor_type)
        if found == expected:
            print("Motor IDs confirmed correct.")
            return
        time.sleep(0.5)
    raise RuntimeError(f"Motor IDs still incorrect after configuration. Expected {expected}, found {found}.")


def wind_with_ramp(motor, direction, total_revs, max_rev_per_s=4.0, accel_rev_per_s2=2.0, dt=0.02):
    """
    Wind `total_revs` revolutions of line onto the spool using a trapezoidal speed
    profile: ramp up to max_rev_per_s, cruise, then ramp back down before stopping.
    The integral of the velocity profile equals total_revs by construction, so the
    correct amount of line is wound regardless of the ramp shape.

    send_cmd_vel expects rad/s, so rev/s values are converted with 2*pi.
    """
    ramp_time = max_rev_per_s / accel_rev_per_s2
    ramp_revs = 0.5 * max_rev_per_s * ramp_time  # revs covered during one ramp

    if 2 * ramp_revs > total_revs:
        # Too little line to reach max speed: triangular profile (ramp up then down).
        ramp_time = sqrt(total_revs / accel_rev_per_s2)
        peak_rev_per_s = accel_rev_per_s2 * ramp_time
        cruise_time = 0.0
    else:
        peak_rev_per_s = max_rev_per_s
        cruise_revs = total_revs - 2 * ramp_revs
        cruise_time = cruise_revs / max_rev_per_s

    def hold(vel_rev_per_s):
        motor.send_cmd_vel(target_velocity=direction * vel_rev_per_s * 2 * pi)
        time.sleep(dt)

    # ramp up
    t = 0.0
    while t < ramp_time:
        hold(accel_rev_per_s2 * t)
        t += dt
    # cruise
    t = 0.0
    while t < cruise_time:
        hold(peak_rev_per_s)
        t += dt
    # ramp down
    t = 0.0
    while t < ramp_time:
        hold(max(peak_rev_per_s - accel_rev_per_s2 * t, 0.0))
        t += dt

    motor.send_cmd_vel(target_velocity=0)


class HoldTorqueAbort(Exception):
    pass


class HoldTorqueRun:
    """Drives one motor through the holding torque measurement, checking every command's
    feedback against the tension, travel and feedback limits."""

    def __init__(self, motor):
        self.motor = motor
        self.raw = None
        self.pos = 0.0  # unwrapped rad since the first feedback
        self.last_frame = None
        self.stale = 0
        self.torque_limit = HOLD_TENSION_LIMIT_N * EMPTY_SPOOL_RADIUS_M

    def tick(self, vel):
        t0 = time.time()
        self.motor.send_cmd_vel(target_velocity=vel)
        time.sleep(max(0.0, 1.0 / HOLD_LOOP_HZ - (time.time() - t0)))
        # the driver replaces its state dict with every feedback frame it decodes
        frame = self.motor.state
        if not frame or frame is self.last_frame:
            self.stale += 1
            if self.stale >= HOLD_STALE_COMMANDS:
                raise HoldTorqueAbort(f'no feedback for {self.stale} commands')
            return {}
        self.stale = 0
        self.last_frame = frame
        torq = frame.get('torq', 0.0)
        if abs(torq) > self.torque_limit:
            raise HoldTorqueAbort(f'torque {torq:+.3f} N.m is about '
                                  f'{abs(torq) / EMPTY_SPOOL_RADIUS_M:.1f} N of line tension')
        raw = frame.get('pos', 0.0)
        if self.raw is not None:
            d = raw - self.raw
            self.pos += d - POS_WRAP_RAD * round(d / POS_WRAP_RAD)
        self.raw = raw
        if abs(self.pos) > HOLD_MAX_TRAVEL_RAD:
            raise HoldTorqueAbort(f'the spool turned {self.pos:+.1f} rad, past the {HOLD_MAX_TRAVEL_RAD} rad limit')
        return frame

    def move_to(self, target, timeout_s=5.0):
        """Run toward target, letting go early enough that the decel lands near it. Speed is
        capped so a short move still gets up to speed and slides before it stops."""
        accel = HOLD_ACCEL * HOLD_RAD_S2_PER_ACCEL
        dist = target - self.pos
        sign = 1.0 if dist > 0 else -1.0
        speed = min(HOLD_SPEED, sqrt(accel * abs(dist)))
        brake = speed * speed / (2 * accel)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            vel = sign * self.tick(sign * speed).get('vel', 0.0)
            if sign * (target - self.pos) <= brake and vel >= 0.8 * speed:
                return
        raise HoldTorqueAbort('the spool did not get up to speed, it may be blocked')

    def hold_and_read(self, timeout_s=2.0):
        """Command zero until the shaft has been still for HOLD_STILL_S, then read the torque."""
        deadline = time.time() + timeout_s
        still_since = None
        while time.time() < deadline:
            if abs(self.tick(0.0).get('vel', 1.0)) < HOLD_STILL_VEL:
                still_since = still_since or time.time()
                if time.time() - still_since >= HOLD_STILL_S:
                    break
            else:
                still_since = None
        torques = []
        while len(torques) < 5:  # a feedback outage ends this through tick's abort
            torq = self.tick(0.0).get('torq')
            if torq is not None:
                torques.append(torq)
        return statistics.fmean(torques)

    def run(self, reel_in_direction):
        """Returns ((after positive, after negative), (sd positive, sd negative)) in N.m.

        Walks once around a revolution in the reel-in direction, so the net motion takes up slack
        rather than paying out loose line. The held torque also varies with where the rotor
        stops, faster than the station spacing, so each stop is at a random spot within its
        station: evenly spaced stops sample that pattern at the same phase every time and bias
        the whole run by wherever it started.
        """
        self.motor.enable()
        time.sleep(0.1)
        self.motor.ensure_control_mode('VEL')
        self.motor.set_acceleration(HOLD_ACCEL)
        self.motor.set_deceleration(-HOLD_ACCEL)
        for _ in range(20):
            self.tick(0.0)
        held = {1: [], -1: []}
        step = 2 * pi / HOLD_STATIONS
        for i in range(HOLD_STATIONS):
            base = reel_in_direction * (i + random.random()) * step
            for sign, target in ((reel_in_direction, base + reel_in_direction * HOLD_STROKE_RAD),
                                 (-reel_in_direction, base)):
                self.move_to(target)
                held[sign].append(self.hold_and_read())
        for _ in range(5):
            self.tick(0.0)
        return ((statistics.fmean(held[1]), statistics.fmean(held[-1])),
                (statistics.stdev(held[1]), statistics.stdev(held[-1])))


def measure_hold_torques(motors):
    """Measure the holding torque of each spool motor with slack line. Returns
    {motor_id: (after positive, after negative)} for the motors that measured cleanly."""
    print(f"Measuring spool friction. Pull at least {HOLD_SLACK_M * 100:.0f} cm of line off each spool "
          "so both lines hang slack, and make sure nothing touches either spool or its line.")
    print("Each spool turns about one revolution back and forth and reels some of that slack back in.")
    input("Press Enter when ready...")

    results = {}
    for motor, reel_in_direction, name, _ in motors:
        while True:
            print(f"  Measuring the {name} motor...")
            try:
                (after_pos, after_neg), (sd_pos, sd_neg) = HoldTorqueRun(motor).run(reel_in_direction)
            except HoldTorqueAbort as e:
                print(f"  Stopped: {e}.")
            else:
                if max(sd_pos, sd_neg) > HOLD_MAX_SD_NM:
                    print(f"  Readings too scattered (sd {sd_pos:.4f}, {sd_neg:.4f} N.m). "
                          "The spool or line is probably touching something.")
                elif not after_pos > 0 > after_neg:
                    print(f"  Implausible result ({after_pos:+.4f}, {after_neg:+.4f} N.m).")
                else:
                    results[motor.motor_id] = (after_pos, after_neg)
                    print(f"  {name} motor: {after_pos:+.4f} N.m after turning positive, "
                          f"{after_neg:+.4f} N.m after turning negative.")
                    break
            finally:
                for m, _, _, _ in motors:
                    m.disable()
            if input(f"  Pull {HOLD_SLACK_M * 100:.0f} cm of slack off the {name} spool again, "
                     "clear anything touching it, and retry? y/n").strip().lower() != 'y':
                print(f"  The {name} motor keeps its previous value, or the default if it has none.")
                break
    return results


def test_camera():
    print('Starting Camera...')

    stream_command = """
    /usr/bin/rpicam-vid -t 0 -n \
      --width=1920 --height=1080 \
      -o tcp://0.0.0.0:8888?listen=1&tcp_nodelay=1 \
      --codec libav \
      --libav-format mpegts \
      --autofocus-mode continuous \
      --bitrate 2000kbps
    """.split()

    # get my ip address
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    addr = s.getsockname()[0]
    s.close()
    print('Please run the following on your host machine and confirm good video, then close the video window by pressing q.')
    print(f'========\n\nffplay -fast -fflags nobuffer -flags low_delay "tcp://{addr}:8888"\n\n========')

    subprocess.run(stream_command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--long', action='store_true',
                        help="Wind extra line: 20 m on the lower spool, 12 m on the upper spool.")
    args = parser.parse_args()

    # The cranebot service grabs the can bus and camera, so it must be stopped first.
    print('Stopping cranebot service...')
    subprocess.run(["sudo", "systemctl", "stop", "cranebot.service"])

    print('Setting up can bus interface')
    controller = DaMiaoController(channel="can0", bustype="socketcan")

    ensure_motor_ids(controller)

    # prepare to wind line on each motor.
    lower_motor = controller.add_motor(motor_id=0x01, feedback_id=0x01, motor_type=MOTOR_TYPE)
    upper_motor = controller.add_motor(motor_id=0x02, feedback_id=0x02, motor_type=MOTOR_TYPE)
    lower_motor.disable()
    upper_motor.disable()
    # lower spool needs more line because it goes around the eyelet
    lower_length, upper_length = (20.0, 12.0) if args.long else (15.0, 7.5)
    if args.long:
        print(f"--long: winding {lower_length} m on the lower spool and {upper_length} m on the upper spool.")
    motors = [
        (lower_motor, -1, 'lower', lower_length),
        (upper_motor, 1, 'upper', upper_length),
    ]

    # Differentiate power anchors from regular anchors before winding line.
    has_powerline = input("Does this anchor have a powerline spool? y/n").strip().lower() == 'y'
    if has_powerline:
        anchor_type = "arpeggio power anchor"
        component = "power-anchor"
    else:
        anchor_type = "arpeggio anchor"
        component = "anchor"

    # Record what differentiates this anchor: which spool it carries, and how much line went on.
    # The server needs the winding to pick the right full spool diameter, and a spool wound long
    # with the thick power line ends up nearly 14 mm fatter than a short one.
    winding = server_conf.WINDING_LONG if args.long else server_conf.WINDING_SHORT
    server_conf.write_server_conf(anchor_type, winding=winding)

    # Give this Pi a hostname unique to its role so the two anchors and the
    # gripper in a setup don't all share one hostname.
    set_component_hostname(component)

    for motor, direction, name, length in motors:
        val = input(f"Do you need to wind the {name} motor? y/n")
        if val == 'y':
            radius = 0.0362
            circumfrence = 2*pi*radius
            revs = length / circumfrence

            # A powerline, when this anchor has one, always goes on the upper spool
            # (the server's 'high' spool); every other spool takes fishing line.
            if has_powerline and name == 'upper':
                print("The powerline must be spliced onto the spool according to the guide at")
                print("  https://neufangled.com/docs/arpeggio_anchor_build_guide/#splice-the-line")
                print("If you need to do that now, power off and splice, then rerun this script.")
                print("The powerline wire is live during winding. Don't cut it.")
                input("Hold the source spool so it can spin. Press Enter when ready to wind...")
            else:
                print("Tie fishing line to the spool's tie off point with a buntline hitch "
                      "and hold the source spool so it can spin.")
                input("Press Enter when ready to wind...")
            try:
                motor.enable()
                wind_with_ramp(motor, direction, revs, max_rev_per_s=4.0)
            finally:
                motor.send_cmd_vel(target_velocity=0)
                motor.disable()
            print("With the end of the line passing through the sunglasses part, "
                  "tie on a carabiner with a palomar knot.")
        else:
            continue

    # Friction in each spool motor biases its tension reading by up to a newton, depending on
    # the way it last turned. Measured here, after winding, while the line can still be slack.
    if input("Do you want to measure spool friction? y/n").strip().lower() == 'y':
        holds = measure_hold_torques(motors)
        if holds:
            server_conf.write_server_conf(anchor_type, winding=winding, hold_torques=holds)
            print(f"Recorded spool friction for motor(s) {sorted(holds)} in {server_conf.CONF_PATH}.")

    if input("Do you want to run the camera test? y/n").strip().lower() == 'y':
        test_camera()


if __name__ == "__main__":
    main()