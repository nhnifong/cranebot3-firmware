# script for Arpeggio Anchor setup
# Set motor IDs and wind up the correct length of line on each spool
# test camera

import time
import socket
import subprocess
from damiao_motor import DaMiaoController
from math import pi, sqrt

import nf_robot.common.definitions as model_constants

MOTOR_TYPE = "G6215"
FEEDBACK_ID_REGISTER = 7  # MST_ID
MOTOR_ID_REGISTER = 8  # ESC_ID
MOTOR_ID_SCAN_RANGE = range(0x01, 0x11)  # motor id 0x00 is reserved and bricks the motor if set
ANCHOR_MOTOR_TARGETS = [
    # (label, target motor_id, target feedback_id)
    ("lower", 1, 1),
    ("upper", 2, 2),
]


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

        # Verify it persisted. The motor may now answer at its old OR new id, so
        # match on the feedback value rather than the id. A persisted feedback_id is
        # our proxy that store_parameters committed all params (including ESC_ID).
        verify = scan_motors(controller, motor_type=motor_type)
        if len(verify) == 1 and target_feedback_id in verify.values():
            answering_id = next(iter(verify))
            print(f"  Set {label} motor to motor_id={target_motor_id}, feedback_id={target_feedback_id} "
                  f"(currently answering at id {answering_id}).")
            return motor_id_changed

        if len(verify) == 1:
            current_motor_id, current_feedback_id = next(iter(verify.items()))
        print(f"  Write did not persist (attempt {attempt}/{attempts}), retrying...")

    raise RuntimeError(f"Failed to configure {label} motor after {attempts} attempts (writes not persisting to flash).")


def configure_feedback_in_place(controller, label, target_motor_id, target_feedback_id, motor_type=MOTOR_TYPE, attempts=3):
    """
    Set a motor's feedback_id (MST_ID) in place, while both motors are connected,
    without changing its motor_id (ESC_ID). Because the motor keeps the id it
    already answers on, addressing target_motor_id reaches only that motor even
    with the other motor present on the bus. Used for the lower motor, which stays
    on the factory motor_id (1), so it never needs to be unplugged on its own.
    """
    for attempt in range(1, attempts + 1):
        controller.motors.clear()
        controller._motors_by_feedback.clear()
        controller.flush_bus()
        motor = controller.add_motor(motor_id=target_motor_id, feedback_id=0x00, motor_type=motor_type)
        time.sleep(0.1)
        controller.poll_feedback()

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
        print(f"  Write did not persist (attempt {attempt}/{attempts}), retrying...")

    raise RuntimeError(f"Failed to configure {label} motor after {attempts} attempts (writes not persisting to flash).")


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
    motor_id_changed = configure_one_motor(
        controller, "upper", upper_motor_id, upper_feedback_id, motor_type=motor_type)

    input("Plug in both motors, then press Enter...")

    if motor_id_changed:
        print("The upper motor's motor_id (ESC_ID) was changed; that only takes effect after a power cycle.")
        input("Power cycle both motors now by pulling the barrel jack connector and re-plugging it, then press Enter. If you are not powering the pi any other way, just come back and start this script again, and it will pick up where it left off.")

    # With the upper motor now on id 2, the lower motor is the only one still on the
    # factory id 1, so set its feedback_id in place without unplugging anything.
    configure_feedback_in_place(
        controller, "lower", lower_motor_id, lower_feedback_id, motor_type=motor_type)

    print("Confirming final motor IDs...")
    found = {}
    for _ in range(3):  # motors may need a moment to come up after a power cycle
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


def test_camera():
    print('Starting Camera...')

    stream_command = """
    /usr/bin/rpicam-vid -t 0 -n \
      --width=1920 --height=1080 \
      -o tcp://0.0.0.0:8888?listen=1 \
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
    motors = [
        (lower_motor, -1, 'lower', 15.0), # lower spool needs more line because it goes around the eyelet
        (upper_motor, 1, 'upper', 7.5),
    ]

    # Differentiate power anchors from regular anchors before winding line.
    if input("Does this anchor have a powerline spool? y/n").strip().lower() == 'y':
        anchor_type = "arpeggio power anchor"
        full_diameter = model_constants.damiao_full_spool_diameter_power_line
    else:
        anchor_type = "arpeggio anchor"
        full_diameter = model_constants.damiao_full_spool_diameter_fishing_line

    # Write the file that differentiates power anchors from regular anchors
    with open('/opt/robot/server.conf', 'w') as f:
        f.write(anchor_type + '\n')

    for motor, direction, name, length in motors:
        val = input(f"Do you need to wind the {name} motor? y/n")
        if val == 'y':
            radius = 0.0362
            circumfrence = 2*pi*radius
            revs = length / circumfrence

            input("When ready press Enter...")
            try:
                motor.enable()
                wind_with_ramp(motor, direction, revs, max_rev_per_s=4.0)
            finally:
                motor.send_cmd_vel(target_velocity=0)
                motor.disable()
        else:
            continue

    if input("Do you want to run the camera test? y/n").strip().lower() == 'y':
        test_camera()


if __name__ == "__main__":
    main()