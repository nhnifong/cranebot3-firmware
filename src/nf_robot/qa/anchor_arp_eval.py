# script for Arpeggio Anchor setup
# Set motor IDs and wind up the correct length of line on each spool
# test camera

import time
import argparse
from damiao_motor import DaMiaoController
from math import pi

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
    Returns {motor_id: feedback_id} for every motor that responded, leaving the
    controller's motor map exactly as it was before the call.
    """
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


def configure_one_motor(controller, label, target_motor_id, target_feedback_id, motor_type=MOTOR_TYPE):
    """
    Prompt for a single motor to be connected, then write its motor_id (ESC_ID) and
    feedback_id (MST_ID) to the requested targets. Returns True if the motor_id
    (ESC_ID) was changed, since that requires a power cycle to take effect.
    """
    if target_motor_id == 0:
        raise ValueError("target_motor_id cannot be 0 (motor_id 0 bricks the motor)")

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

    motor = controller.add_motor(motor_id=current_motor_id, feedback_id=0x00, motor_type=motor_type)
    time.sleep(0.1)
    controller.poll_feedback()

    needs_store = False
    if current_feedback_id != target_feedback_id:
        motor.write_register(FEEDBACK_ID_REGISTER, target_feedback_id)
        needs_store = True

    motor_id_changed = current_motor_id != target_motor_id
    if motor_id_changed:
        motor.write_register(MOTOR_ID_REGISTER, target_motor_id)
        needs_store = True

    if needs_store:
        time.sleep(0.05)
        motor.store_parameters()
        time.sleep(0.1)

    print(f"  Set {label} motor to motor_id={target_motor_id}, feedback_id={target_feedback_id}.")

    controller.motors.pop(motor.motor_id, None)
    controller._motors_by_feedback.pop(motor.motor_id, None)
    return motor_id_changed


def ensure_motor_ids(controller, motor_type=MOTOR_TYPE, targets=ANCHOR_MOTOR_TARGETS):
    """
    Make sure the anchor's two motors are set to their expected motor_id/feedback_id.
    If they're already correct, returns immediately. Otherwise walks the user through
    plugging in each motor individually to set its IDs, then both together to confirm.
    """
    expected = {motor_id: feedback_id for _, motor_id, feedback_id in targets}

    print("Scanning for connected motors...")
    if scan_motors(controller, motor_type=motor_type) == expected:
        print("Motor IDs already correct.")
        return

    print("Motor IDs need to be configured.")
    motor_id_changed = False
    for label, target_motor_id, target_feedback_id in targets:
        if configure_one_motor(controller, label, target_motor_id, target_feedback_id, motor_type=motor_type):
            motor_id_changed = True

    input("Plug in both motors, then press Enter...")

    if motor_id_changed:
        print("A motor_id (ESC_ID) was changed; that only takes effect after a power cycle.")
        input("Power cycle both motors now, then press Enter once they're back on...")

    print("Confirming final motor IDs...")
    found = scan_motors(controller, motor_type=motor_type)
    if found != expected:
        raise RuntimeError(f"Motor IDs still incorrect after configuration. Expected {expected}, found {found}.")
    print("Motor IDs confirmed correct.")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--power", action="store_true",
                        help="Configures this anchor as the one which has the power line")
    args = parser.parse_args()
    if args.power:
        anchor_type = "arpeggio power anchor"
        full_diameter=model_constants.damiao_full_spool_diameter_power_line
    else:
        anchor_type = "arpeggio anchor"
        full_diameter=model_constants.damiao_full_spool_diameter_fishing_line

    # Write the file that differentiates power anchors from regular anchors
    with open('/opt/robot/server.conf', 'w') as f:
        f.write(anchor_type + '\n')

    print('Setting up can bus interface')
    controller = DaMiaoController(channel="can0", bustype="socketcan")

    ensure_motor_ids(controller)

    # prepare to wind line on each motor.
    lower_motor = controller.add_motor(motor_id=0x01, feedback_id=0x01, motor_type=MOTOR_TYPE)
    upper_motor = controller.add_motor(motor_id=0x02, feedback_id=0x02, motor_type=MOTOR_TYPE)
    lower_motor.disable()
    upper_motor.disable()
    motors = [
        (lower_motor, -1, 'lower', 14.0), # lower spool needs more line because it goes around the eyelet
        (upper_motor, 1, 'upper', 7.0),
    ]

    for motor, direction, name, length in motors:
        val = input(f"Do you need to wind the {name} motor? y/n")
        if val == 'y':
            radius = 0.0362
            circumfrence = 2*pi*radius
            revs = length / circumfrence
            rads = revs*2*pi
            wind_speed = 6
            seconds = rads/wind_speed

            input("When ready press Enter...")
            try:
                motor.enable()
                motor.send_cmd_vel(target_velocity=direction*wind_speed)
                time.sleep(seconds)
            finally:
                motor.send_cmd_vel(target_velocity=0)
                motor.disable()
        else:
            continue


if __name__ == "__main__":
    main()