# script for Arpeggio Anchor setup
# Set motor IDs and wind up the correct length of line on each spool
# test camera

import time
import argparse
from damiao_motor import DaMiaoController
from math import pi

import nf_robot.common.definitions as model_constants

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

    # scan for motors

    # if two are connected, check the ids. if correct, proceed, if not tell the user to plug in only the upper motor.
    # if one is connected, move it, ask user if it was the upper motor.
    # if so, set its ID to the upper motor ID.
    # if not, set its ID to the lower motor ID. 
    
    # once both motor ids are set.
    # prepare to wind line on each motor.

    lower_motor = controller.add_motor(motor_id=0x01, feedback_id=0x01, motor_type="H6220")
    upper_motor = controller.add_motor(motor_id=0x02, feedback_id=0x02, motor_type="H6220")
    lower_motor.disable()
    upper_motor.disable()
    motors = [
        (lower_motor, -1, 'lower'),
        (upper_motor, 1, 'upper'),
    ]

    for motor, direction, name in motors:
        val = input(f"Do you need to wind the {name} motor? y/n")
        if val == 'y':
            radius = 0.0362
            length_to_wind = 7.00
            circumfrence = 2*pi*radius
            revs = length_to_wind / circumfrence
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