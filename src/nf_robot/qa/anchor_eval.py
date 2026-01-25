# script to wind up the correct length of line on on an anchor
import time
import RPi.GPIO as GPIO
import argparse

import nf_robot.common.definitions as model_constants
from nf_robot.robot.mks42c_motor import MKSSERVO42C
from nf_robot.robot.spools import SpiralCalculator

def main():

	parser = argparse.ArgumentParser(description="Stringman Component")
	parser.add_argument(
	        'anchor_type',
	        type=str,
	        default='anchor',
	        choices=['anchor', 'power_anchor'],
	        help="The type of anchor to set (choices: anchor, power_anchor)"
	    )
	args = parser.parse_args()

	# Write the file that differentiates power anchors from regular anchors
	if args.anchor_type == 'anchor':
		full_diameter=model_constants.full_spool_diameter_fishing_line
	elif args.anchor_type == 'power anchor':
		full_diameter=model_constants.full_spool_diameter_power_line

	with open('/opt/robot/server.conf', 'w') as f:
		f.write('args.anchor_type' + '\n')

	empty_diameter=model_constants.empty_spool_diameter
	full_length=model_constants.assumed_full_line_length
	gear_ratio=20/51
	sc = SpiralCalculator(empty_diameter, full_diameter, full_length, gear_ratio, -1)

	motor = MKSSERVO42C()
	assert(motor.ping())

	SWITCH_PIN = 18
	GPIO.setmode(GPIO.BCM)
	GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

	print("Click switch to begin winding spool")
	count = 0
	tight = False
	while count < 50 and not tight:
	    time.sleep(0.1)
	    tight = GPIO.input(SWITCH_PIN) == 0
	assert tight, 'switch never registered any click'
	print('switch functioning normally, begin winding')

	try:
		_, revs = motor.getShaftAngle()
		finish_revs = sc.calc_za_from_length(0, revs)
		while revs > finish_revs:
			print(f'revs={revs} finish={finish_revs}')
			motor.runConstantSpeed(-4)
			time.sleep(0.2)
			_, revs = motor.getShaftAngle()
	finally:
		motor.runConstantSpeed(0)

	print('test complete. camera not checked.')

if __name__ == "__main__":
    main()