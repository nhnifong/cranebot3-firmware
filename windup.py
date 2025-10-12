# script to wind up the correct length of line on on an anchor
import model_constants
from motor_control import MKSSERVO42C
from spools import SpiralCalculator
import time

empty_diameter=model_constants.empty_spool_diameter
full_diameter=model_constants.full_spool_diameter_power_line
full_length=model_constants.assumed_full_line_length
gear_ratio=20/51
sc = SpiralCalculator(empty_diameter, full_diameter, full_length, gear_ratio, -1)

motor = MKSSERVO42C()
assert(motor.ping())

try:
	revs = motor.getShaftAngle()
	finish_revs = sc.calc_za_from_length(0, revs)
	while revs > finish_revs:
		print(f'revs={revs} finish={finish_revs}')
		motor.runConstantSpeed(-4)
		time.sleep(0.2)
		revs = motor.getShaftAngle()
finally:
	motor.runConstantSpeed(0)