# script to wind up the correct length of line on on an anchor
import model_constants
from motor_control import MKSSERVO42C
from spools import SpiralCalculator
import time
import RPi.GPIO as GPIO

with open('server.conf', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if not line.startswith('#') and line:  # Check if line is not a comment and is not empty
            component_type = line
            logging.info(f'Starting cranebot server of type {component_type}')
            break

if component_type == 'anchor':
	full_diameter=model_constants.full_spool_diameter_fishing_line
elif component_type == 'power anchor':
	full_diameter=model_constants.full_spool_diameter_power_line

empty_diameter=model_constants.empty_spool_diameter
full_length=model_constants.assumed_full_line_length
gear_ratio=20/51
sc = SpiralCalculator(empty_diameter, full_diameter, full_length, gear_ratio, -1)

motor = MKSSERVO42C()
assert(motor.ping())

SWITCH_PIN = 18
print("Click switch to begin winding spool")
count = 0
tight = False
while count < 50 and not tight
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