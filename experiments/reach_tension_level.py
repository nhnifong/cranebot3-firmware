from motor_control import MKSSERVO42C
import time

m = MKSSERVO42C()

def runUntilLevel(measurement_speed, speed, level):
	_, raw = m.getShaftError()
	smoothed = raw
	motor.runConstantSpeed(measurement_speed)
	try:
		for i in range(30):
			_, raw = m.getShaftError()
			smoothed = smoothed * 0.9 + raw * 0.1
			print(f'raw={raw} smoothed={smoothed}')
			time.sleep(1/30)
		m.runConstantSpeed(speed)
		while abs(smoothed) < level:
			_, raw = m.getShaftError()
			smoothed = smoothed * 0.95 + raw * 0.05
			print(f'raw={raw} smoothed={smoothed}')
			time.sleep(1/30)
	except e:
		m.runConstantSpeed(0)
		raise e
	m.runConstantSpeed(0)
