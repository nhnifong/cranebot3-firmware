from motor_control import MKSSERVO42C
import time

m = MKSSERVO42C()

def runUntilLevel(level)
	_, raw = m.getShaftError()
	smoothed = raw
	for i in range(30):
		_, raw = m.getShaftError()
		smoothed = smoothed * 0.9 + raw * 0.1
		print(f'raw={aerr} smoothed={smoothed}')
		time.sleep(1/30)
	m.runConstantSpeed(-1)
	while aerr < level:
		try:
			_, raw = m.getShaftError()
			smoothed = smoothed * 0.9 + raw * 0.1
			print(f'raw={aerr} smoothed={smoothed}')
			time.sleep(1/30)
		except KeyboardInterrupt:
			m.runConstantSpeed(0)
	m.runConstantSpeed(0)