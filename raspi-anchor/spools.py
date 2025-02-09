from motor_control import MKSSERVO42C
from math import pi
import asyncio
import time

DEFAULT_MICROSTEPS = 16
# A speed of 1 is this many revs/sec
SPEED1_REVS = 30000.0/(DEFAULT_MICROSTEPS * 200)/60;
METER_PER_REV = SPOOL_DIAMETER_MM * pi * 0.001;

class SpoolController:
	def __init(self):
		self.motor = MKSSERVO42C()
		# Meters of line that were spooled out when zeroAngle was set.
		self.lineAtStart = 1.9
		self.zeroAngle = 0
		# last set speed in motor units (-127 to 127)
		self.motor_speed = 0

	def setReferenceLength(length):
		"""
		Provide an external observation of the current line length
		"""
		self.lineAtStart = length
		self.zeroAngle = self.motor.getShaftAngle()

	def currentLineLength(self):
		"""
		return current unspooled line in meters
		"""
		return METER_PER_REV * (float(self.motor.getShaftAngle()) - self.zeroAngle) / ANGLE_RESOLUTION + self.lineAtStart

	async def slowStop(self):
		direction = self.speed / self.speed
		while abs(self.speed) > 0:
			self.speed -= direction
			self.motor.runConstantSpeed(self.speed)
			await asyncio.sleep(0.05)
		self.motor.stop()

	async def trackingLoop(self):
		"""
		Constantly try to match the position and speed given in an array
		"""
		while True:
			await asyncio.sleep(0.03)