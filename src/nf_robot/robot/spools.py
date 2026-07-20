import math

class SpiralCalculator:
    def __init__(self, empty_diameter, full_diameter, full_length, gear_ratio, motor_orientation):
        self.empty_diameter = empty_diameter * 0.001 # millimeter to meters
        self.gear_ratio = gear_ratio # spool rotations per encoder rotation 
        # a negative motor orientation means that negative speeds make the line shorter.
        self.motor_orientation = motor_orientation
        self.zero_angle = 0

        # since line accumulates on the spool in a spiral, the amount of wrapped line is an exponential function of the spool angle.
        self.recalc_k_params(full_diameter, full_length)

    def set_zero_angle(self, zero_a):
        self.zero_angle = zero_a

    def recalc_k_params(self, full_diameter, full_length):
        self.full_diameter = full_diameter * 0.001
        self.full_length = full_length
        # since line accumulates on the spool in a spiral, the amount of wrapped line is an exponential function of the spool angle.
        self.diameter_diff = self.full_diameter - self.empty_diameter
        if self.diameter_diff > 0:
            self.k1 = (self.empty_diameter * self.full_length) / self.diameter_diff
            self.k2 = (math.pi * self.gear_ratio * self.diameter_diff) / self.full_length
        else:
            self.k1 = self.empty_diameter * self.full_length / 1e-9 # Avoid division by zero
            self.k2 = (math.pi * self.gear_ratio * 1e-9) / self.full_length

    def calc_za_from_length(self, length, angle):
        """ Given an observed length and current angle, what would the zero angle be, all other things being equal?"""
        # how many revs must the motor have turned since empty be to have this length
        spooled_length = self.full_length - length
        relative_angle = math.log(spooled_length / self.k1 + 1) / self.k2
        angle *= self.motor_orientation
        return angle - relative_angle

    def get_spooled_length(self, motor_angle_revs):
        relative_angle = self.motor_orientation * motor_angle_revs - self.zero_angle # shaft angle relative to zero_angle
        if self.diameter_diff == 0:
            return relative_angle * self.gear_ratio * math.pi * self.empty_diameter
        else:
            return self.k1 * (math.exp(self.k2 * relative_angle) - 1)

    def get_unspooled_length(self, motor_angle_revs):
        return self.full_length - self.get_spooled_length(motor_angle_revs)

    def get_unspool_rate(self, motor_angle_revs):
        relative_angle = self.motor_orientation * motor_angle_revs - self.zero_angle

        if self.diameter_diff == 0:
            return math.pi * self.empty_diameter * self.gear_ratio
        else:
            effective_spool_diameter =  self.empty_diameter + self.diameter_diff * (self.get_spooled_length(motor_angle_revs) / self.full_length)
            return math.pi * effective_spool_diameter * self.gear_ratio
