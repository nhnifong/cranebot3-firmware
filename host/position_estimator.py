import numpy as np
# import scipy.optimize as optimize
from scipy.interpolate import BSpline

# rather than assuming the position of the gripper is the last position obtained from a charuco board,
# instead we fit a model defining it's positoin in time to a bunch of measurments from given instants.
# charuco measurements provide direct positional estimates to fit the curve to, but we also use the IMU's
# acceleration and pose data, and past (recorded) and future (planned) spool line lengths.

# Consider network with a node for every measurable variable and every variable in the model.
# there are two kinds of links between nodes.
# 1. directional calculation, where one variable can be computed from the other.
# 2. equality, where the two variables are supposed to represent the same thing, and the error between them should be part of the cost function.

class CDPR_position_estimator:
    def __init__(self, anchor_points, platform_mass, platform_inertia, gravity=9.81):
        """
        Initializes the CDPR (Cable Driven Parallel Robot)
        All units are SI. these vales are assumed to be obtained from the auto calibration step

        Args:
            anchor_points: A numpy array of shape (n_cables, 3) representing the 3D coordinates of the cable anchor points.
            gantry_mass: The mass of the gantry. I'm calling the little connection point the gantry
            gantry_inertia: The inertia tensor of the gantry (3x3 matrix).
            gripper_mass: The mass of the gantry. I'm calling the little connection point the gantry
            griper_inertia: The inertia tensor of the gantry (3x3 matrix).
            gravity: Acceleration due to gravity.
        """
        self.anchor_points = np.array(anchor_points)
        self.n_cables = self.anchor_points.shape[0]
        self.platform_mass = platform_mass
        self.platform_inertia = np.array(platform_inertia)
        self.gravity = gravity

	def calc_line_length(self, time, anchor_index, gantry_pos_spline):
		"""
		Return a function that gives the length that an anchor line would be at a given time, from the gantry position spline
		Not applicable to the gripper winch line

		Args:
			anchor_index: which line to estimate
			gantry_pos_spline: BSpline. the model of the gantry position in time.
		"""

		# evaluate the gantry position spline at a given instant and measure distance to anchor
		return lambda time: np.linalg.norm(self.anchor_points[anchor_index] - gantry_pos_spline(time), axis=1)

	def calc_spline_accel(self, position_spline):
		"""
		Calculate what the acceleration would be from a position model
		"""
		return spl.derivative(2) # amazing

	def calc_winch_line_len(gantry_pos_spline, gripper_pos_spline):
		"""
		Return a function giving the winch line length at a point in time
		"""
		return lambda time: np.linalg.norm(gantry_pos_spline(time) - gripper_pos_spline(time), axis=1)

	def calc_gripper_accel_from_forces(self, gantry_pos_spline, gripper_pos_spline):
		"""
		Calculate the acceleration on the gripper based on the forces we would expect it to experience due to
		it being a pendulum hanging from a moving object

		three forces act on it
		gravity acts to accelerate the mass down.
		a tension force acts in the direction of the gantry equal to the component of gravity opposite that direction.
		an inertial force acts on the gripper opposite the direction of acceleration of the gantry

		Args:
			gantry_pos_spline: model of gantry position through time.
			gripper_pos_spline: model of gripper position through time.
		"""
		return None

