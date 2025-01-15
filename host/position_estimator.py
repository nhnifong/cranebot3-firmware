import numpy as np
import scipy.optimize as optimize
from scipy.interpolate import BSpline

# rather than assuming the position of the gripper is the last position obtained from a charuco board,
# instead we fit a model defining it's positoin in time to a bunch of measurments from given instants.
# charuco measurements provide direct positional estimates to fit the curve to, but we also use the IMU's
# acceleration and pose data, and past (recorded) and future (planned) spool line lengths.

# Consider network with a node for every measurable variable and every variable in the model. (see vars_diagram.jpg)
# there are two kinds of links between nodes.
# 1. directional calculation, where one variable can be computed from the other.
# 2. equality, where the two variables are supposed to represent the same thing, and the error between them should be part of the cost function.

class CDPR_position_estimator:
    def __init__(self, anchor_points, platform_mass, platform_inertia, gravity=9.81):
        """
        Initializes the CDPR (Cable Driven Parallel Robot)
        All units are SI. these vales are assumed to be obtained from the auto calibration step

        The base interval of the splines is always (0,1)
        the base interval could be made to be something like (unix_now-10000, unix_now+10000) but it might cause floating point
        imprecision. when moving forward in time, it would be ideal not to start over again with new control points, but just keep the old ones.
        since they already closely approximate the real motion. So the control points could be made to represent fixed instants in time,
        and the spline evaluated and points further and further to the right, but we would run out of space.
        so, would it be easier for the optimizer if we constant inch all the points forwards, making them represent points in time that
        move relative to the measurements, or should we keep them fixed in time relative to the measurements, occationaly throwing out the oldest
        and making a new point by extrapolating from the spline?

        And a third way - let the control points refer to instants in time that move relative to the measurements, but whenever that time window moves
        set all the control points to some new value by evaluating the curve at points to the right. But control points don't lie on the spline...

		base interval 0-1
        control point 0

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

        # number of control points in position splines
        self.n_grip_pts = 4 # more because it can swing around
        self.n_gant_pts = 3 # less because it's *supposed* to move in a plane

        self.spline_degree = 3 # cublic splines

		# Calling the spline constructor is 23x slower than updating the control points in place.
		# So it is critical that we only create this once and then update the control points in the cost function.
		self.gripper_pos_spline = BSpline(clamped_knot_vector(self.n_grip_pts), control_points_gripper, self.spline_degree)
		self.gantry_pos_spline = BSpline(clamped_knot_vector(self.n_gant_pts), control_points_gantry, self.spline_degree)

        self.weights = np.array([
        	1, # gantry position from charuco
        	1, # gripper position from charuco
        	1, # gripper inertial measurements
        	1, # calculated forces
        	1, # winch line record
        	1, # winch line plan
        	1, # anchor line record
        	1, # anchor line plan
        ])

	def anchor_line_length(self, anchor_index, gantry_pos_spline):
		"""
		Return a function that gives the length that an anchor line would be at a given time, from the gantry position spline
		Not applicable to the gripper winch line

		Args:
			anchor_index: which line to estimate
			gantry_pos_spline: BSpline. the model of the gantry position in time.
		"""

		# evaluate the gantry position spline at a given instant and measure distance to anchor
		return lambda time: np.linalg.norm(self.anchor_points[anchor_index] - gantry_pos_spline(time))

	def calc_spline_accel(self, position_spline):
		"""
		Calculate what the acceleration would be from a position model
		"""
		return position_spline.derivative(2) # amazing

	def winch_line_len(gantry_pos_spline, gripper_pos_spline):
		"""
		Return a function giving the winch line length at a point in time
		"""
		return lambda time: np.linalg.norm(gantry_pos_spline(time) - gripper_pos_spline(time))

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

	def error_meas(self, pos_model_func, position_measurements):
		"""
		Return the mean distance between the given position model and all the position measurements
		(assuming they are pre-filtered to the model time horizon).
		this works both for the position splines and the line length functions.

		TODO: in theory if I express this in a certain way, then autograd can differentiate it for me.

		Args:
			pos_model_func: model (function) that returns an N-dimensional point when evaluated at a time T, such as a BSpline
			position_measurements: An array of shape (n_measurements, N+1) representing measurement time and an N-dimensional point (TXYZ).
				time must be the first element in each row
		"""
		# calculate distance between measured position and predicted position, sum over all measurements.
		total = sum(np.linalg.norm(meas[1:] - pos_model_func(meas[0])) for meas in position_measurements)
		return total / len(position_measurements)

	def clamped_knot_vector(self, n_control_pts, base_interval=(0,1)):
		"""
		Return a clamped knot vector suitable for constructing a cubic spline with n_control_pts
		"""
		np.concatenate(
			[base_interval[0]] * self.spline_degree,
			np.linspace(base_interval[0], base_interval[1], len(control_points_gripper) - self.spline_degree + 1),
			[base_interval[1]] * self.spline_degree
		)

	def cost_function(self, model_parameters, measurements_filtered):
		"""
		Return the total error between a model and measurements as a scalar

		Args:
			model_parameters: the array of numbers that defines the model. all spline control points
			measurements_filtered: dictionary of arrays of measurements, only those within prediction horizon.
			    n_measurements can be different for every array
				
				gantry_position: shape (n_measurements, 4) TXYZ
				gripper_position: shape (n_measurements, 4) TXYZ
				imu_accel: shape (n_measurements, 4) each row TXYZ
				winch_line_record: shape (n_measurements, 2) TL
				winch_line_plan: shape (n_measurements, 2) TL
				anchor_line_record: shape (n_cables, n_measurements, 2) TL
				anchor_line_plan: shape (n_cables, n_measurements, 2) TL
		"""

		if len(model_parameters) != 3 * (self.n_grip_pts + self.n_gant_pts):
			raise ValueError("model_parameters incorrect size")

		# Model parameters are the control points for the position curves.
		# directly update the control points of the bsplines
		gripper_pos_spline.c = model_parameters[:3*self.n_grip_pts].reshape((self.n_grip_pts, 3))
		gantry_pos_spline.c = model_parameters[3*self.n_grip_pts : (3*self.n_grip_pts+3*self.n_gant_pts)].reshape((self.n_gant_pts, 3))

		gantry_accel_func = calc_spline_accel(gripper_pos_spline)
		winch_line_func = winch_line_len(gantry_pos_spline, gripper_pos_spline)
		anchor_line_funcs = [anchor_line_length(i, gantry_pos_spline) for i in range(self.n_cables)]

		errors = np.array([
			# error between gantry position model and observation
			error_meas(gantry_pos_spline, measurements_filtered['gantry_position']),
			# error between gripper position model and observation
			error_meas(gripper_pos_spline,  measurements_filtered['gripper_position']),
			# error between gripper acceleration model and observation
			error_meas(gantry_accel_func,  measurements_filtered['imu_accel']),
			# error between gripper acceleration model and acceleration from calculated forces.
			error_meas(gantry_accel_func, calc_gripper_accel_from_forces()),
			# error between model and recorded winch line lengths
			error_meas(winch_line_func, measurements_filtered['winch_line_record']),
			# error between model and planned winch line lengths
			error_meas(winch_line_func, measurements_filtered['winch_line_plan']),
			# error between model and recorded anchor line lengths for every anchor
			sum([error_meas(anchor_line_funcs[i], measurements_filtered['anchor_line_record'][i]) for i in range(self.n_cables)]) / self.n_cables
			# error between model and planned anchor line lengths for every anchor
			sum([error_meas(anchor_line_funcs[i], measurements_filtered['anchor_line_plan'][i]) for i in range(self.n_cables)]) / self.n_cables
		])

		return sum(errors * self.weights)

	def estimate(self):
		"""
		Find curves that tell us the positions of the gripper and gantry over a fixed time window
		given a particular set of observations, and planned motor movments.
		by minimizing self.cost_function

		later this can be used to find a particular set of planned motor movements that result in the gripper having a
		specific position, and zero velocity, at a specific time, while minimizing motor effort.
		that later cost function would be the sum of
			positional error to the goal point at the future time
			velocity magnitude at the future time
			motor effort

		there appears to be a choice as to whether to combine this cost function with that one and minimize it in one step. not sure what to do there.
		preception and action aren't so different are they. Jeff Hinton would like that.
		"""
		model_size = 3 * (self.n_grip_pts + self.n_gant_pts)

		# Initial guess for control points. assume straight lines in reasonable location on first try
		# or assume spline from last time we called this.
        parameter_initial_guess = np.zeros(model_size)

        result = optimize.minimize(
            self.cost_function,
            parameter_initial_guess,
            args=(measurements_filtered,),
            method='SLSQP', # Suitable for constrained optimization
            bounds=((-10, 10),)*model_size # Example bounds on control inputs. Change to room size with some buffer around it.
        )

        # make splines from optimal control points
        # optimal_control_points = result.x.reshape((self.N, 3))
        # u = U_optimal[0]

        # use splines to calculate position at any point in the time interval.
