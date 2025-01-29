import numpy as np
import scipy.optimize as optimize
from scipy.interpolate import BSpline
from scipy.special import softmax
import scipy.integrate as integrate
from time import time
from observer import Observer

# X, Y are horizontal
# positive Z points at the ceiling.

# rather than assuming the position of the gripper is the last position obtained from a charuco board,
# instead we fit a model defining it's positoin in time to a bunch of measurments from given instants.
# charuco measurements provide direct positional estimates to fit the curve to, but we also use the IMU's
# acceleration and pose data, and past (recorded) and future (planned) spool line lengths.

# Consider network with a node for every measurable variable and every variable in the model. (see vars_diagram.jpg)
# there are two kinds of links between nodes.
# 1. directional calculation, where one variable can be computed from the other.
# 2. equality, where the two variables are supposed to represent the same thing, and the error between them should be part of the cost function.
class CDPR_position_estimator:
    def __init__(self, observer, anchor_points, gravity=9.81):
        """
        Initializes the CDPR (Cable Driven Parallel Robot)
        All units are SI. these vales are assumed to be obtained from the auto calibration step

        The base interval of the splines is always (0,1)

        Args:
            observer: instance of Observer where measurements are stored/collected
            anchor_points: A numpy array of shape (n_cables, 3) representing the 3D coordinates of the cable anchor points.
        """
        self.observer = observer
        self.anchor_points = np.array(anchor_points)
        self.n_cables = self.anchor_points.shape[0]
        self.gripper_mass = 0.4 # kg
        self.gantry_mass = 0.06 # kg
        # self.platform_inertia = np.array(platform_inertia)
        self.gravity = np.array([0,0,-1*gravity])

        # number of control points in position splines
        self.n_ctrl_pts = 4
        # cublic splines
        self.spline_degree = 3

        # these are just initial guesses of the locations
        gant_p = np.mean(self.anchor_points, axis=0) + np.array([0,0,-2])
        control_points_gantry = np.array([gant_p for i in range(self.n_ctrl_pts)])
        grip_p = gant_p + np.array([0,0,-2])
        control_points_gripper = np.array([grip_p for i in range(self.n_ctrl_pts)])
        # a single control point for the motor line plan
        lines = [np.linalg.norm(a, gant_p) for a in self.anchor_points]
        lines.append(np.linalg.norm(gant_p, grip_p))
        # four copies of it become the control points for the plan spline, meaning our initial motor plan is to hold still at this spot
        control_points_lines = np.array([lines for i in range(self.n_ctrl_pts)])

        self.knots = self.clamped_knot_vector(self.n_ctrl_pts)

        # Calling the spline constructor is 23x slower than updating the control points in place.
        # So it is critical that we only create this once and then update the control points in the cost function.
        self.gripper_pos_spline = BSpline(self.knots, control_points_gripper, self.spline_degree, True)
        self.gantry_pos_spline = BSpline(self.knots, control_points_gantry, self.spline_degree, True)

        # this spline represent the motor plan as a path through a space with self.n_cables+1 dimensions
        # only positive values. one cable length per anchor, and one for the winch
        # even though the time domain of this spline is the whole -10 to +10 like the others, only the future part
        # is considered in cost functions
        self.motor_spline = BSpline(self.knots, control_points_lines, self.spline_degree, True)

        lpos = 16 # maximum meters from origin than a position spline control point can be
        mpos = 10 # maximum meters from origin that a motor spline control point can be 
        self.bounds = [(-lpos, lpos)] * self.n_ctrl_pts * 6
        self.bounds.extend([(0, mpos)] * self.n_ctrl_pts * (self.n_cables + 1))

        self.weights = softmax(np.array([
            1, # gantry position from charuco
            1, # gripper position from charuco
            1, # gripper inertial measurements
            1, # calculated forces
            1, # winch line record
            1, # anchor line record
            1, # winch line plan
            1, # anchor line plan
            1, # total energy of gripper
            1, # total energy of gantry
            1, # desired gripper location
        ]))

        now = time()
        self.horizon_s = 10
        self.time_domain = np.array([now - self.horizon_s, now + self.horizon_s])

    def filter_measurements(self):
        """
        return views of measurement arrays that contain only items within self.time_domain
        """

    def anchor_line_length(self, gantry_pos_spline):
        """
        Return a function that gives the lengths of all anchor lines at a given time based on the the gantry position spline
        Not applicable to the gripper winch line

        returns a vector with one term for each anchor line

        Args:
            anchor_index: which line to estimate
            gantry_pos_spline: BSpline. the model of the gantry position in time.
        """

        # evaluate the gantry position spline at a given instant and measure distance to anchor
        return lambda time: np.array([
            np.linalg.norm(self.anchor_points[anchor_index] - gantry_pos_spline(time)) 
            for anchor_index in range(self.n_cables)])

    def winch_line_len(self, gantry_pos_spline, gripper_pos_spline):
        """
        Return a function giving the winch line length at a point in time
        """
        return lambda time: np.linalg.norm(self.gantry_pos_spline(time) - self.gripper_pos_spline(time))

    def calc_gripper_accel_from_forces(self, steps):
        """
        Calculate the acceleration on the gripper based on the forces we would expect it to experience due to
        it being a pendulum hanging from a moving object

        two forces act on it
        gravity acts to accelerate the mass down.
        a tension force acts in the direction of the gantry equal to the component of gravity opposite that direction.

        Args:
            steps: number of discrete points at which to measure forces.
        """
        gantry_accel_func = self.gripper_pos_spline.derivative(2)
        results = []

        for time in np.linspace(0,1,steps):
            gant_pos = self.gantry_pos_spline(time)
            grip_pos = self.gripper_pos_spline(time)

            # normalized direction vector from gantry pointing towards gripper
            direction = np.linalg.norm(grip_pos-gant_pos)
            # component of gravity pointing in that direction * -1
            tension_acc = np.dot(self.gravity, direction) * -1
            # mass cancelled out of this equation.

            results.append(np.concatenate([[time], tension_acc + self.gravity]))
        return np.array(results)

    def mechanical_energy(self, position_spline, mass_kg):
        """
        return a function that gives the kinetic energy + potential energy of an object over the model window.
        The function returns a scalar in Joules
        """
        velocity = position_spline.derivative(1)
        return lambda t: mass_kg * (
            0.5 * np.linalg.norm(velocity(t)) ** 2 # kinetic energy
            + 9.81 * position_spline(t)[2] # potential energy
        )

    def clamped_knot_vector(self, n_ctrl_pts, base_interval=(0,1)):
        """
        Return a clamped knot vector suitable for constructing a cubic spline with n_ctrl_pts
        """
        return np.concatenate([
            [base_interval[0]] * self.spline_degree,
            np.linspace(base_interval[0], base_interval[1], n_ctrl_pts - self.spline_degree + 1),
            [base_interval[1]] * self.spline_degree
        ])

    def set_splines_from_params(self, params):
        grip_pos_model_size = self.n_ctrl_pts * 3
        gant_pos_model_size = self.n_ctrl_pts * 3
        motors_model_size = self.n_ctrl_pts * (self.n_cables + 1)
        if len(params) != (grip_pos_model_size + gant_pos_model_size + motors_model_size):
            raise ValueError("position model_parameters incorrect size")
        # Model parameters are the control points for the position curves.
        # directly update the control points of the bsplines
        start = 0
        self.gripper_pos_spline.c = params[start : grip_pos_model_size].reshape((self.n_ctrl_pts, 3))
        start += grip_pos_model_size
        self.gantry_pos_spline.c = params[start : start + gant_pos_model_size].reshape((self.n_ctrl_pts, 3))
        start += gant_pos_model_size
        self.motor_spline.c = params[start : start + motors_model_size].reshape((self.n_ctrl_pts, self.n_cables + 1))

    def model_time(self, t):
        """
        Convert a floating point number of seconds since the epoch into a time relative to the base interval of the model splines.
        assumes base interval is (0,1)
        """
        return (t - self.time_domain[0]) / (self.time_domain[1] - self.time_domain[0])

    def error_meas(self, pos_model_func, position_measurements, normalize_time=True):
        """
        Return the mean distance between the given position model and all the position measurements in the given array
        this works both for the position splines and the line length functions.

        TODO: in theory if I express this in a certain way, then autograd can differentiate it for me.

        Args:
            pos_model_func: model (function) that returns an N-dimensional point when evaluated at a time T, such as a BSpline
            position_measurements: An array of shape (n_measurements, N+1) representing measurement time and an N-dimensional point (TXYZ).
                time must be the first element in each row
                time is a floating point number of seconds since the epoch
        """
        # calculate distance between measured position and predicted position, sum over all measurements.
        times = position_measurements[:,0]
        if normalize_time:
            times = list(map(self.model_time, times))
        expected = np.array(list(map(pos_model_func, times)))
        total = sum(np.linalg.norm(position_measurements[:,1:] - expected, axis=0))
        return total / len(position_measurements)

    def cost_function(self, model_parameters):
        """
        Return the total error between a model and measurements as a scalar

        Args:
            model_parameters: the array of numbers that defines the model. all spline control points
        """
        self.set_splines_from_params(model_parameters)
        
        gantry_accel_func = self.gripper_pos_spline.derivative(2)
        winch_line_func = self.winch_line_len(self.gantry_pos_spline, self.gripper_pos_spline)
        anchor_line_func = self.anchor_line_length(self.gantry_pos_spline)
        steps = 100
        # create discrite approximations of the motor line plan functions in the same format as measurements
        # TODO rather than using now (0) as the starting point of where we eval the plan, maybe use a point slightly in the past.
        winch_line_plan = np.array([[t, self.motor_spline(t)[-1]] for t in np.linspace(0, self.time_domain[1], steps)])
        anchor_line_plan = np.array([[t, self.motor_spline(t)[:-1]] for t in np.linspace(0, self.time_domain[1], steps)])

        errors = np.array([
            # error between gantry position model and observation
            self.error_meas(self.gantry_pos_spline, self.observer.gantry_position.arr),
            # error between gripper position model and observation
            self.error_meas(self.gripper_pos_spline,  self.observer.gripper_position.arr),
            # error between gripper acceleration model and observation
            self.error_meas(gantry_accel_func,  self.observer.imu_accel.arr),
            # error between gripper acceleration model and acceleration from calculated forces.
            self.error_meas(gantry_accel_func, self.calc_gripper_accel_from_forces(steps=steps), normalize_time=False),
            # error between model and recorded winch line lengths
            self.error_meas(winch_line_func, self.observer.winch_line_record.arr),
            # error between model and recorded anchor line lengths
            # TODO have one measurement array for each because records will come in at different instants for each line
            self.error_meas(anchor_line_func, self.observer.anchor_line_record.arr),
            # error between model and planned winch line lengths
            self.error_meas(winch_line_func, winch_line_plan, normalize_time=False),
            # error between model and planned anchor line lengths
            self.error_meas(anchor_line_func, anchor_line_plan, normalize_time=False),
            # integral of the mechanical energy of the moving parts from now till the end in Joule*seconds
            integrate.quad(self.mechanical_energy(gripper_pos_spline, self.gripper_mass), 0, self.horizon_s),
            integrate.quad(self.mechanical_energy(gantry_pos_spline, self.gantry_mass), 0, self.horizon_s),
            # error between position model and desired future locations
            self.error_meas(self.gripper_pos_spline, self.observer.gripper_position_desired.arr),
            
            # penalty for pulling the motors against eachother by raising the gantry too high
            # penalty for letting the gripper touch the floor
            # penalty for letting the gantry touch an imaginary plane 6 feet from the ground (heuristic for dodging furniture)
            # penalty for exceeding max gripper acceleration since it could shake loose the payload
            # penalty for unspooling so fast you make a birdsnest
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
        model_size = (self.n_cables + 1 + 6) * self.n_ctrl_pts

        # Initial guess for control points. Always pick up where we left off
        parameter_initial_guess = np.concatenate([
            self.gripper_pos_spline.c.copy().reshape(-1),
            self.gantry_pos_spline.c.copy().reshape(-1),
            self.motor_spline.c.copy().reshape(-1),
        ])

        start = time()
        result = optimize.minimize(
            self.cost_function,
            parameter_initial_guess,
            method='SLSQP', # Suitable for constrained optimization
            bounds=self.bounds
        )
        print(f"minimization step took {time() - start} seconds")

        # set splines from optimal model params
        self.set_splines_from_params(result.x)

        # now you can use splines to calculate position at any point in the time interval, such as this instant.
        normalized_time = self.model_time(time())
        return self.gripper_pos_spline(normalized_time)

    def move_spline_domain_fast(self, spline, offset):
        """
        Move the domains of the splines in self forward by the given offset by adding the offset to their knots
        you can't do this forever because it gets inaccurate after the domain gets far away from zero

        TODO: keep track of base interval for time calc
        """
        spline.t += offset

    def move_spline_domain_robust(self, spline, offset):
        """
        Move the domains of the splines in self forward by the given offset by transforming the control points
        by a matrix that causes the new spline to equal the old spline in the sampled region.
        offset must be positive
        the base interval remains 0-1
        """
        num_control_points = len(spline.c)
        num_eval_points = num_control_points * 10 # Higher than control points for better accuracy

        eval_points = np.linspace(0, 1-offset, num_eval_points)
        eval_matrix = BSpline.design_matrix(eval_points, self.knots, self.spline_degree)
        
        shifted_eval_points = eval_points + offset
        shifted_eval_matrix = BSpline.design_matrix(shifted_eval_points, self.knots, self.spline_degree)

        # Use least squares because eval_matrix is not square
        new_control_points = np.linalg.lstsq(eval_matrix.todense(), shifted_eval_matrix @ spline.c, rcond=None)[0]

    def move_to_present(self):
        """
        Move the spline domains, and the measurement time domain to be centered at the present.
        """
        old = self.time_domain[0] + self.horizon_s
        now = time()
        self.time_domain = np.array([now - self.horizon_s, now + self.horizon_s])
        domain_offset = (now - old) / (self.horizon_s * 2)

        # if the knots are near 0
        #   move_spline_domain_fast(self.gripper_pos_spline, domain_offset)
        #   move_spline_domain_fast(self.gantry_pos_spline, domain_offset)
        # else:
        self.move_spline_domain_robust(self.gripper_pos_spline, domain_offset)
        self.move_spline_domain_robust(self.gantry_pos_spline, domain_offset)
        self.move_spline_domain_robust(self.motor_spline, domain_offset)
