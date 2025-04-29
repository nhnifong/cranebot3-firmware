import numpy as np
import scipy.optimize as optimize
from scipy.interpolate import BSpline
from scipy.special import softmax
from time import time
from data_store import DataStore
from cv_common import compose_poses
import model_constants
from functools import partial
import asyncio
import signal
import sys
from math import pi
from config import Config
# import cProfile

default_weights = np.array([
    6.0, # gantry position from charuco
    6.0, # gripper position from charuco
    0.0, # gripper inertial measurements
    6.0, # calculated forces
    9.0, # winch line record
    4.0, # anchor line record
    0.75, # total energy of gripper
    0.75, # total energy of gantry
    5.0, # desired gripper location
    0.45, # spline jolt
    1.0, # ideal gantry height
    1.0, # spool_speed_limit
    1.0, # winch_speed_limit
    0.2, # gripper peak accel
    0.2, # gantry peak accel
])

weight_names = [
    'gantry markers',
    'gripper markers',
    'gripper IMU',
    'calculated forces',
    'winch line',
    'anchor lines',
    'gripper energy',
    'gantry energy',
    'goal locations',
    'spline_jolt',
    'ideal_gantry_height',
    'spool_speed_limit',
    'winch_speed_limit',
    'gripper_peak_accel',
    'gantry_peak_accel',
]

# the ideal gantry height is heigh enough not to hit you in the head, but otherwise as low as possible to maximize payload capacity
ideal_gantry_height = 1.75 # meters

# maximum line length speed we can command from the anchor motor
max_line_speed = 0.6234 # m/s
# maximum line speed we can command from the winch motor
max_winch_speed = (61/60)*(0.02*pi) # m/s

def find_intersection(positions, lengths):
    """Triangulation by least squares
    returns scipy result object with .success and .x
    """
    # this code may benefit from some noise
    noise = np.random.normal(0, 1e-6, positions.shape)
    positions = positions + noise
    # Initial guess for the intersection point (e.g., the mean of the positions)
    initial_guess = np.mean(positions, axis=0)
    initial_guess[2] -= 1

    def error_function(intersection, positions, lengths):
        distances = np.linalg.norm(positions - intersection, axis=1)
        errors = distances - lengths
        return errors

    return optimize.least_squares(error_function, initial_guess, args=(positions, lengths))

def get_simplified_position(datastore, anchor_positions):
    """
    Calculate a gantry position based solely on the last line record from each anchor and the anchor positions
    """
    lengths = []
    for i, alr in enumerate(datastore.anchor_line_record):
        lengths.append(alr.getLast()[1])
    if sum(lengths) == 0:
        return [0,0,0], False
    anchor_positions = np.array(anchor_positions)
    lengths = np.array(lengths)
    result = find_intersection(anchor_positions, lengths)
    position = [0,0,0]
    if result.success:
        position = result.x
    return position, result.success

# X, Y are horizontal
# positive Z points at the ceiling.

# rather than assuming the position of the gripper is the last position obtained from a charuco board,
# instead we fit a model defining it's position in time to a bunch of measurments from given instants.
# charuco measurements provide direct positional estimates to fit the curve to, but we also use the IMU's
# acceleration and pose data, and past (recorded) and future (planned) spool line lengths.

# Consider network with a node for every measurable variable and every variable in the model. (see vars_diagram.jpg)
# there are two kinds of links between nodes.
# 1. directional calculation, where one variable can be computed from the other.
# 2. equality, where the two variables are supposed to represent the same thing, and the error between them should be part of the cost function.
class CDPR_position_estimator:
    def __init__(self, datastore, to_ui_q, to_pe_q, to_ob_q):
        """
        Initializes the CDPR (Cable Driven Parallel Robot)
        All units are SI. these vales are assumed to be obtained from the auto calibration step

        The base interval of the splines is always (0,1)

        Args:
            datastore: instance of DataStore where measurements are stored/collected
            anchor_points: A numpy array of shape (n_cables, 3) representing the 3D coordinates of the cable anchor points.
        """
        self.run = True # run main loop
        self.datastore = datastore
        self.to_ui_q = to_ui_q
        self.to_pe_q = to_pe_q
        self.to_ob_q = to_ob_q
        self.snapshot = {}
        self.n_cables = self.datastore.n_cables
        self.config = Config()
        self.anchor_points = np.array([
            [-2,2, 3],
            [ 2,2, 3],
            [ -1,2,-2],
            [ -2,2,-2],
        ], dtype=float)
        for i, a in enumerate(self.config.anchors):
            self.anchor_points[i] = np.array(compose_poses([a.pose, model_constants.anchor_grommet])[0])
        
        self.gripper_mass = 0.4 # kg
        self.gantry_mass = 0.06 # kg
        self.rough_gripper_speed = 0.7 # m/s
        # self.platform_inertia = np.array(platform_inertia)
        self.gravity = np.array([0,0,-9.81])

        # number of control points in position splines
        self.n_ctrl_pts = 12
        # cublic splines
        self.spline_degree = 3

        # these are just initial guesses of the locations
        gant_p = np.mean(self.anchor_points, axis=0) + np.array([0,0,-0.2])
        control_points_gantry = np.array([gant_p for i in range(self.n_ctrl_pts)])
        grip_p = gant_p + np.array([0,0,-0.2])
        control_points_gripper = np.array([grip_p for i in range(self.n_ctrl_pts)])

        self.knots = self.clamped_knot_vector(self.n_ctrl_pts)
        self.to_ui_q.put({
            'knots': self.knots,
            'spline_degree': self.spline_degree,
        })

        # Calling the spline constructor is 23x slower than updating the control points in place.
        # So it is critical that we only create this once and then update the control points in the cost function.
        self.gripper_pos_spline = BSpline(self.knots, control_points_gripper, self.spline_degree, True)
        self.gantry_pos_spline = BSpline(self.knots, control_points_gantry, self.spline_degree, True)
        self.gantry_velocity = self.gantry_pos_spline.derivative(1)
        self.gripper_velocity = self.gripper_pos_spline.derivative(1)
        self.gantry_accel_func = self.gantry_velocity.derivative(1)
        self.gripper_accel_func = self.gripper_velocity.derivative(1)

        self.bounds = np.concatenate([np.array([(-3,3),(-3,3),(0,2)]) for i in range(self.n_ctrl_pts*2)])
        self.weights = 2**(default_weights-5)

        now = time()
        self.horizon_s = self.datastore.horizon_s
        self.time_domain = (now - self.horizon_s, now + self.horizon_s)

        # precalcualate a regular array of times from 0 till horizon_s for evaluating motor positions
        # in the time domain of the model splines. the domain is 0-1, and we want only the right half, representing the future.
        bot_loop_freq = 30 # hz
        self.future_times = np.linspace(0.5, 1, self.horizon_s * 4)
        self.des_grip_locations = np.array([])

        self.jolt_time = now+1
        self.gantry_jolt_pos = self.gantry_pos_spline(self.model_time(self.jolt_time))
        self.last_intermediate_update = now
        self.time_taken = 1

        # number of steps to use when evaluating certain functions
        self.steps = 100
        # model times at which to evaluate the functions, concentrated near the present
        self.times = np.linspace(-1,1,self.steps)**3 * 0.5 + 0.5


    def anchor_line_length(self, anchor_index, times):
        """
        Return the length of an anchor line at a given array of times based on the the gantry position spline
        Not applicable to the gripper winch line

        times shoud be in the base interval of the model splines

        returns an array of scalars for the line length. one for each time

        Args:
            anchor_index: which line to estimate
            gantry_pos_spline: BSpline. the model of the gantry position in time.
        """

        # evaluate the gantry position spline at every instant in the times aray and measure distance to anchor
        return np.linalg.norm(self.anchor_points[anchor_index] - self.gantry_pos_spline(times), axis=1)

    def winch_line_len(self, times):
        """
        Return the winch line lengths at an array of points in time
        time shoud be in the base interval of the model splines
        """
        return np.linalg.norm(self.gantry_pos_spline(times) - self.gripper_pos_spline(times), axis=1, keepdims=True)

    def calc_gripper_accel_from_forces(self, gant_positions, grip_positions):
        """
        Approximate the expected acceleration on the gripper based on the forces it should experience due to
        it being a pendulum hanging from a moving object
        """
        # vector point from gripper towards gantry
        directions = gant_positions - grip_positions
        # line length
        magnitudes = np.linalg.norm(directions, axis=1, keepdims=True)
        # Replace zeros with a small value
        magnitudes[magnitudes == 0] = 1e-8 
        # null out z axis. The veritical component is completely ignored for now.
        directions[:,2] = 0
        # assume we accelerate towards the point under the gantry
        # the longer the rope, the slower we move.
        accel = directions * (1/np.sqrt(magnitudes / 9.81))**2
        # prepend times
        return np.concatenate([self.times[:, np.newaxis], accel], axis=1)

    def kinetic_energy_vectorized(self, velocity_values, mass_kg):
        """
        Return a function that gives the kinetic energy of an object at an array of times.
        provide precalculated velocities to avoid calculating them again
        """
        return np.mean(0.5 * mass_kg * np.linalg.norm(velocity_values, axis=1) ** 2)

    def spline_jolt(self):
        """Measure the amount of distance that the gantry would move if the last spline were
        updated with the current one at self.jolt_time."""
        return np.linalg.norm(self.gantry_jolt_pos - self.gantry_pos_spline(self.model_time(self.jolt_time)))

    def gantry_height_penalty(self, gantry_position_values):
        return np.sum((gantry_position_values[:,2] - ideal_gantry_height)**2)

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
        spline3d_model_size = self.n_ctrl_pts * 3
        if len(params) != (spline3d_model_size * 2): # we have 2 splines
            raise ValueError("position model_parameters incorrect size.")
        # Model parameters are the control points for the position curves.
        # directly update the control points of the bsplines
        start = 0
        self.gripper_pos_spline.c = params[start : spline3d_model_size].reshape((self.n_ctrl_pts, 3))
        start += spline3d_model_size
        self.gantry_pos_spline.c = params[start : start + spline3d_model_size].reshape((self.n_ctrl_pts, 3))
        self.gantry_velocity = self.gantry_pos_spline.derivative(1)
        self.gripper_velocity = self.gripper_pos_spline.derivative(1)
        # this is faster than running derivative(2) on pos spline
        self.gantry_accel_func = self.gantry_velocity.derivative(1)
        self.gripper_accel_func = self.gripper_velocity.derivative(1)

    def model_time(self, times):
        """
        Convert a floating point number of seconds since the epoch into a time relative to the base interval of the model splines.
        assumes base interval is (0,1)

        pass an array of times. if you have only one pass [t]
        """
        time_domain_diff = self.time_domain[1] - self.time_domain[0]
        return (times - self.time_domain[0]) / time_domain_diff

    def unix_time(self, times):
        """
        Convert a time relative to the base interval of the model splines into a floating point number of seconds since the epoch
        assumes base interval is (0,1)

        pass an array of times. if you have only one pass [t]
        """
        time_domain_diff = self.time_domain[1] - self.time_domain[0]
        return times * time_domain_diff + self.time_domain[0]

    def excessive_speed_penalty(self, positions, velocities):
        """ Return a scalar representing the total penalty for exceeding the speed limit on any anchor line

        positions is a pre-evaluated vector of the gantry position at self.times.
        velocities is a pre-evaluated vector of the gantry velocities at self.times.
        """

        # Calculate direction vectors: (100, 4, 3)
        directions = self.anchor_points[np.newaxis, :, :] - positions[:, np.newaxis, :]
        # Calculate direction norms: (100, 4)
        direction_norms = np.linalg.norm(directions, axis=2)
        # Handle zero norms to prevent division by zero
        direction_norms[direction_norms == 0] = 1e-8  # Replace zeros with a small value
        # Normalize direction vectors: (100, 4, 3)
        direction_units = directions / direction_norms[:, :, np.newaxis]
        # Calculate dot products (velocity projection): (100, 4)
        direction_units = np.transpose(direction_units, (0, 2, 1))
        magnitudes = np.einsum('ij,ijk->ik', velocities, direction_units)
        speeds = np.abs(magnitudes).flatten()
        # apply a nonlinear function that rises sharply above the speed limit
        # penalties = np.where(speeds < max_line_speed, 0, 5000 * (speeds - max_line_speed)**2)
        return np.sum(speeds**2)

    def winch_speed_penalty(self, gripper_positions, gantry_positions, gripper_velocities, gantry_velocities):
        # Calculate the relative position vector
        relative_position = gantry_positions - gripper_positions
        # Calculate the relative velocity vector
        relative_velocity = gantry_velocities - gripper_velocities
        # Calculate the normalized relative position vector.
        relative_position_norm = np.linalg.norm(relative_position, axis=1, keepdims=True)
        # Replace zeros with a small value
        relative_position_norm[relative_position_norm == 0] = 1e-8 
        relative_position_unit = relative_position / relative_position_norm
        # Calculate the relative closing speed.
        closing_speed = np.sum(relative_velocity * relative_position_unit, axis=1)
        speeds = np.abs(closing_speed)
        # apply a nonlinear function that rises sharply above the speed limit
        # penalties = np.where(speeds < max_winch_speed, 0, 5000 * (speeds - max_winch_speed)**2)
        return np.sum(speeds**2)

    def peak_acceleration(self, accel_spline, times):
        return np.max(np.linalg.norm(accel_spline(times), axis=1)) * 0.01

    def error_meas(self, pos_model_func, position_measurements, normalize_time=False, pr=False):
        """
        Return the mean distance between the given position model and all the position measurements in the given array
        this works both for the position splines and the line length functions.

        Args:
            pos_model_func: model (function) that returns an N-dimensional point when evaluated at a time T, such as a BSpline
                the function must support vectorized evaluation
            position_measurements: An array of shape (n_measurements, N+1) representing measurement time and an N-dimensional point (TXYZ).
                time must be the first element in each row
                time is a floating point number of seconds since the epoch
        """
        if position_measurements.shape[0] == 0:
            return 0.0

        # convert array of unix times to array of model times if necessary
        times = position_measurements[:,0]
        if normalize_time:
            time_domain_diff = self.time_domain[1] - self.time_domain[0]
            times = (times - self.time_domain[0]) / time_domain_diff

        # calculate distance between measured position and predicted position, sum over all measurements.
        expected = pos_model_func(times)
        if len(expected.shape) == 1:
            expected = expected.reshape(-1,1)
        distances = np.linalg.norm(position_measurements[:, 1:] - expected, axis=1)
        if pr:
            print(f'measured={position_measurements[:, 1:]}')
            print(f'expected={expected} distances={distances}')
        return np.mean(distances)

    def cost_function(self, model_parameters, p=False, store=False):
        """
        Return the total error between a model and measurements as a scalar

        Args:
            model_parameters: the array of numbers that defines the model. all spline control points
        """
        self.set_splines_from_params(model_parameters)
        
        gantry_positions = self.gantry_pos_spline(self.times)  # Vectorized spline evaluation
        grip_positions = self.gripper_pos_spline(self.times)  # Vectorized spline evaluation

        gantry_vel = self.gantry_velocity(self.times)
        gripper_vel = self.gripper_velocity(self.times)

        mid = int(self.steps/2)

        errors = np.array([
            # error between gantry position model and observation
            self.error_meas(self.gantry_pos_spline, self.snapshot['gantry_position']),
            # error between gripper position model and observation
            self.error_meas(self.gripper_pos_spline,  self.snapshot['gripper_position']),
            # error between gripper acceleration model and observation
            0,#self.error_meas(self.gripper_accel_func,  self.snapshot['imu_accel']),
            # error between gripper acceleration model and acceleration from calculated forces.
            self.error_meas(self.gripper_accel_func, self.calc_gripper_accel_from_forces(gantry_positions, grip_positions)),
            # error between model and recorded winch line lengths
            self.error_meas(self.winch_line_len, self.snapshot['winch_line_record']),
            # error between model and recorded anchor line lengths
            sum([self.error_meas(partial(self.anchor_line_length, anchor_num), self.snapshot['anchor_line_record'][anchor_num])
                for anchor_num in range(self.n_cables)]) / self.n_cables,
            # integral of the kinetic energy of the moving parts from now till the end in Joule*seconds
            0,#self.kinetic_energy_vectorized(gripper_vel, self.gripper_mass),
            0,#self.kinetic_energy_vectorized(gantry_vel, self.gantry_mass),
            # error between position model and desired future locations
            self.error_meas(self.gripper_pos_spline, self.des_grip_locations, normalize_time=True),
            # minimize abrupt change from last spline to this one at the point in time when the minimization step is expected to finish
            self.spline_jolt(),
            # minimize squared distance from ideal gantry height.
            self.gantry_height_penalty(gantry_positions[mid:]),
            # penalty for exceeding maximum spool speed. Evaluate only future positions
            self.excessive_speed_penalty(gantry_positions[mid:], gantry_vel[mid:]),
            # pentalty for excessive winch speed
            self.winch_speed_penalty(grip_positions[mid:], gantry_positions[mid:], gripper_vel[mid:], gantry_vel[mid:]),
            # minimize acceleration of gripper
            self.peak_acceleration(self.gripper_accel_func, self.times),
            # minimize acceleration of gantry
            self.peak_acceleration(self.gantry_accel_func, self.times),


            # penalty for exceeding minimum or maximum line length
            # penalty for gripper diving under the floor.
        ])

        if p:
            print(errors)
        if store:
            self.errors = errors

        s = np.sum(errors * self.weights)+0.01
        return s

    def snapshot_datastore(self):
        """
        Make a snapshot of the arrays in the datastore.

        If this ends up being a performance problem, we could add some bookeeping to the circular arrays to just copy the dirty parts.
        Note that the calls to deepCopy will grab the semaphore for each array during its copy, blocking any thread in the observer process that
        may have a measurement to write to it.
        """
        gantry_pose = self.datastore.gantry_pose.deepCopy()
        gripper_pose = self.datastore.gripper_pose.deepCopy()
        self.snapshot = {
            'gantry_position': gantry_pose[:,[0,4,5,6]],
            'gripper_position': gripper_pose[:,[0,4,5,6]],
            'imu_accel': self.datastore.imu_accel.deepCopy(),
            'winch_line_record': self.datastore.winch_line_record.deepCopy()[:,[0,1]],
            'anchor_line_record': [a.deepCopy()[:,[0,1]] for a in self.datastore.anchor_line_record]
        }
        # convert unix times on all measurements to model times.
        # time domain will be fixed for the duration of this estimate
        time_domain_diff = self.time_domain[1] - self.time_domain[0]
        for key, arr in self.snapshot.items():
            if key=='anchor_line_record':
                for anchor_array in arr:
                    anchor_array[:,0] = self.model_time(anchor_array[:,0])
            else:
                arr[:,0] = self.model_time(arr[:,0])

    def estimate(self):
        """
        Find curves that tell us the positions of the gripper and gantry over a fixed time window
        by minimizing self.cost_function
        """
        self.move_to_present()
        model_size = 6 * self.n_ctrl_pts

        # Initial guess for control points. Always pick up where we left off
        parameter_initial_guess = np.concatenate([
            self.gripper_pos_spline.c.copy().reshape(-1),
            self.gantry_pos_spline.c.copy().reshape(-1),
        ])

        self.snapshot_datastore()
        self.start = time()
        self.des_grip_locations = self.desired_gripper_positions()

        # SLSQP
        # COBYLA
        # L-BFGS-B
        # Powell
        # Nelder-Mead
        result = optimize.minimize(
            self.cost_function,
            parameter_initial_guess,
            method='SLSQP',
            bounds=self.bounds,
            # callback=self.intermediate_result,
            options={
                'disp': False,
                'maxiter': 120,
                # 'ftol':0.0001, # Precision goal for the value of f in the stopping criterion.
                #'eps':0.2, # Step size used for numerical approximation of the Jacobian.
                #'finite_diff_rel_step':, # the relative step size to use for numerical approximation of jac
            },
        )
        self.time_taken = time() - self.start
        try:
            if not result.message.startswith("Iteration limit reached"):
                assert result.success
        except AssertionError:
            print(result)
            return

        if result.nit == 1:
            print('reached minimum in one iteration, this is usually garbage')
            return

        # set splines from optimal model params
        self.set_splines_from_params(result.x)

        # print errors
        self.cost_function(result.x, store=True)

        # Where will the gantry be at the point when the next minimization step finishes?
        self.jolt_time = time() + self.time_taken
        self.gantry_jolt_pos = self.gantry_pos_spline(self.model_time(self.jolt_time))

        unix_times = self.unix_time(self.future_times)

        # evaluate line lengths in the future and put them in a queue for immediate transmission to the robot
        future_anchor_lines = np.array([(unix_times, self.anchor_line_length(anchor, self.future_times))
            for anchor in range(self.n_cables)])
        future_winch_line = np.column_stack((unix_times, self.winch_line_len(self.future_times)))

        update_for_observer = {
            'future_anchor_lines': {'sender':'pe', 'data':future_anchor_lines},
            'future_winch_line': {'sender':'pe', 'data':future_winch_line},
        }
        self.to_ob_q.put(update_for_observer)

        # send control points of position splines to UI for visualization
        update_for_ui = {
            'time_domain': self.time_domain,
            'gripper_path': self.gripper_pos_spline.c,
            'gantry_path': self.gantry_pos_spline.c,
            'minimizer_stats': {
                'errors': softmax(self.errors),
                'data_ts': np.max(self.snapshot['gantry_position'][:,0]), # timestamp of last observed gantry position
                'time_taken': self.time_taken,
            },
            'goal_points': self.des_grip_locations, # each goal is a time and a position
        }
        self.to_ui_q.put(update_for_ui)

    def intermediate_result(self, params):
        now = time()
        if now - self.last_intermediate_update > 0.15:
            self.last_intermediate_update = now
            sp_size = self.n_ctrl_pts * 3
            update_for_ui = {
                'gripper_path': params[0 : sp_size].reshape((self.n_ctrl_pts, 3)),
                'gantry_path': params[sp_size : sp_size*2].reshape((self.n_ctrl_pts, 3)),
            }
            self.to_ui_q.put(update_for_ui)

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
        spline.c = new_control_points

    def move_to_present(self):
        """
        Move the spline domains, and the measurement time domain to be centered at the present.
        """
        old = self.time_domain[0] + self.horizon_s
        now = time()
        self.time_domain = (now - self.horizon_s, now + self.horizon_s)
        domain_offset = (now - old) / (self.horizon_s * 2)

        # if the knots are near 0
        #   move_spline_domain_fast(self.gripper_pos_spline, domain_offset)
        #   move_spline_domain_fast(self.gantry_pos_spline, domain_offset)
        # else:
        self.move_spline_domain_robust(self.gripper_pos_spline, domain_offset)
        self.move_spline_domain_robust(self.gantry_pos_spline, domain_offset)

    def desired_gripper_positions(self):
        """
        Return a list of one or more future desired gripper positions
        """
        desired_positions = []
        linger = 2.00 # seconds to hover over bin or object
        # starting at the clock read before scikit.minimize was called.
        # using time() directly here would make the error function nondeterministic, which many solvers seem to hate.
        t = self.start
        # starting with the highest priority item
        item_index = 0
        # which we may already be holding
        holding = self.holding_something_now()
        while t < self.time_domain[1]:
            if holding:
                # you want to be over the bin for the thing you're holding
                destination = self.gripper_over_bin_location()
            else:
                # you want to be over the next target item
                destination = self.item_priority_list(item_index)
                item_index += 1
            # at a point in the future (distance to the destination) / rough_gripper_speed seconds from now
            travel_time = np.linalg.norm(destination - self.gripper_pos_spline(self.model_time(t))) / self.rough_gripper_speed
            # print(f'would travel to {destination} in {travel_time} seconds')
            t += travel_time
            desired_positions.append(np.concatenate([[t], destination], dtype=float))
            # and also remaining at the destination at a point in the future after a fixed lingering period, if it's still within our time domain
            t += linger
            # desired_positions.append(np.concatenate([[t], destination], dtype=float))
            # assume we will drop/pick up the item at this time.
            # we cannot know whether we will succeed, but have to assume we will for planning
            holding = not holding
        return np.array(desired_positions)

    def holding_something_now(self):
        return True

    def item_priority_list(self, idx):
        pl = np.array([ [-1.5, 0, 0.2],
                        [ 0.9, 1, 0.1],
                        [ -0.8, -1, 0.4],
                        [ 2.9, -0.5, 0.2],
                        [ 2.1, 0.5, 0.1],
                        [ -1.2, 2, 0.1],
                        [ 0.5, 1,0.3]])
        return pl[idx%3]

    def gripper_over_bin_location(self):
        return np.array([0,0.2,1])

    def read_input_queue(self):
        while self.run:
            try:
                update = self.to_pe_q.get()
                if 'anchor_pose' in update:
                    apose = update['anchor_pose']
                    anchor_num = apose[0]
                    print(f'updating the position of anchor {anchor_num} to {apose[1][1]}')
                    self.anchor_points[anchor_num] = np.array(apose[1][1])
                if 'STOP' in update:
                    print("stop running")
                    self.run = False
                    break
                if 'weight_change' in update:
                    idx, val = update['weight_change']
                    self.weights[idx] = 2**(val-5)
            except Exception as e:
                self.run = False
                raise e

    async def main(self):
        read_queue_task = asyncio.create_task(asyncio.to_thread(self.read_input_queue))
        await asyncio.sleep(5)
        print('Starting position estimator')
        while self.run:
            try:
                self.estimate()
                # cProfile.runctx('self.estimate()', globals(), locals())
                # some sleep is necessary or we will not receive updates
                rem = (0.25 - self.time_taken)
                await asyncio.sleep(max(0.02, rem))
            except KeyboardInterrupt:
                print('Exiting')
                return
        result = await read_queue_task

def start_estimator(shared_datastore, to_ui_q, to_pe_q, to_ob_q):
    """
    Entry point to be used when starting this from main.py with multiprocessing
    """
    pe = CDPR_position_estimator(shared_datastore, to_ui_q, to_pe_q, to_ob_q)
    asyncio.run(pe.main())


if __name__ == "__main__":
    from multiprocessing import Queue
    from data_store import DataStore
    datastore = DataStore(horizon_s=10, n_cables=4)
    to_ui_q = Queue()
    to_pe_q = Queue()
    to_ob_q = Queue()

    # without this the program has a chance of blocking on exit
    # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue.cancel_join_thread
    to_ui_q.cancel_join_thread()
    to_pe_q.cancel_join_thread()
    to_ob_q.cancel_join_thread()

    # when running as a standalone process (debug only, linux only), register signal handler
    def stop():
        print("\nWait for clean shutdown")
        to_pe_q.put({'STOP':None})
    async def main():
        pe = CDPR_position_estimator(datastore, to_ui_q, to_pe_q, to_ob_q)
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(getattr(signal, 'SIGINT'), stop)
        await pe.main()
    asyncio.run(main())