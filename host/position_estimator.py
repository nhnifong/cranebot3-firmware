import numpy as np
import scipy.optimize as optimize
from scipy.interpolate import BSpline
from scipy.special import softmax
import scipy.integrate as integrate
from time import time
from data_store import DataStore
from calibration import compose_poses
from functools import partial
import threading

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
    def __init__(self, datastore, min_to_ui_q, to_pe_q, to_ob_q, gravity=9.81):
        """
        Initializes the CDPR (Cable Driven Parallel Robot)
        All units are SI. these vales are assumed to be obtained from the auto calibration step

        The base interval of the splines is always (0,1)

        Args:
            datastore: instance of DataStore where measurements are stored/collected
            anchor_points: A numpy array of shape (n_cables, 3) representing the 3D coordinates of the cable anchor points.
        """
        self.datastore = datastore
        self.min_to_ui_q = min_to_ui_q
        self.to_pe_q = to_pe_q # todo, start thread to read this
        self.to_ob_q = to_ob_q
        self.snapshot = {}
        self.n_cables = self.datastore.n_cables
        self.anchor_points = None
        self.loadAnchorPoses()        
        self.gripper_mass = 0.4 # kg
        self.gantry_mass = 0.06 # kg
        self.rough_gripper_speed = 0.5 # m/s
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

        # additional 1d control points for the gripper rotation spline.
        # rather than modelling the gripper rotation as a changing rodrigues vector with a 3D spline,
        # assume it is always pointed along the winch line, and that it's spinning.
        # model only it's angular displacement about the winch line.
        # we have no control over gripper rotation in this axis, but we can see it from the aruco markers,
        # and it might be nice to know what it is when timing finger movement.
        ctrlp_rotation = np.zeros(self.n_ctrl_pts)

        self.knots = self.clamped_knot_vector(self.n_ctrl_pts)
        self.min_to_ui_q.put({
            'knots': self.knots,
            'spline_degree': self.spline_degree,
        })

        # Calling the spline constructor is 23x slower than updating the control points in place.
        # So it is critical that we only create this once and then update the control points in the cost function.
        self.gripper_pos_spline = BSpline(self.knots, control_points_gripper, self.spline_degree, True)
        self.gantry_pos_spline = BSpline(self.knots, control_points_gantry, self.spline_degree, True)
        self.gripper_rot_spline = BSpline(self.knots, ctrlp_rotation, self.spline_degree, True) # 1 dimensional. represents angle only
        self.gantry_accel_func = self.gripper_pos_spline.derivative(2)

        lpos = 16 # maximum meters from origin than a position spline control point can be
        self.bounds = [(-lpos, lpos)] * self.n_ctrl_pts * 7

        self.weights = softmax(np.array([
            1, # gantry position from charuco
            1, # gripper position from charuco
            1, # gripper local z rotation from charuco
            1, # gripper inertial measurements
            1, # calculated forces
            1, # winch line record
            1, # anchor line record
            1, # total energy of gripper
            1, # total energy of gantry
            2, # desired gripper location
        ]))

        now = time()
        self.horizon_s = self.datastore.horizon_s
        self.time_domain = np.array([now - self.horizon_s, now + self.horizon_s])

        # precalcualate a regular array of times from 0 till horizon_s for evaluating motor positions
        # in the time domain of the model splines. the domain is 0-1, and we want only the right half, representing the future.
        bot_loop_freq = 30 # hz
        self.future_times = np.linspace(0.5, 1, self.horizon_s * 4)

    def loadAnchorPoses(self):
        pts = []
        for i in range(self.n_cables):
            try:
                # read calibration data from file
                saved_info = np.load('anchor_pose_%i' % i)
                anchor_pose = tuple(saved_info['pose'])
                pts.append(compose_poses([anchor_pose, model_constants.anchor_grommet])[0])
            except FileNotFoundError:
                pts.append(np.array([0,5,0]))
        self.anchor_points = np.array(pts)

    def gripper_rotation(self, time):
        """
        return rodrigues vector of gripper rotation
        # TODO this is wrong
        """
        # axis of rotation is assumed to be aligned with the winch line.
        rotational_axis = self.gantry_pos_spline(time) - self.gripper_pos_spline(time)
        # the magnitude of the rodrigues vector represents the angle of rotation in radians.
        return rotational_axis / np.linalg.norm(rotational_axis) * self.gripper_rot_spline(time)

    def anchor_line_length(self, anchor_index, time):
        """
        Return the length of an anchor line at a given time based on the the gantry position spline
        Not applicable to the gripper winch line

        time shoud be in the base interval of the model splines

        returns a scalar for the line length

        Args:
            anchor_index: which line to estimate
            gantry_pos_spline: BSpline. the model of the gantry position in time.
        """

        # evaluate the gantry position spline at a given instant and measure distance to anchor
        return np.linalg.norm(self.anchor_points[anchor_index] - self.gantry_pos_spline(time))

    def winch_line_len(self, time):
        """
        Return the winch line length at a point in time
        time shoud be in the base interval of the model splines
        """
        return np.linalg.norm(self.gantry_pos_spline(time) - self.gantry_pos_spline(time))

    def calc_gripper_accel_from_forces(self, steps):
        """
        Calculate the expected acceleration on the gripper's IMU based on the forces it should experience due to
        it being a pendulum hanging from a moving object

        two forces act on it
        gravity acts to accelerate the mass down.
        a tension force acts in the direction of the gantry equal to the component of gravity opposite that direction.

        Args:
            steps: number of discrete points at which to measure forces.
        """
        results = []
        for time in np.linspace(0,1,steps): # only look in the future
            gant_pos = self.gantry_pos_spline(time)
            # the center of gravity is the location that matters for this calculation
            grip_pos = self.gripper_pos_spline(time)

            # normalized direction vector from gantry pointing towards gripper
            direction = np.linalg.norm(grip_pos - gant_pos)
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
        spline3d_model_size = self.n_ctrl_pts * 3
        spline1d_model_size = self.n_ctrl_pts
        if len(params) != (spline3d_model_size * 2 + spline1d_model_size): # we have 3 splines
            raise ValueError("position model_parameters incorrect size.")
        # Model parameters are the control points for the position curves.
        # directly update the control points of the bsplines
        start = 0
        self.gripper_pos_spline.c = params[start : spline3d_model_size].reshape((self.n_ctrl_pts, 3))
        start += spline3d_model_size
        self.gantry_pos_spline.c = params[start : start + spline3d_model_size].reshape((self.n_ctrl_pts, 3))
        start += spline3d_model_size
        self.gripper_rot_spline.c = params[start : start + spline1d_model_size].reshape((self.n_ctrl_pts,))
        self.gantry_accel_func = self.gripper_pos_spline.derivative(2)

    def model_time(self, t):
        """
        Convert a floating point number of seconds since the epoch into a time relative to the base interval of the model splines.
        assumes base interval is (0,1)
        """
        return (t - self.time_domain[0]) / (self.time_domain[1] - self.time_domain[0])

    def unix_time(self, t):
        """
        Convert a time relative to the base interval of the model splines into a floating point number of seconds since the epoch
        assumes base interval is (0,1)
        """
        return t * (self.time_domain[1] - self.time_domain[0]) + self.time_domain[0]

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
        
        steps = 100

        errors = np.array([
            # error between gantry position model and observation
            self.error_meas(self.gantry_pos_spline, self.snapshot['gantry_position']),
            # error between gripper position model and observation
            self.error_meas(self.gripper_pos_spline,  self.snapshot['gripper_position']),
            # error between gripper rotation model and observation
            self.error_meas(self.gripper_rotation,  self.snapshot['gripper_rotation']),
            # error between gripper acceleration model and observation
            self.error_meas(self.gantry_accel_func,  self.snapshot['imu_accel']),
            # error between gripper acceleration model and acceleration from calculated forces.
            self.error_meas(self.gantry_accel_func, self.calc_gripper_accel_from_forces(steps=steps), normalize_time=False),
            # error between model and recorded winch line lengths
            self.error_meas(self.winch_line_len, self.snapshot['winch_line_record']),
            # error between model and recorded anchor line lengths
            sum([self.error_meas(partial(self.anchor_line_length, anchor_num), self.snapshot['anchor_line_record'][anchor_num])
                for anchor_num in range(self.n_cables)]) / self.n_cables,
            # integral of the mechanical energy of the moving parts from now till the end in Joule*seconds
            integrate.quad(self.mechanical_energy(self.gripper_pos_spline, self.gripper_mass), 0, self.horizon_s)[0],
            integrate.quad(self.mechanical_energy(self.gantry_pos_spline, self.gantry_mass), 0, self.horizon_s)[0],
            # error between position model and desired future locations
            self.error_meas(self.gripper_pos_spline, self.desired_gripper_positions()),
            
            # penalty for pulling the motors against eachother by raising the gantry too high
            # penalty for letting the gripper touch the floor
            # penalty for letting the gantry touch an imaginary plane 6 feet from the ground (heuristic for dodging furniture)
            # penalty for exceeding max gripper acceleration since it could shake loose the payload
            # penalty for unspooling so fast you make a birdsnest
        ])

        return sum(errors * self.weights)

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
            'gantry_position': gantry_pose[:,:4],
            'gantry_rotation': gantry_pose[:,[0,4,5,6]],
            'gripper_position': gripper_pose[:,:4],
            'gripper_rotation': gripper_pose[:,[0,4,5,6]],
            'imu_accel': self.datastore.imu_accel.deepCopy(),
            'winch_line_record': self.datastore.winch_line_record.deepCopy(),
            'anchor_line_record': [a.deepCopy() for a in self.datastore.anchor_line_record]
        }

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
            self.gripper_rot_spline.c.copy().reshape(-1),
        ])

        self.snapshot_datastore()
        start = time()
        result = optimize.minimize(
            self.cost_function,
            parameter_initial_guess,
            method='SLSQP', # Suitable for constrained optimization
            bounds=self.bounds,
            options={'maxiter':1000}
        )
        time_taken = time() - start
        print(f"minimization step took {time_taken} seconds")

        # set splines from optimal model params
        self.set_splines_from_params(result.x)

        # now you can use splines to calculate position at any point in the time interval, such as this instant.
        # normalized_time = self.model_time(time())
        # current_gripper_pos = self.gripper_pos_spline(normalized_time)

        # evaluate line lengths in the future and put them in a queue for immediate transmission to the robot
        future_anchor_lines = [[
            (self.unix_time(t), self.anchor_line_length(anchor, t))
            for t in self.future_times] for anchor in range(self.n_cables)]

        future_winch_line = np.array([
            np.concatenate([[self.unix_time(t)], [self.winch_line_len(t)]])
            for t in self.future_times])
        update_for_observer = {
            'future_anchor_lines': future_anchor_lines,
            'future_winch_line': future_winch_line,
        }
        self.to_ob_q.put(update_for_observer)

        # send control points of position splines to UI for visualization
        update_for_ui = {
            'gripper_path': self.gripper_pos_spline.c,
            'gantry_path': self.gantry_pos_spline.c,
            'minimization_step_seconds': time_taken,
        }
        self.min_to_ui_q.put(update_for_ui)

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
        #   move_spline_domain_fast(self.gripper_rot_spline, domain_offset)
        # else:
        self.move_spline_domain_robust(self.gripper_pos_spline, domain_offset)
        self.move_spline_domain_robust(self.gantry_pos_spline, domain_offset)
        self.move_spline_domain_robust(self.gripper_rot_spline, domain_offset)

    def desired_gripper_positions(self):
        """
        Return a list of one or more future desired gripper positions
        """
        desired_positions = []
        linger = 2.0 # seconds to hover over bin or object
        # starting at the present in unix time
        t = time()
        # starting with the highest priority item
        item_index = 0
        # which we may already be holdin
        holding = self.holding_something_now()
        while t < self.time_domain[1]:
            if holding:
                # you want to be over the bin for the thing you're holding
                destination = self.gripper_over_bin_location()
            else:
                # you want to be over the next target item
                destination = self.item_priority_list(item_todo)
                item_index += 1
            # at a point in the future (distance to the destination) / rough_gripper_speed seconds from now
            travel_time = np.linalg.norm(destination - self.gripper_pos_spline(self.model_time(t))) / self.rough_gripper_speed
            t += travel_time
            desired_positions.append(np.concatenate([[t], destination]))
            # and also remaining at the destination at a point in the future after a fixed lingering period, if it's still within our time domain
            t += linger
            if t < self.time_domain[1]:
                desired_positions.append(np.concatenate([[t], destination]))
            # assume we will drop/pick up the item at this time.
            # we cannot know whether we will succeed, but have to assume we will for planning
            holding = not holding
        return np.array(desired_positions)

    def holding_something_now(self):
        return True

    def item_priority_list(self, idx):
        pl = np.array([ [-1.5,0, 2.2],
                        [ 0.9,0, 1.0],
                        [ 0.2,0,-1.2]])
        return pl[idx]

    def gripper_over_bin_location(self):
        return np.array([0,0.2,1])

    def read_input_queue(self):
        while True:
            update = self.to_pe_q.get()
            if 'anchor_pose' in update:
                apose = update['anchor_pose']
                anchor_num = apose[0]
                print(f'updating the position of anchor {anchor_num} to {apose[1][1]}')
                self.anchor_points[anchor_num] = np.array(apose[1][1])


def start_estimator(shared_datastore, min_to_ui_q, to_pe_q, to_ob_q):
    pe = CDPR_position_estimator(shared_datastore, min_to_ui_q, to_pe_q, to_ob_q)
    reader = threading.Thread(target=pe.read_input_queue)
    reader.start()
    while True:
        pe.estimate()