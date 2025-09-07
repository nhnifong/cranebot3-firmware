"""
Estimate the current position and velocity of the gantry and gripper continuously.

Uses a more simplified approach based on initial experimentation with a focus on speed
"""
import time
import numpy as np
import asyncio
import scipy.optimize as optimize
from math import pi, sqrt, sin, cos
from config import Config
from cv_common import compose_poses
import model_constants
from scipy.spatial.transform import Rotation

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

def sphere_intersection(sphere1, sphere2):
    """
    Calculates the intersection circle of two spheres.

    Args:
        sphere1: Tuple or list containing (center, radius) of the first sphere.
                 center is a numpy array of shape (3,).
        sphere2: Tuple or list containing (center, radius) of the second sphere.
                 center is a numpy array of shape (3,).

    Returns:
        A tuple containing:
            - center (numpy array): Center of the intersection circle.
            - normal_vector (numpy array): Normal vector of the plane containing the circle.
            - radius (float): Radius of the intersection circle.
        Returns None if the spheres do not intersect in a circle.
    """
    c1, r1 = sphere1
    c2, r2 = sphere2

    d_vec = c2 - c1
    d = np.linalg.norm(d_vec)
    # Check for intersection
    if d > r1 + r2 + 1e-9 or d < np.abs(r1 - r2) - 1e-9 or d == 0:
        return None  # No intersection or one sphere inside the other (or same center)
    # Normal vector of the intersection plane
    normal_vector = d_vec / d
    # Distance from center of sphere 1 to the intersection plane
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    # Center of the intersection circle
    center_intersection = c1 + a * normal_vector
    # Radius of the intersection circle
    r_intersection_sq = r1**2 - a**2
    if r_intersection_sq < 0:
        return None # Should not happen if intersection check passes, but for robustness
    r_intersection = np.sqrt(r_intersection_sq)

    return center_intersection, normal_vector, r_intersection


def sphere_circle_intersection(sphere_center, sphere_radius, circle_center, circle_normal, circle_radius):
    """
    Finds the intersection points between a sphere and a circle in 3D.

    Args:
        sphere_center: numpy array (3,).
        sphere_radius: float.
        circle_center: numpy array (3,).
        circle_normal: numpy array (3,), unit vector.
        circle_radius: float.

    Returns:
        A list of numpy arrays (3,) representing the intersection points.
        Returns an empty list if there are no intersection points.
        Returns two identical points for the tangency case.
    """

    # 1. Project sphere's center onto the plane of the circle
    distance_to_circle_plane = np.dot(sphere_center - circle_center, circle_normal)
    projected_sphere_center = sphere_center - distance_to_circle_plane * circle_normal

    # 2. Effective radius of the sphere in the circle's plane
    projected_sphere_radius_squared = sphere_radius**2 - distance_to_circle_plane**2
    if projected_sphere_radius_squared < -1e-9:
        return []  # No intersection with the plane
    projected_sphere_radius = np.sqrt(np.maximum(0, projected_sphere_radius_squared))

    # 3. Find an orthonormal basis for the plane of the circle
    if np.abs(np.dot(circle_normal, np.array([0, 0, 1]))) < 1 - 1e-9:
        u_direction = np.cross(circle_normal, np.array([0, 0, 1]))
    else:
        u_direction = np.cross(circle_normal, np.array([1, 0, 0]))
    u_direction = u_direction / np.linalg.norm(u_direction)
    v_direction = np.cross(circle_normal, u_direction)

    # 4. Centers of the two circles in the 2D basis
    center_offset = projected_sphere_center - circle_center
    x0 = np.dot(center_offset, u_direction)
    y0 = np.dot(center_offset, v_direction)

    # 5. Solve for the intersection of the two 2D circles
    centers_distance = np.sqrt(x0**2 + y0**2)

    if centers_distance > circle_radius + projected_sphere_radius + 1e-9 or centers_distance < np.abs(circle_radius - projected_sphere_radius) - 1e-9:
        return []  # No intersection in the plane

    a_param = (circle_radius**2 - projected_sphere_radius**2 + centers_distance**2) / (2 * centers_distance)
    h_param = np.sqrt(np.maximum(0, circle_radius**2 - a_param**2))

    p2_x = a_param * (x0 / centers_distance)
    p2_y = a_param * (y0 / centers_distance)
    p2_base = circle_center + p2_x * u_direction + p2_y * v_direction

    tangent_vector = -y0 * u_direction + x0 * v_direction
    tangent_vector_norm = np.linalg.norm(tangent_vector)
    tangent_vector = tangent_vector / tangent_vector_norm if tangent_vector_norm > 1e-9 else np.array([1, 0, 0])
    offset = h_param * tangent_vector

    point1 = p2_base + offset
    point2 = p2_base - offset
    return np.array([point1, point2])

def lowest_point_on_circle(circle_center, circle_normal, circle_radius):
    """
    Finds the point on the circle with the lowest z-coordinate.

    Args:
        circle_center: numpy array (3,).
        circle_normal: numpy array (3,), unit vector.
        circle_radius: float.

    Returns:
        numpy array (3,): The point on the circle with the lowest z-coordinate.
    """
    # invalid for circles in a horizontal plane
    if np.sum(circle_normal[:2]) == 0:
        return None

    # The lowest point is in the direction opposite the z-component of the normal vector.
    # We create a vector pointing downwards
    downward_vector = np.array([0, 0, -1])

    # Project the downward vector onto the plane of the circle.
    projection_length = np.dot(downward_vector, circle_normal)
    projected_vector = downward_vector - projection_length * circle_normal

    # Normalize the projected vector.
    projected_vector_normalized = projected_vector / np.linalg.norm(projected_vector)

    # Calculate the lowest point
    lowest_point = circle_center + circle_radius * projected_vector_normalized

    return lowest_point

def find_hang_point(positions, lengths):
    """
    Find the lowest point at which a mass could hang from the given anchor positions without
    the distance to any anchor being longer than the given lengths of available line

    In addition to finding the position, we get an array of bools indicating which lines are slack as a side effect

    If two spheres intersect, they form a circle.
    The lowest point on the circle may be a hang point if only two lines are taut
    if a circle intersects a sphere, it does so at two points, the lower of which may be a hang point.
    Any hang point below the floor is discarded
    Any hang point not inside all spheres is discarded
    take the lowest remaining point

    For a four anchor system, there are six possible sphere-sphere crosses.
    For each circle formed this way, it could intersect with either of the two uninvolved spheres.
    """
    if len(positions) != 4 or len(lengths) != 4:
        raise ValueError
    lengths = lengths + np.repeat(1e-8, 4)
    candidates = []
    for pair in [[0,1], [1,2], [2,3], [3,0], [0,2], [1,3]]:
        # find the intersection of the two spheres in this pair
        circle = sphere_intersection(*[(positions[i], lengths[i]) for i in pair])
        if circle is None:
            continue
        lp = lowest_point_on_circle(*circle)
        if lp is not None:
            if lp[2] > 0:
                candidates.append(lp)
        # intersect this circle with the two uninvoled spheres
        for i in range(4):
            if i not in pair:
                pts = sphere_circle_intersection(positions[i], lengths[i], *circle)
                if len(pts) == 2:
                    # take the lower point
                    lower = pts[np.argmin(pts[:, 2])]
                    if lower[2] > 0:
                        candidates.append(lower)
    if len(candidates) == 0:
        return None
    candidates = np.array(candidates)
    # filter out candidates that are not inside all spheres
    ex_lengths = lengths + 1e-5
    distances = np.linalg.norm(candidates[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
    candidates = candidates[np.all(distances <= ex_lengths[np.newaxis, :], axis=1)]

    if len(candidates) == 0:
        return None

    # line length must exceed distance to point by this much to be considered slack
    # this is the estimate of slackness implied by the line lengths.
    index_of_lowest = np.argmin(candidates[:, 2])
    slack_lines = distances[index_of_lowest] <= (ex_lengths - 0.04)

    return candidates[index_of_lowest], slack_lines


def swing_angle_from_params(t, freq, xamp, yamp, xphase, yphase):
    """
    Evaluate a swing angles at an array of times

    t numpy array of timestamps
    freq in hertz
    amplitude of x andy y waves in meters
    phase offset of waves, -pi to pi
    """
    xangles = np.cos(freq * t * 2 * pi + xphase) * xamp
    yangles = np.sin(freq * t * 2 * pi + yphase) * yamp
    if np.isscalar(t):
        return xangles, yangles
    return np.column_stack([xangles, yangles])

def swing_angle_from_params_transformed(t, freq, xamp, yamp, cos_xph, sin_xph, cos_yph, sin_yph):
    omega_t = freq * t * 2 * np.pi
    cos_omega_t = np.cos(omega_t)
    sin_omega_t = np.sin(omega_t)

    # For x-angle: xamp * cos(omega_t + xphase)
    x_angles = xamp * (cos_omega_t * cos_xph - sin_omega_t * sin_xph)
    # For y-angle: yamp * sin(omega_t + yphase)
    y_angles = yamp * (sin_omega_t * cos_yph + cos_omega_t * sin_yph)

    if np.isscalar(t):
        return x_angles, y_angles
    return np.column_stack([x_angles, y_angles])

def swing_cost_fn(model_params, times, measured_angles):
    predicted_angles = swing_angle_from_params(times, *model_params)
    distances = np.linalg.norm(measured_angles - predicted_angles, axis=1)
    return np.mean(distances**2)

def swing_cost_fn_transformed(model_params_transformed, times, measured_angles):
    predicted_angles = swing_angle_from_params_transformed(times, *model_params_transformed)
    distances = np.linalg.norm(measured_angles - predicted_angles, axis=1)
    return np.mean(distances**2)


def eval_linear_pos(t, starting_time, starting_pos, velocity_vec):
    """
    Evaluate positions on a line at an array of times.

    t - numpy array of timestamps to evaluate line at
    starting_time - timestamp when the object is at starting_pos
    starting_pos - position where movement started
    velocity_vec - velocity of object in units(meters) per second
    """
    elapsed = (t - starting_time).reshape((-1,1))
    return starting_pos + velocity_vec * elapsed

def linear_move_cost_fn(model_params, starting_time, times, observed_positions):
    starting_pos = model_params[0:3]
    velocity_vec = model_params[3:6]
    predicted_positions = eval_linear_pos(times, starting_time, starting_pos, velocity_vec)
    distances = np.linalg.norm(observed_positions - predicted_positions, axis=1)
    return np.mean(distances**2)


class Positioner2:
    def __init__(self, datastore, to_ui_q, observer):
        self.run = True # run main loop
        self.datastore = datastore
        self.to_ui_q = to_ui_q
        self.ob = observer
        self.config = Config()
        self.n_cables = len(self.config.anchors)
        self.anchor_points = np.array([
            [-2,  3, 2],
            [ 2,  3, 2],
            [ -1,-2, 2],
            [ -2,-2, 2],
        ], dtype=float)
        for i, a in enumerate(self.config.anchors):
            self.anchor_points[i] = np.array(compose_poses([a.pose, model_constants.anchor_grommet])[1])
        self.holding = False
        self.data_ts = time.time()

        # the gantry position and velocity estimated from reported line length and speeds.
        self.hang_gant_pos = np.zeros(3, dtype=float)
        self.hang_gant_vel = np.zeros(3, dtype=float)

        # gantry position and velocity as a weighted average of hang pose and visual pos
        self.gant_pos = np.zeros(3, dtype=float)
        self.gant_vel = np.zeros(3, dtype=float)

        # the time at which all reported line speeds became zero.
        # If some nonzero speed was occuring on any line at the last update, this will be None
        self.stop_cutoff = time.time()

        self.swing_params = np.array([
            1, # frequency
            0.01, # x amplitude
            0.01, # y amplitude
            cos(1.5), # cos x phase
            sin(1.5), # sin x phase
            cos(1.5), # cos y phase
            sin(1.5), # sin y phase
        ], dtype=float)

        self.visual_move_start_time = time.time()
        self.visual_move_line_params = np.concatenate([self.hang_gant_pos, self.hang_gant_vel])
        self.visual_pos = np.zeros(3, dtype=float)
        self.visual_vel = np.zeros(3, dtype=float)
        self.grip_pose = (np.zeros(3), np.zeros(3))
        self.slack_lines = [False, False, False, False]
        self.last_visual_data_timestamp = 0

    def set_anchor_points(self, points):
        """refers to the grommet points. shape (4,3)"""
        self.anchor_points = points


    def find_swing(self):
        """When the gantry is still, the IMU's quaternion readout can be used to estimate the gantry swing params

        "still" means you can use any IMU observation that occured when all anchor lines had speed 0.

        A swing in two dimension is defined by
        - the natural frequency, from which I could derive the length.
        - the amplitude in the x dimension
        - the amplitude in the y dimention
        - the phase offset of the x swing
        - the phase offset of the y swing
        """
        imu_readings = self.datastore.imu_rotvec.deepCopy()
        if len(imu_readings) < 20:
            return
        # convert to x and y angle offsets from vertical.
        timestamps = imu_readings[:,0]

        angles = Rotation.from_rotvec(imu_readings[:,1:]).as_euler('xyz')[:,:2]
        if np.sum(angles[-1]) == 0:
            return
        # print(f'angles = {angles[-1]}')

        # get the winch line length and use it to bound the swing frequency
        _, length, _ = self.datastore.winch_line_record.getLast()
        if length == 0:
            return
        swing_freq = 1/(2*pi*sqrt(length/9.81))

        bounds = [
            (swing_freq*0.8, swing_freq*1.2), # frequency +- 20%
            (0, 1), # x amplitude. max possible swing angle in radians
            (0, 1), # y amplitude
            (-1, 1), # cos_xphase
            (-1, 1), # sin_xphase
            (-1, 1), # cos_yphase
            (-1, 1), # sin_yphase
        ]
        initial_guess = self.swing_params.copy()
        initial_guess[0] = swing_freq

        constraints = [
            {'type': 'eq', 'fun': lambda p: p[3]**2 + p[4]**2 - 1.0},  # cos_xph^2 + sin_xph^2 = 1
            {'type': 'eq', 'fun': lambda p: p[5]**2 + p[6]**2 - 1.0}   # cos_yph^2 + sin_yph^2 = 1
        ]

        # print(f'initial_guess = {initial_guess}')

        # given this array of timed x and y angles, compute what the expected angles should have been, and total up the error.
        # then use SLSQP to find the parameters.
        result = optimize.minimize(
            swing_cost_fn_transformed,
            initial_guess,
            args=(timestamps, angles),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'disp': False,
                'maxiter': 100,
                # 'ftol': 1e-6,
                # 'gtol': 1e-9,
            },

        )
        try:
            if not result.message.startswith("Iteration limit reached"):
                assert result.success
        except AssertionError:
            print(result)
            return
        self.swing_params = result.x
        self.swing_params[3] = self.swing_params[3] % (2*pi)
        self.swing_params[4] = self.swing_params[4] % (2*pi)

    def find_visual_move(self):
        """
        fit a line to the last few visual gantry position observations
        The parameters of the line are
          * a starting position
          * a velocity vector
        """
        now = time.time()
        backtime = now-1.5
        cutoff = backtime
        moving = True
        if self.stop_cutoff is not None and now > self.stop_cutoff:
            moving = False
            cutoff = self.stop_cutoff
        data = self.datastore.gantry_pos.deepCopy(cutoff=cutoff)
        if len(data) < 2:
            return False
        times = data[:,0]
        self.last_visual_data_timestamp = np.max(times)
        positions = data[:,1:]

        lower = np.min(self.anchor_points, axis=0)
        lower[2] = 0 # lowest possible gantry position is the floor, not the lowest anchor
        upper = np.max(self.anchor_points, axis=0)
        position_bounds = np.column_stack([lower, upper])

        velocity_guess = positions[-1]-positions[-2]
        speed = 0.5
        if not moving:
            speed = 0.0
            velocity_guess = np.zeros(3)
        velocity_bounds = np.column_stack([np.repeat(-speed, 3), np.repeat(speed, 3)])

        initial_guess = np.concatenate([ positions[-1], velocity_guess ])
        bounds = np.concatenate([position_bounds, velocity_bounds])

        try:
            result = optimize.minimize(
                linear_move_cost_fn,
                initial_guess,
                args=(backtime, times, positions),
                bounds=bounds,
                method='SLSQP',
                options={
                    'disp': False,
                    'maxiter': 100,
                },
            )
        except ValueError as e:
            print(f'{e} \nIf this occurs a few times after a call to set_anchor_points, it is harmless')
            return False
        try:
            if not result.message.startswith("Iteration limit reached"):
                assert result.success
        except AssertionError:
            print(result)
            return False

        self.visual_move_start_time = backtime
        self.visual_move_line_params = result.x
        return True

    async def check_and_recal(self):
        """
        Automatically send line reference length based on visual observation under certain conditions.

        Conditions:
        1. no move command has been sent in the last n seconds
        2. the visually estimated gantry velocity is near zero
        """
        if self.stop_cutoff is None:
            return # currently moving
        if time.time() - self.stop_cutoff < 2:
            return # hasn't been long enough since we stopped
        position = self.visual_move_line_params[0:3]
        velocity = self.visual_move_line_params[3:6]
        if np.linalg.norm(velocity) > 0.005: # meters per second
            return # looks like it's moving visually, probably just video latency.

        lengths = np.linalg.norm(self.anchor_points - position, axis=1)
        print(f'auto line calibration lengths={lengths}')
        await self.ob.sendReferenceLengths(lengths)


    async def restimate(self):
        """
        Perform estimations that are meant to occur at a slower rate.
        """
        while self.run:
            try:
                visual_found = self.find_visual_move()
                # currently testing other means of calibrating zero angle. don't overwrite them.
                # if visual_found:
                #     # if visual move was estimated successfully, we may be able to use it to automatically update reference lengths
                #     asyncio.create_task(self.check_and_recal())
                self.find_swing()
                await asyncio.sleep(0.2)
            except Exception as e:
                self.run = False
                raise e

    def estimate(self):
        """
        Estimate current gantry and gripper position and velocity
        """
        self.start = time.time()
        z = np.zeros(3, dtype=float)

        # Look at the last report for each anchor line.
        # time, length, speed, tight
        records = np.array([alr.getLast() for alr in self.datastore.anchor_line_record])
        lengths = np.array(records[:,1])
        speeds = np.array(records[:,2])
        tight = np.array(records[:,3])
        
        # nothing has been recorded
        if sum(lengths) == 0:
            self.time_taken = time.time() - self.start
            return False

        # timestamp of the last record used to produce this estimate. used for latency feedback
        data_ts = np.max(records[0])
        # print(f'find hang point with lens {lengths}, data_ts > self.data_ts: {data_ts > self.data_ts}')
        # only perform work in this block when line records actually change
        if data_ts > self.data_ts:
            self.data_ts = data_ts

            # extrapolate the current length based on the time elapsed since the measurement and the speed at the time.
            # TODO account for clock desync before turning this part on.
            # elapsed = self.start - records[:,0]
            # offsets = records[:,1] + records[:,2] * elapsed
            # lengths += offsets

            # if any line is measured to be slack,
            # make its length effectively infinite so it won't play a part in the hang position
            lengths[tight < 0.5] = 100

            # calculate hang point
            result = find_hang_point(self.anchor_points, lengths)
            if result is None:
                # print(f'estimate bailed because it failed to calc a hang point with lengths {lengths}')
                self.time_taken = time.time() - self.start
                # self.hang_gant_pos will not be updated this time around.
            else:
                self.hang_gant_pos, slack_lines = result

                # this represents a prediction of which lines are slack, it may not match reality.
                # if this prediction says a line is tight but measured slackness says otherwise, the hang point is probably quite wrong.
                # self.slack_lines = result[1]
                self.slack_lines = np.logical_not(tight)

                if sum(speeds) == 0:
                    self.hang_gant_vel = z
                    # if the gantry just now stopped moving, record the time.
                    if self.stop_cutoff is None:
                        self.stop_cutoff = time.time()
                else:
                    self.stop_cutoff = None
                    # repeat for a position some small increment in the future to get the gantry velocity
                    increment = 0.1 # seconds
                    lengths += speeds * increment
                    result = find_hang_point(self.anchor_points, lengths)
                    if result is None:
                        # print(f'estimate failed to calc a hang point the second time from lengths {lengths}')
                        self.time_taken = time.time() - self.start
                        self.hang_gant_vel = np.zeros((3,))
                    else:
                        self.hang_gant_vel = result[0] - self.hang_gant_pos
                        self.hang_gant_vel = self.hang_gant_vel / increment

        # use information both from hang position and visual observation
        self.visual_vel = self.visual_move_line_params[3:6]
        print(f'visualo_vel={np.linalg.norm(self.visual_vel)} m/s')
        eval_time = time.time()
        # if self.stop_cutoff is not None and self.stop_cutoff > self.last_visual_data_timestamp:
        #     print(f'extrapolate visual velocity for {eval_time-self.last_visual_data_timestamp}s because we know gantry stopped')
        #     eval_time = self.stop_cutoff
        self.visual_pos = eval_linear_pos(
            np.array([eval_time]),
            self.visual_move_start_time,
            self.visual_move_line_params[0:3],
            self.visual_vel,
        )[0]


        self.gant_pos = self.hang_gant_pos * 0.7 + self.visual_pos * 0.3
        self.gant_vel = self.hang_gant_vel * 0.7 + self.visual_vel * 0.3

        # figure out gripper position.

        # get the last IMU reading from the gripper, take only the z axis
        last_rotvec = self.datastore.imu_rotvec.getLast()[1:]
        grv = Rotation.from_rotvec(last_rotvec).as_euler('xyz')
        last_z = grv[2]

        # print(f'gripper tilt {grv[0:2]}')
        # print(f'self.swing_params = {self.swing_params}')

        # predict the x and y based on swing parameters
        pre_x, pre_y = swing_angle_from_params_transformed(time.time(), *self.swing_params)
        current_rotation = Rotation.from_euler('xyz', (pre_x, pre_y, last_z)).as_rotvec()

        # get the winch line length
        timestamp, length, speed = self.datastore.winch_line_record.getLast()

        # starting at the gantry, rotate the frame of reference by the gripper rotation
        # and translate along the negative z axis by the rope length
        self.grip_pose = compose_poses([
            (z, self.gant_pos),
            (current_rotation, np.array([0,0,-length], dtype=float)),
        ])

        self.time_taken = time.time() - self.start
        return True

    def send_positions(self):
        # send control points of position splines to UI for visualization
        update_for_ui = {
            'pos_estimate': {
                'gantry_pos': self.gant_pos,
                'gantry_vel': self.gant_vel,
                'gripper_pose': self.grip_pose,
                'slack_lines': self.slack_lines,
                },
            'minimizer_stats': {
                # 'errors': softmax(self.errors),
                'data_ts': self.data_ts,
                'time_taken': self.time_taken,
                },
            'pos_factors_debug': {
                # position as esimated by visual observations
                'visual_pos': self.visual_pos,
                'visual_vel': self.visual_vel,
                # position as estimated by line length only
                'hang_pos': self.hang_gant_pos,
                'hang_vel': self.hang_gant_vel,
                },
            # 'goal_points': self.des_grip_locations, # each goal is a time and a position
        }
        self.to_ui_q.put(update_for_ui)

    def notify_update(self, update):
        if 'holding' in update:
            self.holding = update['holding']

    async def hang_update():
        while True:
            await time.sleep(0.03)
            await self.hang_data_event.wait()
            measured_position, tights = find_hang_point()
            # measurement_time could be the average time of the line records used to calc hang point
            self.kf.update(measurement_time, measured_position, sensor)
            self.hang_data_event.reset()

    async def main(self):
        print('Starting position estimator')
        rest_task = asyncio.create_task(self.restimate())
        while self.run:
            try:
                self.estimate()
                self.send_positions()
                # cProfile.runctx('self.estimate()', globals(), locals())
                # some sleep is necessary or we will not receive updates
                rem = (1/20 - self.time_taken)
                await asyncio.sleep(max(0.005, rem))
            except KeyboardInterrupt:
                print('Exiting')
                return
        result = await rest_task
