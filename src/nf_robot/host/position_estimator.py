"""
Estimate the current position and velocity of the gantry and gripper continuously.

Uses a more simplified approach based on initial experimentation with a focus on speed
"""
import time
import numpy as np
import asyncio
import cv2
import scipy.optimize as optimize
from math import pi, sqrt, sin, cos
from scipy.spatial.transform import Rotation

import nf_robot.common.definitions as model_constants
from nf_robot.common.kalman_filter import KalmanFilter
from nf_robot.host.frequency_estimator import SwingFrequencyEstimator
from nf_robot.generated.nf import telemetry, common
from nf_robot.common.util import *
from nf_robot.common.pose_functions import compose_poses, gripper_imu_inv

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

    # Project sphere's center onto the plane of the circle
    distance_to_circle_plane = np.dot(sphere_center - circle_center, circle_normal)
    projected_sphere_center = sphere_center - distance_to_circle_plane * circle_normal

    # Effective radius of the sphere in the circle's plane
    projected_sphere_radius_squared = sphere_radius**2 - distance_to_circle_plane**2
    if projected_sphere_radius_squared < -1e-9:
        return []  # No intersection with the plane
    projected_sphere_radius = np.sqrt(np.maximum(0, projected_sphere_radius_squared))

    # Find an orthonormal basis for the plane of the circle
    if np.abs(np.dot(circle_normal, np.array([0, 0, 1]))) < 1 - 1e-9:
        u_direction = np.cross(circle_normal, np.array([0, 0, 1]))
    else:
        u_direction = np.cross(circle_normal, np.array([1, 0, 0]))
    u_direction = u_direction / np.linalg.norm(u_direction)
    v_direction = np.cross(circle_normal, u_direction)

    # Centers of the two circles in the 2D basis
    center_offset = projected_sphere_center - circle_center
    x0 = np.dot(center_offset, u_direction)
    y0 = np.dot(center_offset, v_direction)

    # Solve for the intersection of the two 2D circles
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
    def __init__(self, datastore, observer):
        """
        continuous position estimation
        recalculates hang point only when the data affecting it changes and consideres it a measurement'
        considers also any visually obtained position a measurement
        uses all measurements to update a kalman filter with each measurement type having it's own covariance matrix learned from data
        
        main consists of three tasks in a task group
        * task to periodically predict forwards the kalman filter model
        * task to wait for updates that affect hang point and recalculate it, then update filter
        * task to wait for updates that affect visual estimates and update filter.
        """
        self.run = False # false until main is called
        self.datastore = datastore
        self.ob = observer
        self.config = observer.config
        self.n_cables = len(self.config.anchors)
        self.work_area = None
        self.anchor_points = np.array([
            [-2,  3, 2],
            [ 2,  3, 2],
            [ -1,-2, 2],
            [ -2,-2, 2],
        ], dtype=float)
        # save grommet points
        print(f'before {self.config.anchors[0].pose} after {poseProtoToTuple(self.config.anchors[0].pose)}')
        anchor_points = np.array([compose_poses([poseProtoToTuple(a.pose), model_constants.anchor_grommet])[1] for a in self.config.anchors])
        self.set_anchor_points(anchor_points)
        self.data_ts = time.time()

        # the gantry position and velocity estimated from reported line length and speeds.
        self.hang_pos = np.zeros(3, dtype=float)
        self.hang_vel = np.zeros(3, dtype=float)

        # gantry position and velocity as last determined by the kalman filter
        self.gant_pos = np.zeros(3, dtype=float)
        self.gant_vel = np.zeros(3, dtype=float)

        # last visual estimate of gantry position
        self.visual_pos = np.zeros(3, dtype=float)
        self.visual_vel = np.zeros(3, dtype=float)

        # try:
        #     self.gant_pos = np.load('gant_pos.npy')
        # except:
        #     pass

        # the time at which all reported line speeds became zero.
        # If some nonzero speed was occuring on any line at the last update, this will be None
        self.stop_cutoff = time.time()

        # last estimate of the gripper pose
        self.grip_pose = (np.zeros(3), np.zeros(3))

        # last information on whether the gripper is thought to be holding something
        self.holding = False
        self.finger_pressure_rising = asyncio.Event()
        self.finger_pressure_falling = asyncio.Event()

        # event coupled to gripper tipping from vertical
        self.tip_over = asyncio.Event()

        # hang point based prediction of slack lines (currently unused)
        self.slack_lines = [False, False, False, False]

        self.last_visual_cutoff = time.time()

        # Expected standard deviation in gantry acceleration
        acceleration_std_dev=0.001
        # noise in the position biases used for each sensor. the biases can be 10 to 50 cm in my experience.
        # this small constant reflects the assumtion that the bias for a camera is expected to change at least 10x slower than the gantry position.
        bias_std_dev=0.0001
        # noise in visual observations of gantry position from a single camera 
        visual_noise_std_dev=0.01 # experimentally confirmed.
        hang_noise_std_dev=0.015 # noise in hang position as a sensor. This is odd, it will be tightly near some position but make large jumps in z occationally.
        commanded_vel_std_dev=0.01 # the velocity that was commanded of the robot is also considered a sensor. the actual movement may differ from it somewhat.

        self.visual_noise_covariance = np.diag([visual_noise_std_dev**2] * 3)
        self.hang_noise_covariance = np.diag([hang_noise_std_dev**2] * 3)
        self.vel_noise_covariance = np.diag([commanded_vel_std_dev**2] * 3)

        # Initialize the Kalman filter
        self.sensor_names = ['v0', 'v1' ,'v2' ,'v3', 'hang']
        self.kf = KalmanFilter(self.sensor_names, acceleration_std_dev, bias_std_dev)

        # recorded amount of time take to perform various update steps
        self.predict_time_taken = 0
        self.visual_time_taken = 0
        self.hang_time_taken = 0

        # last commanded gantry velocity.
        self.commanded_vel = np.zeros(3)
        self.commanded_vel_ts = time.time()

        # gripper swing freqency estimator
        self.swing_est = SwingFrequencyEstimator(hysteresis=0.04, smoothing_factor=0.15)

        # Gripper type (pilot or arp)
        self.gripper_type = 'pilot'

    def set_gripper_type(self, t):
        self.gripper_type = t

    def set_anchor_points(self, points):
        """refers to the grommet points. shape (4,3)"""
        assert points.shape == (4, 3)
        self.anchor_points = points

        # create and save a 2D contour to be used for containment checking.
        # it must be a valid clockwise polygon so points are sorted by angle relative to centroid
        anchor_2d = self.anchor_points[:, 0:2]
        centroid = np.mean(anchor_2d, axis=0)
        angles = np.arctan2(anchor_2d[:, 1] - centroid[1], anchor_2d[:, 0] - centroid[0])
        self.work_area = anchor_2d[np.argsort(angles)].astype(np.float32)

    def point_inside_work_area_2d(self, point):
        return True 
        if self.work_area is None:
            return False
        in_2d = cv2.pointPolygonTest(self.work_area, (float(point[0]), float(point[1])), False) > 0

    def point_inside_work_area(self, point):
        return True # todo seems broken
        in_2d = self.point_inside_work_area_2d(point)
        min_anchor_z = np.min(self.anchor_points[:, 2])
        in_z = 0 < point[2] < min_anchor_z
        return in_2d and in_z

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

    async def predict_forwards(self):
        while self.run:
            start_time = time.time()
            await asyncio.sleep(1/60)
            # advance the prediction to the current time.
            self.kf.predict_present()
            self.gant_pos = self.kf.state_estimate[:3].copy()
            self.gant_vel = self.kf.state_estimate[3:6].copy()
            self.predict_time_taken = time.time()-start_time
            self.estimate_gripper()
            self.detect_grip()
            self.send_positions()
            await self.ob.flush_tele_buffer()

    async def update_visual(self):
        while self.run:
            start_time = time.time()
            # wait at least this long
            await asyncio.sleep(1/30)
            # and wait for new visual data if necessary
            await self.datastore.gantry_pos_event.wait()
            self.datastore.gantry_pos_event.clear()
            # grab any new data
            data = self.datastore.gantry_pos.deepCopy(cutoff=self.last_visual_cutoff)
            if len(data) == 0:
                continue
            self.last_visual_cutoff = np.max(data[:,0])

            # update filter with every datapoint
            for measurement in data:
                timestamp = measurement[0]
                anchor_num = int(measurement[1])
                this_visual_pos = measurement[2:]
                self.kf.update(this_visual_pos, timestamp, self.visual_noise_covariance, 'position', self.sensor_names[anchor_num])

                # average this visual position into self.visual_pos. this is purely for the UI
                self.visual_pos = self.visual_pos * 0.9 + this_visual_pos * 0.1

            self.visual_time_taken = time.time()-start_time

    async def update_hang(self):
        while self.run:
            start_time = time.time()
            # wait at least this long
            await asyncio.sleep(1/30)
            # and wait for new data affecting hang point if necessary
            await self.datastore.anchor_line_record_event.wait()
            self.datastore.anchor_line_record_event.clear()

            # Look at the last report for each anchor line.
            # time, length, speed, tight
            records = np.array([alr.getLast() for alr in self.datastore.anchor_line_record])
            lengths = np.array(records[:,1])
            speeds = np.array(records[:,2])
            tight = np.array(records[:,3])

            # average timestamp of the four lines contributing to this hang point.
            data_ts = np.mean(records[:,0])

            # if any line is measured to be slack,
            # make its length effectively infinite so it won't play a part in the hang position
            lengths[tight < 0.5] = 100

            # calculate hang point
            result = find_hang_point(self.anchor_points, lengths)
            if result is not None:
                self.hang_pos, self.slack_lines = result
                # update kalman filter with this position
                self.kf.update(self.hang_pos, data_ts, self.hang_noise_covariance, 'position', self.sensor_names[-1])
                self.data_ts = data_ts

            self.kf.enforce_bias_constraint()

            # optional hang velocity (unreviewed)
            if sum(speeds) == 0:
                self.hang_vel = np.zeros(3)
                self.kf.update(self.hang_vel, self.commanded_vel_ts, self.vel_noise_covariance, 'velocity')

            self.hang_time_taken = time.time()-start_time

    def record_commanded_vel(self, vel):
        self.commanded_vel = vel
        self.commanded_vel_ts = time.time()
        self.ob.send_ui(last_commanded_vel=telemetry.CommandedVelocity(velocity=fromnp(vel)))

    async def update_commanded_vel(self):
        """provide an observation to the filter based on the commanded velocity"""
        while self.run:
            self.kf.update(self.commanded_vel, self.commanded_vel_ts, self.vel_noise_covariance, 'velocity')
            await asyncio.sleep(1/30)

    def get_pendulum_length(self):
        return self.swing_est.get_pendulum_length()

    def estimate_gripper(self):
        """Estimate attributes of the gripper that depend on its IMU reading"""
        last_imu = self.datastore.imu_quat.getLast()
        ts = last_imu[0]
        if not ts:
            return
        rotation = Rotation.from_quat(last_imu[1:])
        # back out mounting position of IMU
        rotation = rotation * Rotation.from_euler('xyz', [-90, 0, 0], degrees=True)

        # When distance to floor is less than 12cm, Detect Tipping
        if self.datastore.range_record.getLast()[1] < 0.12:
            THRESHOLD_DEGREES = 9
            euler = rotation.as_euler('xyz', degrees=True)
            if abs(euler[0]) > THRESHOLD_DEGREES or abs(euler[1]) > THRESHOLD_DEGREES:
                self.tip_over.set()

        # feed angle to frequency estimator
        rotvec = rotation.as_rotvec()
        self.swing_est.add_rotation_vector(ts, rotvec)

        # get the encoder based winch line length
        _, length, speed = self.datastore.winch_line_record.getLast()

        self.grip_pose = compose_poses([
            (rotvec, self.gant_pos),
            (np.zeros(3), np.array([0,0,-length], dtype=float)),
        ])

    def detect_grip(self):
        """
        Watch for the rising and falling edge in grip pressure and set corresponding asyncio events
        may set self.finger_pressure_rising or self.finger_pressure_falling
        """
        THRESHOLD = 0.4
        HYSTERESIS = 0.04
        pressure = self.datastore.finger.getLast()[2]

        # Detect Rising Edge
        # Only trigger if we aren't currently holding to ensure we capture the
        # transition event rather than the continuous state of being compressed.
        if not self.holding and pressure >= THRESHOLD:
            self.holding = True
            self.finger_pressure_rising.set()

        # Detect Falling Edge
        elif self.holding and pressure <= (THRESHOLD - HYSTERESIS):
            self.holding = False
            self.finger_pressure_falling.set()


    def send_positions(self):
        # send position factors to UI for visualization

        # no longer being sent but available for debug
        # 'kalman_prediction_rate': self.predict_time_taken,
        # 'visual_update_rate': self.visual_time_taken,
        # 'hang_update_rate': self.hang_time_taken,

        self.ob.send_ui(pos_estimate=telemetry.PositionEstimate(
            data_ts=float(self.data_ts),
            gantry_position=fromnp(self.gant_pos),
            gantry_velocity=fromnp(self.gant_vel),
            gripper_pose=common.Pose(rotation=fromnp(self.grip_pose[0]), position=fromnp(self.grip_pose[1])),
            slack=list(map(bool, self.slack_lines)),
        ))
        self.ob.send_ui(pos_factors_debug=telemetry.PositionFactors(
            visual_pos=fromnp(self.visual_pos),
            visual_vel=fromnp(self.visual_vel),
            hanging_pos=fromnp(self.hang_pos),
            hanging_vel=fromnp(self.hang_vel),
        ))

    def notify_update(self, update):
        if 'holding' in update:
            self.holding = update['holding']

    async def main(self):
        print('Starting position estimator')
        self.run = True
        try:
            async with asyncio.TaskGroup() as tg:
                predict_task = tg.create_task(self.predict_forwards())
                visual_task = tg.create_task(self.update_visual())
                hang_task = tg.create_task(self.update_hang())
                comv_task = tg.create_task(self.update_commanded_vel())
                
        except asyncio.exceptions.CancelledError:
            pass
        np.save('gant_pos.npy', self.gant_pos)
        print('All position estimator tasks finished')