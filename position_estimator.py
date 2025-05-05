"""
Estimate the current position and velocity of the gantry and gripper continuously.

Uses a more simplified approach based on initial experimentation with a focus on speed
"""
import time
import numpy as np
import scipy.optimize as optimize
from config import Config
from cv_common import compose_poses
import model_constants

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
    return [point1, point2]

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
    the distance to any anchor being longer than the given length of available line

    In addition to finding the position, returns an array of bools indicating which lines are slack.

    If two spheres intersect, they form a circle.
    The lowest point on the circle may be a hang point if only two lines are taut
    if a circle intersects a sphere, it does so at two points, the lower of which may be a hang point.
    Any hang point that is outside the radius of any sphere is disqualified.

    For a four anchor system, there are six possible sphere-sphere crosses.
    For each circle formed this way, it could intersect with either of the two uninvolved spheres.
    """
    if len(positions) != 4 or len(lengths) != 4:
        raise ValueError
    candidates = []
    for pair in [[0,1], [1,2], [2,3], [3,0], [0,2], [1,3]]:
        # find the intersection of the two spheres in this pair
        circle = sphere_intersection(*[(positions[i], lengths[i]) for i in pair])
        if circle is None:
            continue
        lp = lowest_point_on_circle(*circle)
        candidates.append(lp)
        print(f'lowest circle point {lp} circle = {circle}')
        # intersect this circle with the two uninvoled spheres
        for i in range(4):
            if i not in pair:
                pts = sphere_circle_intersection(positions[i], lengths[i], *circle)
                if len(pts) == 2:
                    # take the lower point
                    if pts[0][2] < pts[1][2]:
                        candidates.append(pts[0])
                    else:
                        candidates.append(pts[1])
    candidates = np.array(candidates)
    print(f'pre filtered candidates {candidates}')
    # filter out candidates that are not inside all spheres
    ex_lengths = lengths + 1e-5
    distances = np.linalg.norm(candidates[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
    candidates = candidates[np.all(distances <= ex_lengths[np.newaxis, :], axis=1)]

    if len(candidates) == 0:
        return None

    # take the lowest point
    index_of_lowest = np.argmin(candidates[:, 2])
    # line length must exceed distance to point by this much to be considered slack
    slack_lines = distances[index_of_lowest] <= (ex_lengths - 0.02)
    return candidates[index_of_lowest], slack_lines


class Positioner2:
    def __init__(self, datastore, to_ui_q, to_pe_q, to_ob_q):
        self.run = True # run main loop
        self.datastore = datastore
        self.to_ui_q = to_ui_q
        self.to_pe_q = to_pe_q
        self.to_ob_q = to_ob_q
        self.config = Config()
        self.n_cables = len(self.config.anchors)
        self.anchor_points = np.array([
            [-2,2, 3],
            [ 2,2, 3],
            [ -1,2,-2],
            [ -2,2,-2],
        ], dtype=float)
        for i, a in enumerate(self.config.anchors):
            self.anchor_points[i] = np.array(compose_poses([a.pose, model_constants.anchor_grommet])[0])
        self.holding = False

    def estimate(self):
        """
        Estimate current gantry and gripper position and velocity
        """

        self.start = time()

        # Look at the last reading for each line.
        # if it is slack, discard it.
        # extrapolate the current length based on the time elapsed since the measurement and the speed at the time.
        lengths = []
        speeds = []
        for i, alr in enumerate(datastore.anchor_line_record):
            time, length, speed, tension = alr.getLast()
            # if tension > self.config.common_anchor_vars['TENSION_TIGHT_THRESH']:
            lengths.append(length)
            speeds.append(speed)

        z = no.zeros(3, dtype=float)

        # we cannot triangulate with fewer than 3 taught lines.
        if len(lengths) < 3:
            return z, z, False
        
        # nothing has been recorded    
        if sum(lengths) == 0:
            return z, z, False

        lengths = np.array(lengths)
        speeds = np.array(speeds)

        # find a position that minimizes the error between what the lengths would be at that position, and what they are now.
        result = find_hang_point(self.anchor_points, lengths)
        if result is None:
            return z, z, False
        gant_position, slack_lines = result

        # repeat for a position some small increment in the future to get the gantry velocity
        increment = 0.1 # seconds
        lengths += speeds * increment
        result = find_hang_point(self.anchor_points, lengths)
        if result is None:
            return gant_position, z, False
        gantry_vel = result[0] - gant_position

        # get the last IMU reading from the gripper
        gripper_rotvec = datastore.imu_rotvec.getLast()
        # get the winch line length
        time, length, speed = datastore.winch_line_record.getLast()
        # starting at the gantry, rotate the frame of reference by the gripper rotation
        # and translate along the negative z axis by the rope length
        swing_vec = compose_poses([
            (z, gant_position),
            (gripper_rotvec, np.array([0,0,-length], dtype=float)),
        ])
        grip_pos = swing_vec[1]

        self.time_taken = time() - self.start


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
                # if 'weight_change' in update:
                #     idx, val = update['weight_change']
                #     self.weights[idx] = 2**(val-5)
                if 'holding' in update:
                    self.holding = update['holding']
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
    pe = Positioner2(shared_datastore, to_ui_q, to_pe_q, to_ob_q)
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
        pe = Positioner2(datastore, to_ui_q, to_pe_q, to_ob_q)
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(getattr(signal, 'SIGINT'), stop)
        await pe.main()
    asyncio.run(main())