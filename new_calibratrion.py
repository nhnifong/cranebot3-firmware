
import numpy as np
import scipy.optimize as optimize
from cv_common import compose_poses
import model_constants
from spools import SpiralCalculator

def calibration_cost_fn(params, observations, spools):
    """Return the mean squared difference between line lengths calculated from encoders, and from visual observation

    params - a pose and zero angle, full diameter, and full length for each anchor
        (3 rot, 3 pos, za, full_d, full_len) * 4.
        the zero angle refers to the encoder position in revolutions where the amount of unspooled line would be zero.
        full diameter refers to the wrapping diameter of the spool when it is full in millimeters
        full length refers to the length of line in meters that is wrapped on the spool when the gantry is reeled in all the way

    observations - a list with one entry per sample position.
        At each sample position, the gantry was still, and the lines were tight.
        Each sample position will have
        - encoder angles for each of the four anchor spool motors
        - four lists of visual observations, one for each of the four anchor cameras. Each visual observation contains
          - the pose of an observed gantry aruco marker in the camera's 3d coordinate system. (the output of cv2.solvePnP)

    spools - a list of four SpiralCalculators to relate zero angles to line lengths.
    """

    # extract parameters
    params = params.reshape((4,9))
    anchor_poses = []
    for i, ap in enumerate(params):
        anchor_poses.append(ap[:6].reshape((2,3)))
        spools[i].set_zero_angle(ap[6])
        spools[i].recalc_k_params(ap[7], ap[8])

    # obtain the position of the anchor grommet relative to the global origin
    anchor_grommets = np.array([compose_poses([      
        pose,                             # the pose of the anchor relative to a chosen origin.
        model_constants.anchor_grommet,   # the pose of the grommet relative to the anchor
    ])[1] for pose in anchor_poses])

    all_errs = []
    for sample in observations:

        # Calculate the line lengths from zero angles and encoder positions at this sample point.
        encoder_based_lengths = [
            spools[i].get_unspooled_length(sample['encoders'][i])
            for i in range(4)]

        # Every visual observation implies a particular gantry position. calculate the line lengths based
        # on this position and the anchor poses in the parameters being optimized.
        for anchor_num, obs_list in enumerate(sample['visuals']):
            for pose in obs_list:

                # obtain the position of the gantry and anchor in the same coodinate space.
                # Specifically the position of the keyring where the four lines meet, which is also the model's origin.
                global_gantry_pose = compose_poses([
                    anchor_poses[anchor_num],               # the pose of the anchor relative to a chosen origin.
                    model_constants.anchor_camera,          # the pose of the camera relative to the anchor
                    pose,                                   # the pose of the aruco marker relative to the camera
                    model_constants.gantry_aruco_front_inv, # the pose of the gantry relative to the aruco marker
                ])

                # calculate the distance between the two line endpoints
                distance = np.linalg.norm(anchor_grommets[anchor_num] - global_gantry_pose[1])

                # calculate the error between the visually implied length and the encoder implied length
                # append it to the list of errors
                all_errs.append(distance - encoder_based_lengths[anchor_num])

    # return the mean squared error
    return np.mean(np.array(all_errs)**2)


def find_cal_params(current_anchor_poses, observations):
    spools = []
    initial_guess = []
    bounds = []
    for apose in current_anchor_poses:
        # initialize a model of the spool to relate zero angle, encoder readinds, and line lengths
        spools.append(SpiralCalculator(empty_diameter=25, full_diameter=27, full_length=7.5, gear_ratio=20/51, motor_orientation=-1))

        guess = [
            *apose[0], # rotation component
            *apose[1], # position component
            0,         # initial guess of zero angle
            27,        # full diameter in millimeters
            7.5,       # full length in meters
        ]
        initial_guess.append(guess)

        anchor_bounds = [
            (guess[0] - 0.2, guess[0] + 0.2), # x component of rotation vector
            (guess[1] - 0.2, guess[1] + 0.2), # y component of rotation vector
            (guess[2] - 0.2, guess[2] + 0.2), # z component of rotation vector
            (-8, 8), # x component of position
            (-8, 8), # y component of position
            ( 1, 6), # z component of position
            (-400, 400), # bounds on zero angle in revs.
            (25.1, 55),  # bounds on full diameter in mm.
            (4, 10),     # bounds on full length in meters
        ]
        bounds.append(anchor_bounds)

    initial_guess =  np.array(initial_guess).flatten()
    bounds =  np.array(bounds).reshape((len(initial_guess), 2))

    result = optimize.minimize(
        calibration_cost_fn,
        initial_guess,
        args=(observations, spools),
        method='SLSQP',
        bounds=bounds,
        options={
            'disp': False,
        })

    try:
        assert result.success
        return result.x.reshape((4, 3, 3))
    except AssertionError:
        print(result)
        return

def order_points_for_low_travel(points):
    """
    Orders a list of 3D points to achieve a low total travel distance
    using an Greedy Bidirectional Nearest Neighbor heuristic.

    This algorithm starts at an arbitrary point, then at each step, it
    finds the closest unvisited point to both the current beginning and end
    of the path. It extends the path from the end that has a shorter
    connection to an unvisited point.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) where N is the
                             number of points, and each row is an (x, y, z)
                             coordinate.

    Returns:
        tuple: A tuple containing:
            - ordered_points (np.ndarray): The points reordered along the path.
            - total_distance (float): The total Euclidean distance of the path.
                                      Returns (empty array, 0.0) if input is empty.

    Function written by AI
    """
    if points.shape[0] == 0:
        print("Input points array is empty. Returning empty ordered_points and 0.0 distance.")
        return np.array([]), 0.0
    if points.shape[0] == 1:
        print("Only one point provided. Returning the point and 0.0 distance.")
        return points, 0.0

    num_points = points.shape[0]
    
    # Keep track of visited points using a boolean array for efficiency
    visited = np.zeros(num_points, dtype=bool)
    
    # Start with the first point (arbitrary initial choice)
    start_index = 0
    
    # The path is maintained as a deque or by managing head and tail indices
    # We'll use a list and manage head/tail pointers
    ordered_indices = [start_index]
    visited[start_index] = True
    
    current_head_idx = start_index # Index of the point at the current beginning of the path
    current_tail_idx = start_index # Index of the point at the current end of the path
    
    total_distance = 0.0

    # Loop until all points are visited (num_points - 1 connections to make)
    for _ in range(num_points - 1):
        min_dist_head = float('inf')
        nearest_to_head_idx = -1
        
        min_dist_tail = float('inf')
        nearest_to_tail_idx = -1
        
        # Find the nearest unvisited neighbor for the current head and tail
        for i in range(num_points):
            if not visited[i]:
                # Distance from current head to candidate point i
                dist_from_head = np.linalg.norm(points[current_head_idx] - points[i])
                if dist_from_head < min_dist_head:
                    min_dist_head = dist_from_head
                    nearest_to_head_idx = i
                
                # Distance from current tail to candidate point i
                dist_from_tail = np.linalg.norm(points[current_tail_idx] - points[i])
                if dist_from_tail < min_dist_tail:
                    min_dist_tail = dist_from_tail
                    nearest_to_tail_idx = i
        
        # Decide which end to extend based on the shorter distance
        # Ensure a valid neighbor was found for at least one end
        if nearest_to_head_idx == -1 and nearest_to_tail_idx == -1:
            # This should only happen if all points are visited, but a safety break
            break

        if nearest_to_head_idx != -1 and (nearest_to_tail_idx == -1 or min_dist_head <= min_dist_tail):
            # Extend from the head
            ordered_indices.insert(0, nearest_to_head_idx) # Prepend to the list
            visited[nearest_to_head_idx] = True
            total_distance += min_dist_head
            current_head_idx = nearest_to_head_idx
        elif nearest_to_tail_idx != -1: # min_dist_tail < min_dist_head
            # Extend from the tail
            ordered_indices.append(nearest_to_tail_idx) # Append to the list
            visited[nearest_to_tail_idx] = True
            total_distance += min_dist_tail
            current_tail_idx = nearest_to_tail_idx
        else:
            # Fallback for unexpected scenarios (e.g., if only one end had a valid neighbor)
            # This logic should cover all cases, but a print for debugging if needed
            print("Warning: Neither head nor tail could find a valid next point. Breaking loop.")
            break
            
    # Convert the list of ordered indices to an array of points
    ordered_points = points[ordered_indices]
    
    return ordered_points, total_distance