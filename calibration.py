import cv2
import cv2.aruco as aruco
import numpy as np
import scipy.optimize as optimize
from time import time, sleep
import glob
from config import Config
from cv_common import compose_poses, gantry_april_inv
import model_constants
from spools import SpiralCalculator
from itertools import combinations
from position_estimator import find_hang_point
import argparse
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#the number of squares on the board (width and height)
board_w = 14
board_h = 9
# side length of one square in meters
board_dim = 0.075

def collect_images_locally_raspi(num_images, resolution_str):
    """
    Collects images locally on a Raspberry Pi using the picamera2 library.
    
    Args:
        num_images (int): The number of images to collect.
        resolution_str (str): The resolution as a string, e.g., "4608x2592".
    """
    try:
        from picamera2 import Picamera2
        from libcamera import Transform, controls
    except ImportError:
        logging.error("picamera2 or libcamera not found. This function is for Raspberry Pi only.")
        return
        
    width, height = map(int, resolution_str.split('x'))

    picam2 = Picamera2()
    capture_config = picam2.create_still_configuration(main={"size": (width, height), "format": "RGB888"})
    picam2.configure(capture_config)
    picam2.start()
    picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.000001, "AfSpeed": controls.AfSpeedEnum.Fast}) 
    logging.info("Started Pi camera.")
    sleep(1)
    for i in range(num_images):
        sleep(1)
        im = picam2.capture_array()
        cv2.imwrite(f"images/cal/cap_{i}.jpg", im)
        sleep(1)
        logging.info(f'Collected ({i+1}/{num_images}) images.')

def collect_images_stream(address, num_images):
    """
    Connects to a video stream and collects a specified number of images.
    
    Args:
        address (str): The network address of the video stream.
        num_images (int): The number of images to collect.
    """
    logging.info(f'Connecting to {address}...')
    cap = cv2.VideoCapture(address)
    logging.debug(f'Video capture object: {cap}')
    last_cap_time = time()
    i = 0
    while i < num_images:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to capture frame from stream. Retrying...")
            return
        if time() > last_cap_time+1:
            fpath = f'images/cal/cap_{i}.jpg'
            cv2.imwrite(fpath, frame)
            i += 1
            logging.info(f'Saved frame to {fpath}')
            last_cap_time = time()

def is_blurry(image, threshold=6.0):
    """
    Checks if an image is too blurry based on Laplacian variance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #ensure grayscale
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

# calibrate interactively
class CalibrationInteractive:
    def __init__(self):
        #Initializing variables
        board_n = board_w * board_h
        self.opts = []
        self.ipts = []
        self.intrinsic_matrix = np.zeros((3, 3), np.float32)
        self.distCoeffs = np.zeros((5, 1), np.float32)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        # prepare object points based on the actual dimensions of the calibration board
        # like (0,0,0), (25,0,0), (50,0,0) ....,(200,125,0)
        self.objp = np.zeros((board_n,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1,2)
        self.objp = self.objp * board_dim
        logging.debug(f'Object points:\n{self.objp}')

        self.images_obtained = 0
        self.image_shape = None
        self.cnt = 0

    def addImage(self, image):
        #Convert to grayscale
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.image_shape = grey_image.shape[::-1]
        #Find chessboard corners
        logging.debug(f'Searching image {self.cnt}')
        self.cnt+=1
        found, corners = cv2.findChessboardCornersSB(grey_image, (board_w,board_h), cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        # found, corners = cv2.findChessboardCorners(grey_image, (board_w,board_h), cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_ADAPTIVE_THRESH)

        if found == True:
            #Add the "true" checkerboard corners
            self.opts.append(self.objp)

            self.ipts.append(corners)
            self.images_obtained += 1 
            logging.info(f"Chessboards obtained: {self.images_obtained}")

            image = cv2.drawChessboardCorners(image, (14,9), corners, found)
        # this resize is only for display and should not affect calibration
        image = cv2.resize(image, (1920, 1080),  interpolation = cv2.INTER_LINEAR)
        cv2.imshow('img', image)
        cv2.waitKey(500)

    def calibrate(self):
        if self.images_obtained < 20:
            logging.error(f'Obtained {self.images_obtained} images of checkerboard. Required 20.')
            raise RuntimeError(f'Obtained {self.images_obtained} images of checkerboard. Required 20')

        logging.info('Running calibrations...')
        ret, self.intrinsic_matrix, self.distCoeff, rvecs, tvecs = cv2.calibrateCamera(
            self.opts, self.ipts, self.image_shape, None, None)

        #Save matrices
        logging.info(f"Camera calibration performed with image resolution: {self.image_shape[0]}x{self.image_shape[1]}.")
        logging.info('Intrinsic Matrix:')
        logging.info(str(self.intrinsic_matrix))
        logging.info('Distortion Coefficients:')
        logging.info(str(self.distCoeff))
        logging.info('Calibration complete.')

        #Calculate the total reprojection error.  The closer to zero the better.
        tot_error = 0
        for i in range(len(self.opts)):
            imgpoints2, _ = cv2.projectPoints(self.opts[i], rvecs[i], tvecs[i], self.intrinsic_matrix, self.distCoeff)
            error = cv2.norm(self.ipts[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            tot_error += error
        terr = tot_error/len(self.opts)
        logging.info(f"Total reprojection error: {terr}")

    def save(self): 
        logging.info('Saving data to configuration.json...')
        config = Config()
        config.intrinsic_matrix = self.intrinsic_matrix
        config.distortion_coeff = self.distCoeff
        config.resolution = (self.image_shape[1], self.image_shape[0])
        config.write()

# calibrate from files locally
def calibrate_from_files():
    ce = CalibrationInteractive()
    for filepath in glob.glob('images/cal/*.jpg'):
        logging.info(f"Analyzing {filepath}")
        image = cv2.imread(filepath)
        ce.addImage(image)
    ce.calibrate()
    ce.save()

def calibrate_from_stream(address):
    logging.info(f'Connecting to {address}...')
    cap = cv2.VideoCapture(address)
    logging.debug(f'Video capture object: {cap}')
    ce = CalibrationInteractive()
    i=0
    while ce.images_obtained < 20:
        ret, frame = cap.read()
        if ret and i%10==0:
            ce.addImage(frame)
        i+=1
    ce.calibrate()
    ce.save()

### the following functions pertain to calibration of the entire assembled robot, not just one camera ###

def params_consistent_with_move(
    starting_pos_visual, ending_pos_visual,
    encoder_lengths1, encoder_lengths2,
    anchor_points):
    """
    Return a score measuring how well params predict an observed move.
    """
    # Calculate the visually-based line lengths for the starting position.
    # This uses the mean gantry position from the first observation.
    starting_lengths_visual = np.linalg.norm(anchor_points - starting_pos_visual, axis=1)

    # Calculate the change in line lengths based on encoder readings between the two points.
    encoder_deltas = encoder_lengths2 - encoder_lengths1

    # Predict the new line lengths by adding the encoder deltas to the visual starting lengths.
    predicted_ending_lengths = starting_lengths_visual + encoder_deltas

    # Using the predicted new lengths, calculate a hypothetical hang point.
    # logging.debug(f'try to find hang point with anchor_points={anchor_points}, lengths={predicted_ending_lengths}')
    hp_result = find_hang_point(anchor_points, predicted_ending_lengths)
    if hp_result is None:
        return 1 # 1 meter
    predicted_hang_point = hp_result[0] # element 0 is position, element 1 is predicted line tightness.

    # Compare the predicted hang point to the observed visual ending position.
    # The smaller the distance, the more consistent the parameters are with the move.
    # logging.debug(f'hp={predicted_hang_point} ep={ending_pos_visual}')
    return np.linalg.norm(predicted_hang_point - ending_pos_visual)


def calibration_cost_fn(params, observations, spools, mode='full', fixed_poses=None):
    """
    Return the mean squared error for the calibration parameters.

    This function can operate in two modes:
    - 'full': Optimizes anchor poses (rotation, x, y) and zero angles.
      `params` is a flat array of 24 values (4 anchors * 6 params).
    - 'zero_angles_only': Optimizes only the zero angles, using pre-calculated poses.
      `params` is a flat array of 4 zero angles. `fixed_poses` must be provided.
    """
    # Unpack parameters based on the operating mode
    if mode == 'full':
        # The incoming `params` is a flat array of 24 elements.
        # Reshape it to our logical (4 anchors, 6 params) structure.
        params_matrix = params.reshape((4, 6))
        # Explicitly unpack each part of the matrix into named variables.
        rodrigues_vectors = params_matrix[:, 0:3]  # Columns 0, 1, 2 are rotation
        positions_xy = params_matrix[:, 3:5]      # Columns 3, 4 are position X, Y
        zero_angles = params_matrix[:, 5]         # Column 5 is the zero angle

        # construct the anchor_poses array.
        anchor_poses = np.zeros((4, 2, 3))
        # Fill the rotation part
        anchor_poses[:, 0, :] = rodrigues_vectors
        # Fill the position part
        anchor_poses[:, 1, 0:2] = positions_xy
        anchor_poses[:, 1, 2] = 2.5  # Set a constant Z value directly

    elif mode == 'zero_angles_only':
        # In 'zero_angles_only' mode, params are just the zero angles
        assert fixed_poses is not None, "fixed_poses must be provided in 'zero_angles_only' mode."
        zero_angles = params
        anchor_poses = fixed_poses
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for i in range(4):
        spools[i].set_zero_angle(zero_angles[i])

    # Proceed with cost calculation
    anchor_grommets = np.array([compose_poses([p, model_constants.anchor_grommet])[1] for p in anchor_poses])

    all_length_errs = []
    all_consistency_errs = []

    # save some results during the first loop for the move consistency calculation.
    intermediate_data = []

    for sample in observations:
        encoder_based_lengths = np.array([spools[i].get_unspooled_length(sample['encoders'][i]) for i in range(4)])
        # meters from the bottom of the gripper to the floor (or furniture)
        # laser_range = sample['laser_range']

        gantry_positions_by_anchor = [[] for _ in range(4)]
        for anchor_num, obs_list in enumerate(sample['visuals']):
            for pose in obs_list:
                global_gantry_pose = compose_poses([
                    anchor_poses[anchor_num],
                    model_constants.anchor_camera,
                    pose,
                    gantry_april_inv,
                ])
                gantry_positions_by_anchor[anchor_num].append(global_gantry_pose[1])

        # Position consistency error (only relevant in full mode)
        if mode == 'full':
            gantry_positions_by_anchor_np = [np.array(positions) for positions in gantry_positions_by_anchor]
            for i, j in combinations(range(4), 2):
                pos_i = gantry_positions_by_anchor_np[i]
                pos_j = gantry_positions_by_anchor_np[j]
                if pos_i.size > 0 and pos_j.size > 0:
                    diffs = pos_i[:, np.newaxis, :] - pos_j[np.newaxis, :, :]
                    distances = np.linalg.norm(diffs, axis=2)
                    all_consistency_errs.append(distances.flatten())

        gantry_positions = np.array([pos for sublist in gantry_positions_by_anchor for pos in sublist])
        if len(gantry_positions) == 0:
            continue # This point should not have even been recorded.

        # Encoder to visual error
        visual_based_lengths = np.linalg.norm(
            anchor_grommets.reshape(1, 4, 3) - gantry_positions.reshape(gantry_positions.shape[0], 1, 3),
            axis=-1
        )
        errors_per_sample = (visual_based_lengths - encoder_based_lengths.reshape(1, 4)).flatten()
        all_length_errs.append(errors_per_sample)

        intermediate_data.append({
            'mean_gantry_pos': np.mean(gantry_positions, axis=0),
            'encoder_lengths': encoder_based_lengths
        })

        # todo: add another error term to constrain scale and z offset.
        # take readings of gripper winch line and laser rangefidner with each sample point.
        # Every gantry position implies a certain floor z level. all of these should be equal.
        # these measurements however would be invalid if the sample point were over a piece of furniture,
        # so this error term may only be done for positions over the origin card.

    # Calculate move consistency error between consecutive points ---
    all_move_consistency_errs = []
    for i in range(1, len(intermediate_data)):
        point1 = intermediate_data[i-1]
        point2 = intermediate_data[i]
        
        # Ensure both consecutive points had valid visual data
        if point1 and point2:
            err = params_consistent_with_move(
                starting_pos_visual=point1['mean_gantry_pos'],
                ending_pos_visual=point2['mean_gantry_pos'],
                encoder_lengths1=point1['encoder_lengths'],
                encoder_lengths2=point2['encoder_lengths'],
                anchor_points=anchor_grommets
            )
            all_move_consistency_errs.append(err)

    # Combine errors and return final cost
    all_errors_combined = []
    if all_length_errs:
        all_errors_combined.append(np.concatenate(all_length_errs))
    if all_consistency_errs:
        all_errors_combined.append(np.concatenate(all_consistency_errs) * 0.5)
    if all_move_consistency_errs:
        all_errors_combined.append(np.array(all_move_consistency_errs) * 0.3)

    if all_errors_combined:
        final_errs_array = np.concatenate(all_errors_combined)
        return np.mean(final_errs_array**2)
    else:
        return 0.0


def find_cal_params(current_anchor_poses, observations, large_spool_index, mode='full'):
    """
    Find optimal calibration parameters based on previously collected data.

    This function can be called in two modes:
    - mode='full': Optimizes anchor poses (rotation, x, y) and zero angles.
    - mode='zero_angles_only': Uses the provided `current_anchor_poses` as
      fixed values and only optimizes the zero angles.
    """
    spools = []
    average_z = np.mean(current_anchor_poses[:, 1, 2])
    for i in range(4):
        full_diameter = model_constants.full_spool_diameter_power_line if i == large_spool_index else model_constants.full_spool_diameter_fishing_line
        spools.append(SpiralCalculator(
            empty_diameter=model_constants.empty_spool_diameter,
            full_diameter=full_diameter,
            full_length=model_constants.assumed_full_line_length,
            gear_ratio=20/51,
            motor_orientation=-1
        ))

    if mode == 'full':
        initial_guess = []
        bounds = []
        for apose in current_anchor_poses:
            guess = [
                *apose[0],    # xyz rotation component
                apose[1][0],  # x position component
                apose[1][1],  # y position component
                -150,          # initial guess of zero angle
            ]
            initial_guess.append(guess)
            bounds.append([
                (guess[0] - 0.2, guess[0] + 0.2),
                (guess[1] - 0.2, guess[1] + 0.2),
                (guess[2] - 0.2, guess[2] + 0.2),
                (-8, 8),
                (-8, 8),
                (-400, 400),
            ])
        
        initial_guess = np.array(initial_guess).flatten()
        bounds = np.array(bounds).reshape((len(initial_guess), 2))
        args = (observations, spools, 'full', None)
        logging.info(f"Number of observations: {len(observations)}")

    elif mode == 'zero_angles_only':
        initial_guess = np.array([-150.0] * 4)
        bounds = [(-400, 400)] * 4
        # Pass the fixed poses to the cost function via args
        args = (observations, spools, 'zero_angles_only', current_anchor_poses)

    else:
        raise ValueError(f"Unknown optimization mode: {mode}")

    result = optimize.minimize(
        calibration_cost_fn,
        initial_guess,
        args=args,
        method='SLSQP',
        bounds=bounds,
        options={'disp': False, 'maxiter': 10000}
    )

    # --- Process and return results based on the mode ---
    try:
        assert result.success
        if mode == 'full':
            # Unpack the full 24-parameter result
            params = result.x.reshape((4, 6))
            zero_angles = params[:, 5].copy()
            # Reconstruct poses and set the fixed average Z
            poses = params.reshape((4, 2, 3))
            poses[:, 1, 2] = average_z
        else: # mode == 'zero_angles_only'
            # The poses are the ones we passed in, and the result is the zero angles
            poses = current_anchor_poses
            zero_angles = result.x
        
        return poses, zero_angles
    except AssertionError:
        logging.error(f"Optimization failed. Result:\n{result}")
        return None, None

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
        logging.warning("Input points array is empty. Returning empty ordered_points and 0.0 distance.")
        return np.array([]), 0.0
    if points.shape[0] == 1:
        logging.warning("Only one point provided. Returning the point and 0.0 distance.")
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
            logging.warning("Neither head nor tail could find a valid next point. Breaking loop.")
            break
            
    # Convert the list of ordered indices to an array of points
    ordered_points = points[ordered_indices]
    
    return ordered_points, total_distance

def calibrate_poses_from_data():
    import pickle
    with open('collected_cal_data.pickle', 'rb') as f:
        ap, data = pickle.load(f)
    logging.info(f'Anchor poses that were assumed during data collection:\n{ap}')
    poses, zero_a = find_cal_params(ap, data, 2)
    if poses is not None:
        logging.info(f"Optimized poses:\n{poses}")
        logging.info(f"Optimized zero angles:\n{zero_a}")

        bedroom_side_len = 5.334
        side_len = np.linalg.norm( poses[0][1] - poses[2][1] )
        #ideal here meaning the actual length I measured between these two anchors in my bedroom with a tape measure
        # while this is a crude indication, the best test in my experience of the quality of a calibration is the robot's
        # ability to move in cardinal directions while very close to a wall.
        logging.info(f"Side length = {side_len:.3f}m ({(side_len/bedroom_side_len)*100:.2f}% of ideal)")

        config = Config()
        for i, anchor in enumerate(config.anchors):
            anchor.pose = poses[i]
        config.write()
        logging.info('Wrote new anchor poses to configuration.json.')
    else:
        logging.error("Failed to find calibration parameters.")


def main():
    parser = argparse.ArgumentParser(description='Run robot calibration functions. Use --help for more details on each command.')
    parser.add_argument('--mode', type=str, choices=[
        'collect-images-stream',
        'calibrate-from-files',
        'calibrate-poses-from-data',
        'collect-images-locally-raspi',
        'calibrate-from-stream'
    ], required=True, help='Choose the calibration function to run:\n \
            "collect-images-stream" to capture a specified number of images from a network stream; \
            "calibrate-from-files" to run camera calibration on a local set of images; \
            "calibrate-poses-from-data" to optimize robot poses and zero angles from pre-collected data; \
            "collect-images-locally-raspi" to capture a specified number of images from a connected camera on a Raspberry Pi; \
            "calibrate-from-stream" to run camera calibration directly from a network stream until 20 images are collected.')
    parser.add_argument('--address', type=str, default='tcp://192.168.1.151:8888',
                        help='The network address for the video stream (used with stream modes).')
    parser.add_argument('--num-images', type=int, default=50,
                        help='The number of images to collect when using "collect-images-locally-raspi" or "collect-images-stream".')
    parser.add_argument('--resolution', type=str, default='4608x2592',
                        help='The resolution for the camera on the Raspberry Pi (e.g., "4608x2592"). Used with "collect-images-locally-raspi" mode.')

    args = parser.parse_args()

    if args.mode == 'collect-images-locally-raspi':
        collect_images_locally_raspi(args.num_images, args.resolution)
    elif args.mode == 'collect-images-stream':
        collect_images_stream(args.address, args.num_images)
    elif args.mode == 'calibrate-from-files':
        calibrate_from_files()
    elif args.mode == 'calibrate-poses-from-data':
        calibrate_poses_from_data()
    elif args.mode == 'calibrate-from-stream':
        calibrate_from_stream(args.address)

if __name__ == "__main__":
    main()
