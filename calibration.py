import cv2
import cv2.aruco as aruco
import numpy as np
import scipy.optimize as optimize
from time import time, sleep
import glob
from config import Config
from cv_common import compose_poses
import model_constants
from spools import SpiralCalculator
from itertools import combinations

#the number of squares on the board (width and height)
board_w = 14
board_h = 9
# side length of one square in meters
board_dim = 0.075

# when on the raspi, just collect the images. it doesn't have enough ram to analyze them.
def collect_images():
    from picamera2 import Picamera2
    from libcamera import Transform, controls
    picam2 = Picamera2()
    capture_config = picam2.create_still_configuration(main={"size": (4608, 2592), "format": "RGB888"})
    picam2.configure(capture_config)
    picam2.start()
    picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.000001, "AfSpeed": controls.AfSpeedEnum.Fast}) 
    print("started pi camera")
    sleep(1)
    for i in range(50):
        sleep(1)
        im = picam2.capture_array()
        cv2.imwrite(f"images/cal/cap_{i}.jpg", im)
        sleep(1)
        print(f'collected ({i+1}/20)')

def collect_images_stream():
    video_uri = 'tcp://192.168.1.151:8888'
    print(f'Connecting to {video_uri}')
    cap = cv2.VideoCapture(video_uri)
    print(cap)
    i = 0
    while i < 50:
        ret, frame = cap.read()
        if not ret:
            continue
        fpath = f'images/cal/cap_{i}.jpg'
        cv2.imwrite(fpath, frame)
        i += 1
        print(f'saved frame to {fpath}')
        sleep(1)

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
        print(self.objp)

        self.images_obtained = 0
        self.image_shape = None
        self.cnt = 0

    def addImage(self, image):
        #Convert to grayscale
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.image_shape = grey_image.shape[::-1]
        #Find chessboard corners
        print(f'search image {self.cnt}')
        self.cnt+=1
        found, corners = cv2.findChessboardCornersSB(grey_image, (board_w,board_h), cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        # found, corners = cv2.findChessboardCorners(grey_image, (board_w,board_h), cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_ADAPTIVE_THRESH)

        if found == True:
            #Add the "true" checkerboard corners
            self.opts.append(self.objp)

            self.ipts.append(corners)
            self.images_obtained += 1 
            print(f"chessboards obtained {self.images_obtained}")

            image = cv2.drawChessboardCorners(image, (14,9), corners, found)
        # this resize is only for display and should not affect calibration
        image = cv2.resize(image, (2304, 1296),  interpolation = cv2.INTER_LINEAR)
        cv2.imshow('img', image)
        cv2.waitKey(500)

    def calibrate(self):
        if self.images_obtained < 20:
            raise RuntimeError(f'Obtained {self.images_obtained} images of checkerboard. Required 20')

        print('Running Calibrations...')
        ret, self.intrinsic_matrix, self.distCoeff, rvecs, tvecs = cv2.calibrateCamera(
            self.opts, self.ipts, self.image_shape, None, None)

        #Save matrices
        print('Intrinsic Matrix: ')
        print(str(self.intrinsic_matrix))
        print('Distortion Coefficients: ')
        print(str(self.distCoeff))
        print('Calibration complete')

        #Calculate the total reprojection error.  The closer to zero the better.
        tot_error = 0
        for i in range(len(self.opts)):
            imgpoints2, _ = cv2.projectPoints(self.opts[i], rvecs[i], tvecs[i], self.intrinsic_matrix, self.distCoeff)
            error = cv2.norm(self.ipts[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            tot_error += error
        terr = tot_error/len(self.opts)
        print("Total reprojection error: ", terr)

    def save(self): 
        print(f'Saving data to configuration.json')
        config = Config()
        config.intrinsic_matrix = self.intrinsic_matrix
        config.distortion_coeff = self.distCoeff
        config.resolution = (self.image_shape[1], self.image_shape[0])
        config.write()

# calibrate from files locally
def calibrate_from_files():
    ce = CalibrationInteractive()
    for filepath in glob.glob('images/cal/*.jpg'):
        print(f"analyzing {filepath}")
        image = cv2.imread(filepath)
        ce.addImage(image)
    ce.calibrate()
    ce.save()

def calibrate_from_stream():
    video_uri = 'tcp://192.168.1.151:8888'
    print(f'Connecting to {video_uri}')
    cap = cv2.VideoCapture(video_uri)
    print(cap)
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

def calibration_cost_fn(params, observations, spools):
    """Return the mean squared difference between line lengths calculated from encoders, and from visual observation

    params - a reduced pose and zero angle. the pose only contains x and y position
        (3 rot, 2 pos, za) * 4.
        the zero angle refers to the encoder position in revolutions where the amount of unspooled line would be zero.

    observations - a list with one entry per sample position.
        At each sample position, the gantry was still, and the lines were tight.
        Each sample position will have
        - encoder angles for each of the four anchor spool motors
        - four lists of visual observations, one for each of the four anchor cameras. Each visual observation contains
        - the pose of an observed gantry aruco marker in the camera's 3d coordinate system. (the output of cv2.solvePnP)

    spools - a list of four SpiralCalculators to relate zero angles to line lengths.
    """

    # Extract parameters and update spools in a vectorized manner
    params = params.reshape((4, 6))
    zero_angles = params[:, 5].copy()
    anchor_poses = params.reshape((4, 2, 3))
    anchor_poses[:,1,2] = 2.5 # the z position doesn't affect the cost function. any constant will do.

    for i in range(4):
        spools[i].set_zero_angle(zero_angles[i])

    # calculation of anchor grommet positions
    anchor_grommets = np.array([compose_poses([p, model_constants.anchor_grommet])[1] for p in anchor_poses])

    all_errs = []
    for sample in observations:
        # Calculation of encoder-based lengths
        encoder_based_lengths = np.array([spools[i].get_unspooled_length(sample['encoders'][i]) for i in range(4)])

        # Consolidate visual observations and gantry pose calculations
        gantry_poses = []
        for anchor_num, obs_list in enumerate(sample['visuals']):
            # note that it is common for there to be zero observations from one or more of the cameras.
            for pose in obs_list:
                # Calculate global gantry pose from each visual observation
                global_gantry_pose = compose_poses([
                    anchor_poses[anchor_num],
                    model_constants.anchor_camera,
                    pose,
                    model_constants.gantry_aruco_front_inv,
                ])
                gantry_poses.append(global_gantry_pose[1])

        if gantry_poses:
            gantry_poses = np.array(gantry_poses)

            # Distance calculation for all gantry poses against all 4 anchors
            visual_based_lengths = np.linalg.norm(
                anchor_grommets.reshape(1, 4, 3) - gantry_poses.reshape(gantry_poses.shape[0], 1, 3),
                axis=-1)

            # error calculation
            errors_per_sample = (visual_based_lengths - encoder_based_lengths.reshape(1, 4)).flatten()
            all_errs.append(errors_per_sample)

    if all_errs:
        # Concatenate all errors from all samples and calculate mean squared error
        all_errs_array = np.concatenate(all_errs)
        return np.mean(all_errs_array**2)
    else:
        return 0.0 # Return 0 if there are no errors to calculate

def calibration_cost_fn_new(params, observations, spools):
    """
    Return the mean squared error, including line length differences and
    a new cost for inconsistencies between gantry positions implied by different cameras.

    params - a reduced pose and zero angle. the pose only contains x and y position
        (3 rot, 2 pos, za) * 4.
        the zero angle refers to the encoder position in revolutions where the amount of unspooled line would be zero.

    observations - a list with one entry per sample position.
        At each sample position, the gantry was still, and the lines were tight.
        Each sample position will have
        - encoder angles for each of the four anchor spool motors
        - four lists of visual observations, one for each of the four anchor cameras. Each visual observation contains
        - the pose of an observed gantry aruco marker in the camera's 3d coordinate system. (the output of cv2.solvePnP)

    spools - a list of four SpiralCalculators to relate zero angles to line lengths.
    """

    # Extract parameters and update spools in a vectorized manner
    params = params.reshape((4, 6))
    zero_angles = params[:, 5].copy()
    anchor_poses = params.reshape((4, 2, 3))
    anchor_poses[:, 1, 2] = 2.5  # the z position doesn't affect the cost function. any constant will do.

    for i in range(4):
        spools[i].set_zero_angle(zero_angles[i])

    # calculation of anchor grommet positions
    anchor_grommets = np.array([compose_poses([p, model_constants.anchor_grommet])[1] for p in anchor_poses])

    all_length_errs = []
    all_consistency_errs = []

    for sample in observations:
        # Calculation of encoder-based lengths
        encoder_based_lengths = np.array([spools[i].get_unspooled_length(sample['encoders'][i]) for i in range(4)])

        # Consolidate visual observations and calculate implied gantry poses, grouped by anchor camera
        gantry_positions_by_anchor = [[] for _ in range(4)]
        for anchor_num, obs_list in enumerate(sample['visuals']):
            # note that it is common for there to be zero observations from one or more of the cameras.
            for pose in obs_list:
                # Calculate global gantry pose from each visual observation
                global_gantry_pose = compose_poses([
                    anchor_poses[anchor_num],
                    model_constants.anchor_camera,
                    pose,
                    model_constants.gantry_aruco_front_inv,
                ])
                # Store the implied gantry position (the translational part of the pose)
                gantry_positions_by_anchor[anchor_num].append(global_gantry_pose[1])

        # position consistency error
        # This cost penalizes deviations in the calculated gantry position between different anchor cameras.
        gantry_positions_by_anchor_np = [np.array(positions) for positions in gantry_positions_by_anchor]
        
        # Iterate over every unique pair of anchor cameras
        for i, j in combinations(range(4), 2):
            pos_i = gantry_positions_by_anchor_np[i]
            pos_j = gantry_positions_by_anchor_np[j]
            
            # Only compute if both cameras have observations for this sample
            if pos_i.size > 0 and pos_j.size > 0:
                # Use broadcasting to efficiently compute all pairwise distances
                # between implied gantry positions from camera i and camera j.
                diffs = pos_i[:, np.newaxis, :] - pos_j[np.newaxis, :, :]
                distances = np.linalg.norm(diffs, axis=2)
                all_consistency_errs.append(distances.flatten())
        
        # Encoder to visual error
        # First, flatten the list of poses for the original error calculation
        all_gantry_pos_flat = [pos for sublist in gantry_positions_by_anchor for pos in sublist]

        if all_gantry_pos_flat:
            gantry_positions = np.array(all_gantry_pos_flat)

            # Distance calculation for all gantry poses against all 4 anchors
            visual_based_lengths = np.linalg.norm(
                anchor_grommets.reshape(1, 4, 3) - gantry_positions.reshape(gantry_positions.shape[0], 1, 3),
                axis=-1)

            # error calculation
            errors_per_sample = (visual_based_lengths - encoder_based_lengths.reshape(1, 4)).flatten()
            all_length_errs.append(errors_per_sample)

    # --- Combine both error sources and calculate final cost ---
    all_errors_combined = []
    if all_length_errs:
        all_errors_combined.append(np.concatenate(all_length_errs))
    if all_consistency_errs:
        # apply a weighting factor to this term to balance its influence
        all_errors_combined.append(np.concatenate(all_consistency_errs) * 0.4)
    if all_errors_combined:
        # Concatenate all errors from all sources and samples, then calculate mean squared error
        final_errs_array = np.concatenate(all_errors_combined)
        return np.mean(final_errs_array**2)
    else:
        return 0.0  # Return 0 if there are no errors to calculate


def find_cal_params(current_anchor_poses, observations, large_spool_index):
    """ Find optimal anchor poses and zero angles based on previously collected data.

    anchor Z positions don't affect the cost function.
    that is the whole system would be identical if the floor were lower or higher
    since we never touch the floor at all, so we don't optimize for that value, we just average the z vals
    that were obtained from the origin card observations
    """
    spools = []
    initial_guess = []
    bounds = []
    average_z = np.mean(current_anchor_poses[:,1,2])
    for i, apose in enumerate(current_anchor_poses):
        # Determine the full diameter based on the index
        full_diameter = 47.7 if i == large_spool_index else 27.0
        full_length = 7.5

        # initialize a model of the spool to relate zero angle, encoder readinds, and line lengths
        spools.append(SpiralCalculator(
            empty_diameter=25,
            full_diameter=full_diameter,
            full_length=full_length,
            gear_ratio=20/51,
            motor_orientation=-1
        ))

        guess = [
            *apose[0],    # xyz rotation component
            apose[1][0],  # x position component
            apose[1][1],  # y position component
            -150,         # initial guess of zero angle
        ]
        initial_guess.append(guess)

        anchor_bounds = [
            (guess[0] - 0.2, guess[0] + 0.2), # x component of rotation vector
            (guess[1] - 0.2, guess[1] + 0.2), # y component of rotation vector
            (guess[2] - 0.2, guess[2] + 0.2), # z component of rotation vector
            (-8, 8), # x component of position
            (-8, 8), # y component of position
            (-400, 400), # bounds on zero angle in revs.
        ]
        bounds.append(anchor_bounds)

    initial_guess =  np.array(initial_guess).flatten()
    bounds =  np.array(bounds).reshape((len(initial_guess), 2))

    result = optimize.minimize(
        calibration_cost_fn_new,
        initial_guess,
        args=(observations, spools),
        method='SLSQP',
        bounds=bounds,
        options={
            'disp': False,
            'maxiter': 1000000,
        })

    try:
        assert result.success
        # rot is a rodrruigez vector. pos is a position.
        poses = result.x.reshape((4, 6)) # move last number out, it's not part of the pose.
        zero_angles = poses[:,5].copy()
        poses[:,5] = average_z
        poses = poses.reshape((4,2,3))
        # return four poses and four zero angles. [(*rot, *pos), ...], [spool_za, ...]
        return poses, zero_angles
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

if __name__ == "__main__":
    # calibrate_from_stream()
    # collect_images_stream()
    # calibrate_from_files()

    import pickle
    with open('tests/collected_cal_data.pickle', 'rb') as f:
        ap, data = pickle.load(f)
    print(f'Anchor poses that were assumed during calibration \n{ap}')
    result_params = find_cal_params(ap, data, 2)
    print(result_params)