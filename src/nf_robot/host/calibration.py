import cv2
import numpy as np
import scipy.optimize as optimize
import time
import glob
from itertools import combinations
import argparse
import logging

from nf_robot.robot.spools import SpiralCalculator
from nf_robot.common.pose_functions import *
import nf_robot.common.definitions as model_constants
from nf_robot.common.cv_common import *
from nf_robot.common.config_loader import *
from nf_robot.host.position_estimator import find_hang_point
from nf_robot.generated.nf import config

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
    time.sleep(1)
    for i in range(num_images):
        time.sleep(1)
        im = picam2.capture_array()
        cv2.imwrite(f"images/cal/cap_{i}.jpg", im)
        time.sleep(1)
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
    last_cap_time = time.time()
    i = 0
    while i < num_images:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to capture frame from stream. Retrying...")
            return
        if time.time() > last_cap_time+1:
            fpath = f'images/cal2/cap_{i}.png'
            cv2.imwrite(fpath, frame)
            i += 1
            logging.info(f'Saved frame to {fpath}')
            last_cap_time = time.time()

def is_blurry(image, threshold=6.0):
    """
    Checks if an image is too blurry based on Laplacian variance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #ensure grayscale
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

# calibrate interactively
class CalibrationInteractive:
    def __init__(self, config_file):
        #Initializing variables
        board_n = board_w * board_h
        self.opts = []
        self.ipts = []
        self.intrinsic_matrix = np.zeros((3, 3), np.float32)
        self.distCoeffs = np.zeros((5, 1), np.float32)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        self.config_file = config_file

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
        # ret, self.intrinsic_matrix, self.distCoeff, rvecs, tvecs = cv2.calibrateCamera(
        #     self.opts, self.ipts, self.image_shape, None, None)

        # Initialize the Matrix with the Image Center
        # This tells OpenCV: "Start assuming the lens is perfectly centered"
        w, h = self.image_shape
        self.intrinsic_matrix = np.array([
            [1000.0, 0.0,    w / 2.0], # f_x estimate, 0, c_x
            [0.0,    1000.0, h / 2.0], # 0, f_y estimate, c_y
            [0.0,    0.0,    1.0    ]
        ], dtype=np.float32)

        # Use Flags to Lock the Center
        # CALIB_USE_INTRINSIC_GUESS: Use the matrix above as the starting point
        # CALIB_FIX_PRINCIPAL_POINT: Do NOT move c_x and c_y during optimization
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT

        ret, self.intrinsic_matrix, self.distCoeff, rvecs, tvecs = cv2.calibrateCamera(
            self.opts, self.ipts, self.image_shape, 
            self.intrinsic_matrix, # Pass our initialized matrix
            None, 
            flags=flags # Pass our locking flags
        )

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
        logging.info(f'Saving data to {self.config_file}...')
        cfg = load_config(path=self.config_file)
        cfg.camera_cal.intrinsic_matrix = self.intrinsic_matrix.flatten().tolist()
        cfg.camera_cal.distortion_coeff = self.distCoeff.flatten().tolist()
        cfg.camera_cal.resolution.width = self.image_shape[0]
        cfg.camera_cal.resolution.height = self.image_shape[1]
        save_config(cfg, self.config_file)

# calibrate from files locally
def calibrate_from_files(config_file):
    ce = CalibrationInteractive(config_file)
    for filepath in glob.glob('images/cal/*.png'):
        logging.info(f"Analyzing {filepath}")
        image = cv2.imread(filepath)
        ce.addImage(image)
    ce.calibrate()
    ce.save()

def calibrate_from_stream(address, config_file):
    logging.info(f'Connecting to {address}...')
    cap = cv2.VideoCapture(address)
    logging.debug(f'Video capture object: {cap}')
    ce = CalibrationInteractive(config_file)
    i=0
    while ce.images_obtained < 20:
        ret, frame = cap.read()
        if ret and i%10==0:
            ce.addImage(frame)
        i+=1
    ce.calibrate()
    ce.save()

### the following functions pertain to calibration of the entire assembled robot, not just one camera ###

W_ORIGIN = 0.1 # increase this to to make origin errors more expensive
W_PLANAR = 1 # increase this to make anchor height deviations from the average plane more expensive

def multi_card_residuals(x, averages):
    """
    Computes the vector of residuals (differences) for least_squares.
    
    Returns:
        1D numpy array of floats. The optimizer minimizes sum(residuals**2).
        We return raw differences (meters), not squared errors.
    """
    anchor_poses = x.reshape((4, 2, 3))
    residuals = []
    
    # Iterate over every marker (Origin and Cal Assists)
    for marker_name, sightings in averages.items():

        # Calculate world positions for all cameras that saw this marker
        valid_sightings = []
        
        for anchor_idx, marker_pose_cams in enumerate(sightings):
            for marker_pose_cam in marker_pose_cams:
                print(f'marker_pose_cam=\n{marker_pose_cam}')
                if marker_pose_cam is None:
                    continue
                
                # Chain: Anchor -> Camera -> Marker
                pose_list = [
                    anchor_poses[anchor_idx],
                    model_constants.anchor_camera,
                    marker_pose_cam
                ]
                print(f'pose_list=\n{pose_list}')
                if marker_name == 'gantry':
                    pose_list.append(gantry_april_inv)
                pose_in_room = compose_poses(pose_list)
                
                # Extract translation (tvec is index 1)
                valid_sightings.append(pose_in_room[1])

        if not valid_sightings:
            continue
            
        projected_positions = np.array(valid_sightings)
        
        # Calculate Residuals based on marker type
        if marker_name == 'origin':
            # constraint 1: Origin must be at [0,0,0]
            # Residual = Position_Calculated - [0,0,0]
            # We add 3 residuals (dx, dy, dz) per sighting
            
            current_residuals = (projected_positions - np.zeros(3)) * W_ORIGIN
            residuals.extend(current_residuals.flatten())
            
        elif len(projected_positions) > 1:
            # constraint 2: Consistency
            # Residual = Position_Calculated - Average_Position
            
            # Calculate the centroid of where the anchors currently "think" the marker is
            centroid = np.mean(projected_positions, axis=0)
            
            # The error is the distance of each sighting from that shared centroid
            current_residuals = projected_positions - centroid
            residuals.extend(current_residuals.flatten())


        # constrain anchors to a flat plane in the z axis.
        if True:
            # Extract Z coordinates from all 4 anchors
            anchor_zs = anchor_poses[:, 1, 2]
            
            # Calculate the average Z plane
            avg_z = np.mean(anchor_zs)
            
            # Penalize deviation from the average plane
            z_residuals = (anchor_zs - avg_z) * W_PLANAR
            residuals.extend(z_residuals)

    return np.array(residuals)

def optimize_anchor_poses(averages):
    """Finds optimal anchor poses
    Args:
        averages: dict keyed by marker names. each value is a list of four lists of poses corresponding to the four anchors.
            poses represent the pose of the marker in the camera's referene frame.
    """

    # the detection of the origin marker alone is not sufficient to constrain anchor poses with enough accuracy.
    # using the average pose of each marker from each camera,
    # minimize the error between the room position of markers from different perspectives.
    # while also minimizing the error between the origin marker room positions and [0,0,0]

    initial_guesses = []
    # We rely on 'origin' being fully populated to seed the initial guess.
    origin_sightings = averages['origin']
    
    for i in range(4):
        origin_marker_pose = origin_sightings[i][0]
        
        # Calculate anchor pose relative to room by inverting the chain from origin
        # Anchor = Inverse(Origin_in_Cam + Room_relative_to_Origin + Cam_offset)
        guess = invert_pose(compose_poses([
            model_constants.anchor_camera,
            origin_marker_pose,
        ]))
        
        initial_guesses.append(guess)

    # 'lm' (Levenberg-Marquardt) is standard for unconstrained least squares
    print('running least squares optimization')
    result = optimize.least_squares(
        multi_card_residuals,
        np.array(initial_guesses).flatten(),
        args=(averages,),
        method='lm', 
        # max_nfev=1000,
        verbose=0
    )

    if not result.success:
        logging.error(f"Optimization failed. Status: {result.status}, Msg: {result.message}")
        return None

    # Reshape back to (4 anchors, 2 vectors, 3 coords)
    poses = result.x.reshape((4, 2, 3))
    return poses

def main():
    parser = argparse.ArgumentParser(description='Run robot calibration functions. Use --help for more details on each command.')
    parser.add_argument('--mode', type=str, choices=[
        'collect-images-stream',
        'calibrate-from-files',
        'collect-images-locally-raspi',
        'calibrate-from-stream'
    ], required=True, help='Choose the calibration function to run:\n \
            "collect-images-stream" to capture a specified number of images from a network stream; \
            "calibrate-from-files" to run camera calibration on a local set of images; \
            "collect-images-locally-raspi" to capture a specified number of images from a connected camera on a Raspberry Pi; \
            "calibrate-from-stream" to run camera calibration directly from a network stream until 20 images are collected.')
    parser.add_argument('--address', type=str, default='tcp://192.168.1.151:8888',
                        help='The network address for the video stream (used with stream modes).')
    parser.add_argument('--num-images', type=int, default=50,
                        help='The number of images to collect when using "collect-images-locally-raspi" or "collect-images-stream".')
    parser.add_argument('--resolution', type=str, default='4608x2592',
                        help='The resolution for the camera on the Raspberry Pi (e.g., "4608x2592"). Used with "collect-images-locally-raspi" mode.')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path of the config file to write/update with calibrated values.')

    args = parser.parse_args()

    if args.mode == 'collect-images-locally-raspi':
        collect_images_locally_raspi(args.num_images, args.resolution)
    elif args.mode == 'collect-images-stream':
        collect_images_stream(args.address, args.num_images)
    elif args.mode == 'calibrate-from-files':
        calibrate_from_files(args.config)
    elif args.mode == 'calibrate-from-stream':
        calibrate_from_stream(args.address, args.config)

if __name__ == "__main__":
    main()
