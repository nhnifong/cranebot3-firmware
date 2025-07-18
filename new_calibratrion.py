
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


def find_cal_params():
	spools = []
	initial_guess = []
	bounds = []
	for i in range(4):
		# initialize a model of the spool to relate zero angle, encoder readinds, and line lengths
		spools.append(SpiralCalculator(empty_diameter=25, full_diameter=27, full_length=7.5, gear_ratio=20/51, motor_orientation=-1))

		# initial position guesses are made by taking the average of a few observations of an origin card.
	    apose = np.array(invert_pose(compose_poses([model_constants.anchor_camera, average_pose(origin_detections)])))
	    # depending on which quadrant the average anchor pose falls in, constrain the XY rotation,
	    # while still allowing very minor deviation because of crooked mounting and misalignment of the foam shock absorber on the camera.
	    xsign = 1 if apose[1,0]>0 else -1
	    ysign = 1 if apose[1,1]>0 else -1

	    guess = [
	    	0, 0                                   # no x or y component in rotation axis
	    	-xsign*(2-ysign)*pi/4,                 # one of four diagonals. points -Y towards middle of work area
	    	apose[1][0], apose[1][1], apose[1][2], # use position from the averaged pose above
	    	0,                                     # initial guess of zero angle
	    	27,                                    # full diameter in millimeters
	    	7.5,                                   # full length in meters
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
	bounds =  np.array(bounds).flatten()

    result = optimize.minimize(
        calibration_cost_fn,
        initial_guess,
        args=(observations, spools),
        method='SLSQP',
        bounds=bounds,
        options={
            'disp': True,
        },


# TODO overall procedure
# collect observations of origin card aruco marker to get initial guess of anchor poses.
# reel in all lines until the switches indicate they are tight
# use aruco observations of gantry to obtain initial guesses for zero angles
# use this information to perform rough movements
# for at least 15 positions:
#   move to a position.
#   reel in all lines until they are tight
#   save encoder angles
#   collect several visual observations from each camera
# feed collected data to the optimization process above.
# Use the optimization output to update anchor poses and spool params
# move to random locations and determine the quality of the calibration by how often all four lines are tight during and after moves.