
import numpy as np
import scipy.optimize as optimize
from cv_common import compose_poses
import model_constants
from spools import SpiralCalculator

def calc_spool_params(empty_diameter, full_diameter, full_length, gear_ratio):
    # since line accumulates on the spool in a spiral, the amount of wrapped line is an exponential function of the spool angle.\
    # this function is parameterized by the two terms k1 and k2
    diameter_diff = full_diameter - empty_diameter
    assert diameter_diff > 0
    k1 = (empty_diameter * full_length) / diameter_diff
    k2 = (math.pi * gear_ratio * diameter_diff) / full_length
    return k1, k2

def get_spooled_length(motor_angle_revs, k1, k2, zero_angle, motor_orientation):
    relative_angle = motor_orientation * motor_angle_revs - zero_angle # shaft angle relative to zero_angle
    return k1 * (math.exp(k2 * relative_angle) - 1)

def calibration_cost_fn(params, observations, spools):
	"""Return the mean squared difference between line lengths calculated from encoders, and from visual observation

	params - a pose and zero angle for each anchor (3 rot, 3 pos, 1 za) * 4.
		the zero angle refers to the encoder position in revolutions where the amount of unspooled line would be zero.

	observations - a list with one entry per sample position.
		At each sample position, the gantry was still, and the lines were tight.
		Each sample position will have
		- encoder angles for each of the four anchor spool motors
		- four lists of visual observations, one for each of the four anchor cameras. Each visual observation contains
		  - the pose of an observed gantry aruco marker in the camera's 3d coordinate system. (the output of cv2.solvePnP)

	spools - a list of four SpiralCalculators to relate zero angles to line lengths.
	"""

	# extract parameters
	params = params.reshape((4,7))
	anchor_poses = []
	for i, ap in enumerate(params):
		anchor_poses.append(ap[:6].reshape((2,3)))
		spools[i].set_zero_angle(ap[6])

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

	spools = [
		SpiralCalculator(empty_diameter=25, full_diameter=27, full_length=7.5, gear_ratio=20/51, motor_orientation=-1)
		for i in range(4)]

    result = optimize.minimize(
        calibration_cost_fn,
        initial_guess,
        args=(observations, spools),
        method='SLSQP',
        bounds=bounds,
        options={
            'disp': False,
        },