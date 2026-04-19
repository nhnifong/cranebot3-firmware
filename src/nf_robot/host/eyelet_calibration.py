import numpy as np
from scipy import optimize
import logging

from nf_robot.common.pose_functions import *

logger = logging.getLogger(__name__)
import nf_robot.common.definitions as model_constants
from nf_robot.common.cv_common import *

W_ORIGIN = 1.0 # increase this to to make origin errors more expensive
W_PLANAR = 0.9 # increase this to make anchor height deviations from the average plane more expensive
W_DIAMOND_DIST = 0.8 # weight for the distance changes in the diamond pattern
W_DIAMOND_PLANAR = 0.2 # weight for forcing the gantry and eyelets into a single vertical plane
W_EYELET_REG = 0.2 # weight to keep eyelets near their initial 5m guess

# half height and half width of diamond
DIAMOND_SIZE = (0.1, 1.0)

# =============================================================================
# DIAMOND STRATEGY DATA STRUCTURE
# =============================================================================
# diamond_observations = {
#     'bottom': [ [poses_cam0], [poses_cam1] ], # Lists of varying lengths
#     'right':  [ [poses_cam0], [poses_cam1] ],
#     'top':    [ [poses_cam0], [poses_cam1] ],
#     'left':   [ [poses_cam0], [poses_cam1] ]
# }
# 
# Motions (15cm = 0.15m changes):
# bottom -> right: Eyelet 0 (Line 1) shortens by 15cm
# right  -> top:   Eyelet 1 (Line 3) shortens by 15cm
# top    -> left:  Eyelet 0 (Line 1) lengthens by 15cm (back to 'bottom' length)
# left   -> bottom: Eyelet 1 (Line 3) lengthens by 15cm (back to 'bottom' length)
# =============================================================================

def multi_card_residuals(x, raw_obs, diamond_observations, initial_eyelets=None, debug=False, fixed_anchor_poses=None, line_deltas=None):
    """
    Computes the vector of residuals (differences) for least_squares.
    
    If fixed_anchor_poses is None:
        x contains 18 elements: 2 anchors (12) + 2 eyelets (6)
    If fixed_anchor_poses is provided:
        x contains 6 elements: 2 eyelets (6)
    """
    # Unpack state vector based on whether anchors are frozen
    if fixed_anchor_poses is not None:
        anchor_poses = fixed_anchor_poses
        eyelet_positions = x.reshape((2, 3))
    else:
        anchor_poses = x[:12].reshape((2, 2, 3))
        eyelet_positions = x[12:].reshape((2, 3))
    
    residuals = []
    
    # Debug trackers
    cost_origin = 0.0
    cost_planar = 0.0
    cost_diamond_planar = 0.0
    cost_diamond_dist = 0.0
    cost_eyelet_reg = 0.0

    # ---------------------------------------------------------
    # 1. Existing Camera-based Residuals (Origin & Consistency)
    # ---------------------------------------------------------
    for marker_name, sightings in raw_obs.items():
        valid_sightings = []
        
        for anchor_idx, marker_pose_cams in enumerate(sightings):
            for marker_pose_cam in marker_pose_cams:
                if marker_pose_cam is None:
                    continue
                
                # Chain: Anchor -> Camera -> Marker
                pose_list = [
                    anchor_poses[anchor_idx],
                    model_constants.arp_anchor_camera,
                    marker_pose_cam
                ]
                if marker_name == 'gantry':
                    pose_list.append(gantry_april_inv)
                
                pose_in_room = compose_poses(pose_list)
                
                # Extract translation (tvec is index 1)
                valid_sightings.append(pose_in_room[1])

        if not valid_sightings:
            continue
            
        projected_positions = np.array(valid_sightings)
        
        if marker_name == 'origin':
            # constraint 1: Origin must be at [0,0,0]
            current_residuals = (projected_positions - np.zeros(3)) * W_ORIGIN
            residuals.extend(current_residuals.flatten())
            cost_origin += np.sum(current_residuals**2)
            
        elif len(projected_positions) > 1:
            # constraint 2: Consistency between cameras
            centroid = np.mean(projected_positions, axis=0)
            current_residuals = projected_positions - centroid
            residuals.extend(current_residuals.flatten())
            cost_origin += np.sum(current_residuals**2)

    # ---------------------------------------------------------
    # 2. Anchor and Eyelet Z-Plane Constraint
    # ---------------------------------------------------------
        # Extract Z coordinates from the 2 anchors and 2 eyelets
        anchor_zs = anchor_poses[:, 1, 2]
        eyelet_zs = eyelet_positions[:, 2]
        all_zs = np.concatenate([anchor_zs, eyelet_zs])
        
        # Calculate the average Z plane
        avg_z = np.mean(all_zs)
        
        # Penalize deviation from the average plane
        z_residuals = (all_zs - avg_z) * W_PLANAR
        residuals.extend(z_residuals)
        cost_planar += np.sum(z_residuals**2)

    # ---------------------------------------------------------
    # 3. Diamond Kinematic & Distance Residuals
    # ---------------------------------------------------------
    if diamond_observations is not None:
        all_points_per_state = {}
        centroids = {}
        
        # Extract and group all gantry positions by state
        for state, data_arr in diamond_observations.items():
            points = []
            for c, camera_poses in enumerate(data_arr):
                for pose_cam in camera_poses:
                    if pose_cam is None or np.all(pose_cam == 0):
                        continue
                    pose_list = [
                        anchor_poses[c],
                        model_constants.arp_anchor_camera,
                        pose_cam,
                        gantry_april_inv
                    ]
                    pose_in_room = compose_poses(pose_list)
                    points.append(pose_in_room[1])
                    
            all_points_per_state[state] = points
            if points:
                centroids[state] = np.mean(points, axis=0)

        # 3a. Planarity constraint: Both eyelets and all gantry positions share a vertical plane
        # A vertical plane contains the Z axis and the vector between the eyelets
        vec_eyelets = eyelet_positions[1] - eyelet_positions[0]
        # Normal = vec_eyelets CROSS [0,0,1]
        N_plane = np.array([vec_eyelets[1], -vec_eyelets[0], 0.0])
        norm_N = np.linalg.norm(N_plane)
        if norm_N > 1e-6:
            N_plane /= norm_N
        else:
            N_plane = np.array([1.0, 0.0, 0.0]) # Fallback if perfectly stacked
            
        # Penalize the perpendicular distance of EVERY gantry point to this plane
        for state, points in all_points_per_state.items():
            for p in points:
                dist_to_plane = np.dot(p - eyelet_positions[0], N_plane)
                res = dist_to_plane * W_DIAMOND_PLANAR
                residuals.append(res)
                cost_diamond_planar += res**2
                
        # 3b. Distance constraints based on the diamond pattern
        required_states = ['bottom', 'right', 'top', 'left']
        if all(s in centroids for s in required_states):
            c_bot = centroids['bottom']
            c_rig = centroids['right']
            c_top = centroids['top']
            c_lef = centroids['left']

            # Distances to Eyelet 0 (Line 1)
            D0_bot = np.linalg.norm(c_bot - eyelet_positions[0])
            D0_rig = np.linalg.norm(c_rig - eyelet_positions[0])
            D0_top = np.linalg.norm(c_top - eyelet_positions[0])
            D0_lef = np.linalg.norm(c_lef - eyelet_positions[0])

            # Distances to Eyelet 1 (Line 3)
            D1_bot = np.linalg.norm(c_bot - eyelet_positions[1])
            D1_rig = np.linalg.norm(c_rig - eyelet_positions[1])
            D1_top = np.linalg.norm(c_top - eyelet_positions[1])
            D1_lef = np.linalg.norm(c_lef - eyelet_positions[1])

            if line_deltas is not None:
                # Use measured line length changes instead of commanded values
                L1_bot_to_rig = line_deltas['bot_to_rig'][0]
                L3_bot_to_rig = line_deltas['bot_to_rig'][1]
                L1_rig_to_top = line_deltas['rig_to_top'][0]
                L3_rig_to_top = line_deltas['rig_to_top'][1]
                L1_top_to_lef = line_deltas['top_to_lef'][0]
                L3_top_to_lef = line_deltas['top_to_lef'][1]
                # lef_to_bot must close the loop
                L1_lef_to_bot = -(L1_bot_to_rig + L1_rig_to_top + L1_top_to_lef)
                L3_lef_to_bot = -(L3_bot_to_rig + L3_rig_to_top + L3_top_to_lef)
            else:
                half_h, half_w = DIAMOND_SIZE
                # Commanded changes for Eyelet 0 (Line 1)
                L1_bot_to_rig = -(half_w + half_h)
                L1_rig_to_top = (half_w - half_h)
                L1_top_to_lef = (half_w + half_h)
                L1_lef_to_bot = -(half_w - half_h)
                # Commanded changes for Eyelet 1 (Line 3)
                L3_bot_to_rig = (half_w - half_h)
                L3_rig_to_top = -(half_w + half_h)
                L3_top_to_lef = -(half_w - half_h)
                L3_lef_to_bot = (half_w + half_h)
            
            d_res = [
                # Eyelet 0 (Line 1) Constraints
                (D0_rig - D0_bot - L1_bot_to_rig) * W_DIAMOND_DIST,
                (D0_top - D0_rig - L1_rig_to_top) * W_DIAMOND_DIST,
                (D0_lef - D0_top - L1_top_to_lef) * W_DIAMOND_DIST,
                (D0_bot - D0_lef - L1_lef_to_bot) * W_DIAMOND_DIST,
                
                # Eyelet 1 (Line 3) Constraints
                (D1_rig - D1_bot - L3_bot_to_rig) * W_DIAMOND_DIST,
                (D1_top - D1_rig - L3_rig_to_top) * W_DIAMOND_DIST,
                (D1_lef - D1_top - L3_top_to_lef) * W_DIAMOND_DIST,
                (D1_bot - D1_lef - L3_lef_to_bot) * W_DIAMOND_DIST
            ]
            residuals.extend(d_res)
            cost_diamond_dist += sum(r**2 for r in d_res)

    # ---------------------------------------------------------
    # 4. Regularization (Anchor eyelets to initial guesses)
    # ---------------------------------------------------------
    if initial_eyelets is not None:
        reg_residuals = (eyelet_positions - initial_eyelets).flatten() * W_EYELET_REG
        residuals.extend(reg_residuals)
        cost_eyelet_reg += np.sum(reg_residuals**2)

    if debug:
        lines = [
            "--- Residual Costs ---",
            f"Origin/Consistency: {cost_origin:.6f}",
        ]
        if fixed_anchor_poses is None:
            lines.append(f"Anchor Z-Planarity: {cost_planar:.6f}")
        lines += [
            f"Diamond Planarity:  {cost_diamond_planar:.6f}",
            f"Diamond Distances:  {cost_diamond_dist:.6f}",
            f"Eyelet Reg (drift): {cost_eyelet_reg:.6f}",
            "----------------------",
        ]
        logger.info("\n".join(lines))

    return np.array(residuals)

def optimize_arp_anchors(raw_obs, diamond_observations=None, initial_eyelet_guesses=None, fixed_anchor_poses=None, line_deltas=None):
    """
    Finds optimal anchor poses AND external eyelet positions.
    
    Args:
        raw_obs: dict keyed by marker names.
        diamond_observations: dict with keys 'bottom', 'right', 'top', 'left' containing camera obs.
                              Pass None or {} for the first pass (finding anchors only).
        initial_eyelet_guesses: (2, 3) numpy array for initial [x,y,z] positions of the eyelets.
        fixed_anchor_poses: (2, 2, 3) numpy array. If provided, anchor poses are frozen and only eyelets are optimized.
    
    Returns:
        tuple: (optimized_anchor_poses, optimized_eyelet_positions)
    """
    if diamond_observations is None:
        diamond_observations = {}
        
    if fixed_anchor_poses is None:
        initial_guesses = []
        origin_sightings = raw_obs['origin']
        
        # Initialize anchors from origin marker
        for i in range(2):
            origin_marker_pose = origin_sightings[i][0]
            guess = invert_pose(compose_poses([
                model_constants.arp_anchor_camera,
                origin_marker_pose,
            ]))
            initial_guesses.append(guess)
        
        anchor_poses_to_use = np.array(initial_guesses)
        logger.debug(f'initial_anchor_guesses = {initial_guesses}')
    else:
        # Use the provided fixed anchors
        anchor_poses_to_use = fixed_anchor_poses
        logger.info('Using provided fixed_anchor_poses. Anchors will NOT be modified.')
    
    # A point on the wall on the anchor's right side at the same height, about five meters away.
    # this is a diagonal in the anchor's local frame of refernce.
    external_guess = np.array([(0,0,0), (-3.67, -3.57, 0.00)])

    # Initialize eyelet guesses if none provided
    if initial_eyelet_guesses is None:
        initial_eyelet_guesses = np.array([
            compose_poses([anchor_poses_to_use[0], external_guess])[1], # Guess for eyelet 0
            compose_poses([anchor_poses_to_use[1], external_guess])[1]  # Guess for eyelet 1
        ])
        
    # Configure the state vector and args depending on whether we are freezing anchors
    if fixed_anchor_poses is not None:
        x0 = initial_eyelet_guesses.flatten()
        opt_args = (raw_obs, diamond_observations, initial_eyelet_guesses, False, fixed_anchor_poses, line_deltas)
    else:
        initial_anchor_flat = anchor_poses_to_use.flatten()
        initial_eyelet_flat = initial_eyelet_guesses.flatten()
        x0 = np.concatenate([initial_anchor_flat, initial_eyelet_flat])
        opt_args = (raw_obs, diamond_observations, initial_eyelet_guesses, False, None, line_deltas)

    logger.info('Running least squares optimization...')
    result = optimize.least_squares(
        multi_card_residuals,
        x0,
        args=opt_args,
        method='lm', 
        verbose=0
    )

    if not result.success:
        logging.error(f"Optimization failed. Status: {result.status}, Msg: {result.message}")
        return None, None

    # Run one final time with debug=True to print the cost distribution
    logger.info("Final Optimization Costs:")
    multi_card_residuals(result.x, raw_obs, diamond_observations, initial_eyelet_guesses, debug=True, fixed_anchor_poses=fixed_anchor_poses, line_deltas=line_deltas)

    # Reshape back to distinct structures based on the freeze flag
    if fixed_anchor_poses is not None:
        optimized_anchors = fixed_anchor_poses
        optimized_eyelets = result.x.reshape((2, 3))
    else:
        optimized_anchors = result.x[:12].reshape((2, 2, 3))
        optimized_eyelets = result.x[12:].reshape((2, 3))
    
    return optimized_anchors, optimized_eyelets


def analyze_diamond_data(diamond_observations):
    """
    Analyzes the raw diamond observations to determine their inherent planarity 
    and geometric consistency before passing them to the optimizer.
    """
    
    # Anchors frozen from your 2nd pass
    anchor_poses = np.array([
        [[-0.08173866, -0.08760878, -2.27461725],
         [ 3.4444503,  -2.26444289,  2.22807711]],
        [[-0.04984111,  0.02871572,  0.82210996],
         [-2.26615642,  2.69190574,  2.22807715]]
    ])

    logger.info("=== RAW DIAMOND DATA ANALYSIS ===")

    all_points = {0: [], 1: [], 'combined': []}
    centroids = {0: {}, 1: {}, 'combined': {}}

    # 1. Convert all camera observations to physical room coordinates, separated by camera
    for state, data_arr in diamond_observations.items():
        pts = {0: [], 1: []}
        for c, camera_poses in enumerate(data_arr):
            for pose_cam in camera_poses:
                if pose_cam is None or np.all(pose_cam == 0):
                    continue
                pose_list = [
                    anchor_poses[c],
                    model_constants.arp_anchor_camera,
                    pose_cam,
                    gantry_april_inv
                ]
                pose_in_room = compose_poses(pose_list)
                pts[c].append(pose_in_room[1]) # We just care about physical translation
        
        all_points[0].extend(pts[0])
        all_points[1].extend(pts[1])
        all_points['combined'].extend(pts[0] + pts[1])
        
        lines = [f"\nState '{state}':"]
        for c in [0, 1]:
            if pts[c]:
                centroids[c][state] = np.mean(pts[c], axis=0)
                lines.append(f"  Cam {c}: {len(pts[c]):02d} obs. Centroid: {np.round(centroids[c][state], 3)}")
        logger.info("\n".join(lines))
        if pts[0] or pts[1]:
            centroids['combined'][state] = np.mean(pts[0] + pts[1], axis=0)

    if not all_points['combined']:
        logger.warning("No valid points found. Aborting analysis.")
        return

    def analyze_planarity(points_list, label):
        points_arr = np.array(points_list)
        if len(points_arr) < 3:
            return
        
        overall_mean = np.mean(points_arr, axis=0)
        centered_points = points_arr - overall_mean
        
        U, S, Vt = np.linalg.svd(centered_points)
        normal_vector = Vt[2]
        
        # Ensure normal points 'up' (+Z) for easier reading
        if normal_vector[2] < 0:
            normal_vector = -normal_vector

        distances_to_plane = np.dot(centered_points, normal_vector)
        logger.info(
            f"\n=== PLANARITY STATS: {label} ===\n"
            f"Best-fit Plane Normal:     {np.round(normal_vector, 4)}\n"
            f"Mean Absolute Deviation:   {np.mean(np.abs(distances_to_plane))*100:.2f} cm\n"
            f"Max Deviation from Plane:  {np.max(np.abs(distances_to_plane))*100:.2f} cm\n"
            f"RMS Deviation from Plane:  {np.sqrt(np.mean(distances_to_plane**2))*100:.2f} cm\n"
            f"Plane Tilt from Vertical:  {np.degrees(np.arcsin(abs(normal_vector[2]))):.2f} degrees"
        )

    def analyze_kinematics(cents, label):
        req_states = ['bottom', 'right', 'top', 'left']
        if all(s in cents for s in req_states):
            c_bot, c_rig = cents['bottom'], cents['right']
            c_top, c_lef = cents['top'], cents['left']
            logger.info(
                f"\n=== KINEMATIC DIAMOND DISTANCES: {label} ===\n"
                f"Travel Bottom -> Right:  {np.linalg.norm(c_rig - c_bot)*100:.2f} cm\n"
                f"Travel Right  -> Top:    {np.linalg.norm(c_top - c_rig)*100:.2f} cm\n"
                f"Travel Top    -> Left:   {np.linalg.norm(c_lef - c_top)*100:.2f} cm\n"
                f"Travel Left   -> Bottom: {np.linalg.norm(c_bot - c_lef)*100:.2f} cm\n"
                f"\nDiagonal Bottom -> Top:  {np.linalg.norm(c_top - c_bot)*100:.2f} cm\n"
                f"Diagonal Right  -> Left: {np.linalg.norm(c_lef - c_rig)*100:.2f} cm"
            )

    # Run analysis for each subset
    for key, label in [(0, "CAMERA 0"), (1, "CAMERA 1"), ('combined', "COMBINED CAMERAS")]:
        if all_points[key]:
            analyze_planarity(all_points[key], label)
            analyze_kinematics(centroids[key], label)