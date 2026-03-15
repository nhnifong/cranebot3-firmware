import numpy as np
from scipy import optimize
import logging

# --- Assumed external dependencies (same as original code) ---
# compose_poses, invert_pose, model_constants, gantry_april_inv

W_ORIGIN = 0.1 # increase this to to make origin errors more expensive
W_PLANAR = 1 # increase this to make anchor height deviations from the average plane more expensive
W_LINE_CHANGE = 1.0 # weight for the active eyelet line length change
W_KINEMATIC = 1.0 # weight forcing the anchor lines to behave as a rigid swing arc

# =============================================================================
# NEW DATA STRUCTURE DOCUMENTATION & PHYSICAL LAYOUT
# =============================================================================
# PHYSICAL LAYOUT ASSUMPTION: 
# While the math can solve arbitrary configurations, the intended hardware 
# setup places the 2 main anchors in opposite corners of the workspace (e.g., 
# Front-Left and Back-Right). The 2 external ceramic eyelets should be placed 
# in the remaining opposite corners (e.g., Front-Right and Back-Left). 
# This maximizes the kinematic workspace and prevents degenerate geometries.
#
# To solve for the external eyelets, the optimization process requires the 
# `length_change_data` argument. It should be a list of dictionaries, where each 
# dictionary represents one before/after experiment on a single external line.
#
# ASSUMPTION: During this test, the active external line changes length, the two 
# anchor lines remain taut, and the opposite external line is slack.
#
# length_change_data = [
#     {
#         "active_line_idx": 0,  # Int: 0 to 3. 0=Ext 0, 1=Int 0, 2=Ext 1, 3=Int 1
#         "delta_L": -0.05,      # Float: Change in line length in meters
#         
#         # Multiple sightings from multiple cameras BEFORE the change
#         # Index 0 is a list of poses from Anchor 0's camera
#         # Index 1 is a list of poses from Anchor 1's camera
#         "sightings_before": [
#             [pose_c0_obs1, pose_c0_obs2, ...],
#             [pose_c1_obs1, pose_c1_obs2, ...]
#         ],
#         
#         # Multiple sightings from multiple cameras AFTER the change
#         "sightings_after": [
#             [pose_c0_obs1, pose_c0_obs2, ...],
#             [pose_c1_obs1, pose_c1_obs2, ...]
#         ]
#     },
#     # ... multiple experiments ...
# ]
# =============================================================================

def multi_card_residuals(x, averages, length_change_data):
    """
    Computes the vector of residuals (differences) for least_squares.
    
    x contains:
      - 2 anchor poses (2 anchors * 2 vectors (rvec, tvec) * 3 coords = 12 elements)
      - 2 eyelet positions (2 eyelets * 3 coords = 6 elements)
    Total x length: 18
    """
    # Unpack state vector
    anchor_poses = x[:12].reshape((2, 2, 3))
    eyelet_positions = x[12:].reshape((2, 3))
    
    residuals = []
    
    # ---------------------------------------------------------
    # 1. Existing Camera-based Residuals (Origin & Consistency)
    # ---------------------------------------------------------
    for marker_name, sightings in averages.items():
        valid_sightings = []
        
        for anchor_idx, marker_pose_cams in enumerate(sightings):
            for marker_pose_cam in marker_pose_cams:
                if marker_pose_cam is None:
                    continue
                
                # Chain: Anchor -> Camera -> Marker
                pose_list = [
                    anchor_poses[anchor_idx],
                    model_constants.anchor_camera,
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
            
        elif len(projected_positions) > 1:
            # constraint 2: Consistency between cameras
            centroid = np.mean(projected_positions, axis=0)
            current_residuals = projected_positions - centroid
            residuals.extend(current_residuals.flatten())

    # ---------------------------------------------------------
    # 2. Anchor Z-Plane Constraint
    # ---------------------------------------------------------
    if True:
        # Extract Z coordinates from the 2 anchors
        anchor_zs = anchor_poses[:, 1, 2]
        avg_z = np.mean(anchor_zs)
        
        # Penalize deviation from each other
        z_residuals = (anchor_zs - avg_z) * W_PLANAR
        residuals.extend(z_residuals)

    # ---------------------------------------------------------
    # 3. NEW: External Eyelet Kinematic Arc Residuals
    # ---------------------------------------------------------
    def get_pull_point(idx):
        if idx == 0: return eyelet_positions[0]
        if idx == 1: return compose_poses([anchor_poses[0, 1], model_constants.arp_anchor_right_eyelet])[1]
        if idx == 2: return eyelet_positions[1]
        if idx == 3: return compose_poses([anchor_poses[1, 1], model_constants.arp_anchor_right_eyelet])[1]

    for obs in length_change_data:
        active_idx = obs['active_line_idx']
        delta_L = obs['delta_L']
        
        def get_gantry_room_pos(sightings_list_of_lists):
            positions = []
            for anchor_idx, poses_cam in enumerate(sightings_list_of_lists):
                if poses_cam is None:
                    continue
                for pose_cam in poses_cam:
                    if pose_cam is None:
                        continue
                    pose_list = [
                        anchor_poses[anchor_idx],
                        model_constants.anchor_camera,
                        pose_cam,
                        gantry_april_inv
                    ]
                    pose_in_room = compose_poses(pose_list)
                    positions.append(pose_in_room[1])
            return np.mean(positions, axis=0) if positions else None

        pos_before = get_gantry_room_pos(obs['sightings_before'])
        pos_after = get_gantry_room_pos(obs['sightings_after'])
        
        if pos_before is not None and pos_after is not None:
            # --- 3a. Active line distance constraint ---
            pull_point_active = get_pull_point(active_idx)
            dist_before = np.linalg.norm(pos_before - pull_point_active)
            dist_after = np.linalg.norm(pos_after - pull_point_active)
            calc_delta = dist_after - dist_before
            
            residuals.append((calc_delta - delta_L) * W_LINE_CHANGE)

            # --- 3b. Kinematic Arc Constraints ---
            # The line directly across is slack, so we ignore it.
            # The other two lines are taut, so their distance change must be 0.
            slack_idx = (active_idx + 2) % 4
            taut_indices = [i for i in range(4) if i not in (active_idx, slack_idx)]
            
            for t_idx in taut_indices:
                pull_point_taut = get_pull_point(t_idx)
                dist_before_t = np.linalg.norm(pos_before - pull_point_taut)
                dist_after_t = np.linalg.norm(pos_after - pull_point_taut)
                
                calc_delta_t = dist_after_t - dist_before_t
                residuals.append(calc_delta_t * W_KINEMATIC)

    return np.array(residuals)

# A point on the wall on the anchor's right side, about five meters away.
external_guess = ((0,0,0), (-3.67, -3.57, 0.06))

def optimize_arp_anchors(averages, length_change_data, initial_eyelet_guesses=None):
    """
    Finds optimal anchor poses and external eyelet positions for arpeggio anchor
    
    Args:
        averages: dict keyed by marker names.
        length_change_data: list of dicts describing line length adjustments.
        initial_eyelet_guesses: (2, 3) numpy array for initial [x,y,z] positions of the eyelets.
    
    Returns:
        tuple: (optimized_anchor_poses, optimized_eyelet_positions)
    """
    initial_guesses = []
    origin_sightings = averages['origin']
    
    # We now only have 2 anchors to initialize
    for i in range(2):
        origin_marker_pose = origin_sightings[i][0]
        guess = invert_pose(compose_poses([
            model_constants.anchor_camera,
            origin_marker_pose,
        ]))
        initial_guesses.append(guess)
        
    print(f'initial_anchor_guesses = {initial_guesses}')
    
    # Initialize eyelet guesses if none provided
    if initial_eyelet_guesses is None:
        initial_eyelet_guesses = np.array([
            compose_poses([initial_guesses[0], external_guess])[1], # Guess for eyelet 0
            compose_poses([initial_guesses[1], external_guess])[1]  # Guess for eyelet 1
        ])
        
    # Flatten everything into a single 1D state vector for the optimizer
    initial_anchor_flat = np.array(initial_guesses).flatten()
    initial_eyelet_flat = np.array(initial_eyelet_guesses).flatten()
    x0 = np.concatenate([initial_anchor_flat, initial_eyelet_flat])

    print('running least squares optimization (anchors + eyelets)')
    result = optimize.least_squares(
        multi_card_residuals,
        x0,
        args=(averages, length_change_data),
        method='lm', 
        verbose=0
    )

    if not result.success:
        logging.error(f"Optimization failed. Status: {result.status}, Msg: {result.message}")
        return None, None

    # Reshape back to distinct structures
    optimized_anchors = result.x[:12].reshape((2, 2, 3))
    optimized_eyelets = result.x[12:].reshape((2, 3))
    
    return optimized_anchors, optimized_eyelets