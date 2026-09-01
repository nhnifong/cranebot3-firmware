import os
import cv2
from pupil_apriltags import Detector
import numpy as np
import time
from nf_robot.common.config_loader import *
import nf_robot.common.definitions as model_constants
from scipy.spatial.transform import Rotation
from nf_robot.generated.nf import config as nf_config
import functools

# The marker IDs will correspond to the index in this list.
MARKER_NAMES = [
    'origin',
    'gantry',
    'gamepad',
    'hamper',
    'trash',
    'cal_assist_1',
    'cal_assist_2',
    'cal_assist_3',
    'gamepad_back',
    'hamper_back',
    'trash_back',
    'toys',
    'toys_back',
    'park_target',
] # next tag id 14

CAL_MARKERS = set(['origin', 'cal_assist_1', 'cal_assist_2', 'cal_assist_3'])
OTHER_MARKERS = set(['gamepad', 'hamper', 'trash', 'toys', 'gamepad_back', 'hamper_back', 'trash_back', 'toys_back'])

# AprilTag images are typically downloaded, not generated in code.
# We are using the tag36h11 tag family.
# The images for new tags can be downloaded at
# https://github.com/AprilRobotics/apriltag-imgs/tree/master/tag36h11

DEFAULT_MARKER_SIZE = 0.0945 # The default side length of markers in meters
CAL_MARKER_SIZE = 0.1690
# Define the physical size of any markers that are not the default size.
SPECIAL_SIZES = {
    'origin': CAL_MARKER_SIZE,
    'cal_assist_1': CAL_MARKER_SIZE,
    'cal_assist_2': CAL_MARKER_SIZE,
    'cal_assist_3': CAL_MARKER_SIZE,
    'gantry':       0.0915,
    'park_target':       0.0464,
}

# Scales every marker's assumed physical size, and so scales every distance the cameras
# report: solvePnP places a marker at whatever range makes it subtend the pixels it does, so
# a marker assumed bigger is placed further away. Below 1 pulls every sighting closer.
#
# Measured, not guessed. Fitting a hang survey (experiments/hang_survey_fit.py) against the
# line lengths, which have their own scale from the spool encoders, showed the camera
# distances standing about 6% long: sweeping this constant while refitting the geometry at
# each value put the leave-one-out error at 86 mm here at 1.00 and 62 mm at 0.94, with a clean
# minimum between 0.935 and 0.945.
#
# What is 6% off is still unknown. The gantry marker has been measured with a ruler and the
# cameras checkerboard calibrated, and both agree with the constants above, so the cause is
# something they share -- a corner convention in the detector, a resolution or crop mismatch
# between the calibration and the stream -- rather than either of those two. This corrects the
# symptom, in one place, until the cause turns up.
#
# It was measured from gantry marker sightings alone and is applied to every marker, which is
# right if the error is on the camera side and wrong if it is that one tag. Changing it moves
# every vision-derived position, so anchor poses fitted under a different value do not carry
# over: recalibrate after touching it.
GLOBAL_MARKER_SIZE_BIAS = 0.94

# These are the 3D corner points of a generic marker of size 1x1 meter.
# We will scale this based on the actual marker size.
BASE_MARKER_POINTS = np.array([
    [-0.5, -0.5, 0],
    [ 0.5, -0.5, 0],
    [ 0.5,  0.5, 0],
    [-0.5,  0.5, 0]
], dtype=np.float32)

# Pre-calculate marker points for known sizes.
DEFAULT_OBJ_POINTS = BASE_MARKER_POINTS * DEFAULT_MARKER_SIZE * GLOBAL_MARKER_SIZE_BIAS
SPECIAL_OBJ_POINTS = {
    name: BASE_MARKER_POINTS * size * GLOBAL_MARKER_SIZE_BIAS
    for name, size in SPECIAL_SIZES.items()
}

# Gripper stream is now 684x384 (full-sensor 16:9 FOV). Setting the target shape to match means
# process_frame does not resize, so AprilTag detection and the UI get the whole wide field of
# view. Downstream ML models resize their own inputs. (Was a square 384x384 center crop.)
SF_TARGET_SHAPE = (684, 384)

# Stringman tags are from the 'tag36h11' family.
# increase quad_decimate to improve speed at the cost of distance
detector = Detector(families="tag36h11", quad_decimate=1.0)

MIN_CROP_HALF = 64   # smallest search window, and the fallback when apparent size is unknown
CROP_MARGIN = 2.0    # window half-size as a multiple of the tag's apparent half-extent


def tag_half_extent(corners):
    """Half-width of the axis-aligned box around a tag's corners, in pixels."""
    return float(np.abs(corners - corners.mean(axis=0)).max())


def crop_window(center, half_extent, frame_shape):
    """Bounds (x1, y1, x2, y2) of the search window around a previously seen tag.

    The window scales with the tag's apparent size. A fixed window silently stops
    working once the tag outgrows it - the detector needs the whole quad plus
    surrounding background to find it - which is what happens as the camera closes
    in on a card. CROP_MARGIN also leaves room for the tag to move between frames.
    Clamped to the frame, so an approach that fills the view degrades into a
    full-frame scan rather than a miss.
    """
    cx, cy = center
    if half_extent:
        half = max(MIN_CROP_HALF, int(CROP_MARGIN * half_extent))
    else:
        half = MIN_CROP_HALF
    height, width = frame_shape[:2]
    return (
        max(0, int(cx - half)),
        max(0, int(cy - half)),
        min(width, int(cx + half)),
        min(height, int(cy + half)),
    )

def _locate_markers(im, K, D):
    try:
        # AprilTag detection works on grayscale images.
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        detections = detector.detect(gray)
        
        if not detections:
            return []

        results = []
        for detection in detections:
            marker_id = detection.tag_id
            corners = detection.corners

            try:
                name = MARKER_NAMES[marker_id]
            except IndexError:
                # Saw a tag that's not part of the defined system
                print(f'Unknown AprilTag spotted with id {marker_id}')
                continue
            
            # Look up the scaled object points specific to this tag
            obj_points = SPECIAL_OBJ_POINTS.get(name, DEFAULT_OBJ_POINTS)
            
            # Use solvePnP to get the rotation and translation vectors (rvec, tvec)
            # This gives the pose of the marker relative to the camera.
            # The coordinate system has the origin at the camera center. The z-axis points from the camera center out the camera lens.
            # The x-axis is to the right in the image taken by the camera, and y is down. The tag's coordinate frame is centered at the center of the tag.
            # From the viewer's perspective, the x-axis is to the right, y-axis down, and z-axis is out of the tag.
            _, r, t = cv2.solvePnP(obj_points, corners, K, D, False, cv2.SOLVEPNP_IPPE_SQUARE)
            
            results.append({
                'n': name,
                'p': (r.reshape((3,)), t.reshape((3,))), # pose tuple. numpy arrays. numpy supposedly has fast pickle hooks
                'center': tuple(detection.corners.mean(axis=0)),
                'half_extent': tag_half_extent(detection.corners),
            })
        return results
    except Exception as e:
        print(e)

def _locate_markers_in_crops(crops_data, K, D):
    """
    Detects AprilTags by evaluating small, pre-cropped images based on previous detections
    
    Args:
        crops_data (list): A list of dictionaries containing 'crop' (the small ndarray),
            'x1', 'y1' (the crop's offset in the original frame), and 'name' (expected tag).
        K (numpy.ndarray): 3x3 Camera intrinsic matrix.
        D (numpy.ndarray): Camera distortion coefficients.
    """
    results = []
    for data in crops_data:
        gray = cv2.cvtColor(data['crop'], cv2.COLOR_RGB2GRAY)
        
        for detection in detector.detect(gray):
            if detection.tag_id >= len(MARKER_NAMES):
                continue

            name = MARKER_NAMES[detection.tag_id]

            if name != data['name']:
                continue 

            global_corners = detection.corners + np.array([data['x1'], data['y1']])

            _, r, t = cv2.solvePnP(
                SPECIAL_OBJ_POINTS.get(name, DEFAULT_OBJ_POINTS), 
                global_corners, 
                K, 
                D, 
                False, 
                cv2.SOLVEPNP_IPPE_SQUARE
            )

            results.append({
                'n': name,
                'p': (r.reshape((3,)), t.reshape((3,))),
                'center': tuple(global_corners.mean(axis=0)),
                'half_extent': tag_half_extent(global_corners),
            })

    return results

def locate_markers(im, camera_cal: nf_config.CameraCalibration, crops_data=None):
    """
    Detects AprilTags in an image and estimates their pose.
    
    Args:
        im: The input image
        camera_cal: 

    Returns:
        A list of dictionaries, each containing the name, rotation vector (r),
        and translation vector (t) of a detected marker.

    Uses a cropped search window that uses slices like im[y:y+h, x:x+w] based on where tags were seen on
    previous frames. search the whole image only if the tag was not seen on the previous frame.
    """

    # Use passed object for camera calibration. keeps function pure for multiprocessing
    mtx = np.array(camera_cal.intrinsic_matrix).reshape((3,3))
    distortion = np.array(camera_cal.distortion_coeff)
    # Route to the crop handler if crop data was passed over IPC
    if crops_data is not None:
        return _locate_markers_in_crops(crops_data, mtx, distortion)
    elif im is not None:
        return _locate_markers(im, mtx, distortion)
    return []

def project_pixels_to_floor(normalized_pixels, pose, camera_cal: nf_config.CameraCalibration):
    """
    batch project normalized [0,1] pixel coordinates from a camera's point of view to the floor
    make sure you use the camera pose, not just the anchor pose!
    anchor 3 z rot 2.356194490192345
    """
    # Use passed object for camera calibration.
    K = np.array(camera_cal.intrinsic_matrix).reshape((3,3))
    D = np.array(camera_cal.distortion_coeff)
    image_shape = (camera_cal.resolution.width, camera_cal.resolution.height) # (1920, 1080)

    # Undistort Points
    pts = np.array(normalized_pixels, dtype=np.float64) * image_shape
    uv = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D).reshape(-1, 2).T

    # Rotate Rays to World Frame
    rays = cv2.Rodrigues(np.array(pose[0]))[0] @ np.vstack((uv, np.ones(uv.shape[1])))

    # Calculate Intersections with floor
    tvec = np.array(pose[1], dtype=np.float64).reshape(3, 1)
    with np.errstate(divide='ignore'): # Handle potential div/0
        s = -tvec[2] / rays[2]

    # Filter Valid Points and Return
    mask = (s > 0) & (np.abs(rays[2]) > 1e-6)
    return (tvec + s[mask] * rays[:, mask])[:2].T

def project_floor_to_pixels(floor_points, pose, camera_cal: nf_config.CameraCalibration):
    """
    Project world coordinates on the floor (z=0) back to normalized pixel coordinates.
    """
    # Use passed object for camera calibration.
    K = np.array(camera_cal.intrinsic_matrix).reshape((3,3))
    D = np.array(camera_cal.distortion_coeff)
    image_shape = (camera_cal.resolution.width, camera_cal.resolution.height) # (1920, 1080)

    floor_points = np.array(floor_points, dtype=np.float64)
    
    # Create 3D world points by appending z=0
    zeros = np.zeros((floor_points.shape[0], 1))
    object_points = np.hstack((floor_points, zeros))

    # Extract Camera-to-World rotation and translation
    rvec_c2w = np.array(pose[0], dtype=np.float64)
    tvec_c2w = np.array(pose[1], dtype=np.float64).reshape(3, 1)
    
    R_c2w, _ = cv2.Rodrigues(rvec_c2w)

    # Calculate World-to-Camera transformation for cv2.projectPoints
    R_w2c = R_c2w.T
    tvec_w2c = -R_w2c @ tvec_c2w
    
    # Convert rotation matrix back to rvec for projectPoints
    rvec_w2c, _ = cv2.Rodrigues(R_w2c)

    # Project 3D points to 2D pixel coordinates
    # projectPoints returns shape (N, 1, 2), so we reshape to (N, 2)
    image_points, _ = cv2.projectPoints(object_points, rvec_w2c, tvec_w2c, K, D)
    image_points = image_points.reshape(-1, 2)

    # Normalize coordinates to [0, 1] range
    # We divide by the image width and height provided in image_shape
    normalized_pixels = image_points / image_shape

    return normalized_pixels

def get_inward_wall_normal(p: np.ndarray, anchor_points: list[np.ndarray]) -> np.ndarray:
    """
    Given a point p and 4 anchor points (3D), finds the closest 2D wall segment
    and returns a unit vector pointing toward the interior.
    """
    # Project anchors and point to 2D (XY plane)
    anchors_2d = [a[:2] for a in anchor_points]
    p_2d = p[:2]
    
    closest_dist = float('inf')
    best_normal = np.array([0.0, 0.0])
    
    # Calculate centroid to determine "inward" direction
    centroid = np.mean(anchors_2d, axis=0)

    for i in range(len(anchors_2d)):
        p1 = anchors_2d[i]
        p2 = anchors_2d[(i + 1) % len(anchors_2d)]
        
        wall_vec = p2 - p1
        wall_len_sq = np.dot(wall_vec, wall_vec)
        
        # Project p onto the finite segment p1-p2
        # t is the interpolation factor [0, 1]
        t = max(0, min(1, np.dot(p_2d - p1, wall_vec) / wall_len_sq))
        closest_point_on_wall = p1 + t * wall_vec
        
        dist = np.linalg.norm(p_2d - closest_point_on_wall)
        
        if dist < closest_dist:
            closest_dist = dist
            # Standard 2D normal: (dx, dy) -> (-dy, dx)
            normal = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
            
            # Ensure it points toward the centroid
            to_interior = centroid - closest_point_on_wall
            if np.dot(normal, to_interior) < 0:
                normal = -normal
                
            # Normalize for a unit direction
            mag = np.linalg.norm(normal)
            best_normal = normal / mag if mag > 0 else normal
            
    return best_normal
