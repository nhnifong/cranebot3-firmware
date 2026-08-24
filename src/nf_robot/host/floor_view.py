import cv2
import numpy as np

from nf_robot.host.video_streamer import MjpegStreamer, NfVideoStreamer

SIDE_PX = 1000 # width and height of the square output image
EXTENT_M = 5.0 # Size of the floor area rendered in meters

def generate_orthographic_floor_maps(
    valid_anchor_clients,
    camera_cal,
    map_size_px=1800,
    map_extent_meters=10.0
):
    """
    Reprojects camera images from multiple overhead cameras to a top-down
    orthographic floor space projection using analytical homography.

    Channel order is whatever the clients decoded to (rgb24) and is passed through
    untouched, so the result is RGB despite OpenCV's usual convention.

    Args:
        valid_anchor_clients: List of camera clients containing .last_output_frame and .camera_pose
        camera_cal: Camera calibration data to pass into projection
        map_size_px: Output square resolution
        map_extent_meters: How many real-world meters the map_size_px covers (e.g. 10m x 10m)

    Returns:
        combined_rgb: 1800x1800x3 np.ndarray representing the stitched floor images
    """

    # Use float64 in [0,1] space for multiply blend
    combined_rgb = np.ones((map_size_px, map_size_px, 3), dtype=np.float64)
    touched = np.zeros((map_size_px, map_size_px, 1), dtype=bool)

    # Extract calibration matrices once
    K = np.array(camera_cal.intrinsic_matrix).reshape((3, 3))
    D = np.array(camera_cal.distortion_coeff)
    orig_w = camera_cal.resolution.width
    orig_h = camera_cal.resolution.height
    
    for client in valid_anchor_clients:
        rgb_image = client.last_output_frame

        h, w = rgb_image.shape[:2]
        
        # Scale the intrinsic matrix to match the current image resolution
        sx = w / float(orig_w)
        sy = h / float(orig_h)
        K_scaled = K.copy()
        K_scaled[0, :] *= sx
        K_scaled[1, :] *= sy
        
        # Undistort the incoming image
        rgb_undistorted = cv2.undistort(rgb_image, K_scaled, D)

        # Compute Analytical Homography
        rvec = np.array(client.camera_pose[0], dtype=np.float64)
        tvec = np.array(client.camera_pose[1], dtype=np.float64).reshape(3, 1)
        
        # The provided pose represents Camera-to-World (camera's position in world space).
        # We must convert it to World-to-Camera for projection: P_cam = R^T * P_world - R^T * tvec
        R_cam2world, _ = cv2.Rodrigues(rvec)
        R_world2cam = R_cam2world.T
        tvec_world2cam = -R_world2cam @ tvec
        
        # H_floor_to_img maps [X, Y, 1] on the floor (Z=0) to [u, v, 1] in undistorted image pixels
        H_floor_to_img = K_scaled @ np.column_stack((R_world2cam[:, 0], R_world2cam[:, 1], tvec_world2cam))
        
        # Invert to get mapping from Image Pixels to Floor Meters
        H_img_to_floor = np.linalg.inv(H_floor_to_img)
        
        # M maps Floor Meters to Orthographic Map Pixels.
        # It guarantees the origin (0,0) lands exactly at (map_size_px/2, map_size_px/2).
        M = np.array([
            [map_size_px / map_extent_meters, 0, map_size_px / 2.0],
            [0, -map_size_px / map_extent_meters, map_size_px / 2.0],
            [0, 0, 1.0]
        ], dtype=np.float64)
        
        # Final Homography: Undistorted Image Pixels -> Ortho Map Pixels
        H = M @ H_img_to_floor
        
        # Warp the image
        warped_rgb = cv2.warpPerspective(rgb_undistorted, H, (map_size_px, map_size_px))
        
        mask = (warped_rgb.sum(axis=-1, keepdims=True) > 0)
        warped_norm = warped_rgb.astype(np.float64) / 255.0
        # First camera to cover a pixel sets it; subsequent cameras multiply into it
        combined_rgb = np.where(mask & ~touched, warped_norm, combined_rgb)
        combined_rgb = np.where(mask & touched, combined_rgb * warped_norm, combined_rgb)
        touched |= mask

    # Finalize Image Stacking
    # Zero out pixels never covered by any camera, then convert to uint8
    combined_rgb_final = np.where(touched, combined_rgb * 255, 0).astype(np.uint8)

    return combined_rgb_final

# class responsible for combining camera views into a single image on the floor of the room aligned with it's coordinate space.
class FloorView:
    def __init__(self, local_telemetry=False):
        self.local_telemetry = local_telemetry
        frames_sent = 0

    def start(self):
        pass

    def stop(self):
        pass


    