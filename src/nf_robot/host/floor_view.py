import cv2
import numpy as np

from nf_robot.host.video_streamer import MjpegStreamer, VideoStreamer

SIDE_PX = 1000 # width and height of the square output image
EXTENT_M = 5.0 # Size of the floor area rendered in meters

def generate_orthographic_floor_maps(
    valid_anchor_clients, 
    heatmaps_np, 
    camera_cal, 
    map_size_px=1800, 
    map_extent_meters=10.0
):
    """
    Reprojects BGR images and target heatmaps from multiple overhead cameras 
    to a top-down orthographic floor space projection using analytical homography.
    
    Args:
        valid_anchor_clients: List of camera clients containing .last_frame_resized and .camera_pose
        heatmaps_np: List/array of numpy heatmaps corresponding to the clients
        camera_cal: Camera calibration data to pass into projection
        map_size_px: Output square resolution
        map_extent_meters: How many real-world meters the map_size_px covers (e.g. 10m x 10m)
        
    Returns:
        combined_heatmap: 1800x1800 np.ndarray representing the summed floor heatmaps
        combined_bgr: 1800x1800x3 np.ndarray representing the stitched floor images
    """
    
    # Initialize empty combined maps
    combined_heatmap = np.zeros((map_size_px, map_size_px), dtype=np.float32)
    
    # Use float32 for BGR to allow for averaging overlapping regions cleanly
    combined_bgr = np.zeros((map_size_px, map_size_px, 3), dtype=np.float32)
    weight_map = np.zeros((map_size_px, map_size_px, 1), dtype=np.float32)

    # Extract calibration matrices once
    K = np.array(camera_cal.intrinsic_matrix).reshape((3, 3))
    D = np.array(camera_cal.distortion_coeff)
    orig_w = camera_cal.resolution.width
    orig_h = camera_cal.resolution.height
    
    for i, client in enumerate(valid_anchor_clients):
        bgr_image = client.last_frame_resized
        heatmap = heatmaps_np[i]
        
        h, w = bgr_image.shape[:2]
        
        # Scale the intrinsic matrix to match the current image resolution
        sx = w / float(orig_w)
        sy = h / float(orig_h)
        K_scaled = K.copy()
        K_scaled[0, :] *= sx
        K_scaled[1, :] *= sy
        
        # 1. Undistort the incoming BGR image
        bgr_undistorted = cv2.undistort(bgr_image, K_scaled, D)

        # 2. Resize and undistort the heatmap
        if heatmap.shape[:2] != (h, w):
            heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            heatmap_resized = heatmap
        heatmap_undistorted = cv2.undistort(heatmap_resized, K_scaled, D)

        # 3. Compute Analytical Homography
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
        
        # 4. Warp the Heatmap
        warped_heatmap = cv2.warpPerspective(heatmap_undistorted, H, (map_size_px, map_size_px))
        
        # Add to the combined floor heatmap
        combined_heatmap += warped_heatmap
        
        # 5. Warp the BGR Image
        warped_bgr = cv2.warpPerspective(bgr_undistorted, H, (map_size_px, map_size_px))
        
        # Mask out black areas from the warp so we can average overlapping camera views correctly
        mask = (warped_bgr.sum(axis=-1, keepdims=True) > 0).astype(np.float32)
        combined_bgr += warped_bgr * mask
        weight_map += mask

    # 6. Finalize Image Stacking
    # Prevent division by zero in non-visible areas
    weight_map[weight_map == 0] = 1.0 
    
    # Average the BGR overlapping regions and cast back to standard uint8
    combined_bgr_final = (combined_bgr / weight_map).astype(np.uint8)
    
    combined_heatmap_clipped = np.clip(combined_heatmap, 0, 1.0)

    
    return combined_heatmap_clipped, combined_bgr_final

# class responsible for combining camrea views and heatmaps into a single image on the floor of the room aligned with it's coordinate space.
class FloorView:
    def __init__(self, local_telemetry=False):
        self.local_telemetry = local_telemetry
        frames_sent = 0

    def start(self):
        pass

    def stop(self):
        pass


    