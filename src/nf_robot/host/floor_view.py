import cv2
import numpy as np
from collections import OrderedDict

SIDE_PX = 1000 # width and height of the square output image
EXTENT_M = 5.0 # Size of the floor area rendered in meters

# Weights ramp in over this many ortho pixels at the rim of a camera's footprint. The
# blend is nearly winner-take-all, so an abrupt end to one camera's coverage otherwise
# lands as a hard line across the middle of another camera's view.
FEATHER_PX = 12.0

# log1p of a uint8 sample only takes 256 values, so the conversion to log space is a
# lookup that cv2.LUT applies to a whole warp in one threaded pass.
_LOG1P_LUT = np.log1p(np.arange(256, dtype=np.float32)).reshape(256, 1)

# Sums the three channels of a squared-difference image into one.
_CHANNEL_SUM = np.ones((1, 3), dtype=np.float32)

# Sampling grids keyed by pose and calibration. Anchor poses only move when the room is
# recalibrated, so in steady state every frame reuses the grid built on the first one.
_MAP_CACHE = OrderedDict()
_MAP_CACHE_MAX = 8


class _CamMap:
    """Where each ortho pixel inside one camera's floor footprint samples its image."""
    __slots__ = ('x0', 'y0', 'x1', 'y1', 'map1', 'map2', 'mask', 'feather')

    def __init__(self, x0, y0, x1, y1, map1, map2, mask):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.map1, self.map2 = map1, map2
        self.mask = mask
        self.feather = _feathered(mask)


def _feathered(mask):
    """0 outside the footprint, ramping up to 1 FEATHER_PX pixels inside it."""
    if mask.size == 0:
        return mask.astype(np.float32)
    padded = cv2.copyMakeBorder(mask.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    # distanceTransform is >= 1 everywhere inside the mask, so the ramp never reaches
    # zero on a covered pixel and the composite keeps its outer rim.
    return np.minimum(dist * (1.0 / FEATHER_PX), 1.0)


def _world_to_cam(camera_pose):
    """The world's pose in the camera's frame, from the camera's pose in the room."""
    rvec = np.array(camera_pose[0], dtype=np.float64)
    tvec = np.array(camera_pose[1], dtype=np.float64).reshape(3, 1)
    R_cam2world, _ = cv2.Rodrigues(rvec)
    R_world2cam = R_cam2world.T
    tvec_world2cam = -R_world2cam @ tvec
    rvec_world2cam, _ = cv2.Rodrigues(R_world2cam)
    return rvec_world2cam, tvec_world2cam, R_cam2world, tvec


def _footprint_bbox(R_cam2world, tvec, K, D, src_w, src_h, map_size_px, px_per_m):
    """Ortho-pixel bounding box of the floor this camera can see.

    Ray casts the image border down to z=0. A perspective camera's footprint is convex,
    so the border's bounding box is the whole footprint's; if any border ray misses the
    floor then the horizon is in frame and the footprint is unbounded, so take it all.
    """
    n = 32
    xs = np.linspace(0, src_w - 1, n)
    ys = np.linspace(0, src_h - 1, n)
    border = np.concatenate([
        np.stack([xs, np.zeros(n)], axis=1),
        np.stack([xs, np.full(n, src_h - 1.0)], axis=1),
        np.stack([np.zeros(n), ys], axis=1),
        np.stack([np.full(n, src_w - 1.0), ys], axis=1),
    ])
    uv = cv2.undistortPoints(border.reshape(-1, 1, 2), K, D).reshape(-1, 2).T
    rays = R_cam2world @ np.vstack((uv, np.ones(uv.shape[1])))
    with np.errstate(divide='ignore', invalid='ignore'):
        s = -tvec[2] / rays[2]
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        return 0, 0, map_size_px, map_size_px
    hits = tvec + s * rays
    u = hits[0] * px_per_m + map_size_px / 2.0
    v = -hits[1] * px_per_m + map_size_px / 2.0
    x0 = int(np.clip(np.floor(u.min()) - 1, 0, map_size_px))
    x1 = int(np.clip(np.ceil(u.max()) + 2, 0, map_size_px))
    y0 = int(np.clip(np.floor(v.min()) - 1, 0, map_size_px))
    y1 = int(np.clip(np.ceil(v.max()) + 2, 0, map_size_px))
    return x0, y0, x1, y1


def _camera_floor_map(camera_pose, K, D, src_w, src_h, map_size_px, map_extent_meters):
    """Build, or fetch, the grid that takes one camera's frame straight to the floor.

    A single cv2.remap through this grid replaces undistort followed by
    warpPerspective, which costs a full source-resolution intermediate image and
    interpolates the pixels twice. Only the footprint's bounding box is covered, so the
    per-frame blend never touches ortho pixels this camera cannot see.
    """
    key = (np.asarray(camera_pose).tobytes(), K.tobytes(), D.tobytes(),
           src_w, src_h, map_size_px, map_extent_meters)
    cached = _MAP_CACHE.get(key)
    if cached is not None:
        _MAP_CACHE.move_to_end(key)
        return cached

    px_per_m = map_size_px / map_extent_meters
    rvec_w2c, tvec_w2c, R_cam2world, tvec = _world_to_cam(camera_pose)
    x0, y0, x1, y1 = _footprint_bbox(R_cam2world, tvec, K, D, src_w, src_h,
                                     map_size_px, px_per_m)
    cam_map = _CamMap(0, 0, 0, 0, None, None, np.zeros((0, 0), bool))

    if x1 > x0 and y1 > y0:
        # Room coordinates of every ortho pixel in the box, matching the projection the
        # old M matrix applied: room origin at image centre, +y up so v is flipped.
        u = np.arange(x0, x1, dtype=np.float64)
        v = np.arange(y0, y1, dtype=np.float64)
        gx, gy = np.meshgrid((u - map_size_px / 2.0) / px_per_m,
                             -(v - map_size_px / 2.0) / px_per_m)
        pts = np.stack([gx, gy, np.zeros_like(gx)], axis=-1).reshape(-1, 3)

        src_px, _ = cv2.projectPoints(pts, rvec_w2c, tvec_w2c, K, D)
        src_px = src_px.reshape(y1 - y0, x1 - x0, 2)

        # The depth sign has to be tested separately: projectPoints happily maps points
        # behind the camera onto plausible looking pixels.
        R_w2c, _ = cv2.Rodrigues(rvec_w2c)
        depth = gx * R_w2c[2, 0] + gy * R_w2c[2, 1] + tvec_w2c[2, 0]

        sx, sy = src_px[..., 0], src_px[..., 1]
        mask = ((depth > 1e-6) & (sx >= 0) & (sx <= src_w - 1)
                & (sy >= 0) & (sy <= src_h - 1))

        # Tighten to what is really visible; distortion and the depth test both leave
        # the analytic box loose.
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if len(rows) and len(cols):
            r0, r1 = int(rows[0]), int(rows[-1]) + 1
            c0, c1 = int(cols[0]), int(cols[-1]) + 1
            mask = np.ascontiguousarray(mask[r0:r1, c0:c1])
            # Park everything outside the footprint on a coordinate that is safely out
            # of frame. Points near the horizon project arbitrarily far away, and left
            # alone they overflow the fixed point map and wrap back onto real pixels.
            src_px = np.where(mask[..., np.newaxis], src_px[r0:r1, c0:c1], -1.0)
            src_px = np.ascontiguousarray(src_px, dtype=np.float32)
            # Fixed point maps: half the memory of float32 and the faster remap path.
            map1, map2 = cv2.convertMaps(src_px, None, cv2.CV_16SC2)
            cam_map = _CamMap(x0 + c0, y0 + r0, x0 + c1, y0 + r1, map1, map2, mask)

    _MAP_CACHE[key] = cam_map
    if len(_MAP_CACHE) > _MAP_CACHE_MAX:
        _MAP_CACHE.popitem(last=False)
    return cam_map


def _warp(cam_map, image, border):
    """Sample one source image over a camera's floor footprint."""
    return cv2.remap(image, cam_map.map1, cam_map.map2, cv2.INTER_LINEAR,
                     borderMode=border)


def find_background_color(warps, agreement_threshold=30.0, num_clusters=3, max_samples=10000):
    """
    Finds the background color by finding pixels where overlapping images agree,
    sampling those colors, and clustering them.

    warps is one frame's [(_CamMap, warped image), ...].
    """
    agreed_colors = []

    # Compare every pair of overlapping images
    num_images = len(warps)
    for i in range(num_images):
        map_i, img_i = warps[i]
        for j in range(i + 1, num_images):
            map_j, img_j = warps[j]

            # Find where both images have data: intersect the two footprint boxes in
            # ortho space, then intersect the masks inside it.
            x0, y0 = max(map_i.x0, map_j.x0), max(map_i.y0, map_j.y0)
            x1, y1 = min(map_i.x1, map_j.x1), min(map_i.y1, map_j.y1)
            if x1 <= x0 or y1 <= y0:
                continue
            si = (slice(y0 - map_i.y0, y1 - map_i.y0), slice(x0 - map_i.x0, x1 - map_i.x0))
            sj = (slice(y0 - map_j.y0, y1 - map_j.y0), slice(x0 - map_j.x0, x1 - map_j.x0))
            overlap_mask = map_i.mask[si] & map_j.mask[sj]

            if not np.any(overlap_mask):
                continue

            # Using float32 for distance calculation to avoid overflow
            img1_overlap = img_i[si][overlap_mask].astype(np.float32)
            img2_overlap = img_j[sj][overlap_mask].astype(np.float32)

            # Calculate color distance
            diff = img1_overlap - img2_overlap
            dist = np.linalg.norm(diff, axis=-1)

            # Find where they nearly agree
            agree_mask = dist < agreement_threshold

            if np.any(agree_mask):
                # Calculate mean color of agreeing pixels
                mean_colors = (img1_overlap[agree_mask] + img2_overlap[agree_mask]) / 2.0
                agreed_colors.append(mean_colors)

    if not agreed_colors:
        # Fallback if no overlap or no agreement: just return a default (e.g., black or mid-gray)
        return np.array([128, 128, 128], dtype=np.float32)

    # Concatenate all agreed colors from all overlaps
    all_agreed_colors = np.vstack(agreed_colors)

    # Subsample if we have too many points to keep clustering fast
    if len(all_agreed_colors) > max_samples:
        indices = np.random.choice(len(all_agreed_colors), max_samples, replace=False)
        sampled_colors = all_agreed_colors[indices]
    else:
        sampled_colors = all_agreed_colors

    # Ensure data is float32 for cv2.kmeans
    sampled_colors = sampled_colors.astype(np.float32)

    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Avoid errors if we somehow have fewer samples than clusters
    k = min(num_clusters, len(sampled_colors))

    if k == 0:
        return np.array([128, 128, 128], dtype=np.float32)

    _, labels, centers = cv2.kmeans(sampled_colors, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Find the largest cluster
    unique, counts = np.unique(labels, return_counts=True)
    largest_cluster_idx = unique[np.argmax(counts)]

    background_color = centers[largest_cluster_idx]
    return background_color

# Cache the background color and source image brightness so we only compute it once at startup
_cached_bg_color = None
_cached_target_p_low = None
_cached_target_p_high = None
_cached_weight_lut = None

# The exposure percentiles only set one global scale factor, and every third row and
# column of a megapixel map still leaves six figures of samples behind the estimate.
_PERCENTILE_STRIDE = 3


def reset_floor_view_cache():
    """Drop the background and exposure estimates so the next call re-derives them."""
    global _cached_bg_color, _cached_target_p_low, _cached_target_p_high, _cached_weight_lut
    _cached_bg_color = None
    _cached_target_p_low = None
    _cached_target_p_high = None
    _cached_weight_lut = None


def generate_orthographic_floor_maps(
    valid_anchor_clients,
    heatmaps_np,
    camera_cal,
    map_size_px=1800,
    map_extent_meters=10.0
):
    """
    Reprojects camera images and target heatmaps from multiple overhead cameras
    to a top-down orthographic floor space projection using analytical homography.

    Uses a custom blending method:
    Blends overlaps by favoring the color furthest from the dynamically detected background color.
    """
    global _cached_bg_color, _cached_target_p_low, _cached_target_p_high, _cached_weight_lut

    # Extract calibration matrices once
    K = np.array(camera_cal.intrinsic_matrix).reshape((3, 3))
    D = np.array(camera_cal.distortion_coeff)
    orig_w = camera_cal.resolution.width
    orig_h = camera_cal.resolution.height

    # Warp every camera onto the floor up front: the first frame's background estimate
    # and the blend itself both read these.
    warps = []
    for i, client in enumerate(valid_anchor_clients):
        rgb_image = client.last_output_frame
        h, w = rgb_image.shape[:2]

        # Scale the intrinsic matrix to the resolution this camera actually streams
        K_scaled = K.copy()
        K_scaled[0, :] *= w / float(orig_w)
        K_scaled[1, :] *= h / float(orig_h)

        cam_map = _camera_floor_map(client.camera_pose, K_scaled, D, w, h,
                                    map_size_px, map_extent_meters)
        if cam_map.map1 is None:
            continue
        # BORDER_REPLICATE, not a black border: at the footprint rim a constant border
        # bleeds black into the interpolation, and near-black sits so far from the
        # background color that it outweighs every real view and draws the seam as a
        # dark line across the composite.
        warps.append((i, cam_map, _warp(cam_map, rgb_image, cv2.BORDER_REPLICATE)))

    if not warps:
        empty = None if heatmaps_np is None else np.zeros((map_size_px, map_size_px), dtype=np.float32)
        return empty, np.zeros((map_size_px, map_size_px, 3), dtype=np.uint8)

    if _cached_bg_color is None:
        _cached_bg_color = find_background_color([(m, img) for _, m, img in warps])
        _cached_weight_lut = None

        # One-time extraction of target light/dark levels from the first available image
        first_img = valid_anchor_clients[0].last_output_frame
        _cached_target_p_low, _cached_target_p_high = np.percentile(first_img, (1.0, 99.0))

    if _cached_weight_lut is None:
        # Per channel squared distance from the background color, which sums across the
        # channels to the squared distance the blend weight is built from.
        levels = np.arange(256, dtype=np.float32)
        bg = np.asarray(_cached_bg_color, dtype=np.float32).reshape(3)
        _cached_weight_lut = np.stack([(levels - c) ** 2 for c in bg], axis=-1).reshape(256, 1, 3)

    combined_heatmap = None if heatmaps_np is None else np.zeros((map_size_px, map_size_px), dtype=np.float32)

    combined_log_sum = np.zeros((map_size_px, map_size_px, 3), dtype=np.float32)
    weight_sum = np.zeros((map_size_px, map_size_px), dtype=np.float32)

    for i, cam_map, warped_rgb in warps:
        roi = (slice(cam_map.y0, cam_map.y1), slice(cam_map.x0, cam_map.x1))

        # Warp Heatmap. Resized to the camera's resolution so it can ride the camera's
        # own sampling grid instead of needing a second one, and given a zero border
        # because outside its footprint this camera predicts nothing.
        if combined_heatmap is not None:
            heatmap = heatmaps_np[i]
            h, w = valid_anchor_clients[i].last_output_frame.shape[:2]
            if heatmap.shape[:2] != (h, w):
                heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.add(combined_heatmap[roi], _warp(cam_map, heatmap, cv2.BORDER_CONSTANT),
                    combined_heatmap[roi])

        # Distance squared from background color (avoiding slow np.linalg.norm roots for optimization)
        sq_dist = cv2.transform(cv2.LUT(warped_rgb, _cached_weight_lut), _CHANNEL_SUM)

        # Raise the distance to a higher power to make foreground objects stand out more.
        # sq_dist is already distance^2. Squaring it again (distance^4) strongly penalizes the background.
        # You can change the exponent here (e.g., ** 1.5, ** 2, ** 3) to tune the contrast.
        weights = cv2.multiply(sq_dist, sq_dist)
        weights += 1e-5

        # Fade out at the edge of what this camera covers, and drop to zero past it
        cv2.multiply(weights, cam_map.feather, weights)

        # Convert rgb to log domain for geometric mean, accumulating the weighted sum in
        # one pass (np.log1p is the inverse of the np.expm1 further down)
        cv2.accumulateProduct(cv2.LUT(warped_rgb, _LOG1P_LUT),
                              cv2.merge((weights, weights, weights)),
                              combined_log_sum[roi])
        cv2.add(weight_sum[roi], weights, weight_sum[roi])

    # Normalize by total weight (Weighted Geometric Mean). Uncovered pixels divide 0 by
    # the floor rather than by 0, which keeps them at log 0 instead of NaN.
    divisor = cv2.max(weight_sum, 1e-20)
    mean_log = cv2.divide(combined_log_sum, cv2.merge((divisor, divisor, divisor)))

    # Convert back from log space. cv2.exp is the threaded expm1 + 1, and carrying that
    # + 1 through to the contrast stretch below is cheaper than subtracting it here.
    combined_exp = cv2.exp(mean_log)

    # --- Lightness Renormalization to Match Source ---
    # A mask of the pixels that actually received camera data
    valid_mask = cv2.compare(weight_sum, 0.0, cv2.CMP_GT)

    scale, offset = 1.0, 0.0
    if np.any(valid_mask):
        # Take only the colors that were drawn, so the black background cannot skew the
        # math, and only every few pixels, which is plenty for a percentile
        s = _PERCENTILE_STRIDE
        valid_colors = combined_exp[::s, ::s][valid_mask[::s, ::s] > 0]

        # Find the 1st and 99th percentiles of the combined image
        p_low, p_high = np.percentile(valid_colors, (1.0, 99.0)) - 1.0

        # Stretch the contrast to match the cached original image's dynamic range
        if p_high > p_low:  # Prevent division by zero
            scale = (_cached_target_p_high - _cached_target_p_low) / (p_high - p_low)
            offset = _cached_target_p_low - (p_low * scale)

    # Fold the stretch and the outstanding -1 into one pass, then clip to 0-255 and
    # convert to uint8. convertScaleAbs saturates at 255 but mirrors negatives, so the
    # low end is clamped first.
    stretched = cv2.addWeighted(combined_exp, scale, combined_exp, 0.0, offset - scale)
    combined_rgb_final = cv2.convertScaleAbs(cv2.max(stretched, 0.0))

    # Ensure the untouched background remains completely black
    combined_rgb_final = cv2.bitwise_and(combined_rgb_final, combined_rgb_final, mask=valid_mask)

    combined_heatmap_clipped = None if combined_heatmap is None else np.clip(combined_heatmap, 0, 1.0)

    return combined_heatmap_clipped, combined_rgb_final

# class responsible for combining camrea views and heatmaps into a single image on the floor of the room aligned with it's coordinate space.
class FloorView:
    def __init__(self, local_telemetry=False):
        self.local_telemetry = local_telemetry
        frames_sent = 0

    def start(self):
        pass

    def stop(self):
        pass
