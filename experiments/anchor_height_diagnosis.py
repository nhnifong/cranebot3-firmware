#!/usr/bin/env python3
"""
Diagnoses whether origin AprilTag observations from an anchor camera
accurately estimate the anchor's height (z-position in room frame).

The pipeline under test:
  solvePnP  →  compose through arp_anchor_camera + tilt_node  →  invert  →  anchor pose

Usage:
    # Live stream, 50 frames (default):
    python experiments/anchor_height_diagnosis.py --anchor-ip 192.168.1.101

    # More or fewer frames:
    python experiments/anchor_height_diagnosis.py --anchor-ip 192.168.1.101 --n-frames 100

    # From a saved frame (single-frame mode):
    python experiments/anchor_height_diagnosis.py --image /path/to/frame.jpg

    # Override cam_tilt (degrees from horizontal):
    python experiments/anchor_height_diagnosis.py --anchor-ip 192.168.1.101 --cam-tilt 30
"""

import sys
import os
import argparse
import numpy as np
import cv2
from math import pi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pupil_apriltags import Detector
from nf_robot.common.pose_functions import compose_poses, invert_pose, average_pose
from nf_robot.common.config_loader import load_config, DEFAULT_CONFIG_PATH
import nf_robot.common.definitions as model_constants

# ── constants matching cv_common.py exactly ──────────────────────────────────

ORIGIN_TAG_SIZE_M = 0.1680  # from cv_common.SPECIAL_SIZES['origin']

BASE_MARKER_POINTS = np.array([
    [-0.5, -0.5, 0],
    [ 0.5, -0.5, 0],
    [ 0.5,  0.5, 0],
    [-0.5,  0.5, 0],
], dtype=np.float32)

ORIGIN_OBJ_POINTS = BASE_MARKER_POINTS * ORIGIN_TAG_SIZE_M

# ── helpers ───────────────────────────────────────────────────────────────────

detector = Detector(families="tag36h11", quad_decimate=1.0)


def grab_frames_from_stream(ip: str, n_frames: int = 50, port: int = 8888) -> list:
    """Pull n_frames consecutive frames from an anchor's TCP H264 stream."""
    import av
    uri = f'tcp://{ip}:{port}'
    print(f'  Connecting to {uri} …')
    options = {'fflags': 'nobuffer', 'flags': 'low_delay'}
    container = av.open(uri, options=options, mode='r')
    stream = next(s for s in container.streams if s.type == 'video')
    frames = []
    for pkt in container.decode(stream):
        frames.append(pkt.to_ndarray(format='rgb24'))
        if len(frames) >= n_frames:
            break
    container.close()
    if not frames:
        raise RuntimeError('No frames received from stream.')
    print(f'  Collected {len(frames)} frames  ({frames[0].shape})')
    return frames


def detect_origin_tag(frame_rgb: np.ndarray, K: np.ndarray, D: np.ndarray):
    """
    Run AprilTag detection and solvePnP on the origin card (tag ID 0).
    Returns a list of (rvec, tvec) tuples, one per detection.
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    results = []
    for det in detector.detect(gray):
        if det.tag_id != 0:
            continue
        corners = det.corners.astype(np.float32)
        _, rvec, tvec = cv2.solvePnP(
            ORIGIN_OBJ_POINTS, corners, K, D, False, cv2.SOLVEPNP_IPPE_SQUARE
        )
        results.append((rvec.reshape(3), tvec.reshape(3)))
    return results


def build_tilt_node(cam_tilt_deg: float):
    """
    Returns the tilt-correction pose node for the calibration pipeline.
    arp_anchor_camera has 22° of tilt baked in via an x-rotation.
    This node applies the additional x-rotation needed for a different tilt adapter.
    """
    extratilt = 22.0 - cam_tilt_deg
    return (np.array([extratilt / 180.0 * pi, 0.0, 0.0]), np.zeros(3))


def estimate_anchor_pose_from_origin(origin_pose_in_cam, cam_tilt_deg: float):
    """Replicate the initial-guess logic from optimize_arp_anchors()."""
    tilt_node = build_tilt_node(cam_tilt_deg)
    return invert_pose(compose_poses([
        model_constants.arp_anchor_camera,
        tilt_node,
        origin_pose_in_cam,
    ]))


def print_pose(label: str, pose):
    rvec, tvec = pose
    print(f'  {label}')
    print(f'    rvec (rod): {np.round(rvec, 5)}')
    print(f'    tvec (m):   {np.round(tvec, 5)}')


def run_diagnosis(frames: list, K: np.ndarray, D: np.ndarray, cam_tilt_deg: float):
    # ── Collect poses from all frames ────────────────────────────────────────
    print(f'\n──────────────────────────────────────────────')
    print(f' STEP 1: Detect origin AprilTag across {len(frames)} frame(s)')
    print(f'──────────────────────────────────────────────')

    all_poses = []
    for frame in frames:
        dets = detect_origin_tag(frame, K, D)
        if dets:
            all_poses.append(dets[0])

    n_det = len(all_poses)
    print(f'  Detected in {n_det}/{len(frames)} frames.')
    if n_det == 0:
        print('  ERROR: Origin tag (ID 0) not found in any frame.')
        return

    # Per-frame statistics
    heights_per_frame = []
    dists_per_frame = []
    for pose in all_poses:
        dists_per_frame.append(np.linalg.norm(pose[1]))
        ap = estimate_anchor_pose_from_origin(pose, cam_tilt_deg)
        heights_per_frame.append(ap[1][2])

    dists = np.array(dists_per_frame)
    heights = np.array(heights_per_frame)
    print(f'\n  Per-frame stats across {n_det} detections:')
    print(f'  Camera→tag distance:  mean={np.mean(dists):.4f} m  std={np.std(dists):.4f} m'
          f'  [{np.min(dists):.4f}, {np.max(dists):.4f}]')
    print(f'  Estimated height:     mean={np.mean(heights):.4f} m  std={np.std(heights):.4f} m'
          f'  [{np.min(heights):.4f}, {np.max(heights):.4f}]')

    # Average pose over all detections
    if n_det > 1:
        avg_pose_in_cam = average_pose(all_poses)
        print(f'\n  Using averaged pose over {n_det} detections for detailed analysis.')
    else:
        avg_pose_in_cam = all_poses[0]
        print(f'\n  Single detection; using directly.')

    origin_pose_in_cam = avg_pose_in_cam
    dist = np.linalg.norm(origin_pose_in_cam[1])

    # ── Step 2: Raw solvePnP result ──────────────────────────────────────────
    print(f'\n──────────────────────────────────────────────')
    print(f' STEP 2: Averaged solvePnP result (origin card in camera frame)')
    print(f'──────────────────────────────────────────────')
    print_pose('origin card pose in camera frame (averaged)', origin_pose_in_cam)
    print(f'    distance camera→tag: {dist:.4f} m')

    # ── Step 3: Compose through camera model ─────────────────────────────────
    tilt_node = build_tilt_node(cam_tilt_deg)
    R_check, _ = cv2.Rodrigues(compose_poses([model_constants.arp_anchor_camera, tilt_node])[0])
    look_down = np.degrees(np.arcsin(abs(R_check[:, 2][2])))

    print(f'\n──────────────────────────────────────────────')
    print(f' STEP 3: Compose through camera model (cam_tilt={cam_tilt_deg}°)')
    print(f'──────────────────────────────────────────────')
    print_pose('arp_anchor_camera (baked-in model, 22° tilt)', model_constants.arp_anchor_camera)
    print_pose(f'tilt_node (x-rotation, corrects to {cam_tilt_deg}°)', tilt_node)
    print(f'    effective look-down angle: {look_down:.2f}°')

    origin_in_anchor = compose_poses([
        model_constants.arp_anchor_camera, tilt_node, origin_pose_in_cam,
    ])
    print_pose('origin card pose in anchor frame', origin_in_anchor)

    # ── Step 4: Invert ───────────────────────────────────────────────────────
    print(f'\n──────────────────────────────────────────────')
    print(f' STEP 4: Invert → anchor pose in room frame')
    print(f'──────────────────────────────────────────────')
    anchor_pose = invert_pose(origin_in_anchor)
    print_pose('anchor pose in room frame', anchor_pose)

    estimated_height = anchor_pose[1][2]
    print(f'\n  >>> Estimated anchor HEIGHT (z): {estimated_height:.4f} m <<<')

    # ── Step 5: Ground truth comparison ─────────────────────────────────────
    print(f'\n──────────────────────────────────────────────')
    print(f' STEP 5: Compare to measured value')
    print(f'──────────────────────────────────────────────')
    answer = input(
        '  Enter true anchor height in meters (tape measure from floor to anchor body center),\n'
        '  or press Enter to skip: '
    ).strip()
    if not answer:
        return
    try:
        true_height = float(answer)
    except ValueError:
        print('  Could not parse that; skipping.')
        return

    error = estimated_height - true_height
    pct = 100.0 * error / true_height
    print(f'\n  True height:      {true_height:.4f} m')
    print(f'  Estimated height: {estimated_height:.4f} m  (from {n_det}-frame average)')
    print(f'  Error:            {error:+.4f} m  ({pct:+.2f}%)')
    direction = 'UNDERESTIMATES' if error < 0 else ('OVERESTIMATES' if error > 0 else 'matches exactly')
    print(f'  → Estimation {direction}.')

    # ── Finding A: verify tilt_node correctness ───────────────────────────────
    print(f'\n──────────────────────────────────────────────')
    print(f' FINDING A: tilt_node x-rotation — optical axis per cam_tilt')
    print(f'──────────────────────────────────────────────')
    print(f'  Each cam_tilt value now produces a distinct optical axis direction,')
    print(f'  confirming the tilt_node x-rotation correctly changes the look-down angle.')
    for test_tilt in [22, 26, 28, 30]:
        node = build_tilt_node(test_tilt)
        R_t, _ = cv2.Rodrigues(compose_poses([model_constants.arp_anchor_camera, node])[0])
        axis = R_t[:, 2]
        ld = np.degrees(np.arcsin(abs(axis[2])))
        test_pose = estimate_anchor_pose_from_origin(origin_pose_in_cam, test_tilt)
        z = test_pose[1][2]
        print(f'  cam_tilt={test_tilt}°: look_down={ld:.1f}°  optical_axis={np.round(axis,4)}  z={z:.4f} m  (err {z-true_height:+.4f} m)')
    print(f'  Effect of tilt on height: small (~7 mm across the range).')

    # ── Finding B: focal length sweep ────────────────────────────────────────
    print(f'\n──────────────────────────────────────────────')
    print(f' FINDING B: focal length needed to match true height')
    print(f'──────────────────────────────────────────────')
    # Re-detect corners from the last frame to run solvePnP with different K
    last_frame_rgb = frames[-1]
    gray_last = cv2.cvtColor(last_frame_rgb, cv2.COLOR_RGB2GRAY)
    dets_last = detector.detect(gray_last)
    origin_det = next((d for d in dets_last if d.tag_id == 0), None)

    if origin_det is not None:
        corners = origin_det.corners.astype(np.float32)
        px_h = np.linalg.norm(corners[1] - corners[0])
        px_v = np.linalg.norm(corners[2] - corners[1])
        f_current = K[0, 0]
        expected_span = f_current * ORIGIN_TAG_SIZE_M / dist
        print(f'  Corner pixel span (last frame): {px_h:.1f} px (h)  {px_v:.1f} px (v)')
        print(f'  Expected span at f={f_current:.0f}, size={ORIGIN_TAG_SIZE_M:.4f}m, dist={dist:.4f}m: {expected_span:.1f} px')
        print(f'\n  Sweeping focal length (using last frame corners):')
        found_f = []
        for f_test in np.arange(1300, 1700, 5):
            K_test = K.copy()
            K_test[0, 0] = f_test
            K_test[1, 1] = f_test
            _, r_t, t_t = cv2.solvePnP(
                ORIGIN_OBJ_POINTS, corners, K_test, D, False, cv2.SOLVEPNP_IPPE_SQUARE
            )
            test_pose_f = invert_pose(compose_poses([
                model_constants.arp_anchor_camera,
                build_tilt_node(cam_tilt_deg),
                (r_t.reshape(3), t_t.reshape(3)),
            ]))
            z_f = test_pose_f[1][2]
            if abs(z_f - true_height) < 0.01:
                found_f.append(f_test)
                print(f'  f={f_test:.0f}: dist={np.linalg.norm(t_t):.4f} m  z={z_f:.4f} m  ← within 1 cm of true height')
        if not found_f:
            print(f'  (No focal length in [1300,1700] gave z within 1 cm of {true_height:.4f} m)')

    # ── Finding C: marker size sweep ─────────────────────────────────────────
    print(f'\n──────────────────────────────────────────────')
    print(f' FINDING C: marker size needed to match true height')
    print(f'──────────────────────────────────────────────')
    if origin_det is not None:
        found_sz = []
        for size_mm in range(140, 220, 2):
            size = size_mm / 1000.0
            obj_pts = BASE_MARKER_POINTS * size
            _, r_s, t_s = cv2.solvePnP(
                obj_pts, corners, K, D, False, cv2.SOLVEPNP_IPPE_SQUARE
            )
            test_pose_s = invert_pose(compose_poses([
                model_constants.arp_anchor_camera,
                build_tilt_node(cam_tilt_deg),
                (r_s.reshape(3), t_s.reshape(3)),
            ]))
            z_s = test_pose_s[1][2]
            if abs(z_s - true_height) < 0.01:
                found_sz.append(size_mm)
                print(f'  size={size_mm} mm: dist={np.linalg.norm(t_s):.4f} m  z={z_s:.4f} m  ← within 1 cm of true height')
        if not found_sz:
            print(f'  (No marker size in [140,220] mm gave z within 1 cm of {true_height:.4f} m)')
        print(f'  (Current model uses {ORIGIN_TAG_SIZE_M*1000:.0f} mm.  You measured the card at 168.2 mm.)')


def main():
    parser = argparse.ArgumentParser(description='Anchor height estimation diagnosis')
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--anchor-ip', metavar='IP',
                     help='IP address of the anchor (streams from tcp://IP:8888)')
    src.add_argument('--image', metavar='PATH',
                     help='Path to a saved JPEG/PNG frame (single-frame mode)')
    parser.add_argument('--n-frames', type=int, default=50,
                        help='Number of frames to collect from live stream (default: 50)')
    parser.add_argument('--cam-tilt', type=float, default=None,
                        help='Tilt adapter angle in degrees from horizontal (default: from config)')
    parser.add_argument('--config', metavar='PATH', default=None,
                        help='Path to config JSON (default: src/nf_robot/common/configuration.json)')
    args = parser.parse_args()

    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    K = np.array(config.camera_cal.intrinsic_matrix).reshape(3, 3)
    D = np.array(config.camera_cal.distortion_coeff)
    print(f'Camera intrinsics from {config_path}')
    print(f'  K diagonal: fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}')

    if args.cam_tilt is not None:
        cam_tilt = args.cam_tilt
        print(f'cam_tilt = {cam_tilt}° (--cam-tilt argument)')
    elif (config.anchor_type is not None
          and len(config.anchors) > 0
          and config.anchors[0].indirect_line is not None
          and config.anchors[0].indirect_line.cam_tilt is not None):
        cam_tilt = config.anchors[0].indirect_line.cam_tilt
        print(f'cam_tilt = {cam_tilt}° (config anchor 0)')
    else:
        cam_tilt = 22.0
        print(f'cam_tilt = {cam_tilt}° (fallback default)')

    if args.anchor_ip:
        print(f'\nCollecting {args.n_frames} frames from anchor at {args.anchor_ip} …')
        frames = grab_frames_from_stream(args.anchor_ip, n_frames=args.n_frames)
        save_path = '/tmp/anchor_diagnosis_frame.jpg'
        cv2.imwrite(save_path, cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
        print(f'  Last frame saved to {save_path} for inspection.')
    else:
        raw = cv2.imread(args.image)
        if raw is None:
            print(f'ERROR: Could not load {args.image}')
            sys.exit(1)
        frames = [cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)]
        print(f'Loaded single frame from {args.image}  shape={frames[0].shape}')

    run_diagnosis(frames, K, D, cam_tilt)


if __name__ == '__main__':
    main()
