#!/usr/bin/env python

"""Re-render a dataset's ortho floor view from the anchor camera videos behind it.

The ortho view a recording carries was blended by whatever host/floor_view.py did on
the day it was recorded. That blend has since improved - overlaps now resolve by
distance from the room's background color, with the footprint rims feathered, so the
hard seams and dark wedges of the old composite are gone. Nothing about the recording
needs to change to get the new one: the ortho view is a pure function of the two anchor
camera feeds and the room pose of each anchor camera, and a recording keeps both.

What has to be recovered, per source dataset:

  anchor camera poses  Datasets recorded from August 2026 onward carry the anchor poses
                       of the robot that recorded them in an `anchor_poses` feature.
                       Older ones need `anchor_config`, the calibration file the robot
                       was running - the same rule, and the same risk of a config that
                       has been recalibrated since, as camera_goal; see
                       ml/calibrations/readme.md.

  camera intrinsics    Not recorded at all. They come from `anchor_config` when there is
                       one, and otherwise from the stock calibration in config_loader,
                       which is what a robot that has never had a chessboard run has.

  camera tilt          An anchor's pose says where the anchor is, not where its camera
                       looks from: the camera sits on a mount tilted by the anchor's
                       configured cam_tilt, which is neither recorded nor constant
                       across robots (the configs on hand hold 25, 26 and 30 degrees,
                       and one robot's two anchors differ). A few degrees out moves the
                       floor by tens of centimetres, which would misalign every ortho
                       target label, so a wrong guess is not survivable.

                       With no config to read it from, the tilt is recovered from the
                       recording itself: the ortho video already in the dataset was
                       rendered live with the right tilt, so re-rendering one anchor at
                       a range of tilts and phase-correlating each against it picks the
                       true one out with a sharp, unambiguous peak.

Runs as an optional step of lerobot_derive_dataset (recipe key `reblend_ortho`), where
it reads the anchor videos from the source dataset - the camera_mode conversion is
about to drop them - and writes the ortho videos into the derived one.

Usage:
    python src/nf_robot/ml/lerobot_reblend_ortho.py \
        --source_root /path/to/recorded_dataset \
        --dest_root /path/to/derived_dataset \
        [--anchor_config src/nf_robot/ml/calibrations/conf_nick.json] \
        [--preview /tmp/reblend_preview]
"""

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from tqdm import tqdm

from lerobot.datasets.compute_stats import (
    aggregate_stats, auto_downsample_height_width, get_feature_stats, sample_indices,
)

from nf_robot.common.config_loader import create_default_config, load_config
from nf_robot.common.pose_functions import arp_anchor_camera_pose
from nf_robot.common.util import poseProtoToTuple
from nf_robot.generated.nf import config as nf_config
from nf_robot.host.floor_view import EXTENT_M, SIDE_PX, OrthoBlender
from nf_robot.ml import camera_goal
from nf_robot.ml.lerobot_resize_video_feature import open_encoder

ORTHO_KEY = "observation.images.overhead_camera"
ANCHOR_KEYS = ("observation.images.anchor_camera_0", "observation.images.anchor_camera_1")

# The live ortho worker renders at this size and extent (observer._ortho_worker), and
# ortho_target reads room coordinates back out of the result assuming the same. Only
# these values reproduce the view the recording holds and the labels describe.
RENDER_SIZE_PX = SIDE_PX
RENDER_EXTENT_M = EXTENT_M

# The live view is rendered at about this rate into a 30 fps recording, so most recorded
# ortho frames are repeats of the one before. Re-rendering all thirty per second would
# cost three times as much for freshness the policy never sees at eval time. None or 0
# renders every frame.
RENDER_FPS = 10.0

# Tilts to search when no config records the real one, in degrees. Wide enough to cover
# every anchor mount seen so far (25-30) with room either side; a whole degree apart
# because the correlation peak is a degree wide and cam_tilt is set in whole degrees.
TILT_SEARCH_DEG = np.arange(18.0, 39.0, 1.0)
# Frames sampled per calibration when searching. Three agreeing is enough to rule out a
# frame that happened to be blurred or half empty.
TILT_SAMPLE_FRAMES = 3
# The peak must beat the best tilt more than 2 degrees away by this factor. A real peak
# clears it several times over; anything less means the recorded ortho and the re-render
# have nothing in common, which is a wrong pose or wrong intrinsics rather than a wrong
# tilt.
TILT_PEAK_MARGIN = 2.0


def _camera_calibration(intrinsic_matrix, distortion_coeff, width, height):
    """A CameraCalibration from plain numbers, so worker processes need no proto to pickle."""
    cal = nf_config.CameraCalibration()
    cal.resolution = nf_config.Resolution(width=int(width), height=int(height))
    cal.intrinsic_matrix = list(intrinsic_matrix)
    cal.distortion_coeff = list(distortion_coeff)
    return cal


# Frames further ahead than this are reached by seeking rather than by decoding
# everything in between. Below it the decode is cheaper than the seek plus the keyframe
# it has to restart from, and the frame-at-a-time walk that does the real work never
# crosses it anyway.
_SEEK_AHEAD_FRAMES = 60


class _FrameReader:
    """Reads one video file's frames by index, seeking only when it has to.

    Frames are indexed by presentation time rather than by counting decodes, so a seek
    lands where it claims to and a file missing a frame in the middle cannot shift
    everything after it.
    """

    def __init__(self, path, fps):
        self.path = Path(path)
        self.fps = float(fps)
        self._container = None
        self._stream = None
        self._decoder = None
        self._frame = None
        self._index = -1
        self._end = None

    def _restart(self, at_index=None):
        if self._container is None:
            self._container = av.open(str(self.path))
            self._stream = self._container.streams.video[0]
            self._stream.thread_type = "AUTO"
        if at_index is not None:
            self._container.seek(int(round(at_index / self.fps / float(self._stream.time_base))),
                                 stream=self._stream)
        self._decoder = self._container.decode(self._stream)

    def frame(self, index):
        """The frame at `index`, or the last one in the file if it ends before that."""
        if self._decoder is None:
            self._restart()
        if index == self._index or (self._end is not None and index >= self._end):
            return self._frame
        if index < self._index or index > self._index + _SEEK_AHEAD_FRAMES:
            self._restart(at_index=index)
        for av_frame in self._decoder:
            if av_frame.pts is None:
                continue
            self._index = int(round(float(av_frame.pts * self._stream.time_base) * self.fps))
            self._frame = av_frame.to_ndarray(format="rgb24")
            if self._index >= index:
                return self._frame
        # Past the end of a file that holds fewer frames than the episode claims, which a
        # recording that dropped camera frames leaves behind. Hold the last one.
        self._end = self._index + 1
        if self._frame is None:
            raise ValueError(f"{self.path} decoded no frames")
        return self._frame

    def close(self):
        if self._container is not None:
            self._container.close()
            self._container = None
        self._decoder = None


def _read_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def _episode_rows(root: Path) -> list[dict]:
    """One row per episode: its length and where each camera's frames sit, in order.

    `base` is the index of the episode's first frame within that camera's video file,
    which is what the from_timestamp the recorder wrote means in frames.
    """
    import pyarrow.parquet as pq

    fps = float(_read_info(root)["fps"])
    rows: list[dict] = []
    for path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        table = pq.read_table(path)
        names = set(table.schema.names)
        columns = {n: table.column(n).to_pylist() for n in names if not n.startswith("stats/")}
        for i in range(table.num_rows):
            row = {
                "episode_index": columns["episode_index"][i],
                "length": columns["length"][i],
                "videos": {},
            }
            for key in (ORTHO_KEY,) + ANCHOR_KEYS:
                if f"videos/{key}/chunk_index" not in columns:
                    continue
                row["videos"][key] = {
                    "chunk": columns[f"videos/{key}/chunk_index"][i],
                    "file": columns[f"videos/{key}/file_index"][i],
                    "base": int(round(columns[f"videos/{key}/from_timestamp"][i] * fps)),
                }
            rows.append(row)
    rows.sort(key=lambda r: r["episode_index"])
    return rows


def _video_path(root: Path, key: str, chunk: int, file: int) -> Path:
    return root / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"


def _episode_calibration(root: Path, key: str, unpack) -> dict[int, list] | None:
    """Each episode's recorded value of one calibration feature, or None if absent.

    The value is constant within an episode - it only changes when a robot is
    recalibrated between recordings - so the episode's first frame speaks for it.
    """
    import pyarrow.parquet as pq

    if key not in _read_info(root)["features"]:
        return None

    out: dict[int, list] = {}
    for path in sorted(root.glob("data/chunk-*/file-*.parquet")):
        table = pq.read_table(path, columns=["episode_index", key])
        episodes = np.asarray(table.column("episode_index").to_pylist())
        values = np.asarray(table.column(key).to_pylist(), dtype=np.float64)
        for episode in np.unique(episodes):
            if int(episode) in out:
                continue
            unpacked = unpack(values[np.flatnonzero(episodes == episode)[0]])
            if unpacked is not None:
                out[int(episode)] = unpacked
    return out or None


def load_anchor_config(path):
    """(camera_cal, anchor poses, cam_tilts) from the config file of the robot that recorded."""
    config = load_config(Path(path))
    poses = [poseProtoToTuple(anchor.pose) for anchor in config.anchors]
    tilts = [float(anchor.indirect_line.cam_tilt) for anchor in config.anchors]
    return config.camera_cal, poses, tilts


def _render(views, camera_cal, out_w, out_h, blender=None):
    """One ortho frame at the recording's own size, from [(image, camera_pose), ...]."""
    blender = blender or OrthoBlender()
    ortho = blender.render(views, camera_cal, RENDER_SIZE_PX, RENDER_EXTENT_M)
    if (ortho.shape[1], ortho.shape[0]) != (out_w, out_h):
        # The recorder downsizes whatever the live view sends it with a plain
        # cv2.resize, so matching it keeps the re-render on the same footing.
        ortho = cv2.resize(ortho, (out_w, out_h))
    return ortho


def _alignment(reference_gray, ortho):
    """How strongly a re-rendered ortho lines up with the recorded one, 0 to 1."""
    gray = cv2.cvtColor(ortho, cv2.COLOR_RGB2GRAY).astype(np.float32)
    _, response = cv2.phaseCorrelate(reference_gray, gray)
    return float(response)


def recover_cam_tilts(samples, anchor_poses, camera_cal, out_w, out_h):
    """Each anchor's cam_tilt, read back off the ortho video the recording already has.

    samples is [(recorded ortho frame, [anchor frame, ...]), ...]. Each anchor is warped
    onto the floor alone and correlated against the recorded composite: over the floor
    that anchor covers, the composite is essentially that anchor's own view, so the
    correlation peaks where the tilt is right and stays in the noise everywhere else.
    """
    references = [cv2.cvtColor(ortho, cv2.COLOR_RGB2GRAY).astype(np.float32)
                  for ortho, _ in samples]
    tilts = []
    for anchor, pose in enumerate(anchor_poses):
        scores = []
        for tilt in TILT_SEARCH_DEG:
            camera_pose = arp_anchor_camera_pose(pose, tilt)
            scores.append(float(np.median([
                _alignment(reference,
                           _render([(frames[anchor], camera_pose)], camera_cal, out_w, out_h))
                for reference, (_, frames) in zip(references, samples)
            ])))
        # A tilt whose footprint falls off the map correlates against nothing and can
        # come back as a nan, which would win an argmax outright.
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        best = int(np.argmax(scores))
        # The runner-up has to come from a genuinely different tilt: the peak is about a
        # degree wide, so its own shoulders would otherwise look like competition.
        rivals = [s for i, s in enumerate(scores) if abs(TILT_SEARCH_DEG[i] - TILT_SEARCH_DEG[best]) > 2.0]
        runner_up = max(rivals) if rivals else 0.0
        if scores[best] < TILT_PEAK_MARGIN * max(runner_up, 1e-6):
            raise ValueError(
                f"anchor_camera_{anchor}: no cam_tilt reproduces the recorded ortho view "
                f"(best {TILT_SEARCH_DEG[best]:.0f} deg scores {scores[best]:.3f} against "
                f"{runner_up:.3f} elsewhere). The anchor poses or the intrinsics are wrong "
                f"for this dataset, not just the tilt."
            )
        logging.info(
            f"anchor_camera_{anchor}: recovered cam_tilt {TILT_SEARCH_DEG[best]:.0f} deg "
            f"(alignment {scores[best]:.3f}, next best elsewhere {runner_up:.3f})"
        )
        tilts.append(float(TILT_SEARCH_DEG[best]))
    return tilts


def _sample_frames(source_root: Path, episode: dict, fps: float, count: int):
    """[(recorded ortho frame, [anchor frame, ...]), ...] spread across one episode."""
    readers = {
        key: _FrameReader(_video_path(source_root, key, episode["videos"][key]["chunk"],
                                      episode["videos"][key]["file"]), fps)
        for key in (ORTHO_KEY,) + ANCHOR_KEYS
    }
    try:
        # Skip the first tenth: an episode opens with the gantry still parked over the
        # floor, which is the least informative view of it.
        offsets = np.linspace(episode["length"] * 0.1, episode["length"] - 1, count).astype(int)
        return [
            (readers[ORTHO_KEY].frame(episode["videos"][ORTHO_KEY]["base"] + int(offset)),
             [readers[key].frame(episode["videos"][key]["base"] + int(offset)) for key in ANCHOR_KEYS])
            for offset in offsets
        ]
    finally:
        for reader in readers.values():
            reader.close()


def _pose_key(poses):
    """Hashable identity of one calibration, so episodes sharing it are grouped."""
    return np.round(np.concatenate([np.concatenate(p) for p in poses]), 6).tobytes()


def resolve_camera_poses(source_root, episodes, fps, out_w, out_h, anchor_config=None,
                         cam_tilt=None):
    """Each episode's anchor camera poses in the room, ready to warp with.

    Returns (camera_cal, {episode: [camera_pose, ...]}, {episode: [cam_tilt, ...]}).
    Anchor poses recorded with the data win over a config file, matching camera_goal:
    they are the calibration that was actually running, where a config is only right if
    nothing has been recalibrated since. The tilts follow the same rule.
    """
    config_cal, config_poses, config_tilts = (None, None, None)
    if anchor_config is not None:
        config_cal, config_poses, config_tilts = load_anchor_config(anchor_config)

    camera_cal = config_cal or create_default_config().camera_cal
    if config_cal is None:
        logging.warning(
            "No anchor_config for this dataset, so the stock camera intrinsics are "
            "assumed; a robot with its own chessboard calibration would differ."
        )

    recorded = _episode_calibration(source_root, camera_goal.ANCHOR_POSES_KEY,
                                    camera_goal.unpack_anchor_poses)
    recorded_tilts = _episode_calibration(source_root, camera_goal.ANCHOR_CAM_TILT_KEY,
                                          camera_goal.unpack_anchor_cam_tilt) or {}
    if recorded is None and config_poses is None:
        raise ValueError(
            f"{source_root} has no '{camera_goal.ANCHOR_POSES_KEY}' feature and no "
            f"anchor_config was given, so nothing says where its anchor cameras were."
        )

    anchor_poses = {
        row["episode_index"]: (recorded or {}).get(row["episode_index"], config_poses)
        for row in episodes
    }
    missing = [e for e, p in anchor_poses.items() if p is None or len(p) < len(ANCHOR_KEYS)]
    if missing:
        raise ValueError(
            f"{len(missing)} episode(s) record no anchor poses and no anchor_config covers "
            f"them, starting at episode {missing[0]}"
        )

    # One tilt search per distinct calibration rather than per episode: a dataset changes
    # calibration only where it changed robots or was recalibrated mid-run.
    by_calibration: dict[bytes, list[dict]] = {}
    for row in episodes:
        by_calibration.setdefault(_pose_key(anchor_poses[row["episode_index"]]), []).append(row)

    camera_poses: dict[int, list] = {}
    cam_tilts: dict[int, list] = {}
    for group in by_calibration.values():
        first = group[0]["episode_index"]
        poses = anchor_poses[first]
        # Most specific first: what the recipe pinned, then what the recording itself
        # says, then the config that stood in for it, then the recording's own ortho view.
        if cam_tilt is not None:
            tilts = [float(t) for t in cam_tilt]
        elif first in recorded_tilts:
            tilts = recorded_tilts[first]
            logging.info(f"anchor camera tilts recorded with the data: {tilts}")
        elif config_tilts is not None:
            tilts = config_tilts
        else:
            logging.info(
                f"Recovering anchor camera tilt from the recorded ortho view of episode "
                f"{first} ({len(group)} episode(s) share its calibration)"
            )
            tilts = recover_cam_tilts(
                _sample_frames(source_root, group[0], fps, TILT_SAMPLE_FRAMES),
                poses, camera_cal, out_w, out_h,
            )
        resolved = [arp_anchor_camera_pose(pose, tilt) for pose, tilt in zip(poses, tilts)]
        for row in group:
            camera_poses[row["episode_index"]] = resolved
            cam_tilts[row["episode_index"]] = list(tilts)

    return camera_cal, camera_poses, cam_tilts


def _episode_stats(frames: list[np.ndarray]) -> dict:
    """One episode's stats for the ortho feature, in the shape lerobot writes them.

    This is compute_episode_stats' image branch on frames already in hand rather than
    read back off disk: per-channel over a (batch, channel, height, width) stack,
    rescaled to 0-1 and left with the leading channel axis lerobot squeezes it to.

    `count` therefore comes out as a frame count, where a dataset recorded under an
    older lerobot may carry a pixel count for the same field. It only weights the
    aggregate over episodes, and every episode here is counted the same way.
    """
    array = np.stack(frames)
    stats = get_feature_stats(array, axis=(0, 2, 3), keepdims=True)
    return {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in stats.items()}


def _reader(readers, source_root, key, episode, fps):
    """The open reader for the file one episode's frames of `key` live in."""
    placement = episode["videos"][key]
    cache_key = (key, placement["chunk"], placement["file"])
    if cache_key not in readers:
        readers[cache_key] = _FrameReader(
            _video_path(Path(source_root), key, placement["chunk"], placement["file"]), fps)
    return readers[cache_key]


def _render_ortho_file(job: dict) -> tuple[str, int, list[float]]:
    """Re-render every ortho frame of one output video file. Runs in a worker process."""
    fps = job["fps"]
    out_w, out_h = job["out_w"], job["out_h"]
    camera_cal = _camera_calibration(*job["camera_cal"])
    dest = Path(job["dest_path"])
    tmp = dest.with_suffix(".tmp.mp4")
    # One render per this many frames; the rest repeat it, exactly as the live view's
    # slower cadence repeats into a 30 fps recording.
    stride = max(1, int(round(fps / job["render_fps"]))) if job["render_fps"] else 1

    out, stream = open_encoder(tmp, out_w, out_h, fps, job["vcodec"], job["pix_fmt"],
                               job["crf"], job["g"], job["threads"])
    readers: dict[tuple, _FrameReader] = {}
    alignments: list[float] = []
    stats: dict[int, dict] = {}
    written = 0
    try:
        for episode in job["episodes"]:
            # Every recording is its own room under its own lighting, so the background
            # and exposure the blend keys off are measured afresh for each.
            blender = OrthoBlender()
            views_source = [
                (_reader(readers, job["source_root"], key, episode, fps),
                 episode["videos"][key]["base"],
                 np.asarray(pose, dtype=float))
                for key, pose in zip(ANCHOR_KEYS, job["camera_poses"][str(episode["episode_index"])])
            ]
            # The stats the dataset carries for this feature describe the old pixels, so
            # they are recomputed from the new ones on the way past - on the frames
            # lerobot itself would have sampled, so the result is what it would produce.
            sampled_at = set(sample_indices(episode["length"]))
            sampled: list[np.ndarray] = []
            ortho = None
            for frame in range(episode["length"]):
                if frame % stride == 0:
                    ortho = _render(
                        [(reader.frame(base + frame), pose)
                         for reader, base, pose in views_source],
                        camera_cal, out_w, out_h, blender)
                    if job["check"] and frame == 0:
                        recorded = _reader(readers, job["source_root"], ORTHO_KEY, episode, fps) \
                            .frame(episode["videos"][ORTHO_KEY]["base"])
                        alignments.append(_alignment(
                            cv2.cvtColor(recorded, cv2.COLOR_RGB2GRAY).astype(np.float32), ortho))
                if frame in sampled_at:
                    sampled.append(auto_downsample_height_width(ortho.transpose(2, 0, 1)))
                av_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(ortho), format="rgb24")
                av_frame.pts = written
                av_frame.time_base = Fraction(1, int(fps))
                for packet in stream.encode(av_frame.reformat(format=job["pix_fmt"])):
                    out.mux(packet)
                written += 1
            stats[episode["episode_index"]] = _episode_stats(sampled)
        for packet in stream.encode():
            out.mux(packet)
    finally:
        out.close()
        for reader in readers.values():
            reader.close()

    tmp.replace(dest)
    return job["dest_path"], written, alignments, stats


def _rewrite_ortho_meta(dest_root: Path, files: dict, stats: dict[int, dict]) -> None:
    """Bring the episode metadata into line with the ortho videos that were just written.

    Two things move. The window each episode occupies in its video file: every episode is
    re-rendered at one frame per recorded frame, which is what the recording was supposed
    to hold but not always what it did - a dropped frame left the live ortho short - so
    the windows are recomputed rather than trusted. And the feature's statistics, which
    described the old pixels.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    fps = float(_read_info(dest_root)["fps"])
    windows: dict[int, tuple[float, float]] = {}
    for episodes in files.values():
        offset = 0
        for episode in episodes:
            windows[episode["episode_index"]] = (offset / fps, (offset + episode["length"]) / fps)
            offset += episode["length"]

    columns = {f"videos/{ORTHO_KEY}/from_timestamp": lambda e: windows[e][0],
               f"videos/{ORTHO_KEY}/to_timestamp": lambda e: windows[e][1]}
    for stat in next(iter(stats.values()), {}):
        columns[f"stats/{ORTHO_KEY}/{stat}"] = (
            lambda e, stat=stat: np.asarray(stats[e][stat]).tolist())

    for path in sorted((dest_root / "meta" / "episodes").glob("**/*.parquet")):
        table = pq.read_table(path)
        episodes = table.column("episode_index").to_pylist()
        for key, value_of in columns.items():
            index = table.schema.get_field_index(key)
            if index < 0:
                continue
            field = table.schema.field(index)
            table = table.set_column(index, field,
                                     pa.array([value_of(e) for e in episodes], type=field.type))
        pq.write_table(table, path)

    # meta/stats.json is the same numbers aggregated over every episode; only this
    # feature's entry changes, so the rest are left exactly as they were.
    stats_path = dest_root / "meta" / "stats.json"
    if stats_path.exists() and stats:
        whole = json.loads(stats_path.read_text())
        aggregated = aggregate_stats([{ORTHO_KEY: s} for s in stats.values()])[ORTHO_KEY]
        whole[ORTHO_KEY] = {k: np.asarray(v).tolist() for k, v in aggregated.items()}
        stats_path.write_text(json.dumps(whole, indent=4))


def reblend_ortho(
    source_root: Path,
    dest_root: Path,
    anchor_config=None,
    cam_tilt: list[float] | None = None,
    render_fps: float | None = RENDER_FPS,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    crf: int = 30,
    g: int = 2,
    headroom: int = 0,
    check: bool = True,
) -> None:
    """Replace dest_root's ortho videos with ones re-blended from source_root's anchors.

    The two roots are separate because this runs inside a camera_mode conversion: the
    derived dataset no longer has the anchor cameras the ortho view is made of, so the
    frames come from the recording they were copied from.

    Returns the camera tilts it settled on, per episode.
    """
    source_root, dest_root = Path(source_root), Path(dest_root)
    source_info, dest_info = _read_info(source_root), _read_info(dest_root)

    missing = [k for k in ANCHOR_KEYS + (ORTHO_KEY,) if k not in source_info["features"]]
    if missing:
        raise ValueError(f"{source_root} has no {missing}, so its ortho view cannot be rebuilt")
    if ORTHO_KEY not in dest_info["features"]:
        raise ValueError(f"{dest_root} keeps no '{ORTHO_KEY}'; drop the reblend step for it")

    fps = float(source_info["fps"])
    out_h, out_w = dest_info["features"][ORTHO_KEY]["shape"][:2]
    episodes = _episode_rows(source_root)

    camera_cal, camera_poses, cam_tilts = resolve_camera_poses(
        source_root, episodes, fps, out_w, out_h, anchor_config, cam_tilt)

    # One job per output file, keeping the episode-to-file layout the derived dataset
    # already has so nothing but the pixels and the ortho timestamps changes. Where each
    # episode goes is the destination's business; every frame read comes from the source,
    # whose own files the episode rows above describe.
    destinations = {row["episode_index"]: row["videos"][ORTHO_KEY]
                    for row in _episode_rows(dest_root)}
    files: dict[tuple[int, int], list[dict]] = {}
    for episode in episodes:
        placement = destinations[episode["episode_index"]]
        files.setdefault((placement["chunk"], placement["file"]), []).append(episode)

    workers = max(1, (os.cpu_count() or 1) - headroom)
    active = max(1, min(workers, len(files)))
    threads = max(1, workers // max(1, len(files)))
    logging.info(
        f"Re-blending {len(episodes)} episode(s) of ortho video into {len(files)} file(s) "
        f"with {active} worker(s) x {threads} encoder thread(s)"
    )

    jobs = [{
        "source_root": str(source_root),
        "dest_path": str(_video_path(dest_root, ORTHO_KEY, chunk, file)),
        "episodes": eps,
        # str keys: this crosses a process boundary as JSON-shaped plain data
        "camera_poses": {str(e["episode_index"]): [[p[0].tolist(), p[1].tolist()]
                                                   for p in camera_poses[e["episode_index"]]]
                         for e in eps},
        "camera_cal": (list(camera_cal.intrinsic_matrix), list(camera_cal.distortion_coeff),
                       camera_cal.resolution.width, camera_cal.resolution.height),
        "fps": fps, "out_w": out_w, "out_h": out_h, "render_fps": render_fps,
        "vcodec": vcodec, "pix_fmt": pix_fmt, "crf": crf, "g": g, "threads": threads,
        "check": check,
    } for (chunk, file), eps in sorted(files.items())]

    alignments: list[float] = []
    stats: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=active) as pool:
        futures = {pool.submit(_render_ortho_file, job): job for job in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Re-blending ortho"):
            try:
                path, written, scores, file_stats = future.result()
            except Exception as e:
                raise RuntimeError(
                    f"failed to re-blend {futures[future]['dest_path']}") from e
            alignments += scores
            stats.update(file_stats)
            logging.debug(f"wrote {written} ortho frame(s) to {path}")

    _rewrite_ortho_meta(dest_root, files, stats)

    from lerobot.datasets.video_utils import get_video_info

    first = sorted(files)[0]
    dest_info["features"][ORTHO_KEY]["info"] = get_video_info(
        _video_path(dest_root, ORTHO_KEY, *first))
    (dest_root / "meta" / "info.json").write_text(json.dumps(dest_info, indent=4))

    # Whatever the tilts turned out to be, the derived dataset now says so - a source
    # that had to have them recovered does not have to have them recovered again.
    camera_goal.add_anchor_cam_tilt_feature(dest_root, cam_tilts)

    if alignments:
        # Low scores here mean the re-render does not sit where the recorded one did,
        # which puts every ortho target label somewhere the objects are not.
        weak = [s for s in alignments if s < 0.15]
        logging.info(
            f"Alignment against the recorded ortho view: median {np.median(alignments):.3f}, "
            f"worst {min(alignments):.3f} over {len(alignments)} episode(s)"
        )
        if weak:
            logging.warning(
                f"{len(weak)} episode(s) barely line up with the ortho view they were "
                f"recorded with; check the anchor poses and cam_tilt for this dataset"
            )

    return cam_tilts


def write_preview(source_root: Path, dest_root: Path, output_dir: Path, count: int = 4,
                  anchor_config=None, cam_tilt=None) -> None:
    """Dump recorded-vs-reblended ortho pairs, for eyeballing before paying for a build."""
    source_root, dest_root, output_dir = Path(source_root), Path(dest_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = float(_read_info(source_root)["fps"])
    out_h, out_w = _read_info(dest_root)["features"][ORTHO_KEY]["shape"][:2]
    for path, image in _preview_images(source_root, fps, out_w, out_h, count,
                                       anchor_config, cam_tilt):
        cv2.imwrite(str(output_dir / path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    logging.info(f"Wrote {count} recorded-vs-reblended ortho pair(s) to {output_dir}")


def _preview_images(source_root, fps, out_w, out_h, count, anchor_config=None, cam_tilt=None):
    episodes = _episode_rows(source_root)
    camera_cal, camera_poses, _ = resolve_camera_poses(
        source_root, episodes, fps, out_w, out_h, anchor_config, cam_tilt)
    picks = np.linspace(0, len(episodes) - 1, min(count, len(episodes))).astype(int)
    for pick in picks:
        episode = episodes[int(pick)]
        recorded, frames = _sample_frames(source_root, episode, fps, 1)[0]
        ortho = _render(list(zip(frames, camera_poses[episode["episode_index"]])),
                        camera_cal, out_w, out_h)
        yield f"episode-{episode['episode_index']:04d}.png", np.hstack([recorded, ortho])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_root", required=True,
                        help="Recorded dataset, the one that still has the anchor cameras")
    parser.add_argument("--dest_root", default=None,
                        help="Dataset whose ortho videos are replaced. Defaults to the source, "
                             "which rewrites it in place; --preview never writes either way")
    parser.add_argument("--anchor_config", default=None,
                        help="Config file of the robot that recorded it (conf_*.json)")
    parser.add_argument("--cam_tilt", nargs="+", type=float, default=None,
                        help="Pin each anchor's camera tilt in degrees instead of reading "
                             "it from the config or recovering it from the recording")
    parser.add_argument("--render_fps", type=float, default=RENDER_FPS,
                        help="Renders per second; frames in between repeat the last one. "
                             "0 renders every frame (default: %(default)s)")
    parser.add_argument("--headroom", type=int, default=0, help="CPU cores to leave free")
    parser.add_argument("--preview", default=None,
                        help="Write recorded-vs-reblended ortho pairs here and stop, "
                             "without touching the dataset")
    args = parser.parse_args()

    dest_root = Path(args.dest_root or args.source_root)
    if args.preview:
        write_preview(Path(args.source_root), dest_root, Path(args.preview),
                      anchor_config=args.anchor_config, cam_tilt=args.cam_tilt)
        return

    reblend_ortho(
        source_root=Path(args.source_root),
        dest_root=dest_root,
        anchor_config=args.anchor_config,
        cam_tilt=args.cam_tilt,
        render_fps=args.render_fps or None,
        headroom=args.headroom,
    )


if __name__ == "__main__":
    main()
