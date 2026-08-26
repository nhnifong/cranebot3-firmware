#!/usr/bin/env python

"""Hand-label ortho floor frames, starting from what the model already thinks.

Teleop gives one label a frame - the thing the operator actually grabbed - and says
nothing about the rest of the floor. This is where the other kind of label comes from:
frames where *every* target is marked, so the absence of one means the floor really is
empty there. ortho_target's objectness head has no other source for that.

    python ortho_labeler/server.py --root ~/data/combined_targets_reblend
    python ortho_labeler/server.py --repo_id naavox/combined_targets_reblend

Then open the printed URL. Frames are extracted from the LeRobot dataset once, into
ortho_labeler/frames/, so a restart is instant; --refresh re-extracts them.

The model seeds each frame with its own predictions, at a threshold well below the one
the robot acts on, because deleting a wrong dot is faster than placing a right one. A
seeded dot is amber and carries its probability; anything you place is green. Nothing is
saved until you press save, so a frame you never visit costs nothing.

Labels are written with ortho_target.write_user_labels, one parquet per frame, named for
the frame - so re-labelling one overwrites it rather than leaving both, and the result
merges with the ordinary command:

    python -m nf_robot.ml.ortho_target merge_labels
    python -m nf_robot.ml.ortho_target train --data_root ortho_target_data

No opencv GUI: this repo installs headless opencv, so the tool is a local HTTP server
driven from a browser tab, the same shape the target_heatmap labeler had.

A frame with nothing graspable in it is a real label and this cannot save one - saving
needs at least one target, so use skip. Wiring that up means letting write_user_labels
write an empty row, which is a change to what the UI's action means as well.
"""

import argparse
import json
import logging
import mimetypes
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch

from nf_robot.ml import ortho_target as ot

HERE = Path(__file__).parent
DEFAULT_REPO_ID = "naavox/combined_targets_reblend"
DEFAULT_CACHE = HERE / "frames"
# Below ortho_target.TARGET_THRESHOLD on purpose: the seeds are a starting point to prune,
# not the model's answer, and a missed object costs more here than a spurious dot.
SEED_THRESHOLD = 0.2
SEED_CANDIDATES = 32


def extract_frames(repo_id, root, cache: Path, per_episode, limit, min_coverage):
    """Ortho frames spread across the dataset's episodes, as JPEGs named for their source.

    Evenly spaced within each episode rather than random, so one episode cannot donate two
    frames of the same moment, and blank frames are dropped on the way out - an ortho map
    with almost no coverage is a picture of nothing and wastes a labelling slot.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cache.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset(repo_id=repo_id, root=Path(root) if root else None,
                             force_cache_sync=root is None)
    key = ot.ortho_key()
    if key not in dataset.meta.video_keys:
        raise ValueError(f"'{repo_id}' has no '{key}' feature, so it carries no ortho view. "
                         f"Present: {list(dataset.meta.video_keys)}")

    starts = ot.episode_starts(dataset)
    logging.info(f"Extracting from {dataset.root}: {len(starts)} episode(s)")
    written, blank = 0, 0
    for episode in sorted(starts):
        if limit and written >= limit:
            break
        meta = starts[episode]
        offsets = sorted({int(round(o)) for o in
                          np.linspace(0, max(meta["length"] - 1, 0), per_episode)})
        for offset in offsets:
            if limit and written >= limit:
                break
            bgr = ot.frame_to_bgr(dataset[meta["start"] + offset][key])
            if ot.coverage_fraction(bgr) < min_coverage:
                blank += 1
                continue
            cv2.imwrite(str(cache / f"{episode:06d}-{offset:06d}.jpg"), bgr)
            written += 1
        if written and written % 50 == 0:
            logging.info(f"  {written} frame(s)")
    logging.info(f"Extracted {written} frame(s) to {cache}" + (f", {blank} blank skipped" if blank else ""))
    return written


class Session:
    """Frames to label, what has been done to them, and the model that seeds them."""

    def __init__(self, cache: Path, output: Path, model_path, device, seed_threshold):
        self.cache = cache
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.frames = [p.stem for p in sorted(cache.glob("*.jpg"))]
        if not self.frames:
            raise FileNotFoundError(f"No frames in {cache}; run with --refresh to extract them")

        self.device = device
        self.model_path = model_path
        self.seed_threshold = seed_threshold
        self._model = None
        self._seeds: dict[str, list] = {}
        self.skipped: set[str] = set()
        # Inference and file writes both happen on handler threads.
        self.lock = threading.Lock()
        self.done = threading.Event()

        # Pace is timed from the first save, so time spent reading the page at the start
        # is not counted against the rate.
        self._first_save = None
        self._saved_since = 0

    def label_path(self, frame_id):
        return self.output / f"user-{frame_id}.parquet"

    def image_path(self, frame_id):
        return self.cache / f"{frame_id}.jpg"

    def status(self, frame_id):
        if self.label_path(frame_id).exists():
            return "saved"
        return "skipped" if frame_id in self.skipped else "todo"

    def model(self):
        """The checkpoint, loaded on first use, or None if there is none to seed from.

        A missing or unreadable checkpoint leaves the tool working with empty frames rather
        than failing: hand labelling is the point, and seeds are the convenience. That
        matters most for the checkpoint load_checkpoint refuses - one from before the
        objectness head - because retraining is the fix and there is no reason to be
        unable to label in the meantime.
        """
        if self._model is None:
            self._model = False
            path = Path(self.model_path)
            if not path.exists():
                logging.warning(f"No checkpoint at {path}; frames start with no targets")
            else:
                try:
                    self._model, _ = ot.load_checkpoint(str(path), self.device)
                    logging.info(f"Seeding predictions from {path}")
                except Exception as error:
                    logging.warning(f"Cannot seed from {path}, so frames start with no "
                                    f"targets: {error}")
        return self._model or None

    def targets(self, frame_id):
        """Saved points if this frame has been labelled, otherwise the model's guesses.

        Saved wins, so revisiting a frame shows what was actually written rather than
        silently offering the model's opinion of it again.
        """
        path = self.label_path(frame_id)
        if path.exists():
            import pyarrow.parquet as pq

            points = pq.read_table(path, columns=["points"]).column("points")[0].as_py()
            return [{"u": u, "v": v, "p": None} for u, v in points]

        if frame_id in self._seeds:
            return self._seeds[frame_id]

        model = self.model()
        if model is None:
            return []
        bgr = cv2.imread(str(self.image_path(frame_id)))
        height, width = bgr.shape[:2]
        with self.lock:
            predictions = ot.predict_room_targets(
                model, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.device,
                top_k=SEED_CANDIDATES, min_probability=self.seed_threshold)
        seeds = []
        for x_m, y_m, probability in predictions:
            u, v = ot.room_to_ortho_px(x_m, y_m, width, height)
            seeds.append({"u": round(float(u), 2), "v": round(float(v), 2),
                          "p": round(float(probability), 3)})
        self._seeds[frame_id] = seeds
        return seeds

    def save(self, frame_id, points):
        """Write one frame's targets, replacing whatever it was labelled with before."""
        bgr = cv2.imread(str(self.image_path(frame_id)))
        if bgr is None:
            raise FileNotFoundError(frame_id)
        height, width = bgr.shape[:2]
        # The page works in the frame's own pixels; write_user_labels wants room metres and
        # converts back with the same projection, so this round-trips exactly.
        targets_m = [(*ot.ortho_px_to_room(float(u), float(v), width, height), 0.0)
                     for u, v in points]
        with self.lock:
            path, count = ot.write_user_labels(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), targets_m,
                output_root=self.output, name=frame_id)
            self.skipped.discard(frame_id)
            now = time.time()
            if self._first_save is None:
                self._first_save = now
            self._saved_since += 1
        return path, count

    def progress(self):
        """Counts, and a finishing estimate once there is an interval to measure one from.

        Two saves make one interval, which is the first point a rate exists at all; from
        one save the elapsed time is however long the request took and the estimate would
        read as seconds.
        """
        done = {f for f in self.frames if self.label_path(f).exists()}
        remaining = len(self.frames) - len(done | self.skipped)
        eta = None
        if self._saved_since >= 2 and remaining > 0:
            per_frame = (time.time() - self._first_save) / (self._saved_since - 1)
            eta = remaining * per_frame
        return {
            "saved": len(done),
            "skipped": len(self.skipped - done),
            "total": len(self.frames),
            "remaining": remaining,
            "eta_seconds": eta,
        }


def _send(handler, payload, content_type, status=200):
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(payload)


def _send_json(handler, obj, status=200):
    _send(handler, json.dumps(obj).encode(), "application/json", status)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # the useful lines are logged by the session instead

    @property
    def session(self):
        return self.server.session

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            page = (HERE / "index.html").read_bytes()
            return _send(self, page, "text/html; charset=utf-8")
        if path == "/api/frames":
            return _send_json(self, {
                "frames": [{"id": f, "status": self.session.status(f)} for f in self.session.frames],
                "output": str(self.session.output),
                "progress": self.session.progress(),
            })
        if path.startswith("/api/frames/") and path.endswith("/image"):
            frame_id = path[len("/api/frames/"):-len("/image")]
            image = self.session.image_path(frame_id)
            if frame_id not in self.session.frames or not image.exists():
                return self.send_error(404)
            return _send(self, image.read_bytes(), "image/jpeg")
        if path.startswith("/api/frames/") and path.endswith("/targets"):
            frame_id = path[len("/api/frames/"):-len("/targets")]
            if frame_id not in self.session.frames:
                return self.send_error(404)
            return _send_json(self, {"targets": self.session.targets(frame_id),
                                     "status": self.session.status(frame_id)})
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or "{}")

        if path == "/api/quit":
            self.session.done.set()
            return _send_json(self, {"ok": True})

        if path.startswith("/api/frames/") and path.endswith("/save"):
            frame_id = path[len("/api/frames/"):-len("/save")]
            points = body.get("points") or []
            if frame_id not in self.session.frames:
                return self.send_error(404)
            if not points:
                return _send_json(self, {"error": "a saved frame needs at least one target; "
                                                  "skip it instead"}, status=400)
            _, count = self.session.save(frame_id, points)
            logging.info(f"{frame_id}: {count} target(s)")
            return _send_json(self, {"saved": count, "progress": self.session.progress()})

        if path.startswith("/api/frames/") and path.endswith("/skip"):
            frame_id = path[len("/api/frames/"):-len("/skip")]
            self.session.skipped.add(frame_id)
            return _send_json(self, {"progress": self.session.progress()})

        self.send_error(404)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", default=None,
                        help="local LeRobot dataset directory, for a dataset not on the hub yet")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="where extracted frames live")
    parser.add_argument("--refresh", action="store_true", help="re-extract frames before serving")
    parser.add_argument("--frames_per_episode", type=int, default=3)
    parser.add_argument("--limit", type=int, default=300, help="0 for every frame the spacing gives")
    parser.add_argument("--min_coverage", type=float, default=0.35,
                        help="fraction of the ortho map that must be observed floor")
    parser.add_argument("--model_path", default=ot.DEFAULT_MODEL_PATH)
    parser.add_argument("--seed_threshold", type=float, default=SEED_THRESHOLD)
    parser.add_argument("--output", default=ot.USER_LABEL_ROOT)
    parser.add_argument("--port", type=int, default=8022)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cache = Path(args.cache)
    if args.refresh or not any(cache.glob("*.jpg")):
        extract_frames(args.repo_id, args.root, cache, args.frames_per_episode,
                       args.limit, args.min_coverage)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    session = Session(cache, Path(args.output), args.model_path, device, args.seed_threshold)

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.session = session
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logging.info(f"{len(session.frames)} frame(s) to label; writing to {session.output}")
    logging.info(f"Open http://{args.bind}:{args.port}/  (ctrl-c, or Quit in the page, to stop)")
    try:
        session.done.wait()
    except KeyboardInterrupt:
        pass
    server.shutdown()
    server.server_close()
    progress = session.progress()
    logging.info(f"{progress['saved']} frame(s) labelled in {session.output}")
    logging.info("Merge them with: python -m nf_robot.ml.ortho_target merge_labels"
                 + (f" --source {session.output}" if str(session.output) != ot.USER_LABEL_ROOT else ""))


if __name__ == "__main__":
    main()
