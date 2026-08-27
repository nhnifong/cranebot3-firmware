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

Episodes are taken spread across the whole dataset rather than from the front of it. A
merged dataset is its sources concatenated, so a limit applied to the front stops inside
the first room or two and never reaches the rest. `--add N` extracts another N frames that
are neither cached nor already labelled, which is how a second pass gets new work.

The model seeds each frame with its own predictions, at a threshold well below the one
the robot acts on, because deleting a wrong dot is faster than placing a right one. A
seeded dot is amber and carries its probability; anything you place is green. Nothing is
saved until you press save, so a frame you never visit costs nothing.

After the first frame of a scene the model steps aside: each frame starts from the points
just saved on the previous one, jittered a few pixels, shown in blue. Frames arrive several
to an episode and several episodes to a recording session, and consecutive ones are mostly
the same floor with one object missing, so the labels are nearly right already and the work
is pruning rather than placing. Whether it is the same floor is decided by comparing the
frames, not by their numbering - a merged dataset puts different rooms in consecutive
episodes - so the model seeds each new scene afresh. --no_carry turns this off.

Frames already saved or skipped can be walked past with the "Skip finished" toggle, which
is what a second pass over a part-labelled set wants; the page opens on the first
unlabelled frame regardless.

Labels are written with ortho_target.write_user_labels, one parquet per frame, named for
the frame - so re-labelling one overwrites it rather than leaving both, and the result
merges with the ordinary command:

    python -m nf_robot.ml.ortho_target merge_labels
    python -m nf_robot.ml.ortho_target train --data_root ortho_target_data

No opencv GUI: this repo installs headless opencv, so the tool is a local HTTP server
driven from a browser tab, the same shape the target_heatmap labeler had.

A frame with nothing graspable in it is a label too, and the strongest negative the
dataset can hold: every cell of it is a confirmed no, where an ordinary frame only says
where one operator reached. Press E to save one. It is a separate key from save-and-next
on purpose - space on a frame nobody has looked at would otherwise write "this floor is
bare" about a floor that is not - and skip still means "I am not judging this one",
which is a different statement and stores nothing.
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

# Pixels of gaussian noise on carried-over points. Enough that they read as a suggestion
# to check rather than an answer, and that a target held across a dozen frames does not
# train on one repeated coordinate; small enough to still land on the object.
CARRY_JITTER_PX = 2.0
# How alike two frames must look, as a correlation of 48px thumbnails, for one's labels to
# be worth offering for the other. Consecutive episodes of a session differ by an object
# or two and score above 0.7 even across an episode boundary; the merge seam between two
# recordings in different rooms scores near zero, so anything in the middle separates them.
CARRY_SIMILARITY = 0.5
CARRY_THUMB_PX = 48


def spread_order(count):
    """Every index of a run, ordered so that any prefix of it is spread over the whole run.

    A limit has to cut the episode list somewhere, and cutting it at the front is what
    kept the first labelling pass inside the first two rooms of a merged dataset: sources
    are concatenated, so the front of the episode range is the front of the merge. Taking
    them in this order instead means 300 frames out of 500 episodes covers every room, and
    so does the first 50 of them if blank frames or a smaller limit stop it early.

    The golden ratio is what makes any prefix work rather than just the whole: successive
    multiples of an irrational, wrapped into the unit interval, never fall into a repeating
    pattern, so each new index lands in the largest gap the previous ones left.
    """
    golden = (5 ** 0.5 - 1) / 2
    return np.argsort((np.arange(count) * golden) % 1.0)


def extract_frames(repo_id, root, cache: Path, per_episode, limit, min_coverage, skip=()):
    """Ortho frames spread across the dataset's episodes, as JPEGs named for their source.

    Evenly spaced within each episode rather than random, so one episode cannot donate two
    frames of the same moment, and blank frames are dropped on the way out - an ortho map
    with almost no coverage is a picture of nothing and wastes a labelling slot.

    Whole episodes are taken at a time, so the frames of one arrive together and the
    labeller's carry-over has a run of the same scene to work along. Which episodes is
    decided by spread_order, and `skip` names frames to leave out - already labelled, or
    already in the cache - so a second pass lands on new work rather than the same frames.
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
    episodes = sorted(starts)
    skip = set(skip)
    logging.info(f"Extracting from {dataset.root}: {len(episodes)} episode(s)"
                 + (f", leaving out {len(skip)} frame(s) already in hand" if skip else ""))

    written, blank, seen, first, last = 0, 0, 0, None, None
    for index in spread_order(len(episodes)):
        if limit and written >= limit:
            break
        episode = episodes[int(index)]
        meta = starts[episode]
        offsets = sorted({int(round(o)) for o in
                          np.linspace(0, max(meta["length"] - 1, 0), per_episode)})
        for offset in offsets:
            if limit and written >= limit:
                break
            frame_id = f"{episode:06d}-{offset:06d}"
            if frame_id in skip:
                seen += 1
                continue
            bgr = ot.frame_to_bgr(dataset[meta["start"] + offset][key])
            if ot.coverage_fraction(bgr) < min_coverage:
                blank += 1
                continue
            cv2.imwrite(str(cache / f"{frame_id}.jpg"), bgr)
            written += 1
            first = episode if first is None else min(first, episode)
            last = episode if last is None else max(last, episode)
        if written and written % 50 == 0:
            logging.info(f"  {written} frame(s)")

    notes = [f"{blank} blank"] if blank else []
    if seen:
        notes.append(f"{seen} already in hand")
    logging.info(f"Extracted {written} frame(s) to {cache}"
                 + (f", spanning episodes {first}-{last}" if written else "")
                 + (f" ({', '.join(notes)} skipped)" if notes else ""))
    if limit and written < limit:
        logging.info(f"That is short of the {limit} asked for; the dataset has nothing else "
                     f"left to sample at these settings")
    return written


class Session:
    """Frames to label, what has been done to them, and the model that seeds them."""

    def __init__(self, cache: Path, output: Path, model_path, device, seed_threshold,
                 carry=True, carry_jitter=CARRY_JITTER_PX, carry_similarity=CARRY_SIMILARITY):
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
        # frame -> (what the seeds came from, the seeds). The origin is kept so that a
        # frame seeded before its neighbour was labelled is re-seeded from the label once
        # there is one, rather than serving a stale guess for the rest of the run.
        self._seeds: dict[str, tuple[str | None, list]] = {}
        self.skipped: set[str] = set()

        self.carry = carry
        self.carry_jitter = carry_jitter
        self.carry_similarity = carry_similarity
        self._last_saved: tuple[str, list] | None = None
        self._thumbs: dict[str, np.ndarray] = {}
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

    def _thumb(self, frame_id):
        """A small grey copy of one frame, for asking whether two frames are the same scene."""
        if frame_id not in self._thumbs:
            grey = cv2.imread(str(self.image_path(frame_id)), cv2.IMREAD_GRAYSCALE)
            small = cv2.resize(grey, (CARRY_THUMB_PX, CARRY_THUMB_PX),
                               interpolation=cv2.INTER_AREA).astype(np.float32)
            self._thumbs[frame_id] = small - small.mean()
        return self._thumbs[frame_id]

    def same_scene(self, frame_id, other_id):
        """Whether two frames look like the same floor, as a correlation of their thumbnails.

        Frame ids run episode-major, but a merged dataset puts different rooms in
        consecutive episodes, so adjacency in the list says nothing on its own. What the
        frames look like does: within a recording session the operator removes one object
        at a time and almost everything else holds still, while the seam between two
        recordings changes the room, the lighting and the shape of the covered floor at once.
        """
        a, b = self._thumb(frame_id), self._thumb(other_id)
        scale = float(np.linalg.norm(a) * np.linalg.norm(b))
        return (float((a * b).sum() / scale) if scale else 0.0) >= self.carry_similarity

    def carry_origin(self, frame_id):
        """The frame whose labels should seed this one, or None to fall back to the model."""
        if not self.carry or self._last_saved is None:
            return None
        last_id, _ = self._last_saved
        if last_id == frame_id or not self.same_scene(frame_id, last_id):
            return None
        return last_id

    def _carried(self, frame_id):
        """The last saved frame's points, jittered, for a frame of the same scene.

        Jittered because they are a guess about a different moment: an operator holding
        space through a session would otherwise be writing one coordinate over and over,
        and a dot that has visibly moved asks to be looked at in a way an exact copy does
        not.
        """
        _, points = self._last_saved
        bgr = cv2.imread(str(self.image_path(frame_id)))
        height, width = bgr.shape[:2]
        noise = np.random.normal(0.0, self.carry_jitter, size=(len(points), 2))
        carried = []
        for (u, v), (du, dv) in zip(points, noise):
            carried.append({"u": round(float(np.clip(u + du, 0, width - 1)), 2),
                            "v": round(float(np.clip(v + dv, 0, height - 1)), 2),
                            "p": None, "carried": True})
        return carried

    def _model_seeds(self, frame_id):
        """What the model thinks is on this floor, as points to prune."""
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
                          "p": round(float(probability), 3), "carried": False})
        return seeds

    def targets(self, frame_id):
        """What to show on a frame: its own labels, the last frame's, or the model's.

        Saved wins, so revisiting a frame shows what was actually written rather than
        silently offering a guess about it again. Otherwise the previous frame's labels
        carry over while the scene holds - most frames in a session are the one before it
        with an object missing, so re-placing every dot each time is work the operator has
        already done - and the model seeds the first frame of each new scene.
        """
        path = self.label_path(frame_id)
        if path.exists():
            import pyarrow.parquet as pq

            points = pq.read_table(path, columns=["points"]).column("points")[0].as_py()
            return [{"u": u, "v": v, "p": None, "carried": False} for u, v in points]

        origin = self.carry_origin(frame_id)
        cached = self._seeds.get(frame_id)
        if cached is not None and cached[0] == origin:
            return cached[1]
        seeds = self._carried(frame_id) if origin else self._model_seeds(frame_id)
        self._seeds[frame_id] = (origin, seeds)
        return seeds

    def seeded_from(self, frame_id):
        """Where the points a frame is showing came from, for the page to say so."""
        if self.label_path(frame_id).exists():
            return "saved"
        return "carried" if self.carry_origin(frame_id) else "model"

    def save(self, frame_id, points, allow_empty=False):
        """Write one frame's targets, replacing whatever it was labelled with before.

        allow_empty carries through to write_user_labels, where an empty list means the
        floor really is bare rather than that nothing was placed yet.
        """
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
                output_root=self.output, name=frame_id, allow_empty=allow_empty)
            self.skipped.discard(frame_id)
            self._last_saved = (frame_id, [(float(u), float(v)) for u, v in points])
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
                                     "status": self.session.status(frame_id),
                                     "seeded": self.session.seeded_from(frame_id)})
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
            empty = bool(body.get("empty"))
            if frame_id not in self.session.frames:
                return self.send_error(404)
            if not points and not empty:
                return _send_json(self, {"error": "nothing placed on this frame; save it as "
                                                  "empty floor with E, or skip it with N"},
                                  status=400)
            _, count = self.session.save(frame_id, points, allow_empty=empty)
            logging.info(f"{frame_id}: " + (f"{count} target(s)" if count else "empty floor"))
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
    parser.add_argument("--add", type=int, default=0, metavar="N",
                        help="extract N more frames that are not already cached or labelled, "
                             "keeping the ones already there, and serve the lot")
    parser.add_argument("--frames_per_episode", type=int, default=3)
    parser.add_argument("--limit", type=int, default=300, help="0 for every frame the spacing gives")
    parser.add_argument("--min_coverage", type=float, default=0.35,
                        help="fraction of the ortho map that must be observed floor")
    parser.add_argument("--model_path", default=ot.DEFAULT_MODEL_PATH)
    parser.add_argument("--seed_threshold", type=float, default=SEED_THRESHOLD)
    parser.add_argument("--no_carry", action="store_true",
                        help="seed every frame from the model, instead of carrying the last "
                             "saved frame's points into the next one of the same scene")
    parser.add_argument("--carry_jitter_px", type=float, default=CARRY_JITTER_PX,
                        help="gaussian noise on carried points (default: %(default)s)")
    parser.add_argument("--carry_similarity", type=float, default=CARRY_SIMILARITY,
                        help="how alike two frames must look, 0 to 1, to carry labels "
                             "between them (default: %(default)s)")
    parser.add_argument("--output", default=ot.USER_LABEL_ROOT)
    parser.add_argument("--port", type=int, default=8022)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cache = Path(args.cache)
    output = Path(args.output)
    if args.refresh or args.add or not any(cache.glob("*.jpg")):
        # Frames already labelled are never worth extracting again, and with --add the ones
        # already sitting in the cache are not either: N means N to work on, not N drawn.
        skip = {p.stem[len("user-"):] for p in output.glob("user-*.parquet")}
        if args.add:
            skip |= {p.stem for p in cache.glob("*.jpg")}
        extract_frames(args.repo_id, args.root, cache, args.frames_per_episode,
                       args.add or args.limit, args.min_coverage, skip=skip)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    session = Session(cache, output, args.model_path, device, args.seed_threshold,
                      carry=not args.no_carry, carry_jitter=args.carry_jitter_px,
                      carry_similarity=args.carry_similarity)

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
