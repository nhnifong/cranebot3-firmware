#!/usr/bin/env python

"""What an ortho target dataset holds, and the rows in it most likely to be wrong.

Usage:
    python -m nf_robot.ml.ortho_target.audit
    python -m nf_robot.ml.ortho_target.audit --data_root ortho_target_data --split all
    python -m nf_robot.ml.ortho_target.audit --preview_dir audit_previews
"""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from nf_robot.ml.ortho_target.dataset import (
    DEFAULT_DATASET_ID,
    LABEL_COLUMNS,
    MAX_TARGETS,
    POOL_SPLIT,
    SAME_PICTURE_PEAK,
    SIGNATURE_FINE,
    is_complete,
    picture_signature,
    same_picture_pairs,
    sample_group,
)
from nf_robot.ml.ortho_target.model import ORTHO_EXTENT_M, room_to_ortho_px

# A stored point further than this from its own contacts_m, reprojected, is not the same label.
REPROJECT_TOL_PX = 1.5
# Contacts this close to the room origin look like a zeroed contact rather than a grasp.
ORIGIN_RADIUS_M = 0.03
# Gripper heights a floor grasp can plausibly happen at. Outside it the position estimate
# the label's x, y also came from is suspect.
PLAUSIBLE_Z_M = (-0.10, 0.30)
# Frames closer to the grasp than this mostly show the gripper sitting on the target.
SHORT_LEAD_S = 0.5
# A grasp this soon after the episode starts is more likely pressure noise than a reach.
EARLY_CONTACT_S = 1.0
# Two targets in one complete frame this close together are one object clicked twice.
DOUBLE_CLICK_M = 0.04
# Distinct episodes whose contacts share a spot this small are worth a look together.
CLUSTER_M = 0.02
CLUSTER_MIN_EPISODES = 4
# Frames painted less than this are mostly black, and say little either way.
LOW_COVERAGE = 0.25

# Salience: grey-level std in a box around the label, against random painted boxes of
# the same frame. An object has texture; bare floor has little.
SALIENCE_RADIUS_M = 0.06
SALIENCE_SAMPLES = 48
# Label boxes flatter than this share of the frame's random boxes are flagged.
FLAT_LABEL_RANK = 0.2
# Search range and step for the displacement that makes labels most salient on average.
OFFSET_RANGE_M = 0.20
OFFSET_STEP_M = 0.02

# Two labels of the same picture this close mark the same object.
AGREE_M = 0.10

PREVIEW_TILE = 320
PREVIEW_PER_REASON = 48

REASONS = {
    "reproject_mismatch": "stored point disagrees with its contacts_m",
    "off_map": "label outside the frame",
    "points_contacts_mismatch": "points and contacts_m differ in length",
    "partial_not_one": "teleop row without exactly one label",
    "truncated": f"more than {MAX_TARGETS} targets; training drops the rest",
    "episode_label_mismatch": "rows of one episode carry different labels",
    "near_origin": "contact at the room origin",
    "implausible_height": f"contact z outside {PLAUSIBLE_Z_M} m",
    "short_lead": f"frame under {SHORT_LEAD_S}s before contact",
    "early_contact": f"contact within {EARLY_CONTACT_S}s of episode start",
    "double_click": "two targets in one frame closer than an object",
    "unpainted_label": "label on a pixel no camera painted",
    "low_coverage": f"under {LOW_COVERAGE:.0%} of the frame painted",
    "flat_label": "label on featureless floor",
    "cross_duplicate": "same picture as a row from another episode or run; merge_labels drops teleop ones",
    "label_disagreement": "same picture hand-labelled twice, differently",
    "grasp_not_labelled": "teleop grasp (green) far from the same picture's hand labels (magenta)",
    "duplicate_file_name": "file_name shared with another row",
}


class Findings:
    """Warnings collected while reporting, printed together at the end."""

    def __init__(self):
        self.items = []

    def warn(self, message):
        self.items.append(("WARN", message))

    def fail(self, message):
        self.items.append(("FAIL", message))

    def report(self):
        if not self.items:
            print("\nNo findings.")
            return 0
        print(f"\n{'=' * 78}\nFINDINGS\n{'=' * 78}")
        for level, message in self.items:
            print(f"  [{level}] {message}")
        return sum(1 for level, _ in self.items if level == "FAIL")


def row_kind(row):
    """Which producer wrote a row, from the name each one gives its frames."""
    name = row["file_name"]
    if name.startswith("neg-"):
        return "negative"
    if name.startswith("user-"):
        # The labeler names frames [<source>-]<episode>-<offset>; the UI names them
        # <date>-<time>-<nonce>, and a date is the one leading field eight digits long.
        parts = name[len("user-"):].rsplit(".", 1)[0].split("-")
        ui = len(parts) == 3 and len(parts[0]) == 8 and parts[0].isdigit()
        return "ui" if ui else "labeler"
    return "teleop"


def read_split(split_dir: Path, with_images: bool):
    shards = sorted(split_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {split_dir}")
    rows = []
    columns = list(LABEL_COLUMNS) + (["image"] if with_images else [])
    for shard in shards:
        present = [c for c in columns if c in pq.ParquetFile(shard).schema_arrow.names]
        rows.extend(pq.read_table(shard, columns=present).to_pylist())
    return rows, shards


def histogram_lines(values, bins, low, high, unit="", width=40):
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    peak = counts.max() or 1
    return "\n".join(
        f"    {left:7.2f}..{right:7.2f}{unit} {count:7d} {'#' * int(round(width * count / peak))}"
        for count, left, right in zip(counts, edges[:-1], edges[1:]))


def percentiles(values, qs=(1, 25, 50, 75, 99)):
    return " ".join(f"{v:.3f}" for v in np.percentile(values, qs))


class Row:
    """One dataset row with what the audit learns about it."""

    def __init__(self, split, raw):
        self.split = split
        self.raw = raw
        self.name = raw["file_name"]
        self.kind = row_kind(raw)
        self.complete = is_complete(raw)
        self.group = sample_group(raw)
        self.points = np.asarray(raw["points"] or [], dtype=float).reshape(-1, 2)
        self.contacts = np.asarray(raw["contacts_m"] or [], dtype=float).reshape(-1, 3)
        self.reasons = set()
        self.notes = []
        self.size = None
        self.coverage = None
        self.label_rank = None
        self.signature = None
        # The other answer to the same picture, drawn beside this row's own labels.
        self.partner = None

    def flag(self, reason, note=None):
        self.reasons.add(reason)
        if note:
            self.notes.append(note)

    def lead_s(self):
        """Seconds between this frame and the contact, from the row's own frame rate."""
        raw = self.raw
        if self.complete or not raw["contact_time_s"] or raw["contact_frame_index"] <= 0:
            return None
        fps = raw["contact_frame_index"] / raw["contact_time_s"]
        return (raw["contact_frame_index"] - raw["frame_offset"]) / fps


# ==========================================
# LABELS
# ==========================================

def audit_composition(rows, findings, split):
    print("\n-- producers")
    kinds = Counter(r.kind for r in rows)
    for kind, count in kinds.most_common():
        sub = [r for r in rows if r.kind == kind]
        targets = sum(len(r.points) for r in sub)
        groups = len({r.group for r in sub})
        print(f"   {kind:9s} {count:6d} frames {targets:6d} targets  from {groups:5d} "
              f"episodes/runs ({count / max(groups, 1):.1f} frames each)")

    complete = [r for r in rows if r.complete]
    if not complete:
        findings.fail(f"{split}: no complete frames, so every cell trains on nothing negative "
                      f"and ap@20cm cannot be scored")
    else:
        empty = sum(1 for r in complete if not len(r.points))
        per_frame = Counter(len(r.points) for r in complete)
        print(f"\n-- complete frames: {len(complete)}, {empty} with no targets")
        print("   targets per frame: " + "  ".join(f"{n}:{c}" for n, c in sorted(per_frame.items())))
        labelled = [r for r in complete if r.kind in ("labeler", "ui")]
        if labelled and all(not len(r.points) for r in labelled):
            findings.warn(f"{split}: every hand-labelled frame is empty")

    tasks = Counter(r.raw["task"] for r in rows if r.kind == "teleop")
    if tasks:
        print(f"\n-- teleop tasks: {len(tasks)} distinct")
        for task, count in tasks.most_common(8):
            print(f"   {count:6d}  {task[:66]}")

    names = Counter(r.name for r in rows)
    for r in rows:
        if names[r.name] > 1:
            r.flag("duplicate_file_name")
    shared = sum(1 for n, c in names.items() if c > 1)
    if shared:
        findings.warn(
            f"{split}: {shared} file_name(s) appear on more than one row. Labeler names from "
            f"before the source dataset was part of them say only episode and offset, so two "
            f"datasets that share episode numbers collide - which also makes sample_group "
            f"treat unrelated rooms as one run. Two people labelling one frame collide too.")


def audit_integrity(rows, findings, split):
    for r in rows:
        if len(r.points) != len(r.contacts):
            r.flag("points_contacts_mismatch", f"{len(r.points)} points, {len(r.contacts)} contacts")
        if not r.complete and len(r.points) != 1:
            r.flag("partial_not_one", f"{len(r.points)} labels")
        if len(r.points) > MAX_TARGETS:
            r.flag("truncated", f"{len(r.points)} targets")

    # Every frame of one episode shares the episode's one contact.
    by_episode = defaultdict(list)
    for r in rows:
        if not r.complete:
            by_episode[r.raw["episode_index"]].append(r)
    for ep_rows in by_episode.values():
        if len({tuple(map(tuple, r.contacts.round(3))) for r in ep_rows}) > 1:
            for r in ep_rows:
                r.flag("episode_label_mismatch")

    for reason, level in (("points_contacts_mismatch", "fail"), ("partial_not_one", "fail"),
                          ("episode_label_mismatch", "fail"), ("truncated", "warn")):
        n = sum(1 for r in rows if reason in r.reasons)
        if n:
            getattr(findings, level)(f"{split}: {n} row(s) {REASONS[reason]}")


def audit_contacts(rows, findings, split):
    teleop = [r for r in rows if r.kind == "teleop" and len(r.contacts)]
    if not teleop:
        return
    # One contact per episode, so episodes are not weighted by how many frames they gave.
    episodes = {}
    for r in teleop:
        episodes.setdefault(r.raw["episode_index"], r.contacts[0])
    contacts = np.array(list(episodes.values()))
    print(f"\n-- teleop contacts: {len(contacts)} episodes")
    print(f"   x m percentiles (1/25/50/75/99): {percentiles(contacts[:, 0])}")
    print(f"   y m percentiles (1/25/50/75/99): {percentiles(contacts[:, 1])}")
    print(f"   z m percentiles (1/25/50/75/99): {percentiles(contacts[:, 2])}")
    task_of = {r.raw["episode_index"]: r.raw["task"] for r in teleop}
    by_task = defaultdict(list)
    for ep, c in episodes.items():
        by_task[task_of[ep]].append(c[2])
    print("   z m by task (5/50/95):")
    for task, zs in sorted(by_task.items(), key=lambda kv: -len(kv[1]))[:8]:
        print(f"     {len(zs):5d} ep  {percentiles(zs, (5, 50, 95))}  {task[:40]}")

    for r in teleop:
        x, y, z = r.contacts[0]
        if math.hypot(x, y) < ORIGIN_RADIUS_M:
            r.flag("near_origin", f"contact ({x:+.3f}, {y:+.3f})")
        if not PLAUSIBLE_Z_M[0] <= z <= PLAUSIBLE_Z_M[1]:
            r.flag("implausible_height", f"contact z {z:+.2f}m")
        lead = r.lead_s()
        if lead is not None and lead < SHORT_LEAD_S:
            r.flag("short_lead", f"{lead:.2f}s before contact")
        if r.raw["contact_time_s"] < EARLY_CONTACT_S:
            r.flag("early_contact", f"contact at {r.raw['contact_time_s']:.2f}s")

    leads = [lead for lead in (r.lead_s() for r in teleop) if lead is not None]
    if leads:
        print("   seconds from frame to contact:")
        print(histogram_lines(np.clip(leads, 0, 20), 10, 0, 20, "s"))

    def episode_count(reason):
        return len({r.raw["episode_index"] for r in teleop if reason in r.reasons})

    if n := episode_count("near_origin"):
        findings.fail(f"{split}: {n} episode(s) contact within {ORIGIN_RADIUS_M * 100:.0f}cm of "
                      f"the room origin - the signature of a zeroed contact, not a grasp")
    if n := episode_count("implausible_height"):
        findings.warn(f"{split}: {n} of {len(episodes)} episode(s) make contact at a gripper "
                      f"height outside {PLAUSIBLE_Z_M} m. Nothing is grasped below the floor, "
                      f"so the position estimate was wrong there, and the label's x, y came "
                      f"from the same estimate; the height check below says whether it shows")
    if n := episode_count("early_contact"):
        findings.warn(f"{split}: {n} episode(s) reach pressure within {EARLY_CONTACT_S}s of "
                      f"starting - check they are grasps and not a finger already loaded")
    short = sum(1 for r in teleop if "short_lead" in r.reasons)
    if short > 0.1 * len(teleop):
        findings.warn(f"{split}: {short} of {len(teleop)} teleop frames are under "
                      f"{SHORT_LEAD_S}s from contact, where the gripper covers the target and "
                      f"the model can learn to find the gripper instead")

    # Distinct episodes grasping the same spot: a staged object, or a stuck state reading.
    cells = defaultdict(list)
    for ep, c in episodes.items():
        cells[(round(c[0] / CLUSTER_M), round(c[1] / CLUSTER_M))].append(ep)
    clusters = sorted(((len(eps), key) for key, eps in cells.items()
                       if len(eps) >= CLUSTER_MIN_EPISODES), reverse=True)
    if clusters:
        print(f"   spots grasped by {CLUSTER_MIN_EPISODES}+ episodes "
              f"(within {CLUSTER_M * 100:.0f}cm):")
        for n, (cx, cy) in clusters[:8]:
            print(f"     ({cx * CLUSTER_M:+.2f}, {cy * CLUSTER_M:+.2f}) m  {n} episodes: "
                  f"{sorted(cells[(cx, cy)])[:10]}")
        repeated = sum(n for n, _ in clusters)
        if repeated > 0.1 * len(episodes):
            findings.warn(f"{split}: {repeated} of {len(episodes)} episodes contact one of "
                          f"{len(clusters)} repeated spots; staged objects, or a position "
                          f"reading that stopped updating")


def audit_placement(rows, findings, split):
    """Where labels sit on the map, as a coarse text heatmap in room metres."""
    labelled = [r for r in rows if len(r.contacts)]
    if not labelled:
        return
    bins = 10
    half = ORTHO_EXTENT_M / 2
    grid = np.zeros((bins, bins), dtype=int)
    for r in labelled:
        for x, y, _ in r.contacts:
            i = int(np.clip((half - y) / ORTHO_EXTENT_M * bins, 0, bins - 1))
            j = int(np.clip((x + half) / ORTHO_EXTENT_M * bins, 0, bins - 1))
            grid[i, j] += 1
    shades = " .:-=+*#%@"
    peak = grid.max()
    print(f"\n-- targets on the map ({ORTHO_EXTENT_M / bins:.1f}m cells, top is +y), peak {peak}")
    for line in grid:
        print("    |" + "".join(shades[int(round((len(shades) - 1) * c / peak))] * 2
                             for c in line) + "|")
    for kind in sorted({r.kind for r in labelled}):
        c = np.concatenate([r.contacts for r in labelled if r.kind == kind])
        print(f"   {kind:9s} mean ({c[:, 0].mean():+.2f}, {c[:, 1].mean():+.2f}) m  "
              f"std ({c[:, 0].std():.2f}, {c[:, 1].std():.2f}) m")

    for r in labelled:
        if r.complete and len(r.contacts) > 1:
            d = np.linalg.norm(r.contacts[:, None, :2] - r.contacts[None, :, :2], axis=-1)
            d[np.diag_indices(len(d))] = np.inf
            if d.min() < DOUBLE_CLICK_M:
                r.flag("double_click", f"{d.min() * 100:.1f}cm apart")
    if n := sum(1 for r in rows if "double_click" in r.reasons):
        findings.warn(f"{split}: {n} complete frame(s) have two targets under "
                      f"{DOUBLE_CLICK_M * 100:.0f}cm apart; one object counted twice teaches "
                      f"the head that a single object is two")


# ==========================================
# IMAGES
# ==========================================

def box_std(gray_int, gray_sq_int, u, v, r):
    """Grey std over the (2r+1)^2 box centred on each (u, v), from integral images."""
    h, w = gray_int.shape[0] - 1, gray_int.shape[1] - 1
    x0 = np.clip(u - r, 0, w)
    x1 = np.clip(u + r + 1, 0, w)
    y0 = np.clip(v - r, 0, h)
    y1 = np.clip(v + r + 1, 0, h)
    area = np.maximum((x1 - x0) * (y1 - y0), 1)

    def total(img):
        return img[y1, x1] - img[y0, x1] - img[y1, x0] + img[y0, x0]

    mean = total(gray_int) / area
    return np.sqrt(np.maximum(total(gray_sq_int) / area - mean ** 2, 0.0))


def inspect_image(row, offsets_m, rng):
    """Decode one frame and measure coverage, reprojection and label salience."""
    bgr = cv2.imdecode(np.frombuffer(row.raw["image"], np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        row.flag("undecodable")
        return None
    h, w = bgr.shape[:2]
    row.size = (w, h)
    painted = bgr.any(axis=2)
    row.coverage = float(painted.mean())
    if row.coverage < LOW_COVERAGE:
        row.flag("low_coverage", f"{row.coverage:.0%} painted")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    row.signature = picture_signature(gray)

    for (u, v), (x, y, _) in zip(row.points, row.contacts):
        pu, pv = room_to_ortho_px(x, y, w, h)
        if math.hypot(pu - u, pv - v) > REPROJECT_TOL_PX:
            row.flag("reproject_mismatch", f"point ({u:.0f},{v:.0f}) vs contact ({pu:.0f},{pv:.0f})")
    for u, v in row.points:
        if not (0 <= u < w and 0 <= v < h):
            row.flag("off_map", f"({u:.0f},{v:.0f}) on {w}x{h}")
        else:
            r = 2
            patch = painted[max(0, int(v) - r):int(v) + r + 1, max(0, int(u) - r):int(u) + r + 1]
            if not patch.any():
                row.flag("unpainted_label", f"({u:.0f},{v:.0f})")

    if not len(row.points):
        return None
    radius = max(2, int(round(SALIENCE_RADIUS_M * w / ORTHO_EXTENT_M)))
    gray_f = gray.astype(np.float64)
    gray_int = cv2.integral(gray_f)
    gray_sq_int = cv2.integral(gray_f ** 2)

    # The frame's typical box, measured on painted pixels only.
    ys, xs = np.nonzero(painted)
    if not len(xs):
        return None
    pick = rng.integers(0, len(xs), SALIENCE_SAMPLES)
    background = box_std(gray_int, gray_sq_int, xs[pick], ys[pick], radius)
    scale = float(np.median(background)) or 1.0

    u = np.round(row.points[:, 0]).astype(int)
    v = np.round(row.points[:, 1]).astype(int)
    offsets_px = np.round(offsets_m * w / ORTHO_EXTENT_M).astype(int)
    at_label = box_std(gray_int, gray_sq_int, u, v, radius)
    # Rank of the flattest label against the frame's random boxes.
    row.label_rank = float(np.mean(background < at_label.min()))
    if row.label_rank < FLAT_LABEL_RANK:
        row.flag("flat_label", f"flatter than {1 - row.label_rank:.0%} of the frame")

    # Mean salience of every label displaced by each offset, relative to the frame.
    du, dv = offsets_px[:, 0], offsets_px[:, 1]
    shifted = box_std(gray_int, gray_sq_int,
                      u[:, None] + du[None, :], v[:, None] + dv[None, :], radius)
    return shifted.mean(axis=0) / scale


def audit_images(rows, findings, split, seed):
    rng = np.random.default_rng(seed)
    steps = np.arange(-OFFSET_RANGE_M, OFFSET_RANGE_M + 1e-9, OFFSET_STEP_M)
    offsets_m = np.array([(dx, dy) for dy in steps for dx in steps])
    salience = defaultdict(list)
    sizes = Counter()
    for r in rows:
        if "image" not in r.raw:
            continue
        result = inspect_image(r, offsets_m, rng)
        if r.size:
            sizes[r.size] += 1
        if result is not None:
            salience[r.kind].append(result)
        # The image is the bulk of the memory, and nothing later needs all of them at once.
        r.raw["image"] = None

    print("\n-- frame sizes: " + ", ".join(f"{w}x{h}: {n}" for (w, h), n in sizes.most_common()))
    if len(sizes) > 1:
        findings.warn(f"{split}: frames are stored at {len(sizes)} different sizes. Training "
                      f"resizes them, but a merge that did not match sizes is worth knowing about")

    coverage = np.array([r.coverage for r in rows if r.coverage is not None])
    if len(coverage):
        print(f"\n-- painted fraction percentiles (1/25/50/75/99): {percentiles(coverage)}")
        print(histogram_lines(coverage, 10, 0, 1))

    for reason, level in (("reproject_mismatch", "fail"), ("off_map", "fail"),
                          ("unpainted_label", "warn"), ("undecodable", "fail")):
        n = sum(1 for r in rows if reason in r.reasons)
        if n:
            getattr(findings, level)(f"{split}: {n} row(s) {REASONS.get(reason, reason)}")
    if n := sum(1 for r in rows if "low_coverage" in r.reasons):
        findings.warn(f"{split}: {n} frame(s) under {LOW_COVERAGE:.0%} painted")

    audit_height_vs_salience(rows, findings, split)

    print(f"\n-- label salience: grey texture within {SALIENCE_RADIUS_M * 100:.0f}cm of a label, "
          f"as a multiple of the same frame's typical patch.")
    print("   An object is busier than floor, so ~1 means labels land on nothing in particular.")
    print("   A peak away from (0,0) means labels are systematically displaced from what they mark.")
    ranks = defaultdict(list)
    for r in rows:
        if r.label_rank is not None:
            ranks[r.kind].append(r.label_rank)
    centre = len(offsets_m) // 2
    for kind, maps in sorted(salience.items()):
        mean = np.mean(maps, axis=0)
        best = int(np.argmax(mean))
        flat = sum(1 for x in ranks[kind] if x < FLAT_LABEL_RANK)
        print(f"   {kind:9s} at label {mean[centre]:.2f}x, best {mean[best]:.2f}x at offset "
              f"({offsets_m[best][0] * 100:+.0f}, {offsets_m[best][1] * 100:+.0f}) cm in (+u, +v); "
              f"{flat}/{len(ranks[kind])} labels flatter than {1 - FLAT_LABEL_RANK:.0%} of their frame")
        if len(maps) < 20:
            continue
        if mean[centre] < 1.15:
            findings.warn(f"{split}: {kind} labels are barely busier than the floor around them "
                          f"({mean[centre]:.2f}x); look at the flat_label preview to see whether "
                          f"they mark objects at all")
        if np.linalg.norm(offsets_m[best]) >= 0.06 and mean[best] > 1.1 * mean[centre]:
            findings.warn(
                f"{split}: {kind} labels are most salient {np.linalg.norm(offsets_m[best]) * 100:.0f}cm "
                f"away from where they sit ({offsets_m[best][0] * 100:+.0f}, "
                f"{offsets_m[best][1] * 100:+.0f} cm in u, v). That is either a systematic "
                f"offset in the labels or the gripper beside the target in teleop frames; "
                f"--preview_dir shows which.")


def unmatched(points_a, points_b, tol_m):
    """How many of points_a have no point of points_b within tol_m (room metres, xy)."""
    if not len(points_a):
        return 0
    if not len(points_b):
        return len(points_a)
    d = np.linalg.norm(points_a[:, None, :2] - points_b[None, :, :2], axis=-1)
    return int((d.min(axis=1) > tol_m).sum())


def audit_height_vs_salience(rows, findings, split):
    """Whether implausible contact heights also put the label off the object."""
    teleop = [r for r in rows if r.kind == "teleop" and r.label_rank is not None]
    odd = [r for r in teleop if "implausible_height" in r.reasons]
    fine = [r for r in teleop if "implausible_height" not in r.reasons]
    if len(odd) < 20 or len(fine) < 20:
        return

    def flat_rate(sub):
        return np.mean([r.label_rank < FLAT_LABEL_RANK for r in sub])

    print(f"\n-- flat labels by contact height: {flat_rate(fine):.1%} of plausible-height "
          f"frames, {flat_rate(odd):.1%} of implausible")
    if flat_rate(odd) > 1.5 * flat_rate(fine) and flat_rate(odd) - flat_rate(fine) > 0.05:
        findings.warn(f"{split}: implausible-height teleop labels land on bare floor "
                      f"{flat_rate(odd):.0%} of the time against {flat_rate(fine):.0%}; the "
                      f"position estimate is wrong in x, y too, and those episodes are worth "
                      f"dropping")


def audit_duplicates(rows, findings):
    """Rows showing the same picture. Within an episode they inflate the count; across
    producers they are the same scene answered twice, and the answers can be compared."""
    have = [r for r in rows if r.signature is not None]
    # With train and eval present the pool repeats them, so it is left out.
    if any(r.split != POOL_SPLIT for r in have):
        have = [r for r in have if r.split != POOL_SPLIT]
    if len(have) < 2:
        return
    pairs = same_picture_pairs([r.signature for r in have])

    parent = list(range(len(have)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        parent[find(i)] = find(j)

    print(f"\n{'=' * 78}\nsame picture (no local difference over {SAME_PICTURE_PEAK:g} grey "
          f"levels at {SIGNATURE_FINE}x{SIGNATURE_FINE})\n{'=' * 78}")
    print("   distinct pictures per producer, which is what the frame counts are really worth:")
    for kind in sorted({r.kind for r in have}):
        idx = [i for i, r in enumerate(have) if r.kind == kind]
        print(f"     {kind:9s} {len({find(i) for i in idx}):6d} distinct of {len(idx)} frames")

    cross = [(have[i], have[j]) for i, j in pairs if have[i].group != have[j].group]
    print(f"   {len(pairs) - len(cross)} pair(s) inside one episode or run, {len(cross)} across them")
    leaked = sum(1 for a, b in cross if a.split != b.split)
    if leaked:
        findings.warn(f"{leaked} pair(s) of the same picture from different episodes or runs sit "
                      f"on opposite sides of the train/eval split")

    agreements, disagreements, contradictions, unconfirmed, teleop_checked = [], 0, 0, 0, 0
    gaps = []
    for a, b in cross:
        a.flag("cross_duplicate", f"same as {b.split}/{b.name}")
        b.flag("cross_duplicate", f"same as {a.split}/{a.name}")
        if a.complete and b.complete:
            missing = unmatched(a.contacts, b.contacts, AGREE_M) + unmatched(b.contacts, a.contacts, AGREE_M)
            agreements.append((len(a.points) + len(b.points), missing))
            if missing:
                disagreements += 1
                note = (f"{len(a.points)} vs {len(b.points)} targets, {missing} unmatched "
                        f"({b.split}/{b.name} vs {a.split}/{a.name})")
                a.flag("label_disagreement", note)
                a.partner = b
            if (not len(a.points)) != (not len(b.points)):
                contradictions += 1
        elif a.complete != b.complete:
            partial, complete = (a, b) if b.complete else (b, a)
            teleop_checked += 1
            if len(complete.contacts):
                gaps.append((np.linalg.norm(complete.contacts[:, :2] - partial.contacts[0, :2],
                                            axis=1).min(),
                             "implausible_height" in partial.reasons))
            if unmatched(partial.contacts, complete.contacts, AGREE_M):
                unconfirmed += 1
                note = f"grasp not among {complete.split}/{complete.name}'s {len(complete.points)} targets"
                partial.flag("grasp_not_labelled", note)
                partial.partner = complete

    if agreements:
        total = sum(n for n, _ in agreements)
        missing = sum(m for _, m in agreements)
        print(f"   {len(agreements)} pair(s) of complete frames labelled twice: {disagreements} "
              f"disagree, {missing} of {total} targets have no partner within "
              f"{AGREE_M * 100:.0f}cm")
        if disagreements:
            findings.warn(
                f"{disagreements} of {len(agreements)} twice-labelled pictures disagree by at "
                f"least one target. Those are exactly the frames the head treats as ground "
                f"truth for negatives; see the label_disagreement sheet.")
        if contradictions:
            findings.fail(f"{contradictions} picture(s) are labelled empty once and with "
                          f"targets once")
    if teleop_checked:
        print(f"   {teleop_checked} teleop frame(s) also appear hand-labelled; in "
              f"{unconfirmed} the grasped spot has no hand label within {AGREE_M * 100:.0f}cm")
        if gaps:
            dist = np.array([g for g, _ in gaps]) * 100
            print(f"   grasp to nearest hand label, cm (10/25/50/75/90): "
                  f"{percentiles(dist, (10, 25, 50, 75, 90))}")
            for name, want in (("plausible height", False), ("implausible height", True)):
                sub = [g * 100 for g, odd in gaps if odd == want]
                if sub:
                    print(f"     {name:18s} n={len(sub):4d} median {np.median(sub):5.1f}cm")
        if unconfirmed:
            findings.warn(
                f"{unconfirmed} of {teleop_checked} teleop frames that were also hand-labelled "
                f"have their grasp point outside every hand label. Either the labeller missed "
                f"the object or the contact position is off; see the grasp_not_labelled sheet.")


# ==========================================
# SPLITS AND PREVIEWS
# ==========================================

def audit_leakage(by_split, findings):
    if "train" not in by_split or "eval" not in by_split:
        return
    trained = {r.group for r in by_split["train"]}
    evals = by_split["eval"]
    shared = [r for r in evals if r.group in trained]
    print(f"\n{'=' * 78}\ntrain vs eval\n{'=' * 78}")
    print(f"   {len(shared)} of {len(evals)} eval frames share an episode or labelling run "
          f"with train")
    for kind in sorted({r.kind for r in evals}):
        sub = [r for r in evals if r.kind == kind]
        print(f"     {kind:9s} {sum(1 for r in sub if r.group in trained):5d} / {len(sub)}")
    scored = [r for r in evals if r.complete]
    if not scored:
        findings.fail("eval holds no complete frames, so ap@20cm cannot be computed")
    elif all(r.group in trained for r in scored):
        findings.warn("every complete eval frame has a sibling in train, so the selection "
                      "metric measures rooms the model has seen")


def draw_tile(row, image_bytes, note, partner=None):
    bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    scale = PREVIEW_TILE / max(w, h)
    tile = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    # The partner's labels in magenta, under this row's own in green.
    for points, colour in (((partner.points if partner else []), (255, 0, 255)), (row.points, (0, 255, 0))):
        for u, v in points:
            p = (int(round(u * scale)), int(round(v * scale)))
            cv2.circle(tile, p, 10, colour, 2)
            cv2.drawMarker(tile, p, colour, cv2.MARKER_CROSS, 8, 1)
    canvas = np.zeros((PREVIEW_TILE + 36, PREVIEW_TILE, 3), np.uint8)
    canvas[:tile.shape[0], :tile.shape[1]] = tile
    for k, text in enumerate((f"{row.split}/{row.name}", note[:52])):
        cv2.putText(canvas, text, (4, PREVIEW_TILE + 14 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def contact_sheet(tiles, columns=6):
    blank = np.zeros_like(tiles[0])
    tiles = tiles + [blank] * (-len(tiles) % columns)
    return np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])


def write_previews(root, by_split, preview_dir, per_reason, seed):
    """One contact sheet per reason plus a random sheet per producer, and rows.csv."""
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for split_rows in by_split.values() for r in split_rows]
    rng = np.random.default_rng(seed)

    wanted = {}
    for reason in sorted({x for r in rows for x in r.reasons}):
        flagged = [r for r in rows if reason in r.reasons]
        pick = rng.permutation(len(flagged))[:per_reason]
        wanted[reason] = [flagged[i] for i in sorted(pick)]
    for kind in sorted({r.kind for r in rows}):
        pool = [r for r in rows if r.kind == kind]
        pick = rng.permutation(len(pool))[:per_reason]
        wanted[f"random_{kind}"] = [pool[i] for i in sorted(pick)]

    # Images were dropped after measuring, so read back only the rows being drawn.
    needed = defaultdict(set)
    for chosen in wanted.values():
        for r in chosen:
            needed[r.split].add(r.name)
    images = {}
    for split, names in needed.items():
        for shard in sorted((root / split).glob("*.parquet")):
            table = pq.read_table(shard, columns=["file_name", "image"])
            for name, blob in zip(table.column("file_name").to_pylist(),
                                  table.column("image").to_pylist()):
                if name in names:
                    images[(split, name)] = blob

    for sheet, chosen in wanted.items():
        tiles = []
        for r in chosen:
            blob = images.get((r.split, r.name))
            if blob is None:
                continue
            note = "; ".join(r.notes) or f"{r.kind}, {len(r.points)} target(s)"
            partner = r.partner if sheet in ("grasp_not_labelled", "label_disagreement") else None
            tiles.append(draw_tile(r, blob, note, partner))
        if tiles:
            cv2.imwrite(str(preview_dir / f"{sheet}.jpg"), contact_sheet(tiles))

    def fmt(value, spec):
        return "" if value is None else format(value, spec)

    with open(preview_dir / "rows.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "file_name", "kind", "episode_index", "frame_offset", "task",
                         "targets", "contact_z_m", "lead_s", "coverage", "label_rank",
                         "reasons", "notes"])
        for r in rows:
            z = r.contacts[0, 2] if not r.complete and len(r.contacts) else None
            writer.writerow([r.split, r.name, r.kind, r.raw["episode_index"],
                             r.raw["frame_offset"], r.raw["task"], len(r.points), fmt(z, ".3f"),
                             fmt(r.lead_s(), ".2f"), fmt(r.coverage, ".3f"),
                             fmt(r.label_rank, ".2f"), " ".join(sorted(r.reasons)),
                             "; ".join(r.notes)])
    print(f"\nContact sheets and rows.csv (every row, its measurements and flags) written "
          f"to {preview_dir}")


def summarise_flags(rows, split):
    counts = Counter(x for r in rows for x in r.reasons)
    if not counts:
        return
    print(f"\n-- {split}: flagged rows ({sum(1 for r in rows if r.reasons)} of {len(rows)})")
    for reason, n in counts.most_common():
        print(f"   {reason:24s} {n:6d}  {REASONS.get(reason, '')}")


def audit_split(root, split, findings, with_images, seed):
    rows_raw, shards = read_split(root / split, with_images)
    rows = [Row(split, raw) for raw in rows_raw]
    print(f"\n{'=' * 78}\n{root / split}: {len(rows)} frames in {len(shards)} shard(s)\n{'=' * 78}")
    audit_composition(rows, findings, split)
    audit_integrity(rows, findings, split)
    audit_contacts(rows, findings, split)
    audit_placement(rows, findings, split)
    if with_images:
        audit_images(rows, findings, split, seed)
    return rows


def resolve_root(args):
    if args.data_root:
        return Path(args.data_root)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=args.dataset_id, repo_type="dataset"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_id", default=DEFAULT_DATASET_ID,
                        help="hub dataset to download and audit when --data_root is not given")
    parser.add_argument("--data_root", default=None,
                        help=f"local dataset directory holding {POOL_SPLIT}/, train/ or eval/")
    parser.add_argument("--split", default=None, choices=[POOL_SPLIT, "train", "eval"],
                        help="audit one directory; the default does every one present")
    parser.add_argument("--labels_only", action="store_true",
                        help="skip decoding frames: no coverage, salience, reprojection or "
                             "duplicate checks")
    parser.add_argument("--preview_dir", default=None,
                        help="write a contact sheet per flag and a random one per producer, "
                             "with labels drawn, plus rows.csv of every row's measurements and flags")
    parser.add_argument("--preview_count", type=int, default=PREVIEW_PER_REASON,
                        help="tiles per contact sheet")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = resolve_root(args)
    splits = ([args.split] if args.split
              else [s for s in (POOL_SPLIT, "train", "eval") if (root / s).exists()])
    if not splits:
        parser.error(f"no {POOL_SPLIT}/, train/ or eval/ directory under {root}")

    findings = Findings()
    by_split = {split: audit_split(root, split, findings, not args.labels_only, args.seed)
                for split in splits}
    everything = [r for rows in by_split.values() for r in rows]
    if not args.labels_only:
        audit_duplicates(everything, findings)
    for split, rows in by_split.items():
        summarise_flags(rows, split)
    audit_leakage(by_split, findings)
    if args.preview_dir:
        write_previews(root, by_split, args.preview_dir, args.preview_count, args.seed)
    raise SystemExit(1 if findings.report() else 0)


if __name__ == "__main__":
    main()
