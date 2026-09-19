#!/usr/bin/env python

"""Label distributions of a visual servoing dataset per head and producer, without reading images.

Usage:
    python -m nf_robot.ml.visual_servoing.audit --data_root datasets/visual_servoing
    python -m nf_robot.ml.visual_servoing.audit --data_root datasets/visual_servoing \
        --split train --by_source
"""

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from nf_robot.ml.visual_servoing.mine_teleop import POOL_SPLIT

# Everything except the image, so its bytes are never read.
LABEL_COLUMNS = [
    "split_source", "source_repo_id", "episode_index", "frame_index",
    "seconds_to_grasp", "target_uv", "target_range_m", "grasp_axis_rad",
    "finger", "target_present", "holding", "state",
]

# An axis label distribution this concentrated means a constant already scores well.
AXIS_CONCENTRATION_WARN = 0.70
# Systematic position-label offset worth explaining, as a fraction of the frame (~9px).
UV_BIAS_WARN = 0.02
# below this, a producer's axis labels are constant to within rounding
AXIS_CONSTANT_DEG = 0.5


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
            print("\nNo findings. Every head has labels on both sides of its range.")
            return 0
        print(f"\n{'=' * 78}\nFINDINGS\n{'=' * 78}")
        for level, message in self.items:
            print(f"  [{level}] {message}")
        return sum(1 for level, _ in self.items if level == "FAIL")


def read_labels(split_dir: Path):
    """Every label column of a split, as one table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    shards = sorted(split_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {split_dir}")
    tables = []
    for shard in shards:
        available = [c for c in LABEL_COLUMNS if c in pq.ParquetFile(shard).schema_arrow.names]
        tables.append(pq.read_table(shard, columns=available))
    return pa.concat_tables(tables, promote_options="default"), shards


def column(table, name):
    return table.column(name).to_pylist() if name in table.schema.names else []


def circular_summary(angles):
    """Resultant length and mean of a pi-periodic angle set, and what a constant scores."""
    doubled = 2.0 * np.asarray(angles, dtype=float)
    sin, cos = np.sin(doubled).mean(), np.cos(doubled).mean()
    return float(np.hypot(sin, cos)), math.degrees(math.atan2(sin, cos) / 2.0)


def histogram_line(values, bins, low, high, width=40):
    """One text histogram, scaled to the tallest bar."""
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    peak = counts.max() or 1
    lines = []
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(width * count / peak))
        lines.append(f"    {left:+7.1f}..{right:+7.1f} {count:8d} {bar}")
    return "\n".join(lines)


def audit_axis(table, findings, split, by_source):
    """The grasp axis head: how many labels, how concentrated, and per producer."""
    axis = column(table, "grasp_axis_rad")
    labelled = np.array([a for a in axis if a is not None], dtype=float)
    print(f"\n-- grasp_axis_rad: {len(labelled)}/{len(axis)} labelled "
          f"({len(labelled) / max(len(axis), 1):.1%})")
    if not len(labelled):
        findings.fail(f"{split}: no grasp axis labels at all; that head trains on nothing")
        return

    degrees = np.degrees(labelled)
    near_zero = float(np.mean(np.abs(degrees) < 5.0))
    resultant, mean_deg = circular_summary(labelled)
    print(f"   std {degrees.std():.1f} deg   within 5 deg of zero: {near_zero:.1%}")
    print(f"   concentration R {resultant:.3f}; the best constant prediction is "
          f"{mean_deg:+.2f} deg")
    print(histogram_line(degrees, 12, -90, 90))

    if near_zero > AXIS_CONCENTRATION_WARN:
        findings.warn(
            f"{split}: {near_zero:.0%} of axis labels are within 5 deg of zero, so a head "
            f"that always answers {mean_deg:+.1f} deg is already right most of the time. "
            f"Expect it to do exactly that unless the loss stops rewarding the hedge and "
            f"non-zero rows are weighted up.")

    if by_source:
        print("   by producer:")
        for source in sorted(set(column(table, "split_source"))):
            mask = [s == source for s in column(table, "split_source")]
            sub = np.array([np.degrees(a) for a, keep in zip(axis, mask)
                            if keep and a is not None], dtype=float)
            if not len(sub):
                print(f"     {source:8s} no axis labels")
                findings.warn(f"{split}: producer '{source}' contributes no axis labels")
                continue
            print(f"     {source:8s} n={len(sub):7d} std {sub.std():6.1f} deg  "
                  f"|axis|>5 deg in {np.mean(np.abs(sub) > 5):5.1%}")
            if sub.std() < AXIS_CONSTANT_DEG:
                findings.fail(
                    f"{split}: every axis label from '{source}' is the same value "
                    f"({sub.mean():+.1f} deg). A producer is writing a constant - check "
                    f"its extractor before training anything on this.")


def audit_flags(table, findings, split):
    """The two probability heads, which are only as good as having both classes."""
    for name, key in (("target_present", "target_present"), ("holding", "holding")):
        values = column(table, key)
        labelled = [v for v in values if v is not None]
        positives = sum(1 for v in labelled if v)
        negatives = len(labelled) - positives
        print(f"\n-- {name}: {len(labelled)}/{len(values)} labelled "
              f"({len(labelled) / max(len(values), 1):.1%}), "
              f"{positives} positive / {negatives} negative")
        if not labelled:
            findings.warn(f"{split}: {name} has no labels; that head trains on nothing")
        elif not negatives:
            findings.fail(
                f"{split}: {name} has no negative examples. A binary head with one class "
                f"can only saturate - it will read 1.0 on every frame, including frames "
                f"where the answer is obviously no.")
        elif not positives:
            findings.fail(f"{split}: {name} has no positive examples.")

    # Rows near the grasp instant, where the holding label is a coin flip.
    seconds = [s for s in column(table, "seconds_to_grasp") if s is not None]
    if seconds:
        ambiguous = sum(1 for s in seconds if abs(s) < 0.3)
        print(f"   {ambiguous} rows within 0.3s of the grasp instant, where holding is "
              f"ambiguous either way ({ambiguous / len(seconds):.1%} of timed rows)")


def audit_position(table, findings, split):
    """Where the targets are, and whether they sit off-centre as a body."""
    uv = [v for v in column(table, "target_uv") if v is not None]
    total = table.num_rows
    print(f"\n-- target_uv: {len(uv)}/{total} labelled ({len(uv) / max(total, 1):.1%})")
    if not uv:
        findings.fail(f"{split}: no position labels at all")
        return
    array = np.array(uv, dtype=float)
    centred = array - 0.5
    off_frame = np.mean((array < 0) | (array > 1), axis=0)
    print(f"   mean {array.mean(axis=0).round(3)}  std {array.std(axis=0).round(3)}")
    print(f"   off the visible frame: u {off_frame[0]:.1%}, v {off_frame[1]:.1%}")
    print(f"   signed offset from centre: u {centred[:, 0].mean():+.3f}, "
          f"v {centred[:, 1].mean():+.3f} (frame widths)")
    # Compare this against the model's mean prediction when chasing a systematic lean.
    if abs(centred[:, 0].mean()) > UV_BIAS_WARN:
        findings.warn(
            f"{split}: position labels lean {centred[:, 0].mean():+.3f} frame widths in u. "
            f"Horizontal has no mount offset to explain it, so this is either a real bias "
            f"in the labels or an unbalanced set of approaches.")

    ranges = np.array([r for r in column(table, "target_range_m") if r is not None])
    if len(ranges):
        pcts = np.percentile(ranges, [1, 25, 50, 75, 99]).round(3)
        print(f"   target_range_m percentiles (1/25/50/75/99): {pcts}")


def audit_finger(table, findings, split):
    finger = [f for f in column(table, "finger") if f is not None]
    total = table.num_rows
    print(f"\n-- finger: {len(finger)}/{total} labelled ({len(finger) / max(total, 1):.1%})")
    if not finger:
        findings.warn(f"{split}: finger has no labels; that head trains on nothing")
        return
    array = np.array(finger, dtype=float)
    zero = float(np.mean(np.abs(array) < 1e-6))
    print(f"   mean {array.mean():+.3f}  std {array.std():.3f}  exactly zero {zero:.1%}")
    if zero > 0.6:
        findings.warn(
            f"{split}: {zero:.0%} of finger labels are exactly zero, so a head that always "
            f"answers zero has little to lose. Read its output with that in mind.")


def audit_state(table, findings, split):
    state = [s for s in column(table, "state") if s is not None]
    if not state:
        findings.warn(f"{split}: no state column; the FiLM inputs are missing")
        return
    print("\n-- state (model inputs)")
    for key in ("laser_rangefinder", "finger_angle", "target_force"):
        values = np.array([s[key] for s in state if s.get(key) is not None], dtype=float)
        if not len(values):
            findings.warn(f"{split}: state.{key} is never populated")
            continue
        spread = "constant" if values.std() < 1e-9 else f"std {values.std():.3f}"
        print(f"   {key:18s} mean {values.mean():+.3f}  {spread}  "
              f"range {values.min():.3f}..{values.max():.3f}")
        if values.std() < 1e-9:
            findings.warn(
                f"{split}: state.{key} is the same value in every row, so the model cannot "
                f"be conditioning on it - check the producer that wrote these rows.")


def audit_split(root: Path, split: str, findings: Findings, by_source: bool):
    split_dir = root / split
    table, shards = read_labels(split_dir)
    total = table.num_rows

    print(f"\n{'=' * 78}\n{split_dir}: {total} rows in {len(shards)} shard(s)\n{'=' * 78}")

    sources = Counter(column(table, "split_source"))
    print("\n-- producers")
    for source, count in sources.most_common():
        print(f"   {source:8s} {count:8d} rows ({count / max(total, 1):.1%})")
    if len(sources) < 2:
        only = next(iter(sources), "none")
        findings.warn(
            f"{split}: every row comes from '{only}'. The two producers cover different "
            f"heads - mined rows carry no negatives and synthetic rows carry no finger or "
            f"holding labels - so a single-producer split leaves heads unsupervised. "
            f"Check that both ran and that neither deleted the other's shards.")

    repos = Counter(column(table, "source_repo_id"))
    print(f"\n-- sources: {len(repos)} distinct")
    for repo, count in repos.most_common(8):
        print(f"   {repo[:52]:52s} {count:8d}")
    if len(repos) > 8:
        print(f"   ... and {len(repos) - 8} more")

    episodes = set(zip(column(table, "source_repo_id"), column(table, "episode_index")))
    print(f"\n-- episodes: {len(episodes)} distinct (source, episode_index) pairs")

    audit_position(table, findings, split)
    audit_axis(table, findings, split, by_source)
    audit_flags(table, findings, split)
    audit_finger(table, findings, split)
    audit_state(table, findings, split)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", required=True,
                        help=f"Dataset root holding the {POOL_SPLIT}/ pool, or train/ and eval/")
    parser.add_argument("--split", default=None, choices=[POOL_SPLIT, "train", "eval"],
                        help="Audit one split; the default does every one present. Worth "
                             "running on the pool before dealing it, since a head with "
                             "nothing to learn from is a property of what was built rather "
                             "than of how it was cut")
    parser.add_argument("--by_source", action="store_true", default=True,
                        help="Break the axis histogram down by producer (default on)")
    parser.add_argument("--no_by_source", dest="by_source", action="store_false")
    args = parser.parse_args()

    root = Path(args.data_root)
    splits = ([args.split] if args.split
              else [s for s in (POOL_SPLIT, "train", "eval") if (root / s).exists()])
    if not splits:
        parser.error(f"no {POOL_SPLIT}/, train/ or eval/ directory under {root}")

    findings = Findings()
    for split in splits:
        audit_split(root, split, findings, args.by_source)
    raise SystemExit(1 if findings.report() else 0)


if __name__ == "__main__":
    main()
