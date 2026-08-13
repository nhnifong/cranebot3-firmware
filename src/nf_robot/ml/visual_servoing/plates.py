#!/usr/bin/env python

"""Storage for the raw ingredients of the synthetic visual servoing dataset.

Three capture routines feed this, all of them motion tasks in observer.py:

    fingerplates   the gripper's own fingers, matted out of their background
    floorplates    bare floor at a range of heights, for backgrounds
    objectplates   one object on a green board, for compositing onto them

They are different captures but the same kind of thing - a stack of frames plus the
state that was true when each was taken - so they share a file format and differ only in
which attributes they carry. Frames go in a parquet file as encoded bytes and the run is
described by a line appended to manifest.jsonl beside it, so a directory of captures can
be read without opening any of the parquet files.

What is deliberately *not* here is any matting, keying or segmentation. Those are offline
decisions that will be revised - the finger matte in particular is a per-pixel variance
test whose threshold nobody has tuned yet - and a capture that has already thrown away
the frames it was derived from cannot be revisited without going back to the robot. This
module stores what the camera saw.
"""

import json
import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

KINDS = ("fingerplates", "floorplates", "objectplates")

# Capture quality. These frames are the source every synthetic image is built from and
# are only ever downscaled later, so the encoding wants to be near-lossless: the finger
# matte separates gripper from background by per-pixel variance across a wrist turn, and
# compression noise is variance. Lossless PNG is available by passing image_format="png"
# if q95 turns out to muddy the matte, at roughly ten times the bytes.
PLATE_JPEG_QUALITY = 95

MANIFEST_NAME = "manifest.jsonl"


def plate_schema():
    """One row per captured frame.

    The typed columns are the state every capture type has an opinion about; anything
    specific to one routine goes in `attrs` as JSON rather than growing a column that is
    null for two kinds out of three.
    """
    import pyarrow as pa

    return pa.schema([
        ("image", pa.binary()),
        ("image_format", pa.string()),
        ("kind", pa.string()),
        ("run_id", pa.string()),
        ("captured_at", pa.float64()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("finger_angle", pa.float32()),
        ("wrist_angle", pa.float32()),
        ("laser_rangefinder", pa.float32()),
        ("finger_pressure", pa.float32()),
        ("attrs", pa.string()),
    ])


def encode_plate(image_rgb, image_format="jpeg"):
    """Encode one RGB frame for storage, returning (bytes, format)."""
    bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
    if image_format == "png":
        ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, PLATE_JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"failed to encode a plate as {image_format}")
    return buf.tobytes(), image_format


class PlateWriter:
    """Accumulates one capture run and writes it as a parquet file plus a manifest line.

    Buffered rather than streamed because a run is a few hundred frames taken over a few
    minutes of robot motion, and a partial file from an aborted run is worse than none -
    the manifest would describe a capture that never finished.
    """

    def __init__(self, output_dir, kind: str, image_format: str = "jpeg", notes: str = ""):
        if kind not in KINDS:
            raise ValueError(f"unknown plate kind {kind!r}; expected one of {KINDS}")
        self.dir = Path(output_dir)
        self.kind = kind
        self.image_format = image_format
        self.notes = notes
        self.run_id = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.started_at = time.time()
        self.rows: list[dict] = []

    def add(self, image_rgb, captured_at=None, finger_angle=None, wrist_angle=None,
            laser_rangefinder=None, finger_pressure=None, **attrs):
        """Record one frame and the state that was true when it was taken."""
        image = np.asarray(image_rgb)
        blob, fmt = encode_plate(image, self.image_format)
        self.rows.append({
            "image": blob,
            "image_format": fmt,
            "kind": self.kind,
            "run_id": self.run_id,
            "captured_at": float(captured_at if captured_at is not None else time.time()),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "finger_angle": _f(finger_angle),
            "wrist_angle": _f(wrist_angle),
            "laser_rangefinder": _f(laser_rangefinder),
            "finger_pressure": _f(finger_pressure),
            "attrs": json.dumps(attrs, default=float),
        })

    def __len__(self):
        return len(self.rows)

    def close(self, **run_attrs):
        """Write the parquet file and append the manifest line. Returns the path, or None
        for a run that captured nothing - which leaves no file and no manifest entry."""
        if not self.rows:
            logging.warning(f"{self.run_id}: no frames captured, nothing written")
            return None

        import pyarrow as pa
        import pyarrow.parquet as pq

        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{self.run_id}.parquet"
        pq.write_table(pa.Table.from_pylist(self.rows, schema=plate_schema()), path,
                       compression="snappy", row_group_size=32)

        entry = {
            "run_id": self.run_id,
            "kind": self.kind,
            "file": path.name,
            "frames": len(self.rows),
            "image_format": self.image_format,
            "width": self.rows[0]["width"],
            "height": self.rows[0]["height"],
            "started_at": self.started_at,
            "finished_at": time.time(),
            "notes": self.notes,
            **run_attrs,
        }
        with open(self.dir / MANIFEST_NAME, "a") as f:
            f.write(json.dumps(entry, default=float) + "\n")

        size = sum(len(r["image"]) for r in self.rows)
        logging.info(f"{self.run_id}: wrote {len(self.rows)} frames "
                     f"({size / 1e6:.0f} MB) to {path}")
        return path


def _f(value):
    return None if value is None else float(value)


def read_manifest(output_dir):
    """Every capture run in a directory, newest last."""
    path = Path(output_dir) / MANIFEST_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def read_run(output_dir, run_id, columns=None, decode=True):
    """One run's rows, images decoded to RGB unless decode is False.

    A run of 1080p frames is a couple of hundred megabytes, so pass columns without
    "image" to read the attributes of a capture without paying for its pixels.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(Path(output_dir) / f"{run_id}.parquet", columns=columns)
    rows = table.to_pylist()
    for row in rows:
        if decode and row.get("image") is not None:
            buf = np.frombuffer(row["image"], np.uint8)
            row["image"] = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if "attrs" in row:
            row["attrs"] = json.loads(row["attrs"]) if row["attrs"] else {}
    return rows


def iter_run(output_dir, run_id, decode=True):
    """Yield a run's rows one row group at a time, so a whole capture is never resident.

    PlateWriter uses small row groups precisely so this can walk a 200MB file without
    holding it all, which matters on anything that also has a model loaded.
    """
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(Path(output_dir) / f"{run_id}.parquet")
    for group in range(reader.num_row_groups):
        for row in reader.read_row_group(group).to_pylist():
            if decode:
                buf = np.frombuffer(row["image"], np.uint8)
                row["image"] = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            row["attrs"] = json.loads(row["attrs"]) if row["attrs"] else {}
            yield row


def _label(row):
    parts = []
    for key, fmt in (("finger_angle", "fing {:+.0f}"), ("wrist_angle", "wrist {:.0f}"),
                     ("laser_rangefinder", "rng {:.2f}"), ("finger_pressure", "p {:.2f}")):
        if row.get(key) is not None:
            parts.append(fmt.format(row[key]))
    return "  ".join(parts)


def write_preview(output_dir, run_id, preview_dir, cell_width=480, columns=6, group=24,
                  every=1, full_size=0):
    """Contact sheets of a capture run, plus optionally some frames at full size.

    Cells are downscaled first and the text drawn afterwards, so it renders at the size
    it will actually be read at rather than shrinking into illegibility with the image.
    """
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old in list(preview_dir.glob("*.jpg")) + list(preview_dir.glob("*.png")):
        old.unlink()

    cells, kept = [], 0
    for index, row in enumerate(iter_run(output_dir, run_id)):
        if index % every:
            continue
        bgr = cv2.cvtColor(row["image"], cv2.COLOR_RGB2BGR)
        if kept < full_size:
            cv2.imwrite(str(preview_dir / f"frame{index:04d}.jpg"), bgr)
        scale = cell_width / bgr.shape[1]
        cell = cv2.resize(bgr, (cell_width, int(round(bgr.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
        text = f"{index}  {_label(row)}"
        cv2.putText(cell, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(cell, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cells.append(cell)
        kept += 1

    sheets = 0
    for start in range(0, len(cells), group):
        block = cells[start:start + group]
        h, w = block[0].shape[:2]
        blank = np.full((h, w, 3), 25, dtype=np.uint8)
        block = block + [blank] * (-len(block) % columns)
        sheet = np.vstack([np.hstack(block[r:r + columns]) for r in range(0, len(block), columns)])
        cv2.imwrite(str(preview_dir / f"_sheet_{sheets + 1:02d}.png"), sheet)
        sheets += 1

    logging.info(f"{run_id}: {kept} frames -> {sheets} contact sheet(s) in {preview_dir}")
    return preview_dir


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="plates", help="Directory of capture runs")
    parser.add_argument("--run_id", default=None, help="Which run; defaults to the newest")
    parser.add_argument("--preview_dir", default=None,
                        help="Write contact sheets here (default <dir>/preview)")
    parser.add_argument("--every", type=int, default=1, help="Keep every Nth frame")
    parser.add_argument("--cell_width", type=int, default=480)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--group", type=int, default=24, help="Frames per contact sheet")
    parser.add_argument("--full_size", type=int, default=0,
                        help="Also write this many frames at their captured resolution")
    parser.add_argument("--list", action="store_true", help="List runs and exit")
    args = parser.parse_args()

    runs = read_manifest(args.dir)
    if not runs:
        parser.error(f"no {MANIFEST_NAME} in {args.dir}")
    if args.list:
        for entry in runs:
            minutes = (entry["finished_at"] - entry["started_at"]) / 60
            print(f"{entry['run_id']}  {entry['kind']:12s} {entry['frames']:5d} frames  "
                  f"{entry['width']}x{entry['height']}  {minutes:.1f} min  {entry.get('notes', '')}")
        return

    run_id = args.run_id or runs[-1]["run_id"]
    write_preview(args.dir, run_id, args.preview_dir or Path(args.dir) / "preview",
                  cell_width=args.cell_width, columns=args.columns, group=args.group,
                  every=args.every, full_size=args.full_size)


if __name__ == "__main__":
    main()
