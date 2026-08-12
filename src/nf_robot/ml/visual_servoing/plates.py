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


def read_run(output_dir, run_id):
    """One run's rows, images decoded to RGB."""
    import pyarrow.parquet as pq

    table = pq.read_table(Path(output_dir) / f"{run_id}.parquet")
    rows = table.to_pylist()
    for row in rows:
        buf = np.frombuffer(row["image"], np.uint8)
        row["image"] = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        row["attrs"] = json.loads(row["attrs"]) if row["attrs"] else {}
    return rows
