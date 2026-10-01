#!/usr/bin/env python

"""Cluster the pre-grasp gripper snapshots of a drop_pairs dataset in DINOv2 latent space.

Each snapshot (the `image` column the placer's mine_teleop writes) is passed through the
same frozen DINOv2 trunk the visual servo network reads from, at its input size and
normalization, and represented by one vector from the last hidden state:

    cls            the [CLS] token
    cls_registers  [CLS] beside the mean of the register tokens: the global vector the
                   visual servo's finger/present/holding head sees
    patch_mean     the mean of the patch tokens

Those vectors are clustered with sklearn's AgglomerativeClustering. Writes clusters.json
(the snapshots in each cluster), thumbs/ and clusters.html (a thumbnail contact sheet, one
section per cluster) into the output directory.

Give exactly one of --n_clusters (fixed cluster count) or --distance_threshold
(cut the dendrogram at a distance, letting the count fall out).

Usage:
    python experiments/cluster_images_dino.py \
        --data_root datasets/drop_pairs \
        --n_clusters 40 \
        --output_dir ./pregrasp_clusters
"""

import argparse
import html
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.cluster import AgglomerativeClustering

from nf_robot.ml.dino_trunk import shared_backbone
from nf_robot.ml.image_input import normalize, to_tensor
from nf_robot.ml.visual_servoing.model import DEFAULT_BACKBONE, DEFAULT_IMAGE_SIZE

POOLS = ("cls", "cls_registers", "patch_mean")
THUMB_WIDTH = 224


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_snapshots(data_root, split, limit=None):
    """(rows, jpegs) for every snapshot in the split's drop_pairs shards."""
    shards = sorted(Path(data_root, split).glob("drop_pairs-*.parquet"))
    if not shards:
        raise SystemExit(f"No drop_pairs shards in {Path(data_root, split)}")
    columns = ["image", "source_repo_id", "episode_index", "frame_index",
               "seconds_before_grasp", "task", "state"]
    rows, jpegs = [], []
    for shard in shards:
        for r in pq.read_table(shard, columns=columns).to_pylist():
            jpegs.append(r.pop("image"))
            rows.append(r)
            if limit is not None and len(rows) >= limit:
                return rows, jpegs
    return rows, jpegs


def decode_rgb(jpeg):
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("undecodable snapshot")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def embed_images(jpegs, backbone_id, image_size, pool, device, batch_size, normalize_out=True):
    """One vector per snapshot from the frozen trunk's last hidden state, as (N, D) float32."""
    trunk = shared_backbone(backbone_id).to(device)
    width, height = image_size
    patch = trunk.config.patch_size
    n_patches = (height // patch) * (width // patch)

    vecs = []
    for start in range(0, len(jpegs), batch_size):
        batch = []
        for jpeg in jpegs[start:start + batch_size]:
            rgb = decode_rgb(jpeg)
            if (rgb.shape[1], rgb.shape[0]) != (width, height):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            batch.append(normalize(to_tensor(rgb)))
        with torch.no_grad():
            last = trunk(torch.stack(batch).to(device)).last_hidden_state
        # [CLS] and the register tokens lead the sequence; patches are always the tail.
        cls = last[:, 0]
        if pool == "cls":
            v = cls
        elif pool == "cls_registers":
            extras = last[:, 1:-n_patches]
            registers = extras.mean(dim=1) if extras.shape[1] else torch.zeros_like(cls)
            v = torch.cat([cls, registers], dim=-1)
        else:
            v = last[:, -n_patches:].mean(dim=1)
        vecs.append(v.float().cpu().numpy())
        print(f"  embedded {min(start + batch_size, len(jpegs))}/{len(jpegs)}")

    emb = np.concatenate(vecs, axis=0)
    if normalize_out:
        # Unit-norm vectors make euclidean distance a monotone function of cosine
        # distance, so a --distance_threshold means the same thing across datasets.
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return emb


def snapshot_name(row):
    repo = row["source_repo_id"].replace("/", "__")
    return f"{repo}_ep{row['episode_index']:04d}_f{row['frame_index']:05d}"


def write_thumbs(thumb_dir, rows, jpegs):
    """A small JPEG of each snapshot, named after where it came from; returns the filenames."""
    thumb_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for row, jpeg in zip(rows, jpegs):
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (THUMB_WIDTH, round(h * THUMB_WIDTH / w)), interpolation=cv2.INTER_AREA)
        name = snapshot_name(row) + ".jpg"
        cv2.imwrite(str(thumb_dir / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        names.append(name)
    return names


def write_html(out_path, clusters, rows, thumbs, thumb_px, title, source):
    """Contact sheet of the clustering, one section per cluster.

    Image src paths are relative to the html file so the page works over file://.
    """
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:sans-serif;background:#141414;color:#eee;margin:0;padding:16px}",
        "h1{font-size:18px;font-weight:600}",
        "h2{font-size:15px;font-weight:600;margin:24px 0 8px;position:sticky;top:0;"
        "background:#141414;padding:6px 0;border-bottom:1px solid #333}",
        ".grid{display:flex;flex-wrap:wrap;gap:6px}",
        ".cell{text-align:center;font-size:9px;color:#999}",
        f".cell img{{width:{thumb_px}px;height:{round(thumb_px * 9 / 16)}px;object-fit:cover;"
        "border-radius:3px;display:block;background:#000}",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p>{sum(len(c) for c in clusters)} snapshots in {len(clusters)} clusters "
        f"from {html.escape(source)}</p>",
    ]
    for i, members in enumerate(clusters):
        parts.append(f"<h2>Cluster {i} &mdash; {len(members)} snapshots</h2><div class='grid'>")
        for k in members:
            row = rows[k]
            src = html.escape(f"thumbs/{thumbs[k]}")
            tip = html.escape(f"{row['source_repo_id']} ep {row['episode_index']} "
                              f"frame {row['frame_index']}  {row['task']}")
            parts.append(
                f"<div class='cell'><img src='{src}' loading='lazy' title='{tip}'>"
                f"ep{row['episode_index']} {row['state']['laser_rangefinder']:.2f}m</div>"
            )
        parts.append("</div>")
    parts.append("</body></html>")

    with open(out_path, "w") as f:
        f.write("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default="datasets/drop_pairs",
                        help="drop_pairs dataset written by nf_robot.ml.placer.mine_teleop")
    parser.add_argument("--split", default="all")
    parser.add_argument("--output_dir", required=True,
                        help="directory to write clusters.json, clusters.html and thumbs/ into")
    parser.add_argument("--n_clusters", type=int, help="number of clusters to produce")
    parser.add_argument("--distance_threshold", type=float,
                        help="linkage distance above which clusters are not merged")
    parser.add_argument("--linkage", default="ward", choices=["ward", "complete", "average", "single"])
    parser.add_argument("--metric", default="euclidean",
                        help="distance metric (must be euclidean for ward linkage)")
    parser.add_argument("--pool", default="cls", choices=POOLS,
                        help="which tokens of the last hidden state represent a snapshot")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE,
                        help="DINO backbone id (default: the visual servo network's)")
    parser.add_argument("--image_size", type=int, nargs=2, default=DEFAULT_IMAGE_SIZE,
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="first N snapshots only")
    parser.add_argument("--no_normalize", action="store_true",
                        help="skip L2-normalizing the embeddings before clustering")
    parser.add_argument("--thumb_px", type=int, default=160, help="thumbnail width in the html page")
    parser.add_argument("--device", default=None, help="torch device (default: cuda/mps/cpu as available)")
    args = parser.parse_args()

    if (args.n_clusters is None) == (args.distance_threshold is None):
        parser.error("give exactly one of --n_clusters or --distance_threshold")

    rows, jpegs = load_snapshots(args.data_root, args.split, args.limit)
    device = args.device or pick_device()
    print(f"Embedding {len(rows)} snapshots with {args.backbone} ({args.pool}) on {device}...")
    emb = embed_images(jpegs, args.backbone, args.image_size, args.pool, device,
                       args.batch_size, normalize_out=not args.no_normalize)

    print(f"Clustering (linkage={args.linkage}, metric={args.metric})...")
    labels = AgglomerativeClustering(
        n_clusters=args.n_clusters,
        distance_threshold=args.distance_threshold,
        linkage=args.linkage,
        metric=args.metric,
    ).fit_predict(emb)

    # Largest cluster first, so the html leads with the dominant mode.
    groups = [list(np.flatnonzero(labels == lab)) for lab in np.unique(labels)]
    groups.sort(key=len, reverse=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs = write_thumbs(out_dir / "thumbs", rows, jpegs)

    json_path = out_dir / "clusters.json"
    with open(json_path, "w") as f:
        json.dump({
            "data_root": os.path.abspath(args.data_root),
            "split": args.split,
            "backbone": args.backbone,
            "image_size": list(args.image_size),
            "pool": args.pool,
            "n_clusters": args.n_clusters,
            "distance_threshold": args.distance_threshold,
            "linkage": args.linkage,
            "metric": args.metric,
            "normalized": not args.no_normalize,
            "clusters": [{
                "cluster": i,
                "size": len(g),
                "snapshots": [{
                    "source_repo_id": rows[k]["source_repo_id"],
                    "episode_index": rows[k]["episode_index"],
                    "frame_index": rows[k]["frame_index"],
                    "laser_rangefinder": rows[k]["state"]["laser_rangefinder"],
                    "thumb": thumbs[k],
                } for k in g],
            } for i, g in enumerate(groups)],
        }, f, indent=2)

    html_path = out_dir / "clusters.html"
    source = str(Path(args.data_root, args.split))
    write_html(html_path, groups, rows, thumbs, args.thumb_px,
               f"DINOv2 clusters of {source} pre-grasp snapshots", source)

    print(f"\n{len(groups)} clusters, sizes: {[len(g) for g in groups]}")
    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
