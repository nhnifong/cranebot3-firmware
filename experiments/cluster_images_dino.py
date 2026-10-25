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
section per cluster) into the output directory, plus embeddings.npy (one row per snapshot,
in clusters.json's order) and graph.html: the snapshots on their three largest principal
components as a 3D graph, linking each to its nearest neighbors closer than a slider's
threshold, with the thumbnail of whichever node is under the mouse.

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
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from nf_robot.ml.dino_trunk import shared_backbone
from nf_robot.ml.image_input import normalize, to_tensor
from nf_robot.ml.visual_servoing.model import DEFAULT_BACKBONE, DEFAULT_IMAGE_SIZE

POOLS = ("cls", "cls_registers", "patch_mean")
THUMB_WIDTH = 224
# Nearest neighbors each node keeps a candidate link to in graph.html; the slider shows the
# ones closer than its threshold. All pairs would be tens of millions of lines.
GRAPH_NEIGHBORS = 15


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


def write_graph_html(out_path, emb, clusters, rows, thumbs, title, neighbors=GRAPH_NEIGHBORS):
    """The snapshots on their top three principal components as a 3D graph, nodes colored
    by cluster, linked to their nearest neighbors in the full embedding space.

    Thumbnail paths are relative to the html file so the page works over file://.
    """
    order = [k for members in clusters for k in members]
    cluster_of = [i for i, members in enumerate(clusters) for _ in members]
    x = emb[order]

    pca = PCA(n_components=3).fit(x)
    xyz = pca.transform(x)
    xyz /= np.abs(xyz).max()

    # Distances in the full embedding, not the projection, so a link means the snapshots
    # really are alike. Sorted, so the page shows a threshold as a prefix of the list.
    dist, idx = NearestNeighbors(n_neighbors=neighbors + 1).fit(x).kneighbors(x)
    edges = {}
    for a in range(len(x)):
        for d, b in zip(dist[a, 1:], idx[a, 1:]):
            key = (min(a, b), max(a, b))
            edges[key] = float(d)
    edge_list = sorted(edges.items(), key=lambda kv: kv[1])

    nodes = [{
        "thumb": f"thumbs/{thumbs[k]}",
        "label": f"{rows[k]['source_repo_id']} ep {rows[k]['episode_index']} "
                 f"frame {rows[k]['frame_index']} · {rows[k]['state']['laser_rangefinder']:.2f} m",
    } for k in order]
    data = {
        "title": title,
        "xyz": [round(float(v), 4) for v in xyz.ravel()],
        "cluster": cluster_of,
        "n_clusters": len(clusters),
        "nodes": nodes,
        "edges": [v for (a, b), _ in edge_list for v in (int(a), int(b))],
        "dist": [round(d, 4) for _, d in edge_list],
        "neighbors": neighbors,
        "variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
    }
    page = GRAPH_PAGE.replace("__TITLE__", html.escape(title)).replace(
        "__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    with open(out_path, "w") as f:
        f.write(page)


GRAPH_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; height: 100%; background: #0b0d12; color: #d8dde6; overflow: hidden;
               font: 13px system-ui, sans-serif; }
  #ui { position: fixed; top: 12px; left: 12px; background: rgba(20,24,32,0.85); padding: 12px 14px;
        border-radius: 8px; border: 1px solid #2a3140; width: 300px; }
  #ui h1 { font-size: 14px; margin: 0 0 8px; font-weight: 600; }
  #ui input[type=range] { width: 100%; }
  .muted { color: #8a93a3; }
  #tip { position: fixed; pointer-events: none; display: none; background: #141922;
         border: 1px solid #3a4356; border-radius: 6px; padding: 6px; }
  #tip img { display: block; width: 224px; border-radius: 3px; }
  #tip div { margin-top: 4px; font-size: 11px; color: #aab3c2; max-width: 224px; }
</style>
</head>
<body>
<div id="ui">
  <h1>__TITLE__</h1>
  <div class="muted" id="info"></div>
  <div style="margin-top:10px">link if closer than <b id="thr"></b></div>
  <input type="range" id="slider" min="0" max="1000" value="300">
  <div class="muted" id="count"></div>
  <div class="muted" style="margin-top:8px">drag to orbit · right-drag to pan · scroll to zoom</div>
</div>
<div id="tip"><img id="tipimg"><div id="tiplabel"></div></div>
<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
} }
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const D = __DATA__;
const N = D.cluster.length;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x0b0d12);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.01, 100);
camera.position.set(1.8, 1.3, 2.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// nodes, colored by cluster
const positions = new Float32Array(D.xyz);
const colors = new Float32Array(N * 3);
const color = new THREE.Color();
for (let i = 0; i < N; i++) {
  color.setHSL((D.cluster[i] / D.n_clusters + 0.03) % 1, 0.75, 0.6);
  colors.set([color.r, color.g, color.b], i * 3);
}
const nodeGeometry = new THREE.BufferGeometry();
nodeGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
nodeGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
const nodes = new THREE.Points(nodeGeometry, new THREE.PointsMaterial({
  size: 0.022, vertexColors: true, sizeAttenuation: true }));
scene.add(nodes);

// links, sorted nearest first, so a threshold is just how many of them to draw
const edgeGeometry = new THREE.BufferGeometry();
edgeGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
edgeGeometry.setIndex(D.edges);
const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x7f9cff, transparent: true,
  opacity: 0.18, depthWrite: false, blending: THREE.AdditiveBlending });
scene.add(new THREE.LineSegments(edgeGeometry, edgeMaterial));

// the hovered node
const marker = new THREE.Mesh(new THREE.SphereGeometry(0.02, 16, 12),
  new THREE.MeshBasicMaterial({ color: 0xffffff }));
marker.visible = false;
scene.add(marker);

const lo = D.dist[0], hi = D.dist[D.dist.length - 1];
const slider = document.getElementById("slider");
function upperBound(t) {
  let a = 0, b = D.dist.length;
  while (a < b) { const m = (a + b) >> 1; if (D.dist[m] < t) a = m + 1; else b = m; }
  return a;
}
function applyThreshold() {
  const t = lo + (hi - lo) * slider.value / 1000;
  const n = upperBound(t);
  edgeGeometry.setDrawRange(0, n * 2);
  document.getElementById("thr").textContent = t.toFixed(3);
  document.getElementById("count").textContent =
    `${n.toLocaleString()} links (of each node's ${D.neighbors} nearest)`;
}
slider.addEventListener("input", applyThreshold);
applyThreshold();
const v = D.variance.map(x => (x * 100).toFixed(1) + "%").join(", ");
document.getElementById("info").textContent =
  `${N.toLocaleString()} snapshots · ${D.n_clusters} clusters · PC variance ${v}`;

// hover: the nearest node to the ray, its thumbnail beside the cursor
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.015;
const mouse = new THREE.Vector2();
const tip = document.getElementById("tip"), tipimg = document.getElementById("tipimg");
let hovered = -1;
renderer.domElement.addEventListener("pointermove", e => {
  mouse.set(e.clientX / innerWidth * 2 - 1, -e.clientY / innerHeight * 2 + 1);
  raycaster.setFromCamera(mouse, camera);
  const hit = raycaster.intersectObject(nodes)[0];
  if (!hit) { hovered = -1; tip.style.display = "none"; marker.visible = false; return; }
  const i = hit.index;
  if (i !== hovered) {
    hovered = i;
    tipimg.src = D.nodes[i].thumb;
    document.getElementById("tiplabel").textContent = `cluster ${D.cluster[i]} · ${D.nodes[i].label}`;
    marker.position.fromArray(positions, i * 3);
    marker.visible = true;
  }
  tip.style.display = "block";
  const x = e.clientX + 16, y = e.clientY + 16;
  tip.style.left = Math.min(x, innerWidth - 250) + "px";
  tip.style.top = Math.min(y, innerHeight - 180) + "px";
});

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script>
</body>
</html>
"""


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

    # in the order clusters.json lists the snapshots
    np.save(out_dir / "embeddings.npy", emb[[k for g in groups for k in g]])

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

    graph_path = out_dir / "graph.html"
    write_graph_html(graph_path, emb, groups, rows, thumbs,
                     f"DINOv2 embedding graph of {source} pre-grasp snapshots")

    print(f"\n{len(groups)} clusters, sizes: {[len(g) for g in groups]}")
    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")
    print(f"Wrote {graph_path}")


if __name__ == "__main__":
    main()
