#!/usr/bin/env python

"""Cluster a directory of images in CLIP latent space.

Each image is passed through the CLIP vision encoder and represented by its CLS
token (the pooled class embedding, after the final layernorm). Those vectors are
clustered with sklearn's AgglomerativeClustering. Writes clusters.json (the
filenames in each cluster) and clusters.html (a thumbnail contact sheet, one
section per cluster) into the output directory.

Give exactly one of --n_clusters (fixed cluster count) or --distance_threshold
(cut the dendrogram at a distance, letting the count fall out).

Usage:
    python experiments/cluster_images_clip.py \
        --image_dir ./pregrasp_frames \
        --distance_threshold 8.0 \
        --output_dir ./pregrasp_clusters
"""

import argparse
import html
import json
import os

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from transformers import CLIPImageProcessor, CLIPVisionModel

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_images(paths, model_name, device, batch_size, normalize=True):
    """CLS token of the CLIP vision encoder for each image, as (N, D) float32."""
    processor = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPVisionModel.from_pretrained(model_name).to(device).eval()

    vecs = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        # pooler_output is the CLS token from the last hidden state, post-layernorm.
        vecs.append(out.pooler_output.float().cpu().numpy())
        print(f"  embedded {min(start + batch_size, len(paths))}/{len(paths)}")

    emb = np.concatenate(vecs, axis=0)
    if normalize:
        # Unit-norm vectors make euclidean distance a monotone function of cosine
        # distance, so a --distance_threshold means the same thing across datasets.
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return emb


def write_html(out_path, image_dir, clusters, thumb_px, title):
    """Contact sheet of the clustering, one section per cluster.

    Image src paths are relative to the html file so the page works over file://.
    """
    html_dir = os.path.dirname(os.path.abspath(out_path))
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
        f".cell img{{width:{thumb_px}px;height:{thumb_px}px;object-fit:cover;"
        "border-radius:3px;display:block;background:#000}",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p>{sum(len(c) for c in clusters)} images in {len(clusters)} clusters "
        f"from {html.escape(image_dir)}</p>",
    ]
    for i, names in enumerate(clusters):
        parts.append(f"<h2>Cluster {i} &mdash; {len(names)} images</h2><div class='grid'>")
        for name in names:
            src = os.path.relpath(os.path.join(image_dir, name), html_dir)
            esc = html.escape(src)
            parts.append(
                f"<div class='cell'><img src='{esc}' loading='lazy' title='{html.escape(name)}'>"
                f"{html.escape(name)}</div>"
            )
        parts.append("</div>")
    parts.append("</body></html>")

    with open(out_path, "w") as f:
        f.write("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image_dir", required=True, help="directory of images to cluster")
    parser.add_argument("--output_dir", required=True, help="directory to write clusters.json and clusters.html into")
    parser.add_argument("--n_clusters", type=int, help="number of clusters to produce")
    parser.add_argument("--distance_threshold", type=float,
                        help="linkage distance above which clusters are not merged")
    parser.add_argument("--linkage", default="ward", choices=["ward", "complete", "average", "single"])
    parser.add_argument("--metric", default="euclidean",
                        help="distance metric (must be euclidean for ward linkage)")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="CLIP model id")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_normalize", action="store_true",
                        help="skip L2-normalizing the CLS tokens before clustering")
    parser.add_argument("--thumb_px", type=int, default=112, help="thumbnail size in the html page")
    parser.add_argument("--device", default=None, help="torch device (default: cuda/mps/cpu as available)")
    args = parser.parse_args()

    if (args.n_clusters is None) == (args.distance_threshold is None):
        parser.error("give exactly one of --n_clusters or --distance_threshold")

    paths = sorted(
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if f.lower().endswith(IMAGE_EXTS)
    )
    if not paths:
        raise SystemExit(f"No images found in {args.image_dir}")

    device = args.device or pick_device()
    print(f"Embedding {len(paths)} images with {args.model} on {device}...")
    emb = embed_images(paths, args.model, device, args.batch_size, normalize=not args.no_normalize)

    print(f"Clustering (linkage={args.linkage}, metric={args.metric})...")
    labels = AgglomerativeClustering(
        n_clusters=args.n_clusters,
        distance_threshold=args.distance_threshold,
        linkage=args.linkage,
        metric=args.metric,
    ).fit_predict(emb)

    names = [os.path.basename(p) for p in paths]
    # Largest cluster first, so the html leads with the dominant mode.
    groups = [[names[i] for i in np.flatnonzero(labels == lab)] for lab in np.unique(labels)]
    groups.sort(key=len, reverse=True)

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "clusters.json")
    with open(json_path, "w") as f:
        json.dump({
            "image_dir": os.path.abspath(args.image_dir),
            "model": args.model,
            "n_clusters": args.n_clusters,
            "distance_threshold": args.distance_threshold,
            "linkage": args.linkage,
            "metric": args.metric,
            "normalized": not args.no_normalize,
            "clusters": [{"cluster": i, "size": len(g), "images": g} for i, g in enumerate(groups)],
        }, f, indent=2)

    html_path = os.path.join(args.output_dir, "clusters.html")
    write_html(html_path, os.path.abspath(args.image_dir), groups, args.thumb_px,
               f"CLIP clusters of {os.path.basename(os.path.abspath(args.image_dir))}")

    print(f"\n{len(groups)} clusters, sizes: {[len(g) for g in groups]}")
    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
