#!/usr/bin/env python

"""Map episode indices of a merged dataset back to the source datasets that fed it.

Given the recipe a merged dataset was built from (see lerobot_build_dataset.py),
this works out which source dataset and which source episode index each merged
episode came from, and prints the result as a recipe `merge:` block with
`exclude_episodes` filled in - i.e. paste-ready for excluding bad episodes on the
next build.

Merging concatenates sources in recipe order, so the mapping is just cumulative
episode counts. That is verified rather than assumed: per-episode frame counts of
the merged dataset are compared against the concatenated source frame counts, and
the mapping is refused if they diverge (which would mean the sources changed since
the merged dataset was built).

Episodes to map can be given directly, or pulled from clusters of a
cluster_images_clip.py run over frames named `episode_<merged index>.<ext>`
(as produced by extract_pregrasp_frames.py).

Usage:
    python experiments/map_merged_episodes.py \
        --recipe src/nf_robot/ml/recipes/move_clutter_combined_384_2.yaml \
        --episodes 289-325,448-481

    python experiments/map_merged_episodes.py \
        --recipe src/nf_robot/ml/recipes/move_clutter_combined_384_2.yaml \
        --clusters_json pregrasp_clusters/clusters.json --clusters 6 7
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from nf_robot.ml.lerobot_build_dataset import load_recipe, parse_episode_list

EPISODE_FILENAME_RE = re.compile(r"episode_(\d+)\.[A-Za-z0-9]+$")


def episode_lengths(repo_id: str) -> np.ndarray:
    """Per-episode frame counts of a Hub dataset, ordered by episode index.

    Only meta/episodes is downloaded - a few hundred KB - not the videos.
    """
    root = snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=["meta/episodes/**"])
    files = sorted(glob.glob(os.path.join(root, "meta/episodes/**/*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"{repo_id} has no meta/episodes parquet files")

    indices, lengths = [], []
    for f in files:
        table = pq.read_table(f, columns=["episode_index", "length"])
        indices += table.column("episode_index").to_pylist()
        lengths += table.column("length").to_pylist()
    return np.array(lengths)[np.argsort(indices)]


def build_mapping(source_repo_ids, merged_repo_id):
    """(repo_id, first_merged_index, n_episodes) per source, verified against the merge."""
    per_source = [(repo_id, episode_lengths(repo_id)) for repo_id in source_repo_ids]
    merged = episode_lengths(merged_repo_id)
    concatenated = np.concatenate([lengths for _, lengths in per_source])

    n = min(len(merged), len(concatenated))
    mismatch = np.flatnonzero(merged[:n] != concatenated[:n])
    if len(mismatch):
        raise ValueError(
            f"Source episodes no longer line up with '{merged_repo_id}': episode lengths first "
            f"differ at merged episode {mismatch[0]}. The sources have changed since the merge, "
            f"so indices cannot be mapped back."
        )
    if len(merged) != len(concatenated):
        print(
            f"Warning: {merged_repo_id} has {len(merged)} episodes but its sources now total "
            f"{len(concatenated)}. The first {n} line up, so mapping is only trustworthy below "
            f"merged episode {n}."
        )

    spans, start = [], 0
    for repo_id, lengths in per_source:
        spans.append((repo_id, start, len(lengths)))
        start += len(lengths)
    return spans, len(merged)


def format_ranges(values):
    """Collapse a sorted index list into ints and "first-last" range strings."""
    out, i = [], 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[j] + 1:
            j += 1
        out.append(values[i] if i == j else f"{values[i]}-{values[j]}")
        i = j + 1
    return out


def episodes_from_clusters(clusters_json, cluster_ids):
    with open(clusters_json) as f:
        data = json.load(f)
    wanted = set(cluster_ids)
    episodes = set()
    for cluster in data["clusters"]:
        if cluster["cluster"] not in wanted:
            continue
        for name in cluster["images"]:
            m = EPISODE_FILENAME_RE.search(name)
            if not m:
                raise ValueError(f"Image name '{name}' is not of the form episode_<index>.<ext>")
            episodes.add(int(m.group(1)))
    missing = wanted - {c["cluster"] for c in data["clusters"]}
    if missing:
        raise ValueError(f"{clusters_json} has no cluster(s) {sorted(missing)}")
    return sorted(episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True, help="recipe the merged dataset was built from")
    parser.add_argument("--merged_repo_id", help="merged dataset id (default: the recipe's output_repo_id)")
    parser.add_argument("--episodes", help="merged episode indices, e.g. '3,10-14,27'")
    parser.add_argument("--clusters_json", help="clusters.json from cluster_images_clip.py")
    parser.add_argument("--clusters", type=int, nargs="+", help="cluster ids within --clusters_json to map")
    parser.add_argument("--output_json", help="also write the mapping to this json file")
    args = parser.parse_args()

    if bool(args.episodes) == bool(args.clusters_json):
        parser.error("give exactly one of --episodes or --clusters_json")
    if args.clusters_json and not args.clusters:
        parser.error("--clusters_json requires --clusters")

    recipe = load_recipe(Path(args.recipe))
    source_repo_ids = [s["repo_id"] for s in recipe["merge"]]
    merged_repo_id = args.merged_repo_id or recipe["output_repo_id"]

    if args.episodes:
        merged_episodes = parse_episode_list(args.episodes.split(","))
    else:
        merged_episodes = episodes_from_clusters(args.clusters_json, args.clusters)

    print(f"Mapping {len(merged_episodes)} episode(s) of '{merged_repo_id}' "
          f"back through {len(source_repo_ids)} source(s)...")
    spans, merged_total = build_mapping(source_repo_ids, merged_repo_id)

    print("\nSource layout in the merged dataset:")
    for repo_id, start, count in spans:
        print(f"  {repo_id:44s} merged {start}..{start + count - 1}  ({count} episodes)")

    per_source = {repo_id: [] for repo_id, _, _ in spans}
    unmapped = []
    for ep in merged_episodes:
        for repo_id, start, count in spans:
            if start <= ep < start + count:
                per_source[repo_id].append(ep - start)
                break
        else:
            unmapped.append(ep)

    if unmapped:
        print(f"\nWarning: {len(unmapped)} episode(s) fall outside the sources "
              f"(merged dataset has {merged_total} episodes): {unmapped}")

    print("\nBad episodes by source:")
    for repo_id, _, _ in spans:
        eps = sorted(per_source[repo_id])
        if eps:
            print(f"  {repo_id}: {len(eps)} episodes -> {format_ranges(eps)}")

    print("\nRecipe merge block:\n")
    print("merge:")
    for repo_id, _, _ in spans:
        eps = sorted(per_source[repo_id])
        if not eps:
            print(f"  - {repo_id}")
            continue
        ranges = ", ".join(f'"{r}"' if isinstance(r, str) else str(r) for r in format_ranges(eps))
        print(f"  - repo_id: {repo_id}")
        print(f"    exclude_episodes: [{ranges}]")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({
                "merged_repo_id": merged_repo_id,
                "merged_episodes": merged_episodes,
                "sources": [
                    {"repo_id": repo_id, "merged_first_index": start, "num_episodes": count,
                     "bad_episodes": sorted(per_source[repo_id])}
                    for repo_id, start, count in spans
                ],
            }, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
