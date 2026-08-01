#!/usr/bin/env python

"""Build a LeRobot dataset from other datasets by following a declarative recipe.

This orchestrates the individual dataset tools in this folder into one offline
pipeline, so a merged/derived dataset can be produced from a single recipe file
without any intermediate uploads to the Hugging Face Hub.

Pipeline (in order):
  1. Source each input dataset from the Hub (LeRobotDataset downloads into the
     standard HF cache, i.e. $HF_LEROBOT_HOME / $HF_HOME - point those at your
     external drive if space is tight; nothing is re-downloaded if already cached).
  2. Convert each source into the target camera_mode (drop/resize/crop cameras),
     optionally trimming observation.state and normalizing task strings at the
     same time. This happens FIRST, before the merge, because it shrinks each
     source and the merge only needs matching feature sets - and normalizing
     tasks per source is what makes the merged task table canonical.
  3. Drop each source's excluded (bad) episodes, if the recipe lists any.
  4. Merge the remaining datasets into one (fully offline via merge_datasets).
  5. Optionally run contact-action labeling on the merged dataset.
  6. Optionally recompute stats.
  7. Optionally upload the final dataset to the Hub under output_repo_id.

A validity check runs after every intermediate step; if any produced dataset is
invalid the whole pipeline aborts before doing more work.

Intermediate (per-source converted) datasets are written under --temp_dir so the
space cost can be placed on a drive of your choosing. The final dataset is written
to --output_root.

Recipe format (YAML or JSON), e.g. recipe.yaml:

    output_repo_id: naavox/derivation_test   # id for the final dataset
    camera_mode: gripper_floor_384           # target camera format (see stringman_lerobot._CAMERA_MODES)
    center_crop: false                       # center-crop to target aspect instead of stretching (optional)
    pad_clamp: false                          # center + clamp-pad instead of stretching when target exceeds source (optional)
    merge:                                    # source datasets to merge (>= 1)
      - naavox/test_dataset_3                 # plain repo id: use every episode
      - repo_id: naavox/laptop_test_dataset   # or a mapping, to drop bad episodes
        exclude_episodes: [3, "10-14", 27]    # ints and inclusive "first-last" ranges
    keep_state_features:                      # optional; trim observation.state to these
      - vel_x
      - vel_y
    normalize_tasks: tasks.yaml               # optional; path (relative to the recipe) to a
                                              # task mapping file, or the same {tasks:, map:}
                                              # spec inline. See lerobot_normalize_tasks.py.
    label_contact_actions:                    # optional; omit or set enabled: false to skip
      enabled: true
      pressure_threshold: 0.1
      episode_end_seconds: 1.0
      blend_seconds: 0.5
      rotate_contact_vec: false
    recompute_stats: false                    # optional extra full stats pass (merge already writes stats)
    upload: false                             # optional; may also be forced with --upload

Usage:
    python src/nf_robot/ml/lerobot_build_dataset.py \
        --recipe recipe.yaml \
        --temp_dir /media/nhn/nfdrive/tmp_build \
        --output_root /media/nhn/nfdrive/datasets/derivation_test \
        [--upload]
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets, recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from nf_robot.ml.lerobot_derive_dataset import derive_dataset
from nf_robot.ml.lerobot_label_contact_actions import label_dataset
from nf_robot.ml.lerobot_normalize_tasks import load_mapping
from nf_robot.ml.stringman_lerobot import _CAMERA_MODES, camera_mode_from_features


def _short_name(repo_id: str) -> str:
    """Last path component of a repo id, usable as a directory name."""
    return repo_id.split("/")[-1]


# Video-feature "info" fields that actually determine data compatibility. Other
# keys (video.g/crf/preset/fast_decode/video_backend/extra_options, ...) are
# encoder bookkeeping that varies with the lerobot version a dataset was recorded
# under and does not affect merged data.
_CRITICAL_VIDEO_INFO_KEYS = (
    "video.height",
    "video.width",
    "video.codec",
    "video.pix_fmt",
    "video.fps",
    "video.channels",
)


def normalize_video_info(converted: list[tuple[str, Path]]) -> None:
    """Make the video-feature `info` dicts byte-identical across datasets.

    lerobot's merge requires every source's feature dict to be exactly equal,
    but datasets recorded under different versions carry different (cosmetic)
    encoder metadata. This verifies the critical fields (resolution, codec,
    pixel format, fps) match across all datasets - raising if they genuinely
    differ - then overwrites each dataset's `info` for that feature with a single
    canonical copy so the merge's equality check passes.
    """
    infos = []
    for repo_id, root in converted:
        path = root / "meta" / "info.json"
        infos.append((repo_id, path, json.loads(path.read_text())))

    _, _, ref_info = infos[0]
    video_keys = [k for k, f in ref_info["features"].items() if f.get("dtype") == "video"]

    for key in video_keys:
        canonical = ref_info["features"][key].get("info", {})
        for repo_id, _, info in infos:
            other = info["features"][key].get("info", {})
            for ck in _CRITICAL_VIDEO_INFO_KEYS:
                if other.get(ck) != canonical.get(ck):
                    raise ValueError(
                        f"Datasets are not mergeable: '{key}' {ck} differs "
                        f"({infos[0][0]}={canonical.get(ck)} vs {repo_id}={other.get(ck)})"
                    )
        # Critical fields agree; unify the whole info dict so merge sees equality.
        for _, _, info in infos:
            info["features"][key]["info"] = canonical

    for _, path, info in infos:
        path.write_text(json.dumps(info, indent=4))


def parse_episode_list(spec) -> list[int]:
    """Expand an episode spec into a sorted list of unique episode indices.

    Accepts ints and inclusive "first-last" range strings, so long runs of bad
    episodes stay readable in a recipe: [3, "10-14", 27] -> [3, 10, 11, 12, 13, 14, 27].
    """
    if spec is None:
        return []
    if not isinstance(spec, list):
        raise ValueError(f"'exclude_episodes' must be a list, got {spec!r}")

    out: set[int] = set()
    for item in spec:
        if isinstance(item, bool):
            raise ValueError(f"Invalid episode spec {item!r}")
        if isinstance(item, int):
            out.add(item)
            continue
        if isinstance(item, str) and "-" in item.strip().lstrip("-"):
            first, _, last = item.strip().partition("-")
            try:
                first, last = int(first), int(last)
            except ValueError:
                raise ValueError(f"Invalid episode range {item!r}; expected 'first-last'")
            if last < first:
                raise ValueError(f"Invalid episode range {item!r}; last < first")
            out.update(range(first, last + 1))
            continue
        try:
            out.add(int(item))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid episode spec {item!r}; expected an int or 'first-last'")

    if any(i < 0 for i in out):
        raise ValueError(f"Episode indices must be non-negative: {sorted(i for i in out if i < 0)}")
    return sorted(out)


def normalize_sources(merge_spec: list) -> list[dict]:
    """Normalize recipe 'merge' entries to {'repo_id': str, 'exclude_episodes': [int]}."""
    sources = []
    for entry in merge_spec:
        if isinstance(entry, str):
            sources.append({"repo_id": entry, "exclude_episodes": []})
            continue
        if not isinstance(entry, dict) or "repo_id" not in entry:
            raise ValueError(
                f"Each 'merge' entry must be a repo id string or a mapping with 'repo_id', got {entry!r}"
            )
        unknown = set(entry) - {"repo_id", "exclude_episodes"}
        if unknown:
            raise ValueError(f"Unknown keys in merge entry {entry['repo_id']}: {sorted(unknown)}")
        sources.append({
            "repo_id": entry["repo_id"],
            "exclude_episodes": parse_episode_list(entry.get("exclude_episodes")),
        })

    duplicates = {s["repo_id"] for s in sources if [t["repo_id"] for t in sources].count(s["repo_id"]) > 1}
    if duplicates:
        raise ValueError(f"Repeated source repo ids in 'merge': {sorted(duplicates)}")
    return sources


def load_recipe(path: Path) -> dict:
    """Load and validate a build recipe from YAML or JSON."""
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        recipe = yaml.safe_load(text)
    else:
        recipe = json.loads(text)

    if not isinstance(recipe, dict):
        raise ValueError(f"Recipe {path} must be a mapping at the top level")

    for key in ("output_repo_id", "camera_mode", "merge"):
        if key not in recipe:
            raise ValueError(f"Recipe is missing required key '{key}'")

    if recipe["camera_mode"] not in _CAMERA_MODES:
        raise ValueError(
            f"Unknown camera_mode '{recipe['camera_mode']}'. Valid: {list(_CAMERA_MODES)}"
        )

    if not isinstance(recipe["merge"], list) or not recipe["merge"]:
        raise ValueError("Recipe key 'merge' must be a non-empty list of repo ids")
    # Store the normalized form so callers never deal with both entry shapes.
    recipe["merge"] = normalize_sources(recipe["merge"])

    label = recipe.get("label_contact_actions")
    if label is not None and not isinstance(label, dict):
        raise ValueError("'label_contact_actions' must be a mapping if present")

    keep_state = recipe.get("keep_state_features")
    if keep_state is not None and (
        not isinstance(keep_state, list) or not all(isinstance(n, str) for n in keep_state)
    ):
        raise ValueError("'keep_state_features' must be a list of observation.state component names")

    # A string is a mapping file, resolved relative to the recipe so a recipe and
    # its task mapping travel together; anything else must be the inline spec.
    tasks_spec = recipe.get("normalize_tasks")
    if isinstance(tasks_spec, str):
        mapping_path = Path(tasks_spec)
        if not mapping_path.is_absolute() and not mapping_path.exists():
            mapping_path = path.parent / mapping_path
        tasks, mapping = load_mapping(mapping_path)
        recipe["normalize_tasks"] = {"tasks": tasks, "map": mapping}
    elif tasks_spec is not None:
        if not isinstance(tasks_spec, dict) or "tasks" not in tasks_spec or "map" not in tasks_spec:
            raise ValueError("'normalize_tasks' must be a mapping-file path or a {tasks:, map:} mapping")

    return recipe


def validate_dataset(repo_id: str, root: Path, expected_camera_mode: str | None = None) -> LeRobotDataset:
    """Load a dataset and assert integrity; raise if the pipeline produced junk.

    Checks structural consistency (meta constructs, parquet row counts sum to
    total_frames, action/state feature dims match their names), the expected
    camera mode, that every referenced video file exists and opens, and that a
    frame decodes end-to-end.

    Note: this deliberately does not require each video's frame count to equal
    its episode length. Robot recordings routinely drop a few camera frames, and
    lerobot's own loader tolerates that via timestamp-with-tolerance queries, so
    a stricter check here would reject datasets that train fine.
    """
    import pyarrow.parquet as pq

    logging.info(f"Validating dataset '{repo_id}' at {root}")
    ds = LeRobotDataset(repo_id=repo_id, root=root)

    if ds.meta.total_episodes <= 0:
        raise ValueError(f"Dataset '{repo_id}' has no episodes")
    if ds.meta.total_frames <= 0:
        raise ValueError(f"Dataset '{repo_id}' has no frames")

    # Structural: parquet rows must sum to the declared total_frames. This catches
    # a botched merge/reindex, which is exactly what these pipeline steps risk.
    data_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        raise ValueError(f"Dataset '{repo_id}' has no data parquet files under {root}/data")
    total_rows = sum(pq.read_metadata(f).num_rows for f in data_files)
    if total_rows != ds.meta.total_frames:
        raise ValueError(
            f"Dataset '{repo_id}' parquet rows ({total_rows}) != total_frames ({ds.meta.total_frames})"
        )

    # Structural: 1-D vector features must have shape matching their names length.
    # Catches contact-labeling leaving action shape/names inconsistent.
    for key in ("action", "observation.state"):
        feat = ds.meta.features.get(key)
        if feat and "names" in feat and feat.get("names") is not None:
            if list(feat["shape"]) != [len(feat["names"])]:
                raise ValueError(
                    f"Dataset '{repo_id}' feature '{key}' shape {feat['shape']} "
                    f"does not match {len(feat['names'])} names"
                )

    if expected_camera_mode is not None:
        found = camera_mode_from_features(ds.meta.features)
        if found != expected_camera_mode:
            raise ValueError(
                f"Dataset '{repo_id}' has camera_mode '{found}', expected '{expected_camera_mode}'"
            )

    # Every referenced video file must exist and open with a video stream.
    import av

    for ep_idx in range(ds.meta.total_episodes):
        for key in ds.meta.video_keys:
            try:
                rel = ds.meta.get_video_file_path(ep_idx, key)
            except KeyError:
                continue
            path = root / rel
            if not path.exists():
                raise ValueError(f"Dataset '{repo_id}' missing video file {rel}")
    for key in ds.meta.video_keys:
        rel = ds.meta.get_video_file_path(0, key)
        with av.open(str(root / rel)) as container:
            if not container.streams.video:
                raise ValueError(f"Dataset '{repo_id}' video {rel} has no video stream")

    # End-to-end decode of an early frame (frame 0 is always present even when a
    # recording dropped frames near an episode's end).
    item = ds[0]
    for key in ds.meta.video_keys:
        if key not in item:
            raise ValueError(f"Dataset '{repo_id}' frame 0 missing video key '{key}'")

    logging.info(
        f"OK: '{repo_id}' -> {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames, "
        f"cameras {list(ds.meta.video_keys)}"
    )
    return ds


def build(
    recipe: dict,
    temp_dir: Path,
    output_root: Path,
    upload: bool,
    headroom: int,
    keep_intermediate: bool,
) -> LeRobotDataset:
    output_repo_id = recipe["output_repo_id"]
    camera_mode = recipe["camera_mode"]
    center_crop = bool(recipe.get("center_crop", False))
    pad_clamp = bool(recipe.get("pad_clamp", False))
    sources = normalize_sources(recipe["merge"])
    keep_state_features = recipe.get("keep_state_features")
    normalize_tasks_spec = recipe.get("normalize_tasks")
    label_cfg = recipe.get("label_contact_actions") or {}
    do_label = bool(label_cfg.get("enabled", False))
    do_recompute = bool(recipe.get("recompute_stats", False))
    do_upload = upload or bool(recipe.get("upload", False))

    temp_dir.mkdir(parents=True, exist_ok=True)
    converted_root_base = temp_dir / "converted"

    if output_root.exists():
        raise FileExistsError(f"--output_root {output_root} already exists; remove it or pick another path")

    # Step 1 + 2: source each dataset and convert it to the target camera_mode.
    converted: list[tuple[str, Path]] = []
    for source_spec in sources:
        repo_id = source_spec["repo_id"]
        exclude = source_spec["exclude_episodes"]
        logging.info(f"[{repo_id}] sourcing from Hub cache and converting to '{camera_mode}'")
        source = LeRobotDataset(repo_id=repo_id, root=None)  # downloads/caches under HF_LEROBOT_HOME

        converted_root = converted_root_base / _short_name(repo_id)
        if converted_root.exists():
            shutil.rmtree(converted_root)

        derive_dataset(
            dataset=source,
            camera_mode=camera_mode,
            output_dir=converted_root,
            repo_id=repo_id,
            headroom=headroom,
            center_crop=center_crop,
            pad_clamp=pad_clamp,
            keep_state_features=keep_state_features,
            normalize_tasks_spec=normalize_tasks_spec,
        )
        validate_dataset(repo_id, converted_root, expected_camera_mode=camera_mode)

        # Step 3: drop excluded episodes. Done after conversion so the re-encoding
        # delete_episodes does on videos straddling kept/dropped episodes works on
        # the small target-resolution videos rather than the full-size originals.
        if exclude:
            logging.info(f"[{repo_id}] excluding {len(exclude)} episode(s): {exclude}")
            filtered_root = converted_root_base / f"{_short_name(repo_id)}_filtered"
            if filtered_root.exists():
                shutil.rmtree(filtered_root)
            delete_episodes(
                dataset=LeRobotDataset(repo_id=repo_id, root=converted_root),
                episode_indices=exclude,
                output_dir=filtered_root,
                repo_id=repo_id,
            )
            validate_dataset(repo_id, filtered_root, expected_camera_mode=camera_mode)
            shutil.rmtree(converted_root, ignore_errors=True)
            converted_root = filtered_root

        converted.append((repo_id, converted_root))

    # Step 4: merge the converted datasets into the final output location.
    # First reconcile cosmetic video-metadata differences so the merge's strict
    # feature-equality check passes (raises if critical fields truly differ).
    if len(converted) > 1:
        normalize_video_info(converted)
    logging.info(f"Merging {len(converted)} datasets into '{output_repo_id}' at {output_root}")
    converted_datasets = [LeRobotDataset(repo_id=rid, root=root) for rid, root in converted]
    merge_datasets(converted_datasets, output_repo_id=output_repo_id, output_dir=output_root)
    validate_dataset(output_repo_id, output_root, expected_camera_mode=camera_mode)

    # Step 5: optional contact-action labeling (rewrites the action column + its stats).
    if do_label:
        logging.info(f"Labeling contact actions on '{output_repo_id}'")
        label_dataset(
            root=output_root,
            pressure_threshold=float(label_cfg.get("pressure_threshold", 0.1)),
            episode_end_seconds=float(label_cfg.get("episode_end_seconds", 1.0)),
            rotate_contact_vec=bool(label_cfg.get("rotate_contact_vec", False)),
            blend_seconds=float(label_cfg.get("blend_seconds", 0.5)),
        )
        validate_dataset(output_repo_id, output_root, expected_camera_mode=camera_mode)

    # Step 6: optional full stats recompute (merge already aggregates stats, and
    # labeling recomputes the action stats, so this is off by default).
    if do_recompute:
        logging.info(f"Recomputing stats for '{output_repo_id}'")
        final = LeRobotDataset(repo_id=output_repo_id, root=output_root)
        recompute_stats(final)
        validate_dataset(output_repo_id, output_root, expected_camera_mode=camera_mode)

    # Clean up per-source converted intermediates now that the merge is done.
    if not keep_intermediate:
        logging.info(f"Removing intermediate datasets under {converted_root_base}")
        shutil.rmtree(converted_root_base, ignore_errors=True)

    final = LeRobotDataset(repo_id=output_repo_id, root=output_root)

    # Step 7: optional upload of the final dataset.
    if do_upload:
        logging.info(f"Pushing '{output_repo_id}' to the Hugging Face Hub")
        final.push_to_hub()

    logging.info(f"Done. Built '{output_repo_id}' at {output_root}")
    return final


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--recipe", required=True, help="Path to the recipe file (YAML or JSON)")
    parser.add_argument(
        "--temp_dir", required=True,
        help="Directory for intermediate (per-source converted) datasets; place on a drive with space",
    )
    parser.add_argument(
        "--output_root", default=None,
        help="Root path for the final dataset (default: <temp_dir>/<output name>)",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload the final dataset to the Hub (also settable via recipe 'upload: true')",
    )
    parser.add_argument(
        "--headroom", type=int, default=2,
        help="CPU cores to leave free during conversion; the rest run one single-threaded "
             "encode each (default: 0)",
    )
    parser.add_argument(
        "--keep_intermediate", action="store_true",
        help="Keep the per-source converted datasets under temp_dir instead of deleting them",
    )
    args = parser.parse_args()

    recipe_path = Path(args.recipe)
    recipe = load_recipe(recipe_path)

    temp_dir = Path(args.temp_dir)
    output_root = (
        Path(args.output_root)
        if args.output_root
        else temp_dir / _short_name(recipe["output_repo_id"])
    )

    build(
        recipe=recipe,
        temp_dir=temp_dir,
        output_root=output_root,
        upload=args.upload,
        headroom=args.headroom,
        keep_intermediate=args.keep_intermediate,
    )


if __name__ == "__main__":
    main()
