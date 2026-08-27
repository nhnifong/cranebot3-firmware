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
     optionally re-blending its ortho floor view before the anchor cameras that
     view is made of are dropped, and optionally trimming observation.state and
     normalizing task strings at the same time. This happens FIRST, before the merge, because it shrinks each
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
      - repo_id: naavox/laptop_test_dataset   # or a mapping, to select episodes
        anchor_config: conf_robot.json        # only for action_space: camera_goal, and only
                                              # for datasets recorded before they carried their
                                              # own anchor_poses feature
        include_episodes: ["0-99"]            # optional; keep only these (default: all)
        exclude_episodes: [3, "10-14", 27]    # dropped on top of include_episodes
                                              # both take ints and inclusive "first-last" ranges.
                                              # See lerobot_find_frozen_video.py, which prints
                                              # this block for episodes with a frozen camera.
                                              # A repo id may be listed more than once, to split
                                              # a dataset recorded by two robots into runs.
    keep_state_features:                      # optional; trim observation.state to these
      - vel_x
      - vel_y
    normalize_tasks: tasks.yaml               # optional; path (as given, else relative to the recipe) to a
                                              # task mapping file, or the same {tasks:, map:}
                                              # spec inline. See lerobot_normalize_tasks.py.
    drop_features:                            # optional; non-video features to drop from every
      - anchor_poses                          # source, for extras that only some sources carry
                                              # (the merge demands identical feature sets)
    backfill_anchor_poses: true               # optional; the other way to settle that mismatch
                                              # for anchor_poses - a source recorded before the
                                              # feature existed gains it from its anchor_config,
                                              # instead of the sources that have it losing it
    reblend_ortho:                            # optional; re-render the ortho floor view from
      enabled: true                           # each source's anchor cameras with today's blend,
      render_fps: 10                          # instead of keeping the one it was recorded with.
      cam_tilt: [30, 26]                      # See lerobot_reblend_ortho.py. Every source needs
                                              # anchor poses, from its own anchor_poses feature
                                              # or from an anchor_config, exactly as camera_goal
                                              # does. cam_tilt is optional: it comes from the
                                              # anchor_config, and where there is none it is
                                              # recovered from the recorded ortho view.
    action_space: gripper_vel                 # optional; omit to keep the recorded space.
                                              # camera_goal converts; any other named space is a
                                              # subset of the recorded action components and is
                                              # trimmed to. See stringman_lerobot._ACTION_SPACES.
    trim_to_grasp:                            # optional; cut each episode to just its grasp
      enabled: true                           # (see lerobot_trim_to_grasp.py). Runs on the merged
      pressure_threshold: 0.1                 # dataset and re-encodes all video, so it is slow.
      min_grasp_seconds: 0.3                  # Needs gripper_pos_*/finger_pressure in
      rise_m: 0.10                            # keep_state_features.
      near_radius_m: 0.3
      min_segment_seconds: 1.0
      post_lift_seconds: 1.0                  # carry kept after the lift
    label_contact_actions:                    # optional; omit or set enabled: false to skip
      enabled: true
      mode: contact                           # or "waypoints": target the next position the
                                              # gantry actually stopped at, rather than the
                                              # grasp blended into the episode end
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
import os
import shutil
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets, recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from nf_robot.ml.lerobot_derive_dataset import camera_mode_problems, derive_dataset
from nf_robot.ml.lerobot_label_contact_actions import label_dataset
from nf_robot.ml import camera_goal
from nf_robot.ml import lerobot_reblend_ortho as reblend_ortho
from nf_robot.ml.lerobot_normalize_tasks import load_mapping
from nf_robot.ml.stringman_lerobot import (
    _ACTION_SPACES, _CAMERA_MODES, _FEED_NAMES, camera_mode_from_features,
)
from nf_robot.ml.lerobot_repair_episode_meta import repair as repair_episode_meta
from nf_robot.ml.lerobot_trim_to_grasp import trim_dataset_to_grasp

# The AV1 encoder prints a twenty line configuration banner per video file, straight to
# stderr from libSvtAv1Enc rather than through ffmpeg's log system - so av.logging cannot
# reach it and a merge of a few hundred episodes buries its own progress. SVT_LOG is the
# library's own level: 1 is errors only. Exported before the run to keep the banner.
os.environ.setdefault("SVT_LOG", "1")


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


def resolve_source_revision(repo_id: str) -> str | None:
    """Revision to load a source dataset at, or None for lerobot's default.

    lerobot resolves a dataset to its codebase-version tag (v3.0). A dataset
    published without that tag makes get_safe_version raise RevisionNotFoundError
    - and with huggingface_hub >=1.0 that raise itself dies with "missing keyword
    argument 'response'", burying the real cause. Fall back to main for untagged
    repos; the format check in _load_metadata still rejects genuinely old data.
    """
    from lerobot.datasets.utils import get_repo_versions

    if get_repo_versions(repo_id):
        return None
    logging.warning(
        f"[{repo_id}] has no codebase-version tag on the Hub; loading from 'main'. "
        f"Tag it with: HfApi().create_tag('{repo_id}', tag='v3.0', repo_type='dataset')"
    )
    return "main"


def preflight_sources(sources: list[dict], action_space: str | None = None,
                      reblend: bool = False, backfill: bool = False,
                      camera_mode: str | None = None,
                      pad_clamp: bool = False) -> dict[str, str | None]:
    """Resolve every source's revision before any conversion work starts.

    Conversion takes hours; reaching a broken source at the end wastes all of it.
    These are metadata-only Hub calls, so checking all of them up front is cheap.

    For camera_goal, the ortho re-blend and the anchor_poses backfill, this also settles
    where each source's anchor poses come from: the dataset's own anchor_poses feature, or an anchor_config
    naming the calibration the robot was running. A source with neither cannot be
    converted. The re-blend additionally needs the anchor cameras it rebuilds the view
    out of, which a source recorded with a narrower camera_mode would not have.

    Every source is also checked against the target camera_mode here, and every one that
    fails is reported together: a recipe with one unusable source among twenty should say
    so in one go, before the first conversion, rather than after each hour of work.
    """
    from huggingface_hub import hf_hub_download

    needs_poses = reblend or backfill or action_space == camera_goal.ACTION_SPACE_NAME

    revisions: dict[str, str | None] = {}
    self_describing = []
    unusable: list[str] = []
    for spec in sources:
        repo_id = spec["repo_id"]
        try:
            revisions[repo_id] = resolve_source_revision(repo_id)
        except Exception as e:
            raise RuntimeError(f"Source dataset '{repo_id}' is not usable: {type(e).__name__}: {e}") from e

        info = json.loads(Path(hf_hub_download(
            repo_id=repo_id, filename="meta/info.json", repo_type="dataset",
            revision=revisions[repo_id],
        )).read_text())

        if camera_mode is not None:
            problems = camera_mode_problems(info["features"], camera_mode, pad_clamp)
            if problems:
                unusable.append(f"  {spec['name']} ({repo_id}): " + "; ".join(problems))

        if not needs_poses:
            continue

        if reblend:
            missing = [k for k in reblend_ortho.ANCHOR_KEYS + (reblend_ortho.ORTHO_KEY,)
                       if k not in info["features"]]
            if missing:
                raise ValueError(
                    f"Source '{repo_id}' has no {missing}, so its ortho view cannot be "
                    f"re-blended. Drop 'reblend_ortho' or drop this source."
                )

        if spec.get("anchor_config"):
            continue
        if camera_goal.ANCHOR_POSES_KEY not in info["features"]:
            raise ValueError(
                f"Source '{repo_id}' has no '{camera_goal.ANCHOR_POSES_KEY}' feature and no "
                f"'anchor_config' in the recipe, so nothing says where its anchor cameras "
                f"were. Datasets recorded before that feature existed need the config of "
                f"the robot that recorded them."
            )
        self_describing.append(repo_id)

    if unusable:
        raise ValueError(
            f"{len(unusable)} of {len(sources)} source(s) cannot be converted to "
            f"camera_mode '{camera_mode}'. Drop them from the recipe's 'merge' list, or "
            f"change the recipe so they fit:\n" + "\n".join(unusable)
        )

    logging.info(f"Preflight OK: {len(revisions)} source dataset(s) reachable")
    if self_describing:
        logging.info(f"{len(self_describing)} source(s) carry their own anchor poses: {self_describing}")
    return revisions


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
    """Normalize recipe 'merge' entries to a uniform dict per entry.

    A repo id may appear more than once, so a dataset recorded by two robots can be
    split into runs of episodes that each carry their own anchor_config. Repeats get
    a suffixed 'name', which is what the intermediate directories are keyed on -
    keying them on the repo id would make the second entry overwrite the first.
    """
    sources = []
    # Names of every entry, so a suffixed repeat cannot land on another source's name:
    # a second naavox/move_clutter entry must not be called move_clutter_2 when
    # naavox/move_clutter_2 is also being merged.
    taken = {_short_name(e if isinstance(e, str) else e.get("repo_id", "")) for e in merge_spec}
    sources = []
    used_names: dict[str, int] = {}
    for entry in merge_spec:
        if isinstance(entry, str):
            entry = {"repo_id": entry}
        if not isinstance(entry, dict) or "repo_id" not in entry:
            raise ValueError(
                f"Each 'merge' entry must be a repo id string or a mapping with 'repo_id', got {entry!r}"
            )
        # "name" is produced below, and is accepted on the way back in so that
        # normalizing an already-normalized list is a no-op: load_recipe stores its
        # result in the recipe, and build normalizes again.
        unknown = set(entry) - {"repo_id", "exclude_episodes", "include_episodes", "anchor_config", "name"}
        if unknown:
            raise ValueError(f"Unknown keys in merge entry {entry['repo_id']}: {sorted(unknown)}")

        base = _short_name(entry["repo_id"])
        used_names[base] = used_names.get(base, 0) + 1
        name = base
        suffix = used_names[base]
        while suffix > 1 or name in [s_["name"] for s_ in sources]:
            name = f"{base}_part{suffix}"
            if name not in taken and name not in [s_["name"] for s_ in sources]:
                break
            suffix += 1
        taken.add(name)

        include = parse_episode_list(entry.get("include_episodes"))
        exclude = parse_episode_list(entry.get("exclude_episodes"))
        overlap = sorted(set(include) & set(exclude))
        if overlap and len(overlap) == len(include):
            raise ValueError(
                f"merge entry {entry['repo_id']} excludes every episode it includes"
            )
        sources.append({
            "repo_id": entry["repo_id"],
            "name": name,
            # empty include means "all of them"; exclude applies on top either way
            "include_episodes": include,
            "exclude_episodes": exclude,
            # config of the robot that recorded this source, for action spaces that
            # need its anchor camera poses (camera_goal)
            "anchor_config": entry.get("anchor_config"),
        })
    return sources


def episodes_to_drop(total_episodes: int, include: list[int], exclude: list[int]) -> list[int]:
    """Episodes to delete from a source so only the wanted ones remain."""
    out_of_range = [e for e in include + exclude if e >= total_episodes]
    if out_of_range:
        raise ValueError(
            f"episodes {sorted(set(out_of_range))} are past the end of a source with "
            f"{total_episodes} episodes"
        )
    keep = set(include) if include else set(range(total_episodes))
    return sorted((set(range(total_episodes)) - keep) | set(exclude))


def _resolve_recipe_path(value: str, recipe_path: Path, key: str) -> Path:
    """Resolve a path named in a recipe, as given or relative to the recipe itself."""
    candidate = Path(value)
    tried = [candidate]
    if not candidate.is_absolute() and not candidate.exists():
        candidate = recipe_path.parent / candidate
        tried.append(candidate)
    if not candidate.exists():
        raise FileNotFoundError(
            f"Recipe key '{key}' points at {value!r}, which does not exist. Tried: "
            + ", ".join(str(t.resolve()) for t in tried)
        )
    return candidate


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

    action_space = recipe.get("action_space")
    if action_space is not None and action_space not in _ACTION_SPACES:
        raise ValueError(
            f"Unknown action_space '{action_space}'. Omit it to keep the recorded space, "
            f"or use one of {sorted(_ACTION_SPACES)}"
        )
    reblend = recipe.get("reblend_ortho")
    if reblend is not None and not isinstance(reblend, dict):
        raise ValueError("'reblend_ortho' must be a mapping if present")
    if reblend and reblend.get("enabled"):
        unknown = set(reblend) - {"enabled", "render_fps", "cam_tilt", "check"}
        if unknown:
            raise ValueError(f"Unknown keys in 'reblend_ortho': {sorted(unknown)}")
        if reblend_ortho.ORTHO_KEY not in {
            f"observation.images.{_FEED_NAMES[feed]}" for feed in _CAMERA_MODES[recipe["camera_mode"]]
        }:
            raise ValueError(
                f"'reblend_ortho' rebuilds '{reblend_ortho.ORTHO_KEY}', which camera_mode "
                f"'{recipe['camera_mode']}' does not keep"
            )

    backfill = bool(recipe.get("backfill_anchor_poses", False))
    if backfill and camera_goal.ANCHOR_POSES_KEY in (recipe.get("drop_features") or []):
        raise ValueError(
            f"'backfill_anchor_poses' and dropping '{camera_goal.ANCHOR_POSES_KEY}' in "
            f"'drop_features' ask for opposite things"
        )

    # anchor_config supplies the anchor camera poses, and all of these need them.
    if backfill or action_space == camera_goal.ACTION_SPACE_NAME or (reblend and reblend.get("enabled")):
        # anchor_config is optional: a dataset recorded after the anchor_poses feature
        # was added carries its own calibration, and preflight checks that every source
        # has one source of poses or the other.
        # Resolved here, not at use time, so a wrong path fails before any conversion
        # work. Tried as given first (so repo-root-relative paths work when run from
        # the repo root), then relative to the recipe, so a recipe and its calibration
        # files can travel together.
        for source in recipe["merge"]:
            if source.get("anchor_config"):
                source["anchor_config"] = _resolve_recipe_path(source["anchor_config"], path, "anchor_config")

    keep_state = recipe.get("keep_state_features")
    if keep_state is not None and (
        not isinstance(keep_state, list) or not all(isinstance(n, str) for n in keep_state)
    ):
        raise ValueError("'keep_state_features' must be a list of observation.state component names")

    # A string is a mapping file, resolved relative to the recipe so a recipe and
    # its task mapping travel together; anything else must be the inline spec.
    tasks_spec = recipe.get("normalize_tasks")
    if isinstance(tasks_spec, str):
        tasks, mapping = load_mapping(_resolve_recipe_path(tasks_spec, path, "normalize_tasks"))
        recipe["normalize_tasks"] = {"tasks": tasks, "map": mapping}
    elif tasks_spec is not None:
        if not isinstance(tasks_spec, dict) or "tasks" not in tasks_spec or "map" not in tasks_spec:
            raise ValueError("'normalize_tasks' must be a mapping-file path or a {tasks:, map:} mapping")

    return recipe


def _decode_whole_video(path_str: str) -> tuple[str, str | None]:
    """Decode every packet of one video; return (path, first error description or None).

    Module-level and picklable so it can run in a ProcessPoolExecutor.
    """
    import av

    try:
        with av.open(path_str) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            time_base = stream.time_base
            for packet in container.demux(stream):
                try:
                    for _ in packet.decode():
                        pass
                except Exception as e:  # noqa: BLE001 - any decode failure is a failure
                    at = f"{float(packet.pts * time_base):.2f}s" if packet.pts is not None else "unknown offset"
                    return path_str, f"{type(e).__name__} at {at}: {e}"
    except Exception as e:  # noqa: BLE001 - container-level failure counts too
        return path_str, f"{type(e).__name__}: {e}"
    return path_str, None


def verify_videos_decode(root: Path, repo_id: str, video_keys: list[str]) -> None:
    """Fully decode every video file, raising on the first that fails.

    The cheap frame-0 check in validate_dataset cannot see damage in the middle of a
    file, and re-encoding is exactly what produces that kind of damage: a source that
    decodes clean can come out of the AV1 encode with a handful of broken packets
    somewhere in the middle. That survives the build, uploads, and then blows up
    hundreds of steps into training when a sampler finally lands on those frames.
    """
    from concurrent.futures import ProcessPoolExecutor

    files = sorted({p for key in video_keys for p in (root / "videos" / key).glob("**/*.mp4")})
    if not files:
        return

    logging.info(f"Verifying {len(files)} video file(s) of '{repo_id}' decode end to end")
    workers = max(1, min(os.cpu_count() or 1, len(files)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path_str, error in pool.map(_decode_whole_video, [str(p) for p in files]):
            if error is not None:
                raise ValueError(
                    f"Dataset '{repo_id}' has undecodable video: "
                    f"{Path(path_str).relative_to(root)} -> {error}"
                )
    logging.info(f"OK: all {len(files)} video file(s) of '{repo_id}' decode cleanly")


def validate_dataset(
    repo_id: str,
    root: Path,
    expected_camera_mode: str | None = None,
    decode_videos: bool = False,
) -> LeRobotDataset:
    """Load a dataset and assert integrity; raise if the pipeline produced junk.

    Checks structural consistency (meta constructs, parquet row counts sum to
    total_frames, action/state feature dims match their names), the expected
    camera mode, that every referenced video file exists and opens, and that a
    frame decodes end-to-end.

    decode_videos additionally decodes every video file in full. That is the only
    check that catches corruption in the middle of a re-encoded file, so it is worth
    its cost after any step that re-encodes video.

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

    if decode_videos:
        verify_videos_decode(root, repo_id, list(ds.meta.video_keys))

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
    resume: bool = False,
    verify_decode: bool = True,
) -> LeRobotDataset:
    output_repo_id = recipe["output_repo_id"]
    camera_mode = recipe["camera_mode"]
    center_crop = bool(recipe.get("center_crop", False))
    pad_clamp = bool(recipe.get("pad_clamp", False))
    sources = normalize_sources(recipe["merge"])
    action_space = recipe.get("action_space")
    keep_state_features = recipe.get("keep_state_features")
    normalize_tasks_spec = recipe.get("normalize_tasks")
    drop_features = recipe.get("drop_features")
    reblend_cfg = recipe.get("reblend_ortho") or {}
    do_reblend = bool(reblend_cfg.get("enabled", False))
    backfill_anchor_poses = bool(recipe.get("backfill_anchor_poses", False))
    trim_cfg = recipe.get("trim_to_grasp") or {}
    label_cfg = recipe.get("label_contact_actions") or {}
    do_label = bool(label_cfg.get("enabled", False))
    do_recompute = bool(recipe.get("recompute_stats", False))
    do_upload = upload or bool(recipe.get("upload", False))

    temp_dir.mkdir(parents=True, exist_ok=True)
    converted_root_base = temp_dir / "converted"

    if output_root.exists():
        raise FileExistsError(f"--output_root {output_root} already exists; remove it or pick another path")

    revisions = preflight_sources(sources, action_space, reblend=do_reblend,
                                  backfill=backfill_anchor_poses, camera_mode=camera_mode,
                                  pad_clamp=pad_clamp)

    # Step 1 + 2: source each dataset and convert it to the target camera_mode.
    converted: list[tuple[str, Path]] = []
    for source_spec in sources:
        repo_id = source_spec["repo_id"]
        exclude = source_spec["exclude_episodes"]
        include = source_spec["include_episodes"]
        name = source_spec["name"]

        converted_root = converted_root_base / name
        filtered_root = converted_root_base / f"{name}_filtered"
        # What this source contributes to the merge: the filtered copy when episodes
        # are selected away, else the converted one.
        final_root = filtered_root if (exclude or include) else converted_root

        if resume and final_root.exists():
            try:
                validate_dataset(repo_id, final_root, expected_camera_mode=camera_mode,
                                 decode_videos=verify_decode)
                logging.info(f"[{name}] reusing existing conversion at {final_root}")
                converted.append((repo_id, final_root))
                continue
            except Exception as e:
                logging.warning(f"[{name}] existing conversion at {final_root} is unusable ({e}); redoing it")
                shutil.rmtree(final_root, ignore_errors=True)

        # Any failure in here belongs to one named source, and a recipe with twenty of
        # them should say which - the traceback out of lerobot or ffmpeg never does.
        try:
            logging.info(f"[{name}] sourcing from Hub cache and converting to '{camera_mode}'")
            source = LeRobotDataset(  # downloads/caches under HF_LEROBOT_HOME
                repo_id=repo_id, root=None, revision=revisions[repo_id]
            )

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
                # empty tuple still triggers the conversion, and means "the dataset's own
                # recorded poses are the only source"
                camera_goal_anchor_poses=(
                    (camera_goal.load_anchor_poses(source_spec["anchor_config"])
                     if source_spec["anchor_config"] else ())
                    if action_space == camera_goal.ACTION_SPACE_NAME else None
                ),
                camera_goal_label_cfg=label_cfg,
                drop_features=drop_features,
                reblend_ortho_cfg=reblend_cfg if do_reblend else None,
                anchor_config=source_spec["anchor_config"],
                backfill_anchor_poses=backfill_anchor_poses,
                # camera_goal rewrites the action space itself; every other named space is
                # a subset of the recorded components, so it's a trim.
                keep_action_features=(
                    _ACTION_SPACES[action_space]
                    if action_space and action_space != camera_goal.ACTION_SPACE_NAME else None
                ),
            )
            validate_dataset(repo_id, converted_root, expected_camera_mode=camera_mode,
                             decode_videos=verify_decode)

            # Step 3: drop excluded episodes. Done after conversion so the re-encoding
            # delete_episodes does on videos straddling kept/dropped episodes works on
            # the small target-resolution videos rather than the full-size originals.
            if exclude or include:
                drop = episodes_to_drop(source.meta.total_episodes, include, exclude)
                logging.info(
                    f"[{name}] keeping {source.meta.total_episodes - len(drop)} of "
                    f"{source.meta.total_episodes} episodes"
                )
                if filtered_root.exists():
                    shutil.rmtree(filtered_root)
                delete_episodes(
                    dataset=LeRobotDataset(repo_id=repo_id, root=converted_root),
                    episode_indices=drop,
                    output_dir=filtered_root,
                    repo_id=repo_id,
                )
                validate_dataset(repo_id, filtered_root, expected_camera_mode=camera_mode,
                                 decode_videos=verify_decode)
                shutil.rmtree(converted_root, ignore_errors=True)
                converted_root = filtered_root

        except Exception as e:
            raise RuntimeError(
                f"[{name}] converting '{repo_id}' to '{camera_mode}' failed: {e}"
            ) from e

        converted.append((repo_id, converted_root))

    # Step 4: merge the converted datasets into the final output location.
    # First reconcile cosmetic video-metadata differences so the merge's strict
    # feature-equality check passes (raises if critical fields truly differ).
    if len(converted) > 1:
        normalize_video_info(converted)
    logging.info(f"Merging {len(converted)} datasets into '{output_repo_id}' at {output_root}")
    converted_datasets = [LeRobotDataset(repo_id=rid, root=root) for rid, root in converted]
    merge_datasets(converted_datasets, output_repo_id=output_repo_id, output_dir=output_root)
    # A merge leaves every episode metadata row pointing at the file index it had in its
    # source, while writing them all into one destination file - so any source that had
    # more than one metadata file leaves rows naming a file that was never created. See
    # lerobot_repair_episode_meta for the mechanism. Repaired here, right after the step
    # that breaks it, because the next thing to read an episode's stats is the one that
    # fails, and that can be days later in a split.
    broken = repair_episode_meta(output_root)
    if broken:
        logging.info(f"repaired {broken} episode metadata row(s) after the merge")
    validate_dataset(output_repo_id, output_root, expected_camera_mode=camera_mode,
                     decode_videos=verify_decode)

    # Step 4b: optional trim of every episode down to its grasp segment. Runs on the
    # merged dataset so the expensive decode/re-encode happens once rather than per
    # source. Needs gripper_pos_*/finger_pressure, so keep_state_features must retain
    # them; trim_to_grasp raises if they were dropped.
    if trim_cfg.get("enabled"):
        logging.info(f"Trimming '{output_repo_id}' to grasp segments")
        staging_root = output_root.parent / f"{output_root.name}_untrimmed"
        if staging_root.exists():
            shutil.rmtree(staging_root)
        # The trim writes a fresh dataset, so move the merged one aside and let the
        # trimmed result take the output path.
        shutil.move(str(output_root), str(staging_root))
        try:
            trim_dataset_to_grasp(
                LeRobotDataset(repo_id=output_repo_id, root=staging_root),
                output_dir=output_root,
                repo_id=output_repo_id,
                **{k: float(v) for k, v in trim_cfg.items() if k != "enabled"},
            )
        except BaseException:
            # put the merged dataset back so a failed trim doesn't cost the whole merge
            if not output_root.exists():
                shutil.move(str(staging_root), str(output_root))
            raise
        shutil.rmtree(staging_root, ignore_errors=True)
        validate_dataset(output_repo_id, output_root, expected_camera_mode=camera_mode,
                         decode_videos=verify_decode)

    # Step 5: optional contact-action labeling (rewrites the action column + its stats).
    # camera_goal labels per source (its conversion consumes contact_vec), so the
    # merged dataset has nothing left to label.
    if do_label and action_space == camera_goal.ACTION_SPACE_NAME:
        logging.info("Skipping the post-merge labeling pass; camera_goal labels per source")
    elif do_label:
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
    # force=True: importing lerobot installs its own root handler, which would make
    # a plain basicConfig a no-op and silence this pipeline's progress logging.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )

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
        help="CPU cores to leave free during conversion; the rest are spread across the "
             "encodes, one per file where there are enough files to go round",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse per-source conversions already present under --temp_dir instead of redoing "
             "them; each is re-validated first, and rebuilt if it does not pass",
    )
    parser.add_argument(
        "--keep_intermediate", action="store_true",
        help="Keep the per-source converted datasets under temp_dir instead of deleting them",
    )
    parser.add_argument(
        "--no_verify_decode", action="store_true",
        help="Skip fully decoding every re-encoded video. Faster, but silent mid-file "
             "corruption from the AV1 encode then survives to training time",
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
        resume=args.resume,
        verify_decode=not args.no_verify_decode,
    )


if __name__ == "__main__":
    main()
