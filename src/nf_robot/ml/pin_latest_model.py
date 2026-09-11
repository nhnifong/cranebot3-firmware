#!/usr/bin/env python

"""Point a model's pin at whatever is newest on the hub.

Robots download the published models at a fixed commit (common/model_revisions.json), so
uploading a checkpoint does not change what is already flying. This is the step that says
"and now it should": it reads the tip of the model repo and rewrites that one entry.

Run it after publishing, check the title it prints is the upload you meant, and commit the
file. Nothing here uploads or downloads a checkpoint - it is a metadata call and a write to
one JSON file.

Usage:
    python -m nf_robot.ml.pin_latest_model --visual_servo
    python -m nf_robot.ml.pin_latest_model --targeting
    python -m nf_robot.ml.pin_latest_model --all --dry_run
"""

import argparse
import datetime
import json
import logging

from nf_robot.common.model_revisions import REVISIONS_PATH, load_revisions

# flag name -> the module that owns the repo id, so the id is never written down twice
MODELS = {
    "visual_servo": ("nf_robot.ml.visual_servoing.servo", "SERVO_MODEL_REPOID"),
    "targeting": ("nf_robot.ml.ortho_target", "TARGETING_MODEL_REPOID"),
}


def repo_id_for(name):
    """The repo a model name publishes to, read off the module that defines it."""
    import importlib

    module, attr = MODELS[name]
    return getattr(importlib.import_module(module), attr)


def latest_commit(repo_id):
    """The tip commit of a model repo: (sha, title, date)."""
    from huggingface_hub import HfApi

    head = HfApi().list_repo_commits(repo_id)[0]
    return head.commit_id, head.title, head.created_at.date().isoformat()


def pin(names, dry_run=False, path=REVISIONS_PATH):
    """Rewrite each named model's entry to the tip of its repo. Returns what changed."""
    # Read the file rather than load_revisions() so the comment key survives the rewrite.
    data = json.loads(path.read_text())
    changed = []
    for name in names:
        repo_id = repo_id_for(name)
        revision, title, committed_at = latest_commit(repo_id)
        was = data.get(repo_id, {}).get("revision")
        if was == revision:
            logging.info(f"{name}: already pinned to {revision[:12]} ({title})")
            continue
        data[repo_id] = {
            "revision": revision,
            "pinned_at": datetime.date.today().isoformat(),
            "title": title,
            "committed_at": committed_at,
        }
        changed.append((name, repo_id, was, revision, title))
        logging.info(f"{name}: {(was or 'unpinned')[:12]} -> {revision[:12]} ({title})")

    if changed and not dry_run:
        path.write_text(json.dumps(data, indent=2) + "\n")
        logging.info(f"wrote {path}; commit it to make robots download the new checkpoint")
    elif changed:
        logging.info(f"--dry_run, so {path} is unchanged")
    return changed


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # the hub client logs a line per request at INFO, which buries the one line that matters
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in MODELS:
        parser.add_argument(f"--{name}", action="store_true",
                            help=f"Pin {name} to the newest commit in its repo")
    parser.add_argument("--all", action="store_true", help="Every model in the file")
    parser.add_argument("--dry_run", action="store_true",
                        help="Say what would change without writing it")
    args = parser.parse_args()

    names = list(MODELS) if args.all else [n for n in MODELS if getattr(args, n)]
    if not names:
        parser.error("nothing to pin; pass " +
                     ", ".join(f"--{n}" for n in MODELS) + " or --all")

    current = load_revisions()
    for name in names:
        repo = repo_id_for(name)
        entry = current.get(repo)
        logging.info(f"{name} ({repo}) currently pinned to "
                     f"{entry['revision'][:12] if entry else 'nothing'}")
    pin(names, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
