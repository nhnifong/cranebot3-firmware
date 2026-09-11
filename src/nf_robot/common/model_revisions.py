#!/usr/bin/env python

"""Which hub commit each published model is downloaded at.

hf_hub_download with no revision re-resolves the tip of main on every start, so publishing
a checkpoint would change what every robot in the field runs at its next restart - no
version bump, nothing in the logs, and no way back without knowing which commit the old
behaviour was. The pins live in model_revisions.json beside this module so that bumping one
is a data change that shows up in a diff as a data change.

Write them with `python -m nf_robot.ml.pin_latest_model`, which is the step after
publishing a checkpoint; read them with pinned_revision().
"""

import json
from pathlib import Path

REVISIONS_PATH = Path(__file__).with_name("model_revisions.json")


def load_revisions():
    """repo id -> {revision, pinned_at, ...}, straight from the file.

    Keys starting with an underscore are notes to a reader rather than repos.
    """
    data = json.loads(REVISIONS_PATH.read_text())
    return {repo: entry for repo, entry in data.items() if not repo.startswith("_")}


def pinned_revision(repo_id):
    """The commit `repo_id` is pinned to.

    Raises rather than falling back to the tip: an unpinned repo means a model nobody
    chose a version for, and downloading whatever is newest is the failure this exists to
    prevent, not a reasonable default.
    """
    entry = load_revisions().get(repo_id)
    if entry is None:
        raise KeyError(
            f"No pinned revision for '{repo_id}' in {REVISIONS_PATH}. Add one with "
            f"'python -m nf_robot.ml.pin_latest_model'.")
    return entry["revision"]
