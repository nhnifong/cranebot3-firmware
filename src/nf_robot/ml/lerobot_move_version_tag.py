#!/usr/bin/env python

"""Point a dataset's lerobot version tag at the head of main.

LeRobotDataset resolves a repo by a git tag named after the codebase version in
meta/info.json, not by main. A repo whose content was replaced without moving that tag
therefore keeps serving the old content to every reader, while the new content sits on
main unread - and nothing looks wrong, because the episode count that comes back is the
tag's, not the branch's.

That is what an upload into an existing repo used to do here; upload_dataset now moves
the tag itself. This exists for repos already in that state, where re-uploading the
content just to move a ref would mean pushing the videos again.

Usage:
    python src/nf_robot/ml/lerobot_move_version_tag.py --repo_id naavox/combined_targets_eval
"""

import argparse
import json
import logging


def current_version(repo_id):
    """The codebase version in the repo's meta/info.json on main, which names the tag."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", revision="main",
                           filename="meta/info.json")
    return json.loads(open(path).read())["codebase_version"]


def move_tag(repo_id, dry_run=False):
    from huggingface_hub import HfApi

    api = HfApi()
    version = current_version(repo_id)
    refs = api.list_repo_refs(repo_id, repo_type="dataset")
    main = next((b.target_commit for b in refs.branches if b.name == "main"), None)
    tag = next((t for t in refs.tags if t.name == version), None)

    logging.info(f"{repo_id}: main is {main}, tag {version} is "
                 f"{tag.target_commit if tag else '(absent)'}")
    if tag and tag.target_commit == main:
        logging.info("already pointing at main; nothing to do")
        return False
    if dry_run:
        logging.info(f"would move tag {version} to {main}")
        return True

    if tag:
        api.delete_tag(repo_id, tag=version, repo_type="dataset")
    api.create_tag(repo_id, tag=version, repo_type="dataset", revision="main",
                   tag_message="lerobot codebase version, matching meta/info.json")
    logging.info(f"tag {version} now points at main ({main})")
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", required=True, nargs="+", help="Dataset repo(s) to fix")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for repo_id in args.repo_id:
        move_tag(repo_id, args.dry_run)


if __name__ == "__main__":
    main()
