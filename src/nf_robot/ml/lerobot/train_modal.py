#!/usr/bin/env python

"""Run lerobot-train on Modal cloud infrastructure.

All lerobot-train arguments are forwarded as-is; this script only adds the
Modal-specific flags below.  Paths (--dataset.root, --output_dir, etc.) are
passed through unchanged — if they point inside a Modal Volume, mount that
volume via the Modal dashboard or a separate `modal volume` command.

Modal-specific flags
--------------------
  --gpu_type      GPU type to request (default: H200)
  --timeout_hours Maximum job wall-time in hours (default: 10)
  --detach        When true, submit as a background job and return immediately (default: true)
  --num_workers   DataLoader num_workers; injected into training args when not already set (default: 10)
  --hf_secret     Name of the Modal secret that holds HUGGING_FACE_HUB_TOKEN (default: huggingface)
  --lerobot_ref   Which lerobot to install in the image. The literal "public" installs the
                  published lerobot from PyPI (default). A git ref (commit/branch/tag) of the
                  nhnifong/lerobot fork installs that fork instead. The ref must
                  already be pushed to GitHub for Modal to fetch it. Pin a commit (not a branch):
                  Modal caches the image layer by the pip spec string, so a branch name would keep
                  reusing a stale build, whereas a new commit hash forces a rebuild.

Prerequisites
-------------
  pip install modal
  modal setup           # authenticate once
  modal secret create huggingface HUGGING_FACE_HUB_TOKEN=<your_token>

"""

import argparse
import sys

import modal

# Which lerobot to install in the Modal image. Defaults to the published PyPI lerobot
# so checkpoints are loadable by anyone on the public release. To use the nhnifong/lerobot
# fork (dinov3 encoder + visual cross-attention), pass --lerobot_ref <commit>; pin a commit
# hash (not a branch) so Modal rebuilds the cached image layer. A known-good fork commit is
# 022fe150.
_DEFAULT_LEROBOT_REF = "public"

# ---------------------------------------------------------------------------
# Parse Modal-specific flags at import time so they can be baked into the
# @app.function decorator (Modal 1.x removed with_options).
# parse_known_args is used so unrecognised lerobot-train args are ignored here.
# ---------------------------------------------------------------------------
_modal_parser = argparse.ArgumentParser(add_help=False)
_modal_parser.add_argument("--gpu_type", default="H200")
_modal_parser.add_argument("--timeout_hours", type=float, default=24.0)
_modal_parser.add_argument("--hf_secret", default="huggingface")
_modal_parser.add_argument("--lerobot_ref", default=_DEFAULT_LEROBOT_REF)
_modal_args, _ = _modal_parser.parse_known_args()

_GPU_TYPE = _modal_args.gpu_type
_TIMEOUT_S = int(_modal_args.timeout_hours * 3600)
_HF_SECRET = _modal_args.hf_secret
_LEROBOT_REF = _modal_args.lerobot_ref

# ---------------------------------------------------------------------------
# Modal image — installs lerobot and its training extras with system ffmpeg.
# --lerobot_ref selects the source: "public" -> published PyPI lerobot;
# any other value -> the nhnifong/lerobot fork at that git ref (commit/branch/tag).
# The fork carries the multi_task_dit + dinov3 / use_visual_cross_attention code.
# ---------------------------------------------------------------------------
_LEROBOT_EXTRAS = "lerobot[training,multi_task_dit]"
_LEROBOT_FORK_URL = "git+https://github.com/nhnifong/lerobot.git"
if _LEROBOT_REF.lower() == "public":
    _LEROBOT_SPEC = _LEROBOT_EXTRAS
else:
    _LEROBOT_SPEC = f"{_LEROBOT_EXTRAS} @ {_LEROBOT_FORK_URL}@{_LEROBOT_REF}"

_LEROBOT_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install(_LEROBOT_SPEC)
    # Force classic LFS HTTPS downloads instead of the Xet CAS protocol.
    # hf_xet deadlocks at "Fetching N files: 0%" inside Modal containers
    # (old kernel / restricted egress), hanging dataset loading indefinitely.
    .env({"HF_HUB_DISABLE_XET": "1"})
)

app = modal.App("lerobot-train")
data_volume = modal.Volume.from_name("multitask_dit_data", create_if_missing=True)

# ---------------------------------------------------------------------------
# Remote function — runs inside the Modal container
# ---------------------------------------------------------------------------
@app.function(
    image=_LEROBOT_IMAGE,
    gpu=_GPU_TYPE,
    timeout=_TIMEOUT_S,
    secrets=[modal.Secret.from_name(_HF_SECRET)],
    volumes={"/multitask_dit_data": data_volume},
)
def run_training(train_argv: list[str]) -> None:
    """Execute lerobot-train with the provided argv inside Modal."""
    sys.argv = ["lerobot-train"] + train_argv

    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()

    # train() uses @parser.wrap() which reads sys.argv, so set it first.
    from lerobot.scripts.lerobot_train import train

    train()
    data_volume.commit()


# ---------------------------------------------------------------------------
# Local entrypoint — parses Modal flags and dispatches
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lerobot-train-on-modal",
        description="Run lerobot-train on Modal. All unknown args are forwarded to lerobot-train.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gpu_type", default="H200", help="Modal GPU type")
    parser.add_argument("--timeout_hours", type=float, default=10.0, help="Job timeout in hours")
    parser.add_argument(
        "--detach",
        default="true",
        choices=["true", "false"],
        help="Fire-and-forget: return immediately after submitting",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="DataLoader num_workers (injected into training args if not already set)",
    )
    parser.add_argument(
        "--hf_secret",
        default="huggingface",
        help="Name of the Modal secret containing HUGGING_FACE_HUB_TOKEN",
    )
    parser.add_argument(
        "--lerobot_ref",
        default=_DEFAULT_LEROBOT_REF,
        help=(
            "lerobot to install in the image: a git ref of the nhnifong/lerobot fork, "
            "or 'public' for the published PyPI lerobot. Registered here so it is consumed "
            "rather than forwarded to lerobot-train (the image is actually built from the "
            "import-time parse above)."
        ),
    )

    args, train_argv = parser.parse_known_args()

    # Inject num_workers into training args unless the user already set it.
    if not any(a.startswith("--num_workers") for a in train_argv):
        train_argv = [f"--num_workers={args.num_workers}"] + train_argv

    detach = args.detach.lower() == "true"

    print(f"Submitting lerobot-train to Modal")
    print(f"  gpu_type      : {_GPU_TYPE}")
    print(f"  timeout       : {_modal_args.timeout_hours}h")
    print(f"  detach        : {detach}")
    print(f"  hf_secret     : {_HF_SECRET}")
    print(f"  lerobot       : {_LEROBOT_SPEC}")
    print(f"  training args : {' '.join(train_argv)}")

    with modal.enable_output():
        if detach:
            with app.run(detach=True):
                handle = run_training.spawn(train_argv)
            print(f"\nJob submitted. Function call ID: {handle.object_id}")
            print("Monitor with:  modal app logs lerobot-train")
            print("Or visit:      https://modal.com/apps/lerobot-train")
        else:
            with app.run():
                run_training.remote(train_argv)


if __name__ == "__main__":
    main()
