#!/usr/bin/env python

"""Run a lerobot policy on Modal cloud GPUs and drive the robot through the relay.

This is `stringman_lerobot eval` with the GPU somewhere else. The container loads the
policy, connects out to the telemetry relay at `--server_address` for the control
channel, and pulls the camera feeds over RTSP/TCP from media.neufangled.com using the
stream ticket -- all outbound, so nothing has to be reachable from Modal. Start/stop is
still driven from the robot's control panel exactly as with a local eval; this process
just sits in EVAL_IDLE until you press start.

  python src/nf_robot/ml/lerobot_eval_modal.py \
    --policy_id naavox/neu-298-fastwam-test \
    --server_address wss://neufangled.com \
    --robot_id YOUR_ROBOT_ID \
    --remote_stream_token YOUR_STREAM_TICKET

Flags
-----
  --policy_id           Hub repo id of the policy to run (required)
  --server_address      Relay websocket base, e.g. wss://neufangled.com (required)
  --robot_id            Robot to connect to (required)
  --remote_stream_token Single-use stream ticket from Calibration and Maintenance ->
                        Get Stream Ticket (required)
  --camera_mode         Override the camera setup inferred from the training dataset
  --gpu_type            Modal GPU (default: L40S -- see the sizing note below)
  --timeout_hours       Container wall-time cap (default: 2)
  --detach              Fire-and-forget; default false, because an eval session is
                        interactive and you want the logs and the ability to Ctrl-C
  --hf_secret           Modal secret holding HUGGING_FACE_HUB_TOKEN
  --lerobot_ref         "public" (PyPI) or a commit of the nhnifong/lerobot fork
  --lerobot_extras      Extras installed with lerobot; add the one for your policy type

GPU sizing
----------
L40S (48 GB) is the smallest sensible card for FastWAM. Resident weights are about
26 GB: the MoT DiT (~6.0B params, the 12 GB `model.safetensors`) plus the frozen
UMT5-XXL text encoder (11.4 GB) and Wan VAE (2.8 GB), all bf16. That rules out the
24 GB tier (L4 / A10G) and leaves L40S comfortable headroom for activations. Inference
is the slow part, not the memory: each chunk is 10 denoising steps through a 6B MoT, so
an H100 is worth `--gpu_type H100` if the control loop is falling behind. Smaller
policies (ACT, DiT, SmolVLA) run fine on `--gpu_type L4`.

Prerequisites
-------------
  pip install modal
  modal setup
  modal secret create huggingface HUGGING_FACE_HUB_TOKEN=<your_token>
"""

import argparse
import sys
from pathlib import Path

import modal

_DEFAULT_LEROBOT_REF = "public"
# fastwam pulls in transformers (UMT5) + diffusers (Wan VAE); multi_task_dit is here so
# the same image also runs the DiT checkpoints. Add xvla / smolvla / vla_jepa as needed --
# Modal caches the image layer by the pip spec string, so changing this forces a rebuild.
_DEFAULT_LEROBOT_EXTRAS = "dataset,fastwam,multi_task_dit"

# ---------------------------------------------------------------------------
# Modal-only flags are parsed at import time so they can be baked into the
# @app.function decorator (Modal 1.x removed with_options).
# ---------------------------------------------------------------------------
_modal_parser = argparse.ArgumentParser(add_help=False)
_modal_parser.add_argument("--gpu_type", default="L40S")
_modal_parser.add_argument("--timeout_hours", type=float, default=2.0)
_modal_parser.add_argument("--hf_secret", default="huggingface")
_modal_parser.add_argument("--lerobot_ref", default=_DEFAULT_LEROBOT_REF)
_modal_parser.add_argument("--lerobot_extras", default=_DEFAULT_LEROBOT_EXTRAS)
_modal_args, _ = _modal_parser.parse_known_args()

_GPU_TYPE = _modal_args.gpu_type
_TIMEOUT_S = int(_modal_args.timeout_hours * 3600)
_HF_SECRET = _modal_args.hf_secret
_LEROBOT_REF = _modal_args.lerobot_ref

_LEROBOT_EXTRAS = f"lerobot[{_modal_args.lerobot_extras}]"
_LEROBOT_FORK_URL = "git+https://github.com/nhnifong/lerobot.git"
if _LEROBOT_REF.lower() == "public":
    _LEROBOT_SPEC = _LEROBOT_EXTRAS
else:
    _LEROBOT_SPEC = f"{_LEROBOT_EXTRAS} @ {_LEROBOT_FORK_URL}@{_LEROBOT_REF}"

# nf_robot's own runtime deps for the eval path (pyproject's base + host extras, minus the
# ones only the robot host needs). The package itself is not pip installed -- the local
# working tree is mounted below, so you eval the code you are looking at.
_NF_ROBOT_DEPS = [
    "numpy>=2.0",
    "websockets",
    "betterproto2==0.9.0",
    "opencv-contrib-python-headless>=4.0",
    "av",
    "scipy",
    "huggingface_hub",
]

_NF_SRC_MOUNT = "/opt/nf_src"

_EVAL_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install(*_NF_ROBOT_DEPS)
    .pip_install(_LEROBOT_SPEC)
    # hf_xet deadlocks at "Fetching N files: 0%" inside Modal containers; force plain LFS.
    .env({"HF_HUB_DISABLE_XET": "1"})
)

# This whole module is re-imported inside the container, where the file lands at
# /root/lerobot_eval_modal.py and the repo layout does not exist -- so anything that
# reaches for a local path has to be guarded. The mount is declared locally only; the
# container receives it from the already-registered function spec.
if modal.is_local():
    _REPO_SRC = Path(__file__).resolve().parents[2]  # <repo>/src
    _EVAL_IMAGE = _EVAL_IMAGE.add_local_dir(
        _REPO_SRC / "nf_robot",
        f"{_NF_SRC_MOUNT}/nf_robot",
        ignore=["**/__pycache__"],
    )

app = modal.App("lerobot-eval")
# Persists the HF cache across runs. FastWAM pulls ~26 GB of weights (fastwam_base +
# UMT5-XXL + Wan VAE) on a cold start; with this volume that download happens once.
# HF_LEROBOT_HOME defaults to $HF_HOME/lerobot, so the training dataset's meta/ lands
# here too and eval stops re-fetching it.
hf_cache = modal.Volume.from_name("hf_cache", create_if_missing=True)


@app.function(
    image=_EVAL_IMAGE,
    gpu=_GPU_TYPE,
    timeout=_TIMEOUT_S,
    secrets=[modal.Secret.from_name(_HF_SECRET)],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_eval(
    policy_id: str,
    server_address: str,
    robot_id: str,
    remote_stream_token: str,
    camera_mode: str | None = None,
) -> None:
    """Load the policy and drive the robot until it disconnects."""
    sys.path.insert(0, _NF_SRC_MOUNT)

    from nf_robot.ml.stringman_lerobot import eval_until_disconnected

    # Same URI construction as stringman_lerobot's __main__.
    uri = f"{server_address}/control/{robot_id}"
    if remote_stream_token:
        uri += f"?ticket={remote_stream_token}"

    try:
        eval_until_disconnected(
            uri,
            policy_id,
            robot_id,
            remote_stream_token=remote_stream_token,
            camera_mode=camera_mode,
        )
    finally:
        # Keep whatever was downloaded this run, even if the session errored out.
        hf_cache.commit()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lerobot-eval-on-modal",
        description="Run stringman_lerobot eval on a Modal GPU, driving the robot through the relay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy_id", required=True, help="Hub repo id of the policy to run")
    parser.add_argument(
        "--server_address",
        required=True,
        help="Relay websocket base, not including path (e.g. wss://neufangled.com)",
    )
    parser.add_argument("--robot_id", required=True, help="id of the robot to connect to")
    parser.add_argument(
        "--remote_stream_token",
        required=True,
        help="Single-use stream ticket (run menu -> Calibration and Maintenance -> Get Stream Ticket)",
    )
    parser.add_argument(
        "--camera_mode",
        default=None,
        help=(
            "Override the camera setup inferred from the policy's training dataset "
            "(e.g. gripper_anchors_rect). Validated against _CAMERA_MODES inside the "
            "container, not here, so this module stays importable without lerobot."
        ),
    )
    parser.add_argument("--gpu_type", default="L40S", help="Modal GPU type")
    parser.add_argument("--timeout_hours", type=float, default=2.0, help="Container timeout in hours")
    parser.add_argument(
        "--detach",
        default="false",
        choices=["true", "false"],
        help="Fire-and-forget: return immediately after submitting",
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
            "lerobot to install: a git ref of the nhnifong/lerobot fork, or 'public' for the "
            "PyPI release. Registered here so it is consumed rather than treated as unknown; "
            "the image is actually built from the import-time parse above."
        ),
    )
    parser.add_argument(
        "--lerobot_extras",
        default=_DEFAULT_LEROBOT_EXTRAS,
        help="Comma-separated lerobot extras to install (same note as --lerobot_ref)",
    )

    args = parser.parse_args()
    detach = args.detach.lower() == "true"

    print("Submitting lerobot eval to Modal")
    print(f"  gpu_type   : {_GPU_TYPE}")
    print(f"  timeout    : {_modal_args.timeout_hours}h")
    print(f"  detach     : {detach}")
    print(f"  hf_secret  : {_HF_SECRET}")
    print(f"  lerobot    : {_LEROBOT_SPEC}")
    print(f"  policy     : {args.policy_id}")
    print(f"  robot      : {args.robot_id} via {args.server_address}")

    call_args = (
        args.policy_id,
        args.server_address,
        args.robot_id,
        args.remote_stream_token,
        args.camera_mode,
    )

    with modal.enable_output():
        if detach:
            with app.run(detach=True):
                handle = run_eval.spawn(*call_args)
            print(f"\nSession submitted. Function call ID: {handle.object_id}")
            print("Monitor with:  modal app logs lerobot-eval")
        else:
            with app.run():
                run_eval.remote(*call_args)


if __name__ == "__main__":
    main()
