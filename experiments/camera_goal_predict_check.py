#!/usr/bin/env python

"""Sanity-check a camera_goal policy's predictions on frames from its training data.

Answers two questions that do not need a robot:

  AGREEMENT  the two anchor cameras predict the same physical point in two different
             frames, so transforming both into the room and measuring the gap says how
             much the policy has learned that they describe one goal. The dataset's own
             labels agree to ~1e-7 m, so anything the policy contributes is error.
  IN BOUNDS  whether the fused goal lands inside the work area. A policy that has
             learned nothing predicts near the mean of the training distribution, which
             is inside; one that is broken predicts outside it.

The gripper camera's prediction cannot be placed in the room here: that needs the
gripper's 6D rotation, which the derived dataset trims out of observation.state. Its
magnitude is checked against the anchors' answer instead, which is rotation-free.

Usage:
    python experiments/camera_goal_predict_check.py \
        --policy naavox/xvla_camera_goal_smoke \
        --dataset naavox/move_clutter_camera_goal \
        --num_frames 12 --device cuda
"""

import argparse

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from nf_robot.ml import camera_goal as cg

RENAME_MAP = {
    "observation.images.anchor_camera_0": "observation.images.image",
    "observation.images.gripper_camera": "observation.images.image2",
    "observation.images.anchor_camera_1": "observation.images.image3",
}


def anchor_goal_to_room(action, anchor_num, poses):
    local = np.array([action[n] for n in cg.GOAL_SLOTS[f"anchor_camera_{anchor_num}"]], dtype=float)
    rotvec, position = poses[anchor_num]
    return Rotation.from_rotvec(rotvec).apply(local) + position


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default="naavox/xvla_camera_goal_smoke")
    parser.add_argument("--dataset", default="naavox/move_clutter_camera_goal")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import lerobot.policies.factory  # registers the policy types with PreTrainedConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    dataset = LeRobotDataset(repo_id=args.dataset, root=args.dataset_root)
    names = dataset.meta.features["action"]["names"]
    state_names = dataset.meta.features["observation.state"]["names"]

    cfg = PreTrainedConfig.from_pretrained(args.policy)
    cfg.pretrained_path = args.policy
    cfg.device = args.device
    policy = make_policy(cfg=cfg, ds_meta=dataset.meta, rename_map=RENAME_MAP)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.policy, dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=args.num_frames, replace=False)

    # The work area, from where the labels actually are plus the anchors overhead.
    labels = np.array(dataset.meta.stats["action"]["mean"])
    lo = np.array(dataset.meta.stats["action"]["q01"])
    hi = np.array(dataset.meta.stats["action"]["q99"])

    print(f"{'frame':>7} {'label goal (room)':>26} {'predicted (anchor 0)':>26} "
          f"{'err':>6} {'anchors disagree':>17} {'|grip cam|':>11}")
    errors, spreads, grip_err, in_bounds = [], [], [], []
    for idx in indices:
        item = dataset[int(idx)]
        poses = cg.unpack_anchor_poses(item["anchor_poses"].cpu().numpy())

        batch = {k: v.unsqueeze(0) for k, v in item.items() if isinstance(v, torch.Tensor)}
        truth = batch.pop("action").squeeze(0).cpu().numpy()
        batch["task"] = [item["task"]]
        policy.reset(); preprocessor.reset(); postprocessor.reset()
        chunk = postprocessor(policy.predict_action_chunk(preprocessor(batch)))
        predicted = chunk.squeeze(0)[0].float().cpu().numpy()

        p = dict(zip(names, predicted))
        t = dict(zip(names, truth))
        room_p = [anchor_goal_to_room(p, k, poses) for k in (0, 1)]
        room_t = anchor_goal_to_room(t, 0, poses)

        err = float(np.linalg.norm(room_p[0] - room_t))
        spread = float(np.linalg.norm(room_p[0] - room_p[1]))
        state = np.asarray(item["observation.state"].cpu(), dtype=float)
        gripper = np.array([state[state_names.index(f"gripper_pos_{c}")] for c in "xyz"])
        gcam = abs(float(np.linalg.norm([p[n] for n in cg.GOAL_SLOTS["gripper_camera"]]))
                   - float(np.linalg.norm(room_p[0] - gripper)))
        inside = bool(np.all(predicted[:9] >= lo[:9] - 0.5) and np.all(predicted[:9] <= hi[:9] + 0.5))

        errors.append(err); spreads.append(spread); grip_err.append(gcam); in_bounds.append(inside)
        print(f"{int(idx):7d} {np.array2string(room_t, precision=2, suppress_small=True):>26} "
              f"{np.array2string(room_p[0], precision=2, suppress_small=True):>26} "
              f"{err:6.2f} {spread:17.3f} {gcam:11.2f}")

    print(f"\nfused goal error vs the label:    median {np.median(errors):.2f} m   max {np.max(errors):.2f} m")
    print(f"anchor 0 vs anchor 1 disagreement: median {np.median(spreads):.3f} m   max {np.max(spreads):.3f} m")
    print(f"   (the dataset labels agree to ~1e-7 m, so this is all policy error)")
    print(f"gripper-camera distance mismatch:  median {np.median(grip_err):.2f} m")
    print(f"predictions inside the label range: {sum(in_bounds)}/{len(in_bounds)}")


if __name__ == "__main__":
    main()
