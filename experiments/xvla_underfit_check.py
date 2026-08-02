#!/usr/bin/env python

"""Ask whether a trained X-VLA policy is undertrained, without touching the robot.

Runs the policy on frames it was trained on and compares its predicted action chunk
to what the human actually did on those frames. A model that cannot reproduce its
own training data is underfit, and more training steps are the fix. A model that
fits training data well but behaves badly on the robot has a different problem
(distribution shift, execution, conditioning), and more steps would waste money.

Three questions, three answers:

  1. FIT       normalized error against the demonstrator's actions, in units of each
               channel's standard deviation. 1.0 means the model is no better than
               always predicting the dataset mean; 0.0 is perfect.
  2. PROMPT    how much the predicted actions change when only the task string
               changes. Near zero means language conditioning is being ignored, and
               no amount of extra training on this recipe will fix targeting.
  3. SPEED     magnitude of predicted velocity commands vs the demonstrator's, for
               the "it moves faster than we do" complaint.

Usage (on the training box, where the dataset is already local):
    python experiments/xvla_underfit_check.py \
        --dataset_root /home/nick/data_scratch/move_clutter_rect_for_xvla \
        --device cuda

Without --dataset_root the dataset is pulled from the Hub, which is a large download.
"""

import argparse

import numpy as np
import torch

# Dataset keys -> the slots the checkpoint was trained with. Same map as eval.
RENAME_MAP = {
    "observation.images.anchor_camera_0": "observation.images.image",
    "observation.images.gripper_camera": "observation.images.image2",
    "observation.images.anchor_camera_1": "observation.images.image3",
}

# move_clutter_nick_2 occupies this span of the merged dataset (sources are
# concatenated in recipe order, after per-source exclusions).
DEFAULT_EPISODES = "683-739"


def parse_episodes(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            first, last = part.split("-")
            out.extend(range(int(first), int(last) + 1))
        else:
            out.append(int(part))
    return out


def load_policy(policy_repo_id, dataset, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(policy_repo_id)
    cfg.pretrained_path = policy_repo_id
    cfg.device = device

    # rename_map is required here for the same reason eval needs it: the dataset's
    # camera keys differ from the checkpoint's, and passing it also skips lerobot's
    # visual feature check, which would otherwise reject the mismatch.
    policy = make_policy(cfg=cfg, ds_meta=dataset.meta, rename_map=RENAME_MAP)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_repo_id,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # The rename has to happen inside the saved pipeline; without it the model gets
    # zero-padded views and every result below would be meaningless.
    renames = [
        getattr(s, "rename_map", None) for s in preprocessor.steps
        if "rename" in type(s).__name__.lower()
    ]
    if not any(renames):
        raise RuntimeError(
            "The saved preprocessor has no rename step; the policy would receive no camera "
            "views under the names it expects. Check how the policy was trained."
        )
    print(f"preprocessor rename map: {renames[0]}")
    return cfg, policy, preprocessor, postprocessor


def predict_chunk(policy, preprocessor, postprocessor, item, task):
    """Predicted action chunk in raw units, shape (chunk, action_dim)."""
    batch = {k: v.unsqueeze(0) for k, v in item.items() if isinstance(v, torch.Tensor)}
    batch.pop("action", None)  # never let the policy see the answer
    batch.pop("action_is_pad", None)
    batch["task"] = [task]

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    processed = preprocessor(batch)
    chunk = policy.predict_action_chunk(processed)
    chunk = postprocessor(chunk)
    return chunk.squeeze(0).float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default="naavox/xvla-move-clutter")
    parser.add_argument("--dataset", default="naavox/move_clutter_rect_for_xvla")
    parser.add_argument("--dataset_root", default=None, help="local copy of the dataset (skips the Hub download)")
    parser.add_argument("--episodes", default=DEFAULT_EPISODES,
                        help=f"episodes to sample, e.g. '683-739' (default: move_clutter_nick_2's span)")
    parser.add_argument("--num_frames", type=int, default=32, help="frames sampled across those episodes")
    parser.add_argument("--prompt_frames", type=int, default=6, help="frames used for the prompt-sensitivity test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = parse_episodes(args.episodes)
    meta_only = LeRobotDataset(repo_id=args.dataset, root=args.dataset_root, episodes=episodes)
    fps = meta_only.meta.fps
    chunk_size = 30  # read back from the policy config below; used to shape the GT window

    from lerobot.configs.policies import PreTrainedConfig
    chunk_size = PreTrainedConfig.from_pretrained(args.policy).chunk_size

    # delta_timestamps gives each frame the following chunk_size actions, which is
    # exactly what the policy predicts in one shot.
    dataset = LeRobotDataset(
        repo_id=args.dataset,
        root=args.dataset_root,
        episodes=episodes,
        delta_timestamps={"action": [i / fps for i in range(chunk_size)]},
    )
    print(f"{len(dataset)} frames from {len(episodes)} episode(s), chunk {chunk_size} @ {fps}fps")

    cfg, policy, preprocessor, postprocessor = load_policy(args.policy, dataset, args.device)

    names = dataset.meta.features["action"]["names"]
    action_std = np.array(dataset.meta.stats["action"]["std"], dtype=np.float64)
    tasks = list(dataset.meta.tasks.index)
    print(f"action channels: {names}")
    print(f"tasks in dataset: {tasks}\n")

    indices = np.linspace(0, len(dataset) - 1, args.num_frames).astype(int)

    errors, gt_all, pred_all = [], [], []
    for n, idx in enumerate(indices):
        item = dataset[int(idx)]
        gt = item["action"].float().cpu().numpy()
        pred = predict_chunk(policy, preprocessor, postprocessor, item, item["task"])
        valid = ~item["action_is_pad"].cpu().numpy() if "action_is_pad" in item else np.ones(len(gt), bool)
        errors.append(pred[valid] - gt[valid])
        gt_all.append(gt[valid])
        pred_all.append(pred[valid])
        print(f"  frame {n + 1}/{len(indices)}", end="\r")

    err = np.concatenate(errors)
    gt = np.concatenate(gt_all)
    pred = np.concatenate(pred_all)

    print("\n\n=== 1. FIT on frames the model trained on ===")
    print(f"{'channel':>16} {'norm err':>9} {'demo std':>9} {'pred std':>9}")
    per_channel = []
    for i, name in enumerate(names):
        rmse = float(np.sqrt(np.mean(err[:, i] ** 2)))
        norm = rmse / (action_std[i] + 1e-9)
        per_channel.append(norm)
        print(f"{name:>16} {norm:9.2f} {gt[:, i].std():9.4f} {pred[:, i].std():9.4f}")

    control = [i for i, n in enumerate(names) if not n.startswith("contact_vec") and n != "episode_end"]
    overall = float(np.mean([per_channel[i] for i in control]))
    print(f"\ncontrol channels mean normalized error: {overall:.2f}")
    print("  1.0 = no better than always predicting the dataset mean")
    if overall > 0.85:
        print("  VERDICT: badly underfit. It has not learned the training data itself -> train longer.")
    elif overall > 0.55:
        print("  VERDICT: underfit. More steps should still buy real improvement.")
    else:
        print("  VERDICT: fits training data. More steps are NOT the main problem;")
        print("           look at execution (n_action_steps) and distribution shift instead.")

    print("\n=== 2. PROMPT sensitivity ===")
    spreads = []
    for idx in np.linspace(0, len(dataset) - 1, args.prompt_frames).astype(int):
        item = dataset[int(idx)]
        chunks = np.stack([predict_chunk(policy, preprocessor, postprocessor, item, t) for t in tasks])
        # spread across prompts, per channel, in units of that channel's std
        spread = (chunks.std(axis=0).mean(axis=0) / (action_std + 1e-9))[control].mean()
        spreads.append(spread)
    prompt_spread = float(np.mean(spreads))
    print(f"changing only the task string moves the predicted actions by {prompt_spread:.3f} sigma")
    if prompt_spread < 0.05:
        print("  VERDICT: the prompt is being ignored. Extra training on this recipe will not fix targeting.")
    else:
        print("  VERDICT: the policy does respond to the prompt.")

    print("\n=== 3. SPEED vs the demonstrators ===")
    vel = [i for i, n in enumerate(names) if n.startswith("vel_") or n.startswith("room_vel")]
    gt_mag = np.linalg.norm(gt[:, vel], axis=1)
    pred_mag = np.linalg.norm(pred[:, vel], axis=1)
    for label, arr in [("demo", gt_mag), ("policy", pred_mag)]:
        q = np.percentile(arr, [50, 90, 99])
        print(f"  {label:>6} |vel| median {q[0]:.3f}  p90 {q[1]:.3f}  p99 {q[2]:.3f} m/s")
    ratio = float(np.median(pred_mag) / (np.median(gt_mag) + 1e-9))
    print(f"  policy commands {ratio:.2f}x the demonstrated median speed")
    if ratio > 1.5:
        print("  VERDICT: consistent with an under-converged action head, not a scaling bug")
        print("           (the unnormalizer provably uses the training stats).")


if __name__ == "__main__":
    main()
