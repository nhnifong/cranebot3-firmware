#!/usr/bin/env python
"""Does a camera_goal policy's predicted destination actually depend on the scene?

Runs a trained policy over sampled frames and, for each one, transforms the three
per-camera goal predictions back into the room the same way stringman_lerobot.py
does at eval. Reports:

  - how far the fused prediction is from the label
  - how far the fused prediction is from the constant a mean-predicting model would
    emit (the dataset action mean, fused through the same transform)
  - how much the prediction moves between scenes, against how much the label moves

If the spread of predictions across scenes is far smaller than the spread of labels,
the policy is emitting roughly one destination for every input, and no amount of
eval-side smoothing will make it home in on anything.

  python experiments/camera_goal_predict_check.py \
      --policy naavox/xvla-camera-goal-waypoints-2 \
      --dataset naavox/move_clutter_camera_goal --episodes 0,120,240,360,480,600 \
      --per_episode 6
"""

import argparse
import sys
import pathlib

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from nf_robot.ml import camera_goal


def gripper_rot_6d(state, idx):
    """The gripper's 6D rotation, from the trimmed state.

    The derived dataset drops gripper_rot_* (the conversion consumed them before the
    state trim), so reconstruct the yaw from spin. The gripper hangs, so its tilt is
    small and this only perturbs the gripper-camera channel.
    """
    if "gripper_rot_0" in idx:
        return np.array([state[idx[f"gripper_rot_{k}"]] for k in range(6)], dtype=float)
    m = Rotation.from_euler("z", float(state[idx["spin"]])).as_matrix()
    return np.concatenate([m[:, 0], m[:, 1]])


def per_camera_room_estimates(action, gripper_pos, gripper_rot_6d, anchor_poses):
    """Each camera's goal, transformed back into the room independently."""
    from nf_robot.common.pose_functions import compose_poses
    poses = {
        "gripper_camera": camera_goal.gripper_camera_pose(gripper_pos, gripper_rot_6d),
        "anchor_camera_0": anchor_poses[0],
        "anchor_camera_1": anchor_poses[1],
    }
    out = {}
    for name, slots in camera_goal.GOAL_SLOTS.items():
        point = np.array([action[s] for s in slots], dtype=float)
        out[name] = compose_poses([poses[name], (np.zeros(3), point)])[1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="naavox/xvla-camera-goal-waypoints-2")
    ap.add_argument("--dataset", default="naavox/move_clutter_camera_goal")
    ap.add_argument("--episodes", default="0,120,240,360,480,600")
    ap.add_argument("--per_episode", type=int, default=6)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    import lerobot.policies.factory  # registers the xvla choice before from_pretrained
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors, get_policy_class
    from lerobot.configs.policies import PreTrainedConfig

    episodes = [int(e) for e in args.episodes.split(",") if e.strip()]
    ds = LeRobotDataset(args.dataset, episodes=episodes)
    action_names = ds.meta.info["features"]["action"]["names"]
    state_names = ds.meta.info["features"]["observation.state"]["names"]
    action_mean = np.array(ds.meta.stats["action"]["mean"], dtype=float)

    cfg = PreTrainedConfig.from_pretrained(args.policy)
    policy = get_policy_class(cfg.type).from_pretrained(args.policy, config=cfg)
    policy.to(device).eval()
    # the saved processor config pins device=cuda; override it for whatever is here
    pre, post = make_pre_post_processors(
        cfg, args.policy, dataset_stats=ds.meta.stats,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )

    # eval renames the cameras; mirror the training rename_map
    rename = {"observation.images.anchor_camera_0": "observation.images.image",
              "observation.images.gripper_camera": "observation.images.image2",
              "observation.images.anchor_camera_1": "observation.images.image3"}

    idx = {n: i for i, n in enumerate(state_names)}
    # read episode_index from the parquet column; indexing the dataset itself decodes video
    ep_column = np.array(ds.hf_dataset["episode_index"])
    rows = []
    for ep in episodes:
        ep_items = np.flatnonzero(ep_column == ep).tolist()
        if not ep_items:
            continue
        picks = np.linspace(0, len(ep_items) - 1, args.per_episode).astype(int)
        for j in picks:
            item = ds[ep_items[j]]
            state = item["observation.state"].numpy()
            anchors = camera_goal.unpack_anchor_poses(item["anchor_poses"].numpy())
            gpos = np.array([state[idx["gripper_pos_x"]], state[idx["gripper_pos_y"]],
                             state[idx["gripper_pos_z"]]], dtype=float)
            rot6d = gripper_rot_6d(state, idx)

            batch = {}
            for k, v in item.items():
                if k.startswith("observation.images."):
                    batch[rename.get(k, k)] = v.unsqueeze(0).to(device)
                elif k == "observation.state":
                    batch[k] = v.unsqueeze(0).to(device)
            batch["task"] = [item.get("task", "")]
            # n_action_steps>1 keeps a queue; without a reset most frames would return
            # a stale chunk entry predicted from an earlier image
            policy.reset()
            with torch.inference_mode():
                pred = post(policy.select_action(pre(batch)))
            pred = pred.squeeze(0).float().cpu().numpy()

            pred_a = {n: float(pred[i]) for i, n in enumerate(action_names)}
            gt_a = {n: float(item["action"][i]) for i, n in enumerate(action_names)}
            mean_a = {n: float(action_mean[i]) for i, n in enumerate(action_names)}

            pe = per_camera_room_estimates(pred_a, gpos, rot6d, anchors)
            ge = per_camera_room_estimates(gt_a, gpos, rot6d, anchors)
            me = per_camera_room_estimates(mean_a, gpos, rot6d, anchors)
            rows.append(dict(ep=ep, gpos=gpos,
                             pred=np.median(np.array(list(pe.values())), axis=0),
                             gt=np.median(np.array(list(ge.values())), axis=0),
                             mean=np.median(np.array(list(me.values())), axis=0),
                             pe=pe, ge=ge))

    if not rows:
        print("no frames sampled")
        return

    pred = np.array([r["pred"] for r in rows])
    gt = np.array([r["gt"] for r in rows])
    mean = np.array([r["mean"] for r in rows])
    gpos = np.array([r["gpos"] for r in rows])

    def rms(a):
        return float(np.sqrt((np.linalg.norm(a, axis=1) ** 2).mean()))

    print(f"\n{len(rows)} frames from {len(set(r['ep'] for r in rows))} episodes\n")
    print(f"prediction vs label            {rms(pred - gt):6.3f} m rms")
    print(f"prediction vs mean-predictor   {rms(pred - mean):6.3f} m rms   "
          f"(small = the policy is emitting one destination for everything)")
    print(f"label vs mean-predictor        {rms(gt - mean):6.3f} m rms   "
          f"(how much signal there is to capture)")
    print(f"\nspread across sampled scenes (std per axis, m)")
    print(f"  labels       {np.round(gt.std(axis=0), 3)}")
    print(f"  predictions  {np.round(pred.std(axis=0), 3)}")
    print(f"  gripper      {np.round(gpos.std(axis=0), 3)}")

    dis_p = [float(np.linalg.norm(np.array(list(r['pe'].values())) -
                                  np.median(np.array(list(r['pe'].values())), axis=0), axis=1).mean())
             for r in rows]
    dis_g = [float(np.linalg.norm(np.array(list(r['ge'].values())) -
                                  np.median(np.array(list(r['ge'].values())), axis=0), axis=1).mean())
             for r in rows]
    print(f"\ncamera disagreement about the same point (mean over frames)")
    print(f"  labels       {np.mean(dis_g):.3f} m")
    print(f"  predictions  {np.mean(dis_p):.3f} m")
    print(f"\nerror of each channel used alone, m rms")
    for name in camera_goal.GOAL_SLOTS:
        chan = np.array([r['pe'][name] for r in rows])
        print(f"  {name:16} {rms(chan - gt):6.3f}")
    print(f"  {'median of 3':16} {rms(pred - gt):6.3f}")
    print(f"  {'stay put baseline':16} {rms(gpos - gt):6.3f}   (goal = where the gripper already is)")

    print(f"\nby how far the true goal is")
    reach = np.linalg.norm(gt - gpos, axis=1)
    for lo, hi in [(0, 0.25), (0.25, 0.75), (0.75, 99)]:
        m = (reach >= lo) & (reach < hi)
        if not m.any():
            continue
        parts = "  ".join(f"{n.split('_camera')[0]:>7} {rms(np.array([r['pe'][n] for r in rows])[m] - gt[m]):5.2f}"
                          for n in camera_goal.GOAL_SLOTS)
        print(f"  goal {lo:.2f}-{hi:.2f} m away, n={int(m.sum()):3d}:  {parts}   median {rms(pred[m]-gt[m]):5.2f}")


if __name__ == "__main__":
    main()
