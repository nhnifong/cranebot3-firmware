#!/usr/bin/env python

"""Predict where the robot could reach next in the ortho floor view; readme.md has the pipeline."""

import argparse
import logging

from nf_robot.ml.ortho_target.dataset import (
    DEFAULT_DATASET_ID,
    DEFAULT_NEGATIVES_REPO_ID,
    NEGATIVE_INTERVAL_S,
    DEFAULT_SOURCE_REPO_ID,
    LOCAL_DATASET_ROOT,
    POOL_SPLIT,
    USER_LABEL_DATASET_NAME,
    USER_LABEL_ROOT,
    distill,
    distill_negatives,
    merge_labels,
    split_dataset,
    upload_user_labels,
)
from nf_robot.ml.ortho_target.model import (
    CELL_SIGMA,
    DEFAULT_ATTENTION_LAYERS,
    DEFAULT_BACKBONE,
    DEFAULT_GRID,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_PATH,
)
from nf_robot.ml.ortho_target.training import SELECTION_METRIC, evaluate, train


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    parser = argparse.ArgumentParser(
        prog="python -m nf_robot.ml.ortho_target",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    distill_parser = subparsers.add_parser(
        "distill", help="reduce an ortho LeRobot dataset to one image + contact point per episode"
    )
    distill_parser.add_argument("--repo_id", default=DEFAULT_SOURCE_REPO_ID,
                                help="LeRobot dataset built by recipes/combined_targets_reblend.yaml, whole "
                                     "and unsplit")
    distill_parser.add_argument("--root", default=None,
                                help="local root of that dataset (default: the HF cache, downloading if needed)")
    distill_parser.add_argument("--output", default=LOCAL_DATASET_ROOT,
                                help=f"directory to write the distilled dataset into. Rows land "
                                     f"in its {POOL_SPLIT}/ pool; `split` deals them into train "
                                     f"and eval once the hand labels have been merged in")
    distill_parser.add_argument("--pressure_threshold", type=float, default=0.1,
                                help="finger_pressure above which an episode counts as having made contact")
    distill_parser.add_argument("--frame_offset", type=int, default=0,
                                help="which frame of each episode to keep, counted from its start")
    distill_parser.add_argument("--frames_per_episode", type=int, default=8,
                                help="ortho frames to take from each episode. They share the "
                                     "episode's one label, which is valid for every frame "
                                     "before the grasp because the target does not move")
    distill_parser.add_argument("--frame_stride", type=int, default=4,
                                help="frames between them; too small and they are the same picture")
    distill_parser.add_argument("--min_coverage", type=float, default=0.02,
                                help="skip frames where less than this fraction of the ortho map was "
                                     "painted by any camera")
    distill_parser.add_argument("--limit", type=int, default=0,
                                help="stop after this many samples, for a quick trial run")
    distill_parser.add_argument("--annotate_dir", default=None,
                                help="also write copies with the label drawn on them, to check the projection")

    negatives_parser = subparsers.add_parser(
        "distill_negatives",
        help="add target-free frames from a dataset recorded with nothing in view to the pool")
    negatives_parser.add_argument("--repo_id", default=DEFAULT_NEGATIVES_REPO_ID,
                                  help="LeRobot dataset with an ortho view and no graspable "
                                       "targets in any episode")
    negatives_parser.add_argument("--root", default=None,
                                  help="local root of that dataset (default: the HF cache, "
                                       "downloading if needed)")
    negatives_parser.add_argument("--output", default=LOCAL_DATASET_ROOT,
                                  help=f"distilled dataset directory; frames join its "
                                       f"{POOL_SPLIT}/ pool")
    negatives_parser.add_argument("--interval_s", type=float, default=NEGATIVE_INTERVAL_S,
                                  help="seconds between frames taken from each episode")
    negatives_parser.add_argument("--min_coverage", type=float, default=0.02,
                                  help="skip frames where less than this fraction of the ortho "
                                       "map was painted by any camera")
    negatives_parser.add_argument("--limit", type=int, default=0,
                                  help="stop after this many frames, for a quick trial run")

    upload_labels_parser = subparsers.add_parser(
        "upload_labels", help="push targets saved from the UI to a hub dataset of your own")
    upload_labels_parser.add_argument("--source", default=USER_LABEL_ROOT,
                                      help="directory the UI action wrote them to")
    upload_labels_parser.add_argument("--dataset_id", default=None,
                                      help=f"hub dataset to add them to "
                                           f"(default: <your account>/{USER_LABEL_DATASET_NAME})")
    upload_labels_parser.add_argument("--public", action="store_true",
                                      help="create the repo public. It is private by default "
                                           "because these are pictures of your floor")

    merge_labels_parser = subparsers.add_parser(
        "merge_labels", help="fold targets saved from the UI into the distilled dataset")
    source = merge_labels_parser.add_mutually_exclusive_group()
    source.add_argument("--source", default=USER_LABEL_ROOT,
                        help="local directory of label files to merge")
    source.add_argument("--repo_id", default=None,
                        help="merge a hub dataset of label files instead, e.g. one upload_labels wrote")
    merge_labels_parser.add_argument("--output", default=LOCAL_DATASET_ROOT,
                                     help="distilled dataset directory to merge them into")
    merge_labels_parser.add_argument("--split", default=POOL_SPLIT,
                                     choices=[POOL_SPLIT, "train", "eval"],
                                     help=f"which directory they join. The default is the "
                                          f"{POOL_SPLIT}/ pool, so `split` decides afterwards how "
                                          f"many of them land in eval - which is what makes the "
                                          f"selection metric computable. Naming train or eval "
                                          f"instead puts them wholly on one side")
    merge_labels_parser.add_argument("--tag", default=None,
                                     help="name the contributor these labels came from, so "
                                          "several people's labels of the same dataset do not "
                                          "overwrite each other (default: the --repo_id owner)")
    merge_labels_parser.add_argument("--no_resize", action="store_true",
                                     help="keep the frames at the size they were saved at instead "
                                          "of matching the split's distilled shards")

    split_parser = subparsers.add_parser(
        "split", help="deal the pooled rows into train and eval at random")
    split_parser.add_argument("--data_root", default=LOCAL_DATASET_ROOT,
                              help=f"dataset directory holding the {POOL_SPLIT}/ pool")
    split_parser.add_argument("--eval_fraction", type=float, default=0.1,
                              help="share of rows held out for eval")
    split_parser.add_argument("--seed", type=int, default=0,
                              help="the same seed deals the same split, so a re-deal is "
                                   "reproducible and a new one is a new number")
    split_parser.add_argument("--dataset_id", default=DEFAULT_DATASET_ID,
                              help="hub repo to upload the split dataset to")
    split_parser.add_argument("--upload", action="store_true",
                              help="upload the result, replacing the hub copy. The pool stays "
                                   "local; only train and eval go up")

    def add_data_args(sub):
        sub.add_argument("--dataset_id", default=DEFAULT_DATASET_ID, help="distilled dataset on the hub")
        sub.add_argument("--data_root", default=None,
                         help="local distilled dataset directory (default: download --dataset_id)")
        sub.add_argument("--batch_size", type=int, default=32)
        sub.add_argument("--workers", type=int, default=4)
        sub.add_argument("--top_k", type=int, default=5,
                         help="candidates to decode; the top-k hit rate is the 'picked something "
                              "plausible' measure, which single-answer error cannot capture")
        sub.add_argument("--device", default=None)
        sub.add_argument("--model_path", default=DEFAULT_MODEL_PATH)

    train_parser = subparsers.add_parser("train", help="fit the model on the distilled dataset")
    add_data_args(train_parser)
    train_parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    train_parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    train_parser.add_argument("--grid", type=int, default=DEFAULT_GRID,
                              help="output cells per side; 128 over a 5m map is 3.9cm per cell")
    train_parser.add_argument("--fuse_layers", type=int, default=4,
                              help="how many of the backbone's last blocks to concatenate")
    train_parser.add_argument("--attention_layers", type=int, default=DEFAULT_ATTENTION_LAYERS,
                              help="self-attention blocks over the token grid before "
                                   "upsampling; 0 trains the convolution-only decoder")
    train_parser.add_argument("--no_attention_skip", action="store_true",
                              help="feed the decoder only the attended map, without the "
                                   "pre-attention skip connection")
    train_parser.add_argument("--epochs", type=int, default=60)
    # The learning rate is paired with --batch_size (sqrt scaling).
    train_parser.add_argument("--lr", type=float, default=6e-4)
    train_parser.add_argument("--weight_decay", type=float, default=0.05)
    train_parser.add_argument("--offset_weight", type=float, default=1.0)
    train_parser.add_argument("--cell_sigma", type=float, default=CELL_SIGMA,
                              help="width in cells of the Gaussian the cell head is trained "
                                   "against; 0 restores a one-hot target")
    train_parser.add_argument("--pos_weight", type=float, default=0,
                              help="weight on the objectness positives; 0 uses the balanced "
                                   "value for --grid and --cell_sigma. Raising it finds more "
                                   "of the objects nobody labelled, at more false positives. "
                                   "See objectness_loss")
    train_parser.add_argument("--select_metric", default=SELECTION_METRIC,
                              help="eval metric that picks the saved checkpoint")
    train_parser.add_argument("--translate_px", type=int, default=48,
                              help="max random shift of the map and its label, in model "
                                   "pixels; 0 disables it")
    train_parser.add_argument("--eval_every", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--unfreeze_backbone", action="store_true",
                              help="also train the backbone, at --backbone_lr_scale of the head's rate")
    train_parser.add_argument("--backbone_lr_scale", type=float, default=0.05)

    eval_parser = subparsers.add_parser("evaluate", help="score a checkpoint on a held-out split")
    add_data_args(eval_parser)
    eval_parser.add_argument("--split", default="eval", choices=["train", "eval"])
    eval_parser.add_argument("--tta", action="store_true",
                             help="average predictions over the 8 square symmetries")
    eval_parser.add_argument("--preview_dir", default=None,
                             help="write images with the label and the predictions drawn on")

    args = parser.parse_args()
    if args.command == "distill":
        distill(args)
    elif args.command == "distill_negatives":
        distill_negatives(args)
    elif args.command == "split":
        split_dataset(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "upload_labels":
        upload_user_labels(args.source, args.dataset_id, private=not args.public)
    elif args.command == "merge_labels":
        merge_labels(args)


if __name__ == "__main__":
    main()
