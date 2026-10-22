#!/usr/bin/env python

"""Generate synthetic ortho frames with the grip_o SDXL LoRA and stage them for the ortho target labeler.

Each prompt is a random combination of room type, floor, furniture and clutter, and the LoRA is
weakened to --scales so the base model can paint rooms the LoRA never saw while keeping the ortho
look. At full strength the LoRA just reproduces the training rooms, which the targeting model
already handles.

Frames are generated at 1024, downsized to 512 (the real ortho size) and written as JPEGs into the
labeler's frame cache under a source name of their own, so every label made from them is named
user-<source>-<frame>.parquet and can never be mistaken for a label of a real dataset. Frames are
numbered after the last one already in the cache, so a re-run adds to it rather than replacing it.
manifest.jsonl beside them records the prompt, seed and LoRA scale of each.

Usage (from the repo root):
    python experiments/anchor_diffusion/generate_for_labeling.py --num 300

Then label them, see README.md.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from PIL import Image

from train_lora_sdxl import BASE_MODEL, FP16_VAE

HERE = Path(__file__).parent
SOURCE = "synthetic_sdxl1024_ortho"

ROOMS = [
    "child's bedroom", "master bedroom", "teen bedroom", "woodworking workshop", "garage workshop",
    "kitchen", "narrow hallway", "playroom", "home office", "living room", "dining room",
    "laundry room", "basement rec room", "dorm room", "nursery", "mudroom entryway", "craft room",
    "studio apartment", "classroom", "daycare room",
]
FLOORS = [
    "beige carpet", "grey carpet", "blue shag rug", "light hardwood floor", "dark hardwood floor",
    "white tile floor", "checkered linoleum floor", "polished concrete floor", "terracotta tile floor",
    "patterned persian rug over wood floor", "foam play mats", "bamboo floor", "laminate wood floor",
    "green carpet", "black and white hexagon tile",
]
FURNITURE = [
    "a double bed", "a crib", "bunk beds", "a dining table and chairs", "a kitchen island",
    "a workbench and tool chest", "an office desk and rolling chair", "a sofa and coffee table",
    "a bookshelf", "a dresser", "a washing machine and dryer", "a toy kitchen", "a piano",
    "a bean bag chair", "a shoe rack", "storage bins", "a treadmill",
]
CLUTTER = [
    "toys", "plastic building blocks", "stuffed animals", "toy cars", "laundry", "socks",
    "t-shirts and jeans", "towels", "trash", "crumpled paper", "empty bottles and cans",
    "cardboard boxes", "shoes", "books", "tools and wood scraps", "cables", "board game pieces",
    "balls", "dog toys", "mail and envelopes", "plastic bags", "pillows and blankets",
]
AMOUNTS = [
    ("a clean empty floor", 0), ("a nearly clean floor with one item of", 1),
    ("a few scattered", 2), ("some scattered", 3), ("lots of", 3), ("a very messy floor covered in", 3),
]


def make_prompt(rng, token):
    room, floor = rng.choice(ROOMS), rng.choice(FLOORS)
    furniture = rng.sample(FURNITURE, rng.choice([0, 0, 1, 1, 2]))
    amount, kinds = rng.choice(AMOUNTS)
    parts = [f"{token} ortho top-down view of a {room}", floor]
    parts.append(" and ".join(furniture) if furniture else "no furniture")
    if kinds == 0:
        parts.append(amount)
    else:
        parts.append(f"{amount} {' and '.join(rng.sample(CLUTTER, min(kinds, 2)))}")
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lora", default=str(HERE / "runs" / "grip_o_ortho_r32"))
    parser.add_argument("--num", type=int, default=300)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.4, 0.55, 0.7],
                        help="LoRA strengths, cycled frame by frame")
    parser.add_argument("--cache", default=f"ortho_labeler_frames/{SOURCE}", help="labeler frame cache, relative to the repo root")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--token", default="strngmn")
    parser.add_argument("--seed", type=int, default=None,
                        help="prompt and image seed (default: the first frame number, so each run differs)")
    args = parser.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    existing = [int(p.stem.split("-")[0]) for p in cache.glob("*.jpg")]
    first = max(existing) + 1 if existing else 0
    seed = first if args.seed is None else args.seed
    rng = random.Random(seed)

    vae = AutoencoderKL.from_pretrained(FP16_VAE, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(BASE_MODEL, vae=vae, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.load_lora_weights(args.lora, adapter_name="ortho")
    pipe.set_progress_bar_config(disable=True)

    with open(cache / "manifest.jsonl", "a") as manifest:
        for k in range(args.num):
            prompt = make_prompt(rng, args.token)
            scale = args.scales[k % len(args.scales)]
            pipe.set_adapters(["ortho"], adapter_weights=[scale])
            gen = torch.Generator("cuda").manual_seed(seed * 1000 + k)
            image = pipe(prompt, width=1024, height=1024, num_inference_steps=args.steps,
                         guidance_scale=args.guidance, generator=gen).images[0]
            # episode-offset shaped ids, like the labeler's own extracted frames
            frame_id = f"{first + k:06d}-000000"
            image.resize((512, 512), Image.LANCZOS).save(cache / f"{frame_id}.jpg", quality=95)
            manifest.write(json.dumps({"frame": frame_id, "prompt": prompt, "seed": seed * 1000 + k,
                                       "lora_scale": scale, "kind": "variety",
                                       "lora": os.path.abspath(args.lora)}) + "\n")
            manifest.flush()
            print(f"{k + 1}/{args.num} {frame_id} s={scale} {prompt}", flush=True)
    print(f"wrote {args.num} frames to {cache} ({first:06d} onward)")


if __name__ == "__main__":
    main()
