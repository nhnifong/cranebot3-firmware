#!/usr/bin/env python

"""Can SigLIP 2 separate the objects on the floor from the floor? A no-training probe.

The question this answers, before any of the work of building a prompt-conditioned target
model: does a frozen SigLIP 2 carry enough language-grounded, spatially localized signal
on real ortho floor views that a prompt like "a pile of laundry on the floor" lights up
the laundry. Nothing here trains. It reads frames, scores them against text, renders what
came out, and prints the numbers that say whether to believe it.

    python experiments/siglip2_probe.py --out siglip2_probe
    python experiments/siglip2_probe.py --prompts "a sock" "a shoe" "bare carpet" --frames 16

Frames come from the distilled dataset ortho_target already publishes, so each one arrives
with the operator's grasp point attached - which makes the maps scoreable without anyone
labelling anything new. That point is not the whole answer, since every object on the
floor was a valid grasp and the operator picked one, but a map that cannot rank it above
the bare floor is not carrying object signal either.

Crops, not patches
------------------
`--mode crops`, the default, cuts the frame into overlapping windows, embeds each whole
window the way SigLIP's own zero-shot classification does, and accumulates the scores back
into a full-resolution map. This is the mode that works. `--mode patches` is kept as the
control and does not: it takes per-patch vectors through the pooling head (see
patch_embeddings for how that projection is built) and dots them against text, which
produces salt-and-pepper noise on this domain even though the arithmetic is right.

Two properties of SigLIP make the crop mode work and are easy to get wrong:

  Raw cosine is dominated by a per-crop magnitude term - a busy crop scores higher against
  every prompt, including a deliberately absurd one - so a per-prompt cosine map mostly
  draws "how much stuff is here". The softmax *across the prompt set* cancels it, because
  the bias is shared by every prompt on that crop. Everything below is computed on those
  softmax probabilities.

  That only works if the prompt set spans the frame. At least one prompt has to describe
  empty floor, or the softmax has nowhere to put a crop with nothing in it and the
  probability mass lands on whichever object prompt is least wrong. --control adds an
  absurd prompt that should stay dark everywhere; if it does not, the maps are noise.

How to read the output
----------------------
Per frame a contact sheet PNG, and over the run a table plus summary.json:

  label percentile  Where the grasp point ranks in that prompt's map, 0-100, chance 50.
  hit@Ncm           How often the map's best point lands within N cm of the grasp point.
  control max       The absurd prompt's highest probability anywhere in the frame. This is
                    the noise floor: object prompts have to beat it to mean anything.
  floor/object gap  Median probability the floor prompts assign at the grasp point,
                    subtracted from the object prompts'. Negative means the model thinks
                    the operator reached for bare carpet.

The panels: the frame with the grasp point circled, one per prompt showing that prompt's
probability, a confidence panel holding the winning probability per location, and an
argmax panel colouring each location by which prompt won it.

Panels are drawn on probabilities in [0, 1], so unlike cosines they are comparable across
tiles and across frames, and the per-tile maximum in each subtitle is directly meaningful.

Window size is the parameter that matters most. A window has to contain the object with
enough context to name it and little enough that the object is not a speck: at the default
5 m map, --window 128 is 1.25 m of floor. Several sizes can be given and their maps are
averaged, which is what covers a sock and a laundry pile in one pass.
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from nf_robot.ml.ortho_target import ORTHO_EXTENT_M

DEFAULT_MODEL = "google/siglip2-base-patch16-512"
# The set has to span the frame: object prompts plus at least one that describes empty
# floor, or the softmax has nowhere to put an empty crop. See the module docstring.
DEFAULT_PROMPTS = (
    "a pile of laundry on the floor",
    "a sock on the floor",
    "a shoe on the floor",
    "a toy on the floor",
    "bare empty carpet floor",
    "a bare wooden floor",
)
# Prompts naming empty floor rather than an object, by index into the list above. Scoring
# needs to know which is which; --floor_prompts overrides when the list is replaced.
DEFAULT_FLOOR_PROMPTS = ("bare empty carpet floor", "a bare wooden floor")
# Always in the softmax, never expected to win: whatever probability it reaches is the
# noise floor for that frame.
DEFAULT_CONTROL = "a photograph of a dog"
# SigLIP's text tower was trained on captions, so a bare noun is out of distribution.
DEFAULT_TEMPLATE = "a photo of {}."
# The text tower is trained at a fixed 64 tokens and pools the last position, so anything
# shorter pools over padding and shifts the embedding.
TEXT_LENGTH = 64
DEFAULT_WINDOWS = (128, 192)
DEFAULT_RADII_CM = (20, 50)
FRAME_SIZE = 512


def load_frames(root: Path, split: str, count: int, seed: int):
    """`count` frames from a distilled split, as (BGR, (u, v), name) at FRAME_SIZE."""
    from nf_robot.ml.ortho_target import OrthoTargetDataset

    dataset = OrthoTargetDataset(root, split, image_size=FRAME_SIZE, augment=False)
    rng = np.random.default_rng(seed)
    picks = rng.permutation(len(dataset))[:count]
    frames = []
    for idx in sorted(int(i) for i in picks):
        bgr = dataset.decode(idx)
        height, width = bgr.shape[:2]
        # A frame may carry several labels; the probe scores against the first.
        (u, v) = dataset.samples[idx]["points"][0]
        u, v = u * FRAME_SIZE / width, v * FRAME_SIZE / height
        frames.append((cv2.resize(bgr, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA),
                       (u, v), dataset.samples[idx]["file_name"]))
    return frames


class Siglip:
    """The frozen model, its preprocessing, and embeddings on the CPU for scoring."""

    def __init__(self, model_id, device):
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id)
        self.device = device
        self.model = AutoModel.from_pretrained(model_id).eval().to(device)
        self.tokenizer = processor.tokenizer
        self.mean = torch.tensor(processor.image_processor.image_mean, device=device).view(3, 1, 1)
        self.std = torch.tensor(processor.image_processor.image_std, device=device).view(3, 1, 1)
        self.size = processor.image_processor.size["height"]
        self.patch = self.model.config.vision_config.patch_size
        # SigLIP's own zero-shot use applies this before comparing prompts; without it the
        # softmax over a 0.1-wide cosine spread comes out nearly uniform.
        self.logit_scale = self.model.logit_scale.exp().detach().cpu()

    @staticmethod
    def _pooled(out):
        """Newer transformers hand back the whole model output; the vector is its pooled one."""
        return getattr(out, "pooler_output", out)

    def text(self, prompts, template):
        texts = [template.format(p) if template else p for p in prompts]
        batch = self.tokenizer(texts, padding="max_length", max_length=TEXT_LENGTH,
                               truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeds = self._pooled(self.model.get_text_features(**batch))
        return F.normalize(embeds, dim=-1).cpu(), texts

    def pixels(self, bgr_list):
        batch = []
        for bgr in bgr_list:
            rgb = cv2.cvtColor(cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
            batch.append((tensor - self.mean) / self.std)
        return torch.stack(batch)

    def images(self, bgr_list, batch_size=16):
        """Unit-norm pooled embeddings, one row per image."""
        out = []
        for start in range(0, len(bgr_list), batch_size):
            with torch.no_grad():
                embeds = self._pooled(self.model.get_image_features(
                    pixel_values=self.pixels(bgr_list[start:start + batch_size])))
            out.append(F.normalize(embeds, dim=-1).cpu())
        return torch.cat(out)


def window_origins(size, window, stride):
    """Window start coordinates covering the axis, with the far edge always included."""
    starts = list(range(0, size - window + 1, stride))
    if starts[-1] != size - window:
        starts.append(size - window)
    return starts


def crop_probabilities(siglip, bgr, text, windows, stride_fraction, batch_size):
    """Per-pixel prompt probabilities, as (P, size, size).

    Every window votes over the area it covers and the votes are averaged, so overlapping
    scales combine without any of them having to share a grid. Softmax across prompts runs
    per crop, before accumulation, because that is where the per-crop magnitude bias lives
    - averaging raw cosines first would carry it into the result.
    """
    size = bgr.shape[0]
    total = np.zeros((len(text), size, size), dtype=np.float64)
    count = np.zeros((size, size), dtype=np.float64)

    for window in windows:
        stride = max(1, int(round(window * stride_fraction)))
        origins = [(x, y) for y in window_origins(size, window, stride)
                   for x in window_origins(size, window, stride)]
        crops = [bgr[y:y + window, x:x + window] for x, y in origins]
        cosines = siglip.images(crops, batch_size) @ text.T
        probs = torch.softmax(cosines * siglip.logit_scale, dim=1).numpy()
        for (x, y), row in zip(origins, probs):
            total[:, y:y + window, x:x + window] += row[:, None, None]
            count[y:y + window, x:x + window] += 1.0
    return total / np.maximum(count, 1.0)


def patch_probabilities(siglip, bgr, text, layer, features):
    """The control path: per-patch embeddings against text, as (P, grid, grid).

    Softmaxed across prompts like the crop path, so the two are read the same way.
    """
    vision = siglip.model.vision_model
    pixel_values = siglip.pixels([bgr])
    with torch.no_grad():
        feats = patch_embeddings(vision, pixel_values, features, layer)
        cosines = (F.normalize(feats, dim=-1)[0].cpu() @ text.T)
        probs = torch.softmax(cosines * siglip.logit_scale, dim=1)
    grid = siglip.size // siglip.patch
    return probs.T.reshape(len(text), grid, grid).float().numpy()


def hidden_tokens(vision, pixel_values, layer: int):
    """Patch tokens out of one encoder layer, post-layernormed.

    A forward hook rather than output_hidden_states, which this model silently returns
    None for. The pooling head expects post-layernorm input, so an intermediate layer gets
    it applied too - otherwise --layer measures the missing norm as much as the layer.
    """
    if layer == -1:
        return vision(pixel_values).last_hidden_state

    captured = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = output[0] if isinstance(output, tuple) else output

    handle = vision.encoder.layers[layer].register_forward_hook(hook)
    try:
        vision(pixel_values)
    finally:
        handle.remove()
    return vision.post_layernorm(captured["hidden"])


def patch_embeddings(vision, pixel_values, features: str, layer: int):
    """Per-patch vectors in the joint image-text space, as (B, N, D).

    SigLIP's joint space is the output of the vision tower's attention-pooling head, not
    the patch tokens, so a cosine against a raw patch token compares two different spaces.
    `map` runs that head with its attention collapsed onto one token at a time - exactly
    what head(hidden) would compute if the probe attended to that patch alone - so the
    result is in the joint space by construction. `value` stops at the linear part, which
    is what the real pooled embedding is a weighted sum of, and `raw` skips the head as the
    control that should score at chance.
    """
    hidden = hidden_tokens(vision, pixel_values, layer)
    head = vision.head
    if features == "raw":
        return hidden

    attention = head.attention
    dim = hidden.shape[-1]
    linear = attention.out_proj(F.linear(hidden, attention.in_proj_weight[2 * dim:],
                                         attention.in_proj_bias[2 * dim:]))
    if features == "value":
        return linear
    return linear + head.mlp(head.layernorm(linear))


def px_per_cm(size):
    return size / (ORTHO_EXTENT_M * 100.0)


def score_map(prob, label_uv, size, radii_cm):
    """How well one prompt's map agrees with one grasp point."""
    grid = prob.shape[0]
    scale = size / grid
    u, v = label_uv
    col = int(np.clip(u / scale, 0, grid - 1))
    row = int(np.clip(v / scale, 0, grid - 1))
    flat = prob.reshape(-1)
    at_label = float(prob[row, col])

    best = int(np.argmax(flat))
    bu = (best % grid + 0.5) * scale
    bv = (best // grid + 0.5) * scale
    error_cm = float(np.hypot(bu - u, bv - v) / px_per_cm(size))
    return {
        "label_percentile": float((flat < at_label).mean() * 100.0),
        "label_prob": at_label,
        "argmax_error_cm": error_cm,
        **{f"hit@{r}cm": bool(error_cm <= r) for r in radii_cm},
        "prob_max": float(flat.max()),
        "prob_mean": float(flat.mean()),
    }


def heat_panel(bgr, prob, title):
    """One probability map over a desaturated frame, on a fixed 0..1 scale."""
    size = bgr.shape[0]
    heat = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (size, size), interpolation=cv2.INTER_NEAREST)
    grey = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return label_panel(cv2.addWeighted(grey, 0.45, heat, 0.55, 0), title, f"max {prob.max():.2f}")


def label_panel(panel, title, subtitle=None):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(panel, title[:46], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if subtitle:
        cv2.putText(panel, subtitle, (6, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)
    return panel


def argmax_panel(probs, size, prompts):
    """Which prompt won each location, one hue per prompt in the order they were given."""
    # 255 // n rather than 255 // (n - 1): the hue wheel wraps, so spanning it inclusively
    # would give the first and last prompt the same colour.
    colours = cv2.applyColorMap((np.arange(len(prompts)) * (255 // len(prompts))).astype(np.uint8),
                                cv2.COLORMAP_HSV).reshape(-1, 3)
    panel = colours[probs.argmax(0).astype(np.uint8)]
    panel = cv2.resize(panel, (size, size), interpolation=cv2.INTER_NEAREST)
    return label_panel(panel, "argmax prompt")


def contact_sheet(panels, columns=4):
    rows = []
    for start in range(0, len(panels), columns):
        row = panels[start:start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def probe(args):
    device = torch.device(args.device)
    root = resolve_root(args)
    frames = load_frames(root, args.split, args.frames, args.seed)

    siglip = Siglip(args.model, device)
    prompts = list(args.prompts) + [args.control]
    text, rendered = siglip.text(prompts, args.template)
    floor = set(args.floor_prompts)
    objects = [p for p in args.prompts if p not in floor]
    logging.info(f"{args.model} at {siglip.size}px, mode={args.mode}, "
                 f"windows={args.windows} ({', '.join(f'{w * ORTHO_EXTENT_M * 100 / FRAME_SIZE:.0f}cm' for w in args.windows)})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, controls, gaps = [], [], []

    for bgr, (u, v), name in frames:
        if args.mode == "crops":
            probs = crop_probabilities(siglip, bgr, text, args.windows,
                                       args.stride_fraction, args.batch_size)
        else:
            probs = patch_probabilities(siglip, bgr, text, args.layer, args.features)

        control_max = float(probs[-1].max())
        controls.append(control_max)
        at_label = {p: float(score_map(m, (u, v), FRAME_SIZE, args.radii_cm)["label_prob"])
                    for p, m in zip(prompts, probs)}
        gap = (np.mean([at_label[p] for p in objects])
               - np.mean([at_label[p] for p in floor if p in at_label]))
        gaps.append(float(gap))

        for prompt, prob in zip(prompts, probs):
            rows.append({"frame": name, "prompt": prompt, "control_max": control_max,
                         "object_floor_gap": float(gap),
                         **score_map(prob, (u, v), FRAME_SIZE, args.radii_cm)})

        marked = bgr.copy()
        cv2.circle(marked, (int(u), int(v)), 9, (0, 255, 0), 2)
        panels = [label_panel(marked, name[:46], "grasp point")]
        panels += [heat_panel(bgr, prob, prompt) for prompt, prob in zip(prompts, probs)]
        panels.append(heat_panel(bgr, probs.max(0), "confidence (winning prob)"))
        panels.append(argmax_panel(probs, FRAME_SIZE, prompts))
        cv2.imwrite(str(out_dir / f"{Path(name).stem}.png"), contact_sheet(panels))
        logging.info(f"{name}: control max {control_max:.2f}, object-floor gap {gap:+.3f}")

    report(rows, args, prompts, controls, gaps, rendered, out_dir)


def report(rows, args, prompts, controls, gaps, rendered, out_dir):
    print(f"\n{len(rows) // len(prompts)} frames, mode={args.mode}, model={args.model}")
    header = f"{'prompt':34s} {'label pct':>10s} {'p@label':>8s} {'p max':>8s} {'argmax cm':>10s}"
    for radius in args.radii_cm:
        header += f" {f'hit@{radius}cm':>10s}"
    print(header)
    print("-" * len(header))

    summary = {}
    for prompt in prompts:
        picked = [r for r in rows if r["prompt"] == prompt]
        entry = {
            "median_label_percentile": float(np.median([r["label_percentile"] for r in picked])),
            "median_label_prob": float(np.median([r["label_prob"] for r in picked])),
            "median_prob_max": float(np.median([r["prob_max"] for r in picked])),
            "median_argmax_error_cm": float(np.median([r["argmax_error_cm"] for r in picked])),
            **{f"hit@{r}cm": float(np.mean([p[f"hit@{r}cm"] for p in picked])) for r in args.radii_cm},
        }
        summary[prompt] = entry
        tag = " (control)" if prompt == args.control else ""
        line = (f"{(prompt + tag)[:34]:34s} {entry['median_label_percentile']:10.1f} "
                f"{entry['median_label_prob']:8.3f} {entry['median_prob_max']:8.3f} "
                f"{entry['median_argmax_error_cm']:10.0f}")
        for radius in args.radii_cm:
            line += f" {entry[f'hit@{radius}cm']:10.2f}"
        print(line)

    print(f"\nlabel percentile: 50 is chance. control reaches {np.mean(controls):.2f} at its "
          f"best, which is the noise floor every object prompt has to beat. "
          f"object-floor gap at the grasp point {np.mean(gaps):+.3f} "
          f"(negative means the model calls the grasp point bare floor).")

    payload = {
        "model": args.model, "mode": args.mode, "windows": args.windows,
        "template": args.template, "rendered_prompts": rendered,
        "frames": len(rows) // len(prompts),
        "mean_control_max": float(np.mean(controls)),
        "mean_object_floor_gap": float(np.mean(gaps)),
        "per_prompt": summary, "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(f"\npanels and summary.json in {out_dir}")


def resolve_root(args) -> Path:
    """The same local-wins-over-hub shape ortho_target.resolve_data_root has."""
    if args.data_root:
        return Path(args.data_root)
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo_id=args.dataset_id, repo_type="dataset"))
    logging.info(f"Using {args.dataset_id} from the hub at {root}")
    return root


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    from nf_robot.ml.ortho_target import DEFAULT_DATASET_ID, LOCAL_DATASET_ROOT

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data_root", default=None,
                        help=f"a distilled dataset directory, e.g. {LOCAL_DATASET_ROOT}; "
                             f"without it the hub copy is downloaded")
    parser.add_argument("--dataset_id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=("crops", "patches"), default="crops",
                        help="crops: whole-window pooled embeddings, the mode that works; "
                             "patches: per-patch embeddings, kept as the control")
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS),
                        help="must include at least one prompt describing empty floor")
    parser.add_argument("--floor_prompts", nargs="+", default=list(DEFAULT_FLOOR_PROMPTS),
                        help="which of --prompts name bare floor rather than an object")
    parser.add_argument("--control", default=DEFAULT_CONTROL,
                        help="an absurd prompt that should stay dark; its peak is the noise floor")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE,
                        help="wraps each prompt; --template '{}' to pass them through raw")
    parser.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS),
                        help="crop sizes in frame pixels; their maps are averaged")
    parser.add_argument("--stride_fraction", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--features", choices=("map", "value", "raw"), default="value",
                        help="patches mode only; see patch_embeddings")
    parser.add_argument("--layer", type=int, default=-3,
                        help="patches mode only: encoder layer to read from. Pass it as "
                             "--layer=-3, or argparse reads the minus as a flag")
    parser.add_argument("--radii_cm", type=int, nargs="+", default=list(DEFAULT_RADII_CM))
    parser.add_argument("--out", default="siglip2_probe")
    parser.add_argument("--device", default=default_device())
    probe(parser.parse_args())


if __name__ == "__main__":
    main()
