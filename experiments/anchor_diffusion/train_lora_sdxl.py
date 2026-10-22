#!/usr/bin/env python

"""LoRA fine-tune of SDXL base on anchor camera frames (see README.md).

Single-GPU loop, no accelerate. Captions come from a small fixed set (one per camera), so their
text embeddings are computed once up front; VAE latents for every frame are also precomputed and
kept on the GPU. Only the UNet attention LoRA weights train.

Images are not flipped or cropped: anchor views have fixed geometry and flipping would teach the
model mirror-image rooms.

Usage:
    python experiments/anchor_diffusion/train_lora_sdxl.py \
        --data_dir experiments/anchor_diffusion/data/bedroom-oct15 \
        --output_dir experiments/anchor_diffusion/runs/bedroom-oct15-r32
"""

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.training_utils import compute_snr
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
FP16_VAE = "madebyollin/sdxl-vae-fp16-fix"


def load_records(data_dir):
    with open(os.path.join(data_dir, "metadata.jsonl")) as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def encode_prompts(prompts, tokenizers, text_encoders, device):
    """SDXL prompt embedding: penultimate hidden states of both encoders concatenated, pooled output of the second."""
    embeds, pooled = [], None
    for tokenizer, encoder in zip(tokenizers, text_encoders):
        ids = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(device)
        out = encoder(ids, output_hidden_states=True)
        pooled = out[0]  # only the second encoder's (projection) pooled output is kept
        embeds.append(out.hidden_states[-2])
    return torch.cat(embeds, dim=-1), pooled


@torch.no_grad()
def encode_images(records, data_dir, vae, width, height, device, batch_size=4):
    latents = []
    for start in range(0, len(records), batch_size):
        batch = []
        for r in records[start:start + batch_size]:
            img = Image.open(os.path.join(data_dir, r["file_name"])).convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height), Image.BICUBIC)
            batch.append(torch.from_numpy(np.array(img)).permute(2, 0, 1))
        pixels = torch.stack(batch).to(device, torch.float16) / 127.5 - 1.0
        latents.append(vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor)
        print(f"encoded {min(start + batch_size, len(records))}/{len(records)}", end="\r")
    print()
    return torch.cat(latents)


def save_lora(unet, path):
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(path, unet_lora_layers=state, safe_serialization=True)


@torch.no_grad()
def sample(pipe, prompts, width, height, steps, seed, path, max_images=4):
    gen = torch.Generator("cuda").manual_seed(seed)
    # one prompt per call keeps VAE decode memory flat at 1024px
    # autocast because the UNet is bf16, its LoRA weights fp32 and the VAE fp16
    with torch.autocast("cuda", dtype=torch.bfloat16):
        images = [pipe(p, width=width, height=height, num_inference_steps=steps, generator=gen).images[0]
                  for p in prompts[:max_images]]
    grid = Image.new("RGB", (width * len(images), height))
    for i, im in enumerate(images):
        grid.paste(im, (i * width, 0))
    grid.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--snr_gamma", type=float, default=5.0, help="min-SNR loss weighting, 0 disables")
    parser.add_argument("--caption_dropout", type=float, default=0.05,
                        help="fraction of steps trained with an empty prompt, keeps CFG working")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--sample_steps", type=int, default=30)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert args.width % 8 == 0 and args.height % 8 == 0, "resolution must be a multiple of 8"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizers = [AutoTokenizer.from_pretrained(BASE_MODEL, subfolder=s, use_fast=False) for s in ("tokenizer", "tokenizer_2")]
    text_encoders = [
        CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16),
        CLIPTextModelWithProjection.from_pretrained(BASE_MODEL, subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float16),
    ]
    vae = AutoencoderKL.from_pretrained(FP16_VAE, torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet", variant="fp16", torch_dtype=torch.bfloat16)
    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    for m in (*text_encoders, vae, unet):
        m.requires_grad_(False)
        m.to(device)

    records = load_records(args.data_dir)
    prompts = sorted({r["text"] for r in records})
    print(f"{len(records)} images, captions: {prompts}")

    prompt_embeds, pooled_embeds = encode_prompts(prompts + [""], tokenizers, text_encoders, device)
    prompt_index = torch.tensor([prompts.index(r["text"]) for r in records], device=device)
    empty_index = len(prompts)
    latents = encode_images(records, args.data_dir, vae, args.width, args.height, device)

    # SDXL micro-conditioning: original size, crop top-left, target size. Frames are uncropped at native size.
    add_time_ids = torch.tensor([[args.height, args.width, 0, 0, args.height, args.width]], device=device, dtype=torch.bfloat16)

    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    for p in lora_params:
        p.data = p.data.float()
    print(f"trainable LoRA params: {sum(p.numel() for p in lora_params) / 1e6:.1f}M")
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-2)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / args.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(s, args.max_steps) / args.max_steps)))

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL, unet=unet, vae=vae, text_encoder=text_encoders[0], text_encoder_2=text_encoders[1],
        tokenizer=tokenizers[0], tokenizer_2=tokenizers[1], torch_dtype=torch.bfloat16, variant="fp16")
    pipe.set_progress_bar_config(disable=True)
    sample(pipe, prompts, args.width, args.height, args.sample_steps, args.seed,
           os.path.join(args.output_dir, "samples", "step_000000.jpg"))

    unet.train()
    for step in range(1, args.max_steps + 1):
        idx = torch.randint(0, len(records), (args.batch_size,), device=device)
        x0 = latents[idx].to(torch.bfloat16)
        cap = prompt_index[idx].clone()
        cap[torch.rand(args.batch_size, device=device) < args.caption_dropout] = empty_index

        noise = torch.randn_like(x0)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (args.batch_size,), device=device)
        noisy = scheduler.add_noise(x0, noise, t)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = unet(noisy, t, encoder_hidden_states=prompt_embeds[cap].to(torch.bfloat16),
                        added_cond_kwargs={"text_embeds": pooled_embeds[cap].to(torch.bfloat16),
                                           "time_ids": add_time_ids.expand(args.batch_size, -1)}).sample

        if args.snr_gamma > 0:
            snr = compute_snr(scheduler, t)
            weight = torch.clamp(snr, max=args.snr_gamma) / snr
            loss = (F.mse_loss(pred.float(), noise.float(), reduction="none").mean(dim=(1, 2, 3)) * weight).mean()
        else:
            loss = F.mse_loss(pred.float(), noise.float())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        lr_sched.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 25 == 0:
            print(f"step {step} loss {loss.item():.4f} lr {lr_sched.get_last_lr()[0]:.2e}")
        if step % args.sample_every == 0 or step == args.max_steps:
            unet.eval()
            sample(pipe, prompts, args.width, args.height, args.sample_steps, args.seed,
                   os.path.join(args.output_dir, "samples", f"step_{step:06d}.jpg"))
            unet.train()
        if step % args.save_every == 0 or step == args.max_steps:
            save_lora(unet, os.path.join(args.output_dir, f"checkpoint-{step:06d}"))

    save_lora(unet, args.output_dir)
    print(f"saved LoRA to {args.output_dir}")


if __name__ == "__main__":
    main()
