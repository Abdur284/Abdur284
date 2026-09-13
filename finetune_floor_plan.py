"""LoRA fine-tuning for HAVEN AI's 2D floor-plan Stable Diffusion side.

Single-file, Colab-friendly (T4 or better). LoRA adapts SD's *style* to
Pakistani-blueprint floor plans while ControlNet-MLSD keeps handling the
geometry at inference. Output is a small `.safetensors` you drop next to the
base SD checkpoint — the Colab server in `colab/sd_controlnet_server.ipynb`
can load it with a two-line change (see the bottom of this file).

--------------------------------------------------------------------------
Dataset format
--------------------------------------------------------------------------
A single folder of paired files:

    dataset/
        plan_0001.png
        plan_0001.txt          # caption, one line, e.g.
                               #   "2D top-down architectural floor plan of a
                               #    Pakistani 5 marla house, 3 bedrooms,
                               #    kitchen, drawing room, blueprint style,
                               #    black-and-white line drawing"
        plan_0002.png
        plan_0002.txt
        ...

Any image extension diffusers/PIL can read works (png/jpg/webp). Captions can
also be a single CSV with columns `image,caption` — set `CSV_MODE = True`.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
    pip install "diffusers==0.30.3" "transformers==4.44.2" "accelerate==0.34.2" \
                "peft==0.13.0" "safetensors==0.4.5" "datasets==3.0.1" "pillow" "tqdm"

    python finetune_floor_plan.py                     # uses the CONFIG below
    # or override any value from the env:
    DATASET_DIR=./my_plans OUTPUT_DIR=./lora_out python finetune_floor_plan.py

--------------------------------------------------------------------------
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import save_file
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer


# ============================================================
# Config — override any of these via environment variables.
# ============================================================
@dataclass
class Config:
    base_model: str = os.getenv("BASE_MODEL", "runwayml/stable-diffusion-v1-5")
    dataset_dir: str = os.getenv("DATASET_DIR", "./dataset")
    output_dir: str = os.getenv("OUTPUT_DIR", "./lora_floor_plan")
    csv_mode: bool = os.getenv("CSV_MODE", "false").lower() == "true"
    csv_path: str = os.getenv("CSV_PATH", "./dataset/captions.csv")

    # Image + trainer
    resolution: int = int(os.getenv("RESOLUTION", "512"))
    train_batch_size: int = int(os.getenv("BATCH_SIZE", "1"))
    gradient_accumulation_steps: int = int(os.getenv("GRAD_ACCUM", "4"))
    max_train_steps: int = int(os.getenv("MAX_STEPS", "1500"))
    learning_rate: float = float(os.getenv("LR", "1e-4"))
    lr_scheduler: str = os.getenv("LR_SCHEDULER", "cosine")
    lr_warmup_steps: int = int(os.getenv("WARMUP", "100"))
    seed: int = int(os.getenv("SEED", "42"))
    mixed_precision: str = os.getenv("PRECISION", "fp16")  # "no" | "fp16" | "bf16"

    # LoRA
    lora_rank: int = int(os.getenv("LORA_RANK", "16"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "16"))
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.05"))

    # Prompt drop rate (classifier-free guidance training)
    caption_drop_prob: float = float(os.getenv("CAPTION_DROP", "0.1"))

    # Checkpointing
    save_every: int = int(os.getenv("SAVE_EVERY", "500"))
    validation_prompt: str = os.getenv(
        "VAL_PROMPT",
        "2D top-down architectural floor plan of a Pakistani 5 marla house, "
        "3 bedrooms, kitchen, drawing room, blueprint style, black-and-white line drawing",
    )


# ============================================================
# Dataset
# ============================================================
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _collect_pairs(cfg: Config) -> List[Tuple[Path, str]]:
    """Return a list of (image_path, caption) tuples."""
    if cfg.csv_mode:
        import csv
        pairs = []
        with open(cfg.csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pairs.append((Path(row["image"]), row["caption"]))
        return pairs

    root = Path(cfg.dataset_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory {root!s} not found. Point DATASET_DIR at a folder of "
            "image + .txt caption pairs, or set CSV_MODE=true and CSV_PATH."
        )
    pairs: List[Tuple[Path, str]] = []
    for img in sorted(root.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        cap = img.with_suffix(".txt")
        if not cap.exists():
            # Fall back to a generic caption so a partially-labelled folder still works.
            caption = "2D architectural floor plan, top-down, blueprint, line drawing"
        else:
            caption = cap.read_text(encoding="utf-8").strip()
        pairs.append((img, caption))
    if not pairs:
        raise RuntimeError(f"No images found under {root!s}.")
    return pairs


class FloorPlanDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, str]], tokenizer: CLIPTokenizer, resolution: int, drop_prob: float):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.drop_prob = drop_prob
        self.tf = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1] as SD's VAE expects
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        img_path, caption = self.pairs[i]
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.tf(img)

        # Classifier-free guidance: drop caption sometimes.
        if random.random() < self.drop_prob:
            caption = ""

        ids = self.tokenizer(
            caption, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt",
        ).input_ids[0]

        return {"pixel_values": pixel_values, "input_ids": ids}


# ============================================================
# LoRA plumbing (UNet only — smallest and most impactful)
# ============================================================
UNET_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "proj_in", "proj_out",
    "ff.net.0.proj", "ff.net.2",
]


def attach_lora(unet: UNet2DConditionModel, cfg: Config) -> UNet2DConditionModel:
    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=UNET_TARGET_MODULES,
        init_lora_weights="gaussian",
    )
    return get_peft_model(unet, lora_cfg)


def save_lora(unet, out_dir: str, step: int) -> Path:
    """Save just the LoRA delta weights as a single .safetensors file."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in unet.state_dict().items() if "lora_" in k}
    path = Path(out_dir) / f"lora_step_{step:06d}.safetensors"
    save_file(state, str(path))
    return path


# ============================================================
# Training
# ============================================================
def main():
    cfg = Config()
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": torch.float32}[cfg.mixed_precision]
    weight_dtype = dtype if device == "cuda" else torch.float32

    print(f"[cfg] device={device} dtype={weight_dtype} base={cfg.base_model}")
    print(f"[cfg] dataset={cfg.dataset_dir} output={cfg.output_dir}")

    # ---- Load frozen components + trainable UNet ----
    tokenizer = CLIPTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.base_model, subfolder="text_encoder").to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(cfg.base_model, subfolder="vae").to(device, dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(cfg.base_model, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.base_model, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    unet = attach_lora(unet, cfg)
    unet.print_trainable_parameters()
    unet.to(device)

    # ---- Data ----
    pairs = _collect_pairs(cfg)
    print(f"[data] {len(pairs)} image/caption pairs")
    dataset = FloorPlanDataset(pairs, tokenizer, cfg.resolution, cfg.caption_drop_prob)

    def collate(batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
        }

    loader = DataLoader(dataset, batch_size=cfg.train_batch_size, shuffle=True,
                        collate_fn=collate, num_workers=2, drop_last=True)

    # ---- Optim / schedule ----
    trainable = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)

    total_steps = cfg.max_train_steps
    scheduler = get_scheduler(
        cfg.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=total_steps * cfg.gradient_accumulation_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and dtype == torch.float16))

    # ---- Training loop ----
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    global_step = 0
    unet.train()

    pbar = tqdm(total=total_steps, desc="train")
    while global_step < total_steps:
        for batch in loader:
            with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=dtype if device == "cuda" else torch.float32):
                pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
                input_ids = batch["input_ids"].to(device)

                # Encode to latents.
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                # Sample noise + timesteps.
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text conditioning.
                enc = text_encoder(input_ids)[0]

                # UNet prediction + MSE against noise (v-pred not used for SD 1.5).
                model_pred = unet(noisy, timesteps, encoder_hidden_states=enc).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss = loss / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (global_step + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=float(loss.detach().cpu()) * cfg.gradient_accumulation_steps,
                             lr=scheduler.get_last_lr()[0])

            if global_step % cfg.save_every == 0:
                path = save_lora(unet, cfg.output_dir, global_step)
                print(f"\n[ckpt] saved {path}")
                _validate(cfg, unet, tokenizer, text_encoder, vae, noise_scheduler, device, weight_dtype, global_step)

            if global_step >= total_steps:
                break

    final = save_lora(unet, cfg.output_dir, global_step)
    print(f"[done] final LoRA weights -> {final}")


# ============================================================
# Sanity-check validation: run the pipeline on one prompt and save a preview.
# ============================================================
@torch.no_grad()
def _validate(cfg, unet, tokenizer, text_encoder, vae, noise_scheduler, device, dtype, step):
    unet.eval()
    try:
        pipe = StableDiffusionPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=noise_scheduler, safety_checker=None, feature_extractor=None,
            requires_safety_checker=False,
        ).to(device)
        img = pipe(cfg.validation_prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
        out = Path(cfg.output_dir) / f"sample_step_{step:06d}.png"
        img.save(out)
        print(f"[val ] preview -> {out}")
    except Exception as e:
        print(f"[val ] skipped ({e.__class__.__name__}: {e})")
    finally:
        unet.train()


# ============================================================
# How to load this LoRA at inference (in the Colab SD+ControlNet server)
# ============================================================
#
#   from safetensors.torch import load_file
#   from peft import LoraConfig, set_peft_model_state_dict, get_peft_model
#
#   lora_cfg = LoraConfig(r=16, lora_alpha=16, target_modules=[
#       "to_q","to_k","to_v","to_out.0","proj_in","proj_out","ff.net.0.proj","ff.net.2",
#   ])
#   pipe.unet = get_peft_model(pipe.unet, lora_cfg)
#   set_peft_model_state_dict(pipe.unet, load_file("lora_floor_plan/lora_step_001500.safetensors"))
#   pipe.unet = pipe.unet.merge_and_unload()   # optional: bake LoRA into UNet weights
#
if __name__ == "__main__":
    main()
