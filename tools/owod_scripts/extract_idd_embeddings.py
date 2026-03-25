#!/usr/bin/env python3
"""
Lightweight IDD text embedding extractor.
Uses CLIP text model directly (no mmengine/mmdet/full model load needed).

Reads class names from data/OWOD/ImageSets/IDD/t{task}_known.txt,
encodes them with openai/clip-vit-base-patch32, and saves to
embeddings/uniow-s/idd_t{task}.npy

Usage:
    python tools/owod_scripts/extract_idd_embeddings.py              # all tasks
    python tools/owod_scripts/extract_idd_embeddings.py --task 1     # T1 only
"""
import os
import argparse
import numpy as np
import torch
from pathlib import Path

# Offline mode — compute nodes have no internet
os.environ.setdefault("HF_HOME", "/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoTokenizer, CLIPTextModelWithProjection


CLIP_MODEL = "openai/clip-vit-base-patch32"
SAVE_DIR = Path("embeddings/uniow-idd")
DATA_ROOT = Path("data/OWOD/ImageSets/IDD")

# Base class embedding norms after T1 training are ~4.3-4.5 in BN head space.
# Raw CLIP embeddings are L2-normalized (norm=1.0). This 4x mismatch causes
# novel logits to be structurally lower and gradient signal weaker.
# Rescale novel embeddings to match base operating point.
BASE_EMBED_TARGET_NORM = 4.4


@torch.inference_mode()
def extract_task_embedding(task: int, tokenizer, model, device,
                           rescale_novel: bool = True):
    """Extract CLIP text embeddings for a single task's known classes.

    For T2+, novel class embeddings (indices beyond base count) are rescaled
    to BASE_EMBED_TARGET_NORM so their norms match the base embeddings that
    were calibrated during T1 training (~4.3-4.5). Without this, novel logits
    are structurally 4x lower and receive weaker gradient signal.
    """
    known_file = DATA_ROOT / f"t{task}_known.txt"
    if not known_file.exists():
        print(f"  Skipping task {task}: {known_file} not found")
        return

    with open(known_file) as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"  Task {task}: {len(class_names)} classes → {class_names}")

    # Count base classes from T1 (needed to identify novel class indices)
    t1_known_file = DATA_ROOT / "t1_known.txt"
    num_base = 0
    if t1_known_file.exists():
        with open(t1_known_file) as f:
            num_base = sum(1 for line in f if line.strip())

    # Tokenize and encode — mirrors HuggingCLIPLanguageBackbone.forward()
    tokens = tokenizer(text=class_names, return_tensors="pt", padding=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    outputs = model(**tokens)
    text_embeds = outputs.text_embeds  # [num_classes, 512]
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds.cpu().float().numpy()

    # Rescale novel class embeddings for T2+ to match base norm range.
    # Base embeddings (indices 0..num_base-1) are frozen and will be
    # overwritten by the T1 checkpoint, so their norm here doesn't matter.
    # Novel embeddings (indices num_base..) are trainable and need to start
    # at the same operating point as base embeddings in the BN head.
    num_novel = len(class_names) - num_base
    if rescale_novel and task >= 2 and num_novel > 0:
        print(f"  Rescaling {num_novel} novel embeddings "
              f"(indices {num_base}..{len(class_names)-1}) "
              f"to norm={BASE_EMBED_TARGET_NORM:.1f}")
        for i in range(num_base, len(class_names)):
            norm = np.linalg.norm(text_embeds[i])
            text_embeds[i] = text_embeds[i] / norm * BASE_EMBED_TARGET_NORM

    save_path = SAVE_DIR / f"idd_t{task}.npy"
    np.save(save_path, text_embeds)
    print(f"  Saved: {save_path}  shape={text_embeds.shape}")
    # Print norms for verification
    for i, name in enumerate(class_names):
        print(f"    {name:20s}: norm={np.linalg.norm(text_embeds[i]):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=0,
                        help="Task number (1 or 2). 0 = all tasks (default)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto/cpu/cuda")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading CLIP text model: {CLIP_MODEL}")
    print(f"  Cache dir: {os.environ.get('HF_HOME')}")
    print(f"  Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL)
    model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL).to(device).eval()
    print(f"  Model loaded (embed_dim={model.config.projection_dim})")

    tasks = [args.task] if args.task > 0 else [1, 2]
    for task in tasks:
        extract_task_embedding(task, tokenizer, model, device)

    print("Done!")


if __name__ == "__main__":
    main()
