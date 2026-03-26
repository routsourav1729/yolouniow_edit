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


@torch.inference_mode()
def extract_task_embedding(task: int, tokenizer, model, device):
    """Extract CLIP text embeddings for a single task's known classes."""
    known_file = DATA_ROOT / f"t{task}_known.txt"
    if not known_file.exists():
        print(f"  Skipping task {task}: {known_file} not found")
        return

    with open(known_file) as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"  Task {task}: {len(class_names)} classes → {class_names}")

    # Tokenize and encode — mirrors HuggingCLIPLanguageBackbone.forward()
    tokens = tokenizer(text=class_names, return_tensors="pt", padding=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    outputs = model(**tokens)
    text_embeds = outputs.text_embeds  # [num_classes, 512]
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds.cpu().float().numpy()

    save_path = SAVE_DIR / f"idd_t{task}.npy"
    np.save(save_path, text_embeds)
    print(f"  Saved: {save_path}  shape={text_embeds.shape}")


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
