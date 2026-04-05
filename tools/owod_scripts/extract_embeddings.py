#!/usr/bin/env python3
"""
Generic text embedding extractor for any OWOD dataset.
Reads class names from data/OWOD/ImageSets/{dataset}/t{task}_known.txt,
encodes them with openai/clip-vit-base-patch32 and saves to
embeddings/uniow-{dataset_lower}/

Also generates 'object.npy' and 'object_tuned.npy' wildcard embeddings
if they don't already exist in the target directory.

Usage:
    python tools/owod_scripts/extract_embeddings.py --dataset FOOD_VOC --task 1
    python tools/owod_scripts/extract_embeddings.py --dataset FOOD_VOCCOCO --task 0  # all tasks
    python tools/owod_scripts/extract_embeddings.py --dataset IDD --task 2
"""
import os
import argparse
import numpy as np
import torch
from pathlib import Path

# Offline mode — compute nodes have no internet
os.environ["HF_HOME"] = "/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoTokenizer, CLIPTextModelWithProjection

CLIP_MODEL = "openai/clip-vit-base-patch32"
DATA_ROOT = Path("data/OWOD/ImageSets")

# Map dataset names to embedding directory names
EMBED_DIR_MAP = {
    "IDD": "uniow-idd",
    "FOOD_VOC": "uniow-food-voc",
    "FOOD_VOCCOCO": "uniow-food-voccoco",
    "MOWODB": "uniow-l",
    "SOWODB": "uniow-l",
}

# Map dataset names to their task count
DATASET_TASKS = {
    "IDD": [1, 2],
    "FOOD_VOC": [1, 2],
    "FOOD_VOCCOCO": [1, 2],
    "MOWODB": [1, 2, 3, 4],
    "SOWODB": [1, 2, 3, 4],
}


@torch.inference_mode()
def extract_task_embedding(dataset, task, save_dir, tokenizer, model, device):
    """Extract CLIP text embeddings for a single task's known classes."""
    known_file = DATA_ROOT / dataset / f"t{task}_known.txt"
    if not known_file.exists():
        print(f"  Skipping task {task}: {known_file} not found")
        return

    with open(known_file) as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"  Task {task}: {len(class_names)} classes → {class_names[:5]}...")

    tokens = tokenizer(text=class_names, return_tensors="pt", padding=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    outputs = model(**tokens)
    text_embeds = outputs.text_embeds
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds.cpu().float().numpy()

    save_path = save_dir / f"{dataset.lower()}_t{task}.npy"
    np.save(save_path, text_embeds)
    print(f"  Saved: {save_path}  shape={text_embeds.shape}")


@torch.inference_mode()
def extract_wildcard_embeddings(save_dir, tokenizer, model, device):
    """Generate 'object' and 'object_tuned' wildcard embeddings."""
    for name, text in [("object", "object"), ("object_tuned", "object")]:
        save_path = save_dir / f"{name}.npy"
        if save_path.exists():
            print(f"  {name}.npy already exists, skipping")
            continue

        tokens = tokenizer(text=[text], return_tensors="pt", padding=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        outputs = model(**tokens)
        text_embeds = outputs.text_embeds
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds.cpu().float().numpy()

        np.save(save_path, text_embeds)
        print(f"  Saved: {save_path}  shape={text_embeds.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., FOOD_VOC, FOOD_VOCCOCO, IDD)")
    parser.add_argument("--task", type=int, default=0,
                        help="Task number (1, 2, ...). 0 = all tasks (default)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto/cpu/cuda")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    embed_dir_name = EMBED_DIR_MAP.get(args.dataset, f"uniow-{args.dataset.lower()}")
    save_dir = Path("embeddings") / embed_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Embedding dir: {save_dir}")
    print(f"Loading CLIP text model: {CLIP_MODEL}")
    print(f"  Cache dir: {os.environ.get('HF_HOME')}")
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL)
    model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL).to(device).eval()
    print(f"  Model loaded (embed_dim={model.config.projection_dim})")

    tasks = [args.task] if args.task > 0 else DATASET_TASKS.get(args.dataset, [1, 2])
    for task in tasks:
        extract_task_embedding(args.dataset, task, save_dir, tokenizer, model, device)

    extract_wildcard_embeddings(save_dir, tokenizer, model, device)
    print("Done!")


if __name__ == "__main__":
    main()
