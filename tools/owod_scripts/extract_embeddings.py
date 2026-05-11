#!/usr/bin/env python3
"""
Text embedding extractor for OWOD datasets using the YOLO-World model backbone.

Extracts class embeddings and wildcard ('object') embedding via
model.backbone.forward_text(), matching the approach used by HONDA's
extract_text_feats.py.  For 'object_tuned.npy', extracts the full
embeddings tensor from a trained checkpoint's state_dict.

Usage (extract class + wildcard embeddings):
    python tools/owod_scripts/extract_embeddings.py \\
        --config configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod.py \\
        --ckpt pretrained/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth \\
        --dataset FOOD_VOCCOCO --task 0

Usage (extract tuned anchor embedding from trained model):
    python tools/owod_scripts/extract_embeddings.py \\
        --config configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod.py \\
        --ckpt work_dirs/<run>/best*.pth \\
        --dataset FOOD_VOCCOCO --extract_tuned
"""
import argparse
import glob
import numpy as np
import torch
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner

DATA_ROOT = Path("data/OWOD/ImageSets")

# Map dataset names to embedding directory names
EMBED_DIR_MAP = {
    "IDD": "uniow-idd",
    "FOOD_VOC": "uniow-food-voc",
    "FOOD_VOCCOCO": "uniow-food-voccoco",
    "MOWODB": "uniow-l",
    "SOWODB": "uniow-l",
    "nuOWODB": "uniow-l",
}

# Map dataset names to their task count
DATASET_TASKS = {
    "IDD": [1, 2],
    "FOOD_VOC": [1, 2],
    "FOOD_VOCCOCO": [1, 2],
    "MOWODB": [1, 2, 3, 4],
    "SOWODB": [1, 2, 3, 4],
    "nuOWODB": [1, 2, 3],
}


@torch.inference_mode()
def extract_task_embedding(model, dataset, task, save_dir):
    """Extract text embeddings for a single task's known classes via model backbone."""
    known_file = DATA_ROOT / dataset / f"t{task}_known.txt"
    if not known_file.exists():
        print(f"  Skipping task {task}: {known_file} not found")
        return

    with open(known_file) as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"  Task {task}: {len(class_names)} classes -> {class_names[:5]}...")

    text_feats = model.backbone.forward_text([class_names]).squeeze(0).detach().cpu()

    save_path = save_dir / f"{dataset.lower()}_t{task}.npy"
    np.save(save_path, text_feats.numpy())
    print(f"  Saved: {save_path}  shape={text_feats.shape}")


@torch.inference_mode()
def extract_custom_embedding(model, class_names, save_dir, save_name):
    """Extract embeddings for an explicit ordered prompt list."""
    print(f"  Custom: {len(class_names)} prompts -> {class_names}")
    text_feats = model.backbone.forward_text([class_names]).squeeze(0).detach().cpu()
    save_path = save_dir / save_name
    np.save(save_path, text_feats.numpy())
    print(f"  Saved: {save_path}  shape={text_feats.shape}")


@torch.inference_mode()
def extract_wildcard_embedding(model, save_dir, wildcard='object'):
    """Generate wildcard ('object') embedding via model backbone."""
    save_path = save_dir / f"{wildcard.replace(' ', '_')}.npy"
    text_feats = model.backbone.forward_text([[wildcard]]).squeeze(0).detach().cpu()
    np.save(save_path, text_feats.numpy())
    print(f"  Saved: {save_path}  shape={text_feats.shape}")


def extract_tuned_embedding(ckpt, save_dir, wildcard='object'):
    """Extract the T_anchor (last) embedding row from a trained model's state_dict.

    The saved file must be shape (1, 512) — it is concatenated as the anchor
    embedding in training via anchor_embedding_path.  Saving the full (K+2, 512)
    matrix would cause a dimension mismatch at training time.
    """
    all_embs = torch.load(ckpt, map_location='cpu')['state_dict']['embeddings']
    tuned_feats = all_embs[-1:]  # T_anchor is the last row, shape (1, C)
    save_path = save_dir / f"{wildcard.replace(' ', '_')}_tuned.npy"
    np.save(save_path, tuned_feats.numpy())
    print(f"  Saved: {save_path}  shape={tuned_feats.shape}  norm={torch.norm(tuned_feats, dim=-1).item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Pretrained or trained checkpoint path")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., FOOD_VOC, FOOD_VOCCOCO, IDD)")
    parser.add_argument("--task", type=int, default=0,
                        help="Task number (1, 2, ...). 0 = all tasks (default)")
    parser.add_argument("--wildcard", type=str, default="object",
                        help="Wildcard text for unknown/anchor embedding")
    parser.add_argument("--class-names", type=str, default="",
                        help="Comma-separated ordered prompt names. If set, "
                             "extract exactly these prompts instead of t{task}_known.txt.")
    parser.add_argument("--save-name", type=str, default="",
                        help="Output .npy filename for --class-names, e.g. idd_t2_altveh.npy")
    parser.add_argument("--extract_tuned", action="store_true",
                        help="Extract tuned embedding from trained model state_dict")
    args = parser.parse_args()

    embed_dir_name = EMBED_DIR_MAP.get(args.dataset, f"uniow-{args.dataset.lower()}")
    save_dir = Path("embeddings") / embed_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Embedding dir: {save_dir}")

    if args.extract_tuned:
        extract_tuned_embedding(args.ckpt, save_dir, wildcard=args.wildcard)
    else:
        # Use the pretrain config (no OWOD embedding file dependencies)
        pretrain_config = 'configs/pretrain/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py'
        cfg = Config.fromfile(pretrain_config)
        cfg.work_dir = 'work_dirs/extract_feats'

        runner = Runner.from_cfg(cfg)
        runner.call_hook("before_run")
        runner.load_checkpoint(args.ckpt, map_location='cpu')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = runner.model.to(device)
        model.eval()
        print(f"  Model loaded from: {args.ckpt} on {device}")

        if args.class_names:
            class_names = [x.strip() for x in args.class_names.split(',') if x.strip()]
            save_name = args.save_name or f"{args.dataset.lower()}_custom.npy"
            extract_custom_embedding(model, class_names, save_dir, save_name)
        else:
            # Extract wildcard embedding
            extract_wildcard_embedding(model, save_dir, wildcard=args.wildcard)

            # Extract class embeddings for requested tasks
            tasks = [args.task] if args.task > 0 else DATASET_TASKS.get(args.dataset, [1, 2])
            for task in tasks:
                extract_task_embedding(model, args.dataset, task, save_dir)

    print("Done!")


if __name__ == "__main__":
    main()
