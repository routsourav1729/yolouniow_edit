#!/usr/bin/env python3
"""
SCPI: Support-Calibrated Prompt Interpolation — post-training calibration.

Runs AFTER T2 fine-tuning.  Loads the T2 checkpoint, extracts post-BN visual
features from the K-shot support set, and for each novel class decides:
keep the fine-tuned prompt OR revert to the zero-shot CLIP prompt.

Decision criterion: which prompt (fine-tuned vs zero-shot) has higher mean
cosine alignment with the post-BN visual features?  Both prompts are
L2-normalized before comparison (matching BNContrastiveHead scoring), and
features come from the same T2 model — so the comparison is valid.

Output: .npy with shape [num_known, 512] containing the per-class selected
embeddings (base classes unchanged, novel classes selected per above).

Usage:
    python tools/owod_scripts/scpi_calibrate.py \
        --config configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_wapr.py \
        --checkpoint work_dirs/.../best_owod_Both_epoch_40.pth \
        --zeroshot-emb embeddings/uniow-idd/idd_t2.npy \
        --output embeddings/uniow-idd/idd_t2_scpi.npy
"""
import argparse
import gc
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# Offline mode — must be set before any HF imports
os.environ.setdefault("HF_HOME", "/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.registry import DefaultScope, MODELS
from mmyolo.models import *  # noqa — registers YOLOv5DetDataPreprocessor etc.
import yolo_world  # noqa — registers OWODDetector etc.


def parse_args():
    p = argparse.ArgumentParser(description='SCPI calibration (post-training)')
    p.add_argument('--config', required=True, help='T2 config file')
    p.add_argument('--checkpoint', required=True, help='T2 fine-tuned checkpoint')
    p.add_argument('--zeroshot-emb', required=True,
                   help='Path to zero-shot CLIP embeddings npy (e.g. idd_t2.npy)')
    p.add_argument('--output', required=True, help='Output npy path for calibrated embeddings')
    p.add_argument('--fewshot-k', type=int, default=10)
    p.add_argument('--fewshot-seed', type=int, default=1)
    p.add_argument('--fewshot-dir', default='data/OWOD/iddsplit')
    p.add_argument('--data-root', default='data/OWOD')
    p.add_argument('--dataset', default='IDD')
    return p.parse_args()


def parse_xml_for_class(xml_path, cls_name):
    """Parse VOC XML, return list of [xmin, ymin, xmax, ymax] for cls."""
    tree = ET.parse(xml_path)
    boxes = []
    for obj in tree.findall('object'):
        if obj.find('name').text != cls_name:
            continue
        bbox = obj.find('bndbox')
        box = [float(bbox.find(x).text)
               for x in ('xmin', 'ymin', 'xmax', 'ymax')]
        boxes.append(box)
    return boxes


FPN_STRIDES = [8, 16, 32]
AREA_THRESHOLDS = [96**2, 192**2]


def extract_feat_at_box(box, scale_factor, pad_param, hooked):
    """Extract post-BN feature vector at the center of a GT box."""
    x1 = box[0] * scale_factor[0] + pad_param[2]
    y1 = box[1] * scale_factor[1] + pad_param[0]
    x2 = box[2] * scale_factor[0] + pad_param[2]
    y2 = box[3] * scale_factor[1] + pad_param[0]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = (x2 - x1) * (y2 - y1)

    if area < AREA_THRESHOLDS[0]:
        level = 0
    elif area < AREA_THRESHOLDS[1]:
        level = 1
    else:
        level = 2

    bn_key = f'bn_{level}'
    if bn_key not in hooked:
        return None

    feat_map = hooked[bn_key]  # (1, 512, H, W)
    stride = FPN_STRIDES[level]
    _, C, H, W = feat_map.shape
    gx = max(0, min(int(cx / stride), W - 1))
    gy = max(0, min(int(cy / stride), H - 1))

    return feat_map[0, :, gy, gx].clone()  # (512,)


@torch.no_grad()
def main():
    args = parse_args()

    # Ensure mmengine config env placeholders are always resolvable.
    os.environ.setdefault('DATASET', args.dataset)
    os.environ.setdefault('TASK', '2')
    os.environ.setdefault('THRESHOLD', '0.05')
    os.environ.setdefault('SAVE', 'False')
    os.environ.setdefault('FEWSHOT_DIR', args.fewshot_dir)
    os.environ.setdefault('FEWSHOT_K', str(args.fewshot_k))
    os.environ.setdefault('FEWSHOT_SEED', str(args.fewshot_seed))

    # ── Load config ─────────────────────────────────────────────────────
    cfg = Config.fromfile(args.config)
    DefaultScope.get_instance('scpi_calib', scope_name=cfg.default_scope)

    # ── Read class info ─────────────────────────────────────────────────
    class_text = cfg.get('class_text_path', '')
    with open(class_text) as f:
        all_known = [line.strip() for line in f if line.strip()]

    owod_settings = cfg.owod_settings[args.dataset]
    task = cfg.owod_task
    task_list = owod_settings['task_list']
    num_prev = task_list[task - 1]
    num_cur = task_list[task] - task_list[task - 1]
    novel_classes = all_known[num_prev:num_prev + num_cur]

    print(f'[SCPI] Task {task}: {num_prev} base, {num_cur} novel')
    print(f'[SCPI] Novel classes: {novel_classes}')

    # ── Build model ─────────────────────────────────────────────────────
    model = MODELS.build(cfg.model)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # ── Load T2 checkpoint ──────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    sd = {k: v for k, v in sd.items() if 'text_model' not in k}
    load_result = model.load_state_dict(sd, strict=False)
    if load_result.missing_keys:
        print(f'[SCPI] Missing keys: {load_result.missing_keys[:5]}...')
    model.eval()
    print(f'[SCPI] Model loaded from {args.checkpoint}')
    print(f'[SCPI] T2 embedding norms: {model.embeddings.data.norm(dim=-1)}')

    # ── Load zero-shot CLIP embeddings ──────────────────────────────────
    zeroshot_emb = torch.from_numpy(np.load(args.zeroshot_emb)).float().to(device)
    print(f'[SCPI] Zero-shot embeddings loaded from {args.zeroshot_emb}: '
          f'shape={zeroshot_emb.shape}')

    # ── Build preprocessing pipeline ────────────────────────────────────
    img_scale = (640, 640)
    pipeline = Compose([
        dict(type='LoadImageFromFile'),
        dict(type='YOLOv5KeepRatioResize', scale=img_scale),
        dict(type='LetterResize', scale=img_scale,
             allow_scale_up=False, pad_val=dict(img=114)),
        dict(type='mmdet.PackDetInputs',
             meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                        'scale_factor', 'pad_param')),
    ])

    # ── Register BN hooks ───────────────────────────────────────────────
    head_module = model.bbox_head.head_module
    hooked = {}

    def make_hook(name):
        def hook_fn(module, inp, out):
            hooked[name] = out.detach()
        return hook_fn

    handles = []
    for i, layer in enumerate(head_module.one2one_cls_contrasts):
        if hasattr(layer, 'norm'):
            handles.append(
                layer.norm.register_forward_hook(make_hook(f'bn_{i}')))

    # ── Collect per-class visual features ───────────────────────────────
    ann_dir = os.path.join(args.data_root, 'Annotations', args.dataset)
    seed_dir = os.path.join(args.fewshot_dir, f'seed{args.fewshot_seed}')

    class_features = defaultdict(list)

    for cls_name in novel_classes:
        shot_file = os.path.join(
            seed_dir, f'box_{args.fewshot_k}shot_{cls_name}_train.txt')
        with open(shot_file) as f:
            img_paths = [line.strip() for line in f if line.strip()]

        for img_path in img_paths:
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            xml_path = os.path.join(ann_dir, img_id + '.xml')
            full_img_path = img_path
            if not os.path.exists(full_img_path):
                full_img_path = os.path.join(args.data_root, '..', img_path)

            if not os.path.exists(xml_path) or not os.path.exists(full_img_path):
                print(f'[SCPI] WARNING: missing {xml_path} or {full_img_path}')
                continue

            gt_boxes = parse_xml_for_class(xml_path, cls_name)
            if not gt_boxes:
                continue

            data = dict(img_path=full_img_path, img_id=img_id, instances=[])
            data = pipeline(data)
            img_tensor = data['inputs'].unsqueeze(0).float().to(device) / 255.0
            data_sample = data['data_samples']

            img_feats, txt_feats = model.extract_feat(
                img_tensor, [data_sample])
            head_module.forward_one2one(img_feats, txt_feats)

            meta = data_sample.metainfo
            scale_factor = np.array(meta.get('scale_factor', [1.0, 1.0]))
            pad_param = meta.get('pad_param', (0, 0, 0, 0))

            for box in gt_boxes:
                feat = extract_feat_at_box(box, scale_factor, pad_param, hooked)
                if feat is not None:
                    class_features[cls_name].append(feat)

    # ── Cleanup hooks ───────────────────────────────────────────────────
    for h in handles:
        h.remove()
    hooked.clear()

    # ── Hard-switch: fine-tuned vs zero-shot per novel class ────────────
    print(f'\n[SCPI] {"Class":20s} {"n_feat":>6s} {"s_ft":>7s} '
          f'{"s_zs":>7s} {"action":>20s}')
    print('-' * 70)

    novel_start = num_prev
    for cls_idx, cls_name in enumerate(novel_classes):
        emb_idx = novel_start + cls_idx

        e_ft = model.embeddings[emb_idx].clone()        # fine-tuned from T2 ckpt
        e_zs = zeroshot_emb[emb_idx].clone().to(device)  # CLIP from npy

        feats = class_features.get(cls_name, [])
        if not feats:
            print(f'[SCPI] {cls_name:20s} {0:6d} {"--":>7s} '
                  f'{"--":>7s} {"skip (no feats)":>20s}')
            continue

        feat_stack = torch.stack(feats)
        feat_normed = F.normalize(feat_stack, dim=-1)

        # Compare alignment of both prompts with visual features
        s_ft = (feat_normed @ F.normalize(e_ft, dim=0)).mean().item()
        s_zs = (feat_normed @ F.normalize(e_zs, dim=0)).mean().item()

        if s_zs > s_ft:
            # Zero-shot is better — fine-tuning hurt this class
            e_final = e_zs
            action = 'revert-to-zeroshot'
        else:
            # Fine-tuned is better — keep it
            e_final = e_ft
            action = 'keep-finetuned'

        model.embeddings.data[emb_idx] = e_final

        print(f'[SCPI] {cls_name:20s} {len(feats):6d} {s_ft:7.4f} '
              f'{s_zs:7.4f} {action:>20s}')

    # ── Save calibrated embeddings as npy ───────────────────────────────
    # Save only known-class embeddings (no unk/anchor — those come from
    # the T2 checkpoint at eval time).
    num_known = num_prev + num_cur
    known_emb = model.embeddings[:num_known].detach().cpu().numpy()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.save(args.output, known_emb)
    print(f'\n[SCPI] Saved calibrated embeddings: {args.output}')
    print(f'[SCPI] Shape: {known_emb.shape}')
    print(f'[SCPI] Norms: {torch.from_numpy(known_emb).norm(dim=-1)}')

    # Cleanup
    del model, class_features, pipeline
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
