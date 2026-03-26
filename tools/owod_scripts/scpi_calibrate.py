#!/usr/bin/env python3
"""
SCPI: Support-Calibrated Prompt Interpolation — standalone calibration.

Loads T1 checkpoint, builds the full model in eval mode, extracts visual
centroids from K-shot support images, blends with zero-shot CLIP prompts,
and saves a NEW checkpoint with T2-shaped embeddings [16, 512].

NOTE:
    This script now uses a hard switch for novel classes, not interpolation.
    We do NOT mix visual centroids (post-BN space) with CLIP text embeddings
    (CLIP space), because those spaces are misaligned.

The output checkpoint can be evaluated with the normal eval pipeline
(eval_owod.sbatch) — no hooks, no runtime overhead.

Usage:
    python tools/owod_scripts/scpi_calibrate.py \
        --config configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_wapr.py \
        --checkpoint work_dirs/.../best_owod_Both_epoch_20.pth \
        --output work_dirs/scpi_t2_b10_t0.15.pth \
        --beta 10.0 --tau 0.15
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

# Module-level imports to register all custom mmyolo/yolo_world modules
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.registry import DefaultScope, MODELS
from mmyolo.models import *  # noqa — registers YOLOv5DetDataPreprocessor etc.
import yolo_world  # noqa — registers OWODDetector etc.


def parse_args():
    p = argparse.ArgumentParser(description='SCPI calibration (standalone)')
    p.add_argument('--config', required=True, help='T2 config file')
    p.add_argument('--checkpoint', required=True, help='T1 best checkpoint')
    p.add_argument('--output', required=True, help='Output npy path for calibrated embeddings')
    p.add_argument('--beta', type=float, default=10.0)
    p.add_argument('--tau', type=float, default=0.15)
    p.add_argument(
        '--switch-rule',
        default='alignment-median',
        choices=['alignment-median', 'manual'],
        help=('Hard-switch rule for novel classes: '
              'alignment-median uses per-class CLIP-vs-visual alignment '
              'relative to median; manual uses explicit class lists.'))
    p.add_argument(
        '--manual-zero-shot',
        default='',
        help='Comma-separated novel classes to keep zero-shot (manual mode).')
    p.add_argument(
        '--manual-visual',
        default='',
        help='Comma-separated novel classes to replace with visual centroid (manual mode).')
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
    # scale_factor from YOLOv5KeepRatioResize: (scale_w, scale_h)
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
    # Activate mmyolo registry scope (same as what Runner does internally)
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

    # ── Load T1 checkpoint into model ───────────────────────────────────
    # IMPORTANT: Use state_dict only (NOT EMA). The eval pipeline loads
    # state_dict, so BN running stats and all other params must match
    # exactly what eval will use. EMA has slightly different BN stats
    # which produces different features → bad visual centroids.
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    sd = {k: v for k, v in sd.items() if 'text_model' not in k}
    t1_raw_emb = sd['embeddings'].clone() if 'embeddings' in sd else None
    if 'embeddings' in sd:
        sd['embeddings'] = model.update_embeddings(sd['embeddings'])
        print(f'[SCPI] Merged embeddings: {sd["embeddings"].shape}')
    load_result = model.load_state_dict(sd, strict=False)
    if load_result.missing_keys:
        print(f'[SCPI] Missing keys: {load_result.missing_keys[:5]}...')
    model.eval()
    print(f'[SCPI] Model loaded from {args.checkpoint} (state_dict only)')

    # Patch unknown/anchor from T1 state_dict (not EMA).
    if t1_raw_emb is not None:
        model.embeddings.data[-2] = t1_raw_emb[-2].to(device)   # unknown
        model.embeddings.data[-1] = t1_raw_emb[-1].to(device)   # anchor
        print(f'[SCPI] Patched unk (norm={t1_raw_emb[-2].norm():.4f}) '
              f'and anchor (norm={t1_raw_emb[-1].norm():.4f}) from T1')

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
            full_img_path = img_path  # paths are relative to repo root
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

    # ── Compute hard-switch calibrated embeddings ───────────────────────
    base_norms = model.embeddings[:num_prev].norm(dim=-1).mean().item()
    print(f'[SCPI] Base embedding mean norm: {base_norms:.4f}')
    print(f'[SCPI] switch_rule={args.switch_rule}')

    manual_zero_shot = {c.strip() for c in args.manual_zero_shot.split(',') if c.strip()}
    manual_visual = {c.strip() for c in args.manual_visual.split(',') if c.strip()}

    # Precompute per-class stats first (required for median-based switching).
    class_stats = {}
    alignments = []

    novel_start = num_prev
    for cls_idx, cls_name in enumerate(novel_classes):
        emb_idx = novel_start + cls_idx
        e_zs = model.embeddings[emb_idx].clone()

        feats = class_features.get(cls_name, [])
        if not feats:
            class_stats[cls_name] = dict(
                emb_idx=emb_idx,
                n_feat=0,
                e_zs=e_zs,
                e_vis=None,
                alignment=None,
                coherence=None,
            )
            continue

        feat_stack = torch.stack(feats)
        e_vis = F.normalize(feat_stack.mean(dim=0), dim=0)
        e_zs_normed = F.normalize(e_zs, dim=0)

        # Cross-space alignment (used only as RELATIVE signal across novel classes).
        alignment = F.cosine_similarity(
            e_vis.unsqueeze(0), e_zs_normed.unsqueeze(0), dim=1).item()

        # Intra-class visual coherence for diagnostics.
        feat_normed = F.normalize(feat_stack, dim=-1)
        if feat_normed.shape[0] > 1:
            sim = feat_normed @ feat_normed.T
            sim.fill_diagonal_(0.0)
            coherence = (sim.sum() / (sim.shape[0] * (sim.shape[0] - 1))).item()
        else:
            coherence = 1.0

        class_stats[cls_name] = dict(
            emb_idx=emb_idx,
            n_feat=len(feats),
            e_zs=e_zs,
            e_vis=e_vis,
            alignment=alignment,
            coherence=coherence,
        )
        alignments.append(alignment)

    alignment_median = float(np.median(alignments)) if alignments else 0.0
    print(f'[SCPI] alignment_median={alignment_median:.4f}')
    print(f'[SCPI] {"Class":20s} {"n_feat":>6s} {"align":>7s} '
          f'{"cohere":>7s} {"action":>10s}')
    print('-' * 60)

    for cls_name in novel_classes:
        st = class_stats[cls_name]
        emb_idx = st['emb_idx']
        e_zs = st['e_zs']
        e_vis = st['e_vis']
        n_feat = st['n_feat']
        alignment = st['alignment']
        coherence = st['coherence']

        if n_feat == 0 or e_vis is None:
            print(f'[SCPI] {cls_name:20s} {0:6d} {"--":>7s} '
                  f'{"--":>7s} {"skip":>10s}')
            continue

        if args.switch_rule == 'manual':
            if cls_name in manual_zero_shot and cls_name in manual_visual:
                raise ValueError(
                    f'Class {cls_name} is in both manual-zero-shot and manual-visual.')
            if cls_name in manual_zero_shot:
                use_zero_shot = True
            elif cls_name in manual_visual:
                use_zero_shot = False
            else:
                # Fallback for unspecified classes in manual mode.
                use_zero_shot = alignment >= alignment_median
        else:
            # Auto hard-switch: keep zero-shot for relatively aligned classes,
            # otherwise switch to visual centroid.
            use_zero_shot = alignment >= alignment_median

        if use_zero_shot:
            e_final = F.normalize(e_zs, dim=0) * base_norms
            action = 'zero-shot'
        else:
            e_final = F.normalize(e_vis, dim=0) * base_norms
            action = 'visual'

        model.embeddings.data[emb_idx] = e_final

        print(f'[SCPI] {cls_name:20s} {n_feat:6d} {alignment:7.3f} '
              f'{coherence:7.3f} {action:>10s}')

    # ── Save calibrated embeddings as npy ─────────────────────────────
    # Save only the known-class embeddings (no unk/anchor — those come
    # from the T1 checkpoint at eval time via SCPIHook).
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
