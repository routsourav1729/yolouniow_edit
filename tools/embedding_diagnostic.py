#!/usr/bin/env python
"""Embedding space diagnostic for YOLO-UniOW open-world detection.

Extracts vision features at GT locations, loads prompt embeddings from
a checkpoint, and generates visualizations that diagnose A-OSE (unknown
objects misclassified as known) and embedding alignment issues.

Outputs are saved to  visualizations/embed_diag_t{task}/  by default.

Usage (standalone):
    python tools/embedding_diagnostic.py \
        --config configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_wapr.py \
        --checkpoint work_dirs/.../best_owod_Both_epoch_40.pth \
        --task 2 --max-images 300

Usage (from eval sbatch with ANALYZE=1):
    Automatically called after eval completes; see scripts/eval_owod.sbatch.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Embedding diagnostic for YOLO-UniOW')
    p.add_argument('--config', required=True, help='mmengine config file')
    p.add_argument('--checkpoint', required=True, help='model checkpoint (.pth)')
    p.add_argument('--task', type=int, default=2, help='OWOD task number')
    p.add_argument('--max-images', type=int, default=300,
                   help='max test images for feature extraction')
    p.add_argument('--output-dir', type=str, default=None,
                   help='output dir (default: visualizations/embed_diag_t{task})')
    p.add_argument('--cache', type=str, default=None,
                   help='path to cached features .pt (skip extraction)')
    p.add_argument('--method', choices=['umap', 'tsne'], default='umap',
                   help='dimensionality reduction method')
    p.add_argument('--device', type=str, default='cuda:0')
    return p.parse_args()


# ── Constants ────────────────────────────────────────────────────────────────

# FPN strides and area thresholds for level assignment
FPN_STRIDES = [8, 16, 32]
# area < AREA_THRESH[0] → level 0, < AREA_THRESH[1] → level 1, else level 2
AREA_THRESHOLDS = [96**2, 192**2]

IDD_CLASSES_T1 = ['car', 'motorcycle', 'rider', 'person',
                  'autorickshaw', 'bicycle', 'traffic sign', 'traffic light']
IDD_CLASSES_T2 = ['bus', 'truck', 'tanker_vehicle', 'crane_truck',
                  'street_cart', 'excavator']
IDD_UNKNOWNS = ['pole', 'animal', 'tractor', 'concrete_mixer',
                'pull_cart', 'road_roller']


def get_class_info(task):
    """Return (known_classes, num_prev, num_cur) for given IDD task."""
    if task == 1:
        return list(IDD_CLASSES_T1), 0, 8
    elif task == 2:
        return list(IDD_CLASSES_T1) + list(IDD_CLASSES_T2), 8, 6
    else:
        raise ValueError(f'Unsupported task {task}')


def fpn_level_for_area(area):
    """Assign FPN level (0/1/2) based on box area."""
    if area < AREA_THRESHOLDS[0]:
        return 0
    elif area < AREA_THRESHOLDS[1]:
        return 1
    return 2


# ── XML Parsing ──────────────────────────────────────────────────────────────

def parse_voc_xml(xml_path, known_classes):
    """Parse VOC XML, return list of (original_cls, mapped_cls, bbox)."""
    tree = ET.parse(xml_path)
    objs = []
    for obj in tree.findall('object'):
        cls = obj.find('name').text
        original_cls = cls
        if cls in known_classes:
            mapped_cls = cls
        else:
            mapped_cls = 'unknown'
        bbox = obj.find('bndbox')
        box = [float(bbox.find(x).text) for x in ['xmin', 'ymin', 'xmax', 'ymax']]
        objs.append(dict(original_cls=original_cls, mapped_cls=mapped_cls, bbox=box))
    return objs


# ── Model Loading & Feature Extraction ───────────────────────────────────────

def load_model(config_path, checkpoint_path, device):
    """Load YOLO-UniOW model via mmengine Runner (handles all registry scoping).

    Runner.from_cfg() builds the model but does NOT load the checkpoint
    (that only happens inside runner.test/train). We must load explicitly.
    """
    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint

    cfg = Config.fromfile(config_path)
    cfg.load_from = checkpoint_path
    cfg.work_dir = '/tmp/_embed_diag_dummy'
    cfg.launcher = 'none'

    runner = Runner.from_cfg(cfg)
    model = runner.model

    # Explicitly load checkpoint — Runner.from_cfg does NOT do this
    load_checkpoint(model, checkpoint_path, map_location='cpu')

    # Verify BN stats were loaded (sanity check)
    bn = model.bbox_head.head_module.one2one_cls_contrasts[0].norm
    if bn.running_var.mean().item() > 0.99:
        print('[WARN] BN running_var is ~1.0 — checkpoint may not have loaded correctly')

    model = model.to(device).eval()
    return model, cfg


def extract_features(model, cfg, known_classes, max_images, device):
    """Run model on test images, extract 512-dim features at GT locations.

    Returns:
        features: {mapped_cls: list of 512-dim tensors}
        logits:   {mapped_cls: list of K-dim tensors}
        gt_meta:  {mapped_cls: list of dict with 'original_cls', 'img_id', 'bbox'}
    """
    from mmengine.config import Config

    # Resolve data paths
    data_root = cfg.get('owod_root', 'data/OWOD')
    dataset_name = cfg.get('owod_dataset', 'IDD')
    test_image_set = cfg.owod_settings[dataset_name]['test_image_set']

    ann_dir = os.path.join(data_root, 'Annotations', dataset_name)
    img_dir = os.path.join(data_root, 'JPEGImages', dataset_name)
    split_file = os.path.join(data_root, 'ImageSets', dataset_name,
                              test_image_set + '.txt')

    with open(split_file) as f:
        img_ids = [x.strip() for x in f.readlines() if x.strip()]

    # Subsample
    if max_images > 0 and len(img_ids) > max_images:
        rng = np.random.RandomState(42)
        img_ids = list(rng.choice(img_ids, max_images, replace=False))

    print(f'[extract] {len(img_ids)} images, ann_dir={ann_dir}')

    # Register hooks on one2one cls_preds (raw 512-dim) and cls_contrasts (logits)
    head_module = model.bbox_head.head_module
    hooked = {}

    def make_hook(name):
        def hook_fn(module, inp, out):
            hooked[name] = out.detach()
        return hook_fn

    handles = []
    for i, layer in enumerate(head_module.one2one_cls_preds):
        handles.append(layer.register_forward_hook(make_hook(f'cls_pred_{i}')))
    for i, layer in enumerate(head_module.one2one_cls_contrasts):
        handles.append(layer.register_forward_hook(make_hook(f'cls_logit_{i}')))
        # Also hook the BN inside the contrastive head
        if hasattr(layer, 'norm'):
            handles.append(layer.norm.register_forward_hook(make_hook(f'cls_bn_{i}')))

    # Image preprocessing — only use image transforms, skip LoadAnnotations/PackDetInputs
    # We handle GT parsing from XML ourselves.
    import cv2
    from mmengine.dataset import Compose
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData

    # Filter pipeline: keep only image-related transforms
    img_pipeline_cfg = []
    for t in cfg.owod_val_dataset.pipeline:
        t_type = t.get('type', '')
        # Skip annotation loading and packing
        if 'LoadAnnotations' in t_type or 'PackDetInputs' in t_type:
            continue
        img_pipeline_cfg.append(t)
    # Add a minimal packing at the end
    img_pipeline_cfg.append(dict(type='mmdet.PackDetInputs',
                                 meta_keys=('img_id', 'img_path', 'ori_shape',
                                            'img_shape', 'scale_factor',
                                            'pad_param')))
    pipeline = Compose(img_pipeline_cfg)

    features = defaultdict(list)   # mapped_cls -> list of 512-dim
    logits_all = defaultdict(list)  # mapped_cls -> list of K-dim
    gt_meta = defaultdict(list)     # mapped_cls -> list of metadata dicts

    num_extracted = 0
    for idx, img_id in enumerate(img_ids):
        if idx % 100 == 0:
            print(f'  [{idx}/{len(img_ids)}] extracted {num_extracted} features')

        xml_path = os.path.join(ann_dir, img_id + '.xml')
        img_path = os.path.join(img_dir, img_id + '.jpg')
        if not os.path.exists(xml_path) or not os.path.exists(img_path):
            continue

        gt_objs = parse_voc_xml(xml_path, known_classes)
        if not gt_objs:
            continue

        # Build data dict for pipeline
        data = dict(
            img_path=img_path,
            img_id=img_id,
            # Provide empty instances so PackDetInputs doesn't fail
            instances=[],
        )
        data = pipeline(data)

        # Prepare batch
        img_tensor = data['inputs'].unsqueeze(0).float().to(device)
        # Normalize: model preprocessor does /255
        img_tensor = img_tensor / 255.0

        data_sample = data['data_samples']

        # Forward through model to populate hooks
        with torch.no_grad():
            img_feats, txt_feats = model.extract_feat(img_tensor, [data_sample])
            head_module.forward_one2one(img_feats, txt_feats)

        # Get scale_factor and pad_param for coordinate mapping
        meta = data_sample.metainfo
        scale_factor = np.array(meta.get('scale_factor', [1.0, 1.0]))  # (h_scale, w_scale)
        pad_param = meta.get('pad_param', (0, 0, 0, 0))  # (top, bottom, left, right)

        # For each GT box, extract features
        for obj in gt_objs:
            box = obj['bbox']  # original image coordinates (xmin, ymin, xmax, ymax)
            mapped_cls = obj['mapped_cls']
            original_cls = obj['original_cls']

            # Map box to resized+padded image coordinates
            # scale_factor is (h_scale, w_scale)
            x1 = box[0] * scale_factor[1] + pad_param[2]
            y1 = box[1] * scale_factor[0] + pad_param[0]
            x2 = box[2] * scale_factor[1] + pad_param[2]
            y2 = box[3] * scale_factor[0] + pad_param[0]

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = (x2 - x1) * (y2 - y1)

            level = fpn_level_for_area(area)
            stride = FPN_STRIDES[level]

            # Get feature maps for this level
            bn_key = f'cls_bn_{level}'
            logit_key = f'cls_logit_{level}'

            if bn_key not in hooked or logit_key not in hooked:
                continue

            feat_map = hooked[bn_key]    # (1, 512, H, W)
            logit_map = hooked[logit_key]  # (1, K, H, W)

            _, C, H, W = feat_map.shape
            gx = int(cx / stride)
            gy = int(cy / stride)
            gx = max(0, min(gx, W - 1))
            gy = max(0, min(gy, H - 1))

            feat_vec = feat_map[0, :, gy, gx].cpu()      # (512,)
            logit_vec = logit_map[0, :, gy, gx].cpu()     # (K,)

            features[mapped_cls].append(feat_vec)
            logits_all[mapped_cls].append(logit_vec)
            gt_meta[mapped_cls].append(dict(
                original_cls=original_cls,
                img_id=img_id,
                bbox=box,
            ))
            num_extracted += 1

    # Remove hooks
    for h in handles:
        h.remove()

    print(f'[extract] Done. {num_extracted} total features.')
    for cls in sorted(features.keys()):
        print(f'  {cls}: {len(features[cls])} features')

    # Stack into tensors
    features_t = {k: torch.stack(v) for k, v in features.items() if v}
    logits_t = {k: torch.stack(v) for k, v in logits_all.items() if v}

    return features_t, logits_t, gt_meta


def load_prompt_embeddings(checkpoint_path, known_classes, task):
    """Load prompt embeddings from checkpoint state_dict."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    sd = ckpt if 'state_dict' not in ckpt else ckpt['state_dict']

    # Try both key patterns
    for key in ['embeddings', 'model.embeddings']:
        if key in sd:
            emb = sd[key]  # (num_prompts, 512)
            break
    else:
        raise KeyError('Cannot find embeddings in checkpoint')

    # Map rows to class names: known_classes + ['unknown', 'anchor']
    names = list(known_classes) + ['unknown', 'anchor']
    assert emb.shape[0] >= len(names), \
        f'Embedding has {emb.shape[0]} rows but expected >= {len(names)}'
    prompt_dict = {names[i]: emb[i] for i in range(len(names))}
    return prompt_dict


# ── Cosine Utilities ─────────────────────────────────────────────────────────

def cosine_sim(a, b):
    """Cosine similarity between two 1-D tensors."""
    a = a.float()
    b = b.float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def pairwise_cosine(mat):
    """Pairwise cosine similarity for (N, D) tensor. Returns (N, N)."""
    mat = F.normalize(mat.float(), dim=1)
    return (mat @ mat.T).numpy()


# ── Visualization ────────────────────────────────────────────────────────────

def make_colors(known_classes, num_prev):
    """Generate color dict: base=blues/greens, novel=reds/oranges, unknown=grays."""
    colors = {}
    base_cmap = plt.cm.Set2
    novel_cmap = plt.cm.Set1
    for i, cls in enumerate(known_classes[:num_prev]):
        colors[cls] = base_cmap(i / max(num_prev, 1))
    for i, cls in enumerate(known_classes[num_prev:]):
        colors[cls] = novel_cmap(i / max(len(known_classes) - num_prev, 1))
    colors['unknown'] = (0.5, 0.5, 0.5, 1.0)
    colors['anchor'] = (0.2, 0.2, 0.2, 1.0)
    return colors


def run_projection(all_vecs, all_labels, method='umap'):
    """Project all_vecs (N, D) to 2D via UMAP or t-SNE.

    All vectors (vision features AND prompts) are L2-normalized before
    projection so they live on the unit hypersphere. UMAP/t-SNE with
    metric='cosine' then captures angular relationships — which is exactly
    what the dot-product scoring in BNContrastiveHead cares about.

    No mean-subtraction: BN features and L2-normed prompts should be
    projected in the same space so we can see where prompts sit relative
    to their class feature clouds.
    """
    mat = all_vecs.numpy().astype(np.float32)

    # L2 normalize everything — this is what matters for cosine similarity
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms

    if method == 'umap':
        try:
            from umap import UMAP
            reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine',
                           random_state=42)
        except ImportError:
            print('  [warn] umap-learn not installed, falling back to t-SNE')
            method = 'tsne'
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, metric='cosine', perplexity=30,
                       random_state=42)
    coords = reducer.fit_transform(mat)
    return coords


def fig1_embedding_overview(coords_dict, prompt_coords, colors, out_dir, method):
    """Fig 1: All classes in one UMAP/tSNE plot."""
    fig, ax = plt.subplots(figsize=(16, 12))

    for cls, xy in coords_dict.items():
        c = colors.get(cls, (0.5, 0.5, 0.5, 1.0))
        ax.scatter(xy[:, 0], xy[:, 1], c=[c], s=8, alpha=0.3, label=cls)

    for cls, xy in prompt_coords.items():
        c = colors.get(cls, (0.2, 0.2, 0.2, 1.0))
        marker = 'p' if cls in ('unknown', 'anchor') else '*'
        ax.scatter(xy[0], xy[1], c=[c], s=300, marker=marker,
                   edgecolors='black', linewidths=1.5, zorder=10)
        ax.annotate(cls, (xy[0], xy[1]), fontsize=7, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points')

    ax.set_title(f'Embedding Space Overview ({method.upper()})', fontsize=14)
    ax.legend(loc='best', fontsize=7, ncol=2, markerscale=3)
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'embedding_overview.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved embedding_overview.png')


def fig2_per_class(features, prompt_dict, known_classes, colors, out_dir,
                   coords_all, labels_all, method):
    """Fig 2: Per-class feature cloud + prompt + cosine stats."""
    per_class_dir = os.path.join(out_dir, 'per_class')
    os.makedirs(per_class_dir, exist_ok=True)

    # Compute centroids in original space
    centroids = {}
    for cls, feats in features.items():
        centroids[cls] = feats.mean(dim=0)

    all_classes = list(features.keys())

    for cls in all_classes:
        if cls not in features or len(features[cls]) < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        # This class
        mask = np.array(labels_all) == cls
        if mask.sum() == 0:
            plt.close(fig)
            continue
        xy_cls = coords_all[mask]
        c = colors.get(cls, (0.5, 0.5, 0.5, 1.0))
        ax.scatter(xy_cls[:, 0], xy_cls[:, 1], c=[c], s=15, alpha=0.5,
                   label=f'{cls} features')

        # Top 3 nearest classes by centroid cosine distance
        if cls in centroids:
            dists = []
            for other_cls, other_ctr in centroids.items():
                if other_cls == cls:
                    continue
                d = cosine_sim(centroids[cls], other_ctr)
                dists.append((other_cls, d))
            dists.sort(key=lambda x: -x[1])
            for neighbor_cls, _ in dists[:3]:
                mask_n = np.array(labels_all) == neighbor_cls
                if mask_n.sum() > 0:
                    xy_n = coords_all[mask_n]
                    ax.scatter(xy_n[:, 0], xy_n[:, 1], c=[(0.8, 0.8, 0.8, 0.4)],
                               s=5, alpha=0.2, label=f'{neighbor_cls} (neighbor)')

        # Prompt embedding
        if cls in prompt_dict:
            # Find prompt coord
            prompt_idx = None
            for i, lbl in enumerate(labels_all):
                if lbl == f'PROMPT_{cls}':
                    prompt_idx = i
                    break
            if prompt_idx is not None:
                px, py = coords_all[prompt_idx]
                ax.scatter(px, py, c=[c], s=400, marker='*',
                           edgecolors='black', linewidths=2, zorder=10,
                           label=f'{cls} prompt')

        # Cosine stats text
        stats_lines = []
        if cls in prompt_dict and cls in centroids:
            stats_lines.append(
                f'cos(prompt, centroid) = {cosine_sim(prompt_dict[cls], centroids[cls]):.4f}')
        if cls in centroids:
            # Intra-class variance
            feats_norm = F.normalize(features[cls].float(), dim=1)
            ctr_norm = F.normalize(centroids[cls].float().unsqueeze(0), dim=1)
            cos_to_ctr = (feats_norm @ ctr_norm.T).squeeze()
            stats_lines.append(f'intra-class cos (mean) = {cos_to_ctr.mean().item():.4f}')
            stats_lines.append(f'intra-class cos (std)  = {cos_to_ctr.std().item():.4f}')
            stats_lines.append(f'n = {len(features[cls])}')

        if stats_lines:
            ax.text(0.02, 0.98, '\n'.join(stats_lines), transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title(f'{cls} — Embedding Space ({method.upper()})', fontsize=12)
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(per_class_dir, f'{cls.replace(" ", "_")}.png'), dpi=120)
        plt.close(fig)

    print(f'  Saved per_class/ plots')


def fig3_aose_confusion(logits_all, gt_meta, known_classes, out_dir):
    """Fig 3: A-OSE confusion matrix — unknown GT features scored by each known class."""
    if 'unknown' not in logits_all or len(logits_all['unknown']) == 0:
        print('  [skip] No unknown features for A-OSE confusion matrix')
        return

    unk_logits = logits_all['unknown']  # (N_unk, K)
    unk_meta = gt_meta['unknown']

    # Get original unknown class names
    orig_classes = sorted(set(m['original_cls'] for m in unk_meta))

    # For each unknown GT, find which known class has highest logit (simulating A-OSE)
    num_known = len(known_classes)
    # Confusion: rows = original unknown classes, cols = known classes
    confusion = np.zeros((len(orig_classes), num_known), dtype=int)
    orig_to_idx = {c: i for i, c in enumerate(orig_classes)}

    scores = torch.sigmoid(unk_logits[:, :num_known])  # (N, num_known)
    top_known = scores.argmax(dim=1)  # which known class scores highest

    for i, meta in enumerate(unk_meta):
        row = orig_to_idx[meta['original_cls']]
        col = top_known[i].item()
        confusion[row, col] += 1

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(confusion, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(num_known))
    ax.set_xticklabels(known_classes, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(orig_classes)))
    ax.set_yticklabels(orig_classes, fontsize=9)

    # Annotate cells
    for i in range(len(orig_classes)):
        for j in range(num_known):
            if confusion[i, j] > 0:
                ax.text(j, i, str(confusion[i, j]), ha='center', va='center',
                        fontsize=8, color='black' if confusion[i, j] < confusion.max() * 0.7 else 'white')

    ax.set_xlabel('Known class (absorbing unknown as)', fontsize=11)
    ax.set_ylabel('True unknown class', fontsize=11)
    ax.set_title('A-OSE Confusion: Which known class absorbs each unknown type\n'
                 '(based on highest logit at GT unknown locations)', fontsize=12)
    fig.colorbar(im, ax=ax, label='count')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'aose_confusion_matrix.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved aose_confusion_matrix.png')


def fig4_score_distributions(logits_all, gt_meta, features, known_classes,
                             num_prev, out_dir):
    """Fig 4: Score distributions — for top A-OSE offenders, compare
    known-class GT scores vs unknown GT scores at that class channel."""
    if 'unknown' not in logits_all or len(logits_all['unknown']) == 0:
        print('  [skip] No unknown features for score distributions')
        return

    unk_logits = logits_all['unknown']  # (N_unk, K)
    num_known = len(known_classes)
    unk_scores = torch.sigmoid(unk_logits[:, :num_known])  # (N, num_known)

    # Find top 5 classes by total unknown-absorbing score mass
    unk_score_sum = unk_scores.sum(dim=0).numpy()
    top5_idx = np.argsort(-unk_score_sum)[:5]

    n_plots = len(top5_idx)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), squeeze=False)

    for plot_i, cls_idx in enumerate(top5_idx):
        ax = axes[0, plot_i]
        cls_name = known_classes[cls_idx]

        # Scores of unknown GT at this class channel
        unk_cls_scores = unk_scores[:, cls_idx].numpy()

        # Scores of known GT for this class at this class channel
        if cls_name in logits_all and len(logits_all[cls_name]) > 0:
            known_logits = logits_all[cls_name]
            known_cls_scores = torch.sigmoid(known_logits[:, cls_idx]).numpy()
        else:
            known_cls_scores = np.array([])

        bins = np.linspace(0, 1, 40)
        if len(known_cls_scores) > 0:
            ax.hist(known_cls_scores, bins=bins, alpha=0.6, color='tab:blue',
                    label=f'{cls_name} GT', density=True)
        ax.hist(unk_cls_scores, bins=bins, alpha=0.6, color='tab:red',
                label='unknown GT', density=True)

        ax.set_title(f'{cls_name}\n(unk absorbed: {unk_score_sum[cls_idx]:.0f})',
                     fontsize=10)
        ax.set_xlabel('sigmoid score', fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)

    fig.suptitle('Score distributions at GT locations: known vs unknown', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'score_distributions.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved score_distributions.png')


def fig5_cosine_heatmaps(prompt_dict, features, known_classes, out_dir):
    """Fig 5: Side-by-side cosine heatmaps — prompt×prompt and centroid×centroid."""
    names = [c for c in known_classes if c in prompt_dict]
    if len(names) < 2:
        print('  [skip] Not enough classes for cosine heatmaps')
        return

    # Prompt pairwise
    prompt_mat = torch.stack([prompt_dict[c] for c in names])
    prompt_cos = pairwise_cosine(prompt_mat)

    # Centroid pairwise
    centroids = []
    centroid_names = []
    for c in names:
        if c in features and len(features[c]) > 0:
            centroids.append(features[c].mean(dim=0))
            centroid_names.append(c)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: prompt×prompt
    im1 = ax1.imshow(prompt_cos, cmap='RdBu_r', vmin=-0.3, vmax=1.0, aspect='equal')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax1.text(j, i, f'{prompt_cos[i, j]:.2f}', ha='center', va='center',
                     fontsize=7)
    ax1.set_title('Prompt ↔ Prompt cosine similarity', fontsize=11)
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    # Right: centroid×centroid
    if len(centroids) >= 2:
        ctr_mat = torch.stack(centroids)
        ctr_cos = pairwise_cosine(ctr_mat)
        im2 = ax2.imshow(ctr_cos, cmap='RdBu_r', vmin=-0.3, vmax=1.0, aspect='equal')
        ax2.set_xticks(range(len(centroid_names)))
        ax2.set_xticklabels(centroid_names, rotation=45, ha='right', fontsize=8)
        ax2.set_yticks(range(len(centroid_names)))
        ax2.set_yticklabels(centroid_names, fontsize=8)
        for i in range(len(centroid_names)):
            for j in range(len(centroid_names)):
                ax2.text(j, i, f'{ctr_cos[i, j]:.2f}', ha='center', va='center',
                         fontsize=7)
        ax2.set_title('Vision centroid ↔ centroid cosine similarity', fontsize=11)
        fig.colorbar(im2, ax=ax2, shrink=0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'cosine_heatmaps.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved cosine_heatmaps.png')


def fig6_decision_boundary(logits_all, gt_meta, known_classes, out_dir):
    """Fig 6: For top 3 A-OSE confusing pairs, scatter
    score_to_known vs score_to_unknown for unknown GT features."""
    if 'unknown' not in logits_all or len(logits_all['unknown']) == 0:
        print('  [skip] No unknown features for decision boundary')
        return

    unk_logits = logits_all['unknown']
    unk_meta = gt_meta['unknown']
    num_known = len(known_classes)
    unk_idx = num_known  # unknown class index in logits

    scores = torch.sigmoid(unk_logits)  # (N, K+extra)
    known_scores = scores[:, :num_known]
    unk_score = scores[:, unk_idx] if scores.shape[1] > unk_idx else torch.zeros(len(scores))

    # Count which known class absorbs most unknowns
    top_known = known_scores.argmax(dim=1)
    absorb_count = defaultdict(int)
    for k in top_known.numpy():
        absorb_count[k] += 1
    top3 = sorted(absorb_count.keys(), key=lambda k: -absorb_count[k])[:3]

    fig, axes = plt.subplots(1, len(top3), figsize=(5 * len(top3), 5), squeeze=False)

    for plot_i, cls_idx in enumerate(top3):
        ax = axes[0, plot_i]
        cls_name = known_classes[cls_idx]

        # Unknowns whose top known class is this one
        mask = (top_known == cls_idx).numpy()
        x = known_scores[mask, cls_idx].numpy()
        y = unk_score[mask].numpy()

        # Color by original class
        orig_names = [unk_meta[i]['original_cls'] for i in range(len(mask)) if mask[i]]
        unique_orig = sorted(set(orig_names))
        orig_cmap = plt.cm.tab10
        for oi, orig in enumerate(unique_orig):
            omask = np.array([o == orig for o in orig_names])
            ax.scatter(x[omask], y[omask], s=15, alpha=0.6,
                       c=[orig_cmap(oi)], label=orig)

        # Diagonal: above = unknown wins, below = known class wins
        lim = max(x.max() if len(x) > 0 else 0.5,
                  y.max() if len(y) > 0 else 0.5) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='decision boundary')

        ax.set_xlabel(f'score({cls_name})', fontsize=9)
        ax.set_ylabel('score(unknown)', fontsize=9)
        ax.set_title(f'{cls_name} (absorbs {absorb_count[cls_idx]} unk)', fontsize=10)
        ax.legend(fontsize=7, loc='upper left')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

    fig.suptitle('Decision boundary: known class score vs unknown score\n'
                 '(points below diagonal → misclassified as known)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'decision_boundary_top3.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved decision_boundary_top3.png')


def write_analysis(features, logits_all, gt_meta, prompt_dict, known_classes,
                   num_prev, out_dir):
    """Write analysis.txt with quantitative summary."""
    lines = []
    lines.append('=' * 80)
    lines.append('EMBEDDING DIAGNOSTIC — QUANTITATIVE SUMMARY')
    lines.append('=' * 80)
    lines.append('')

    # Prompt embedding norms (important: base ~4.3, novel ~1.0 reveals training asymmetry)
    lines.append('PROMPT EMBEDDING NORMS')
    lines.append('-' * 50)
    for cls in known_classes + ['unknown', 'anchor']:
        if cls in prompt_dict:
            n = prompt_dict[cls].norm().item()
            cls_type = 'base' if cls in IDD_CLASSES_T1 else \
                       'novel' if cls in IDD_CLASSES_T2 else cls
            lines.append(f'  {cls:20s} ({cls_type:6s}): norm = {n:.4f}')
    lines.append('')

    # Centroids (raw BN space)
    centroids = {}
    for cls, feats in features.items():
        if len(feats) > 0:
            centroids[cls] = feats.mean(dim=0)

    # Global mean for centering (the dominant direction shared by all BN features)
    all_feats = torch.cat([features[c] for c in features], dim=0)
    global_mean = all_feats.mean(dim=0)
    cos_with_global = F.cosine_similarity(all_feats, global_mean.unsqueeze(0), dim=-1)
    lines.append(f'BN FEATURE SPACE ANALYSIS')
    lines.append(f'-' * 50)
    lines.append(f'  Global mean norm:          {global_mean.norm():.4f}')
    lines.append(f'  Avg feature norm:          {all_feats.norm(dim=-1).mean():.4f}')
    lines.append(f'  cos(feat, global_mean):    mean={cos_with_global.mean():.4f}, '
                 f'std={cos_with_global.std():.4f}')
    lines.append(f'  (values near 1.0 = features collapsed into a cone; '
                 f'discriminative info is in the residual)')
    lines.append('')

    # Centered centroids
    centered_centroids = {c: centroids[c] - global_mean for c in centroids}
    # L2-normalized prompts (what the head uses)
    l2_prompts = {c: F.normalize(prompt_dict[c].float().unsqueeze(0), dim=-1).squeeze()
                  for c in prompt_dict}

    # Per-class table with CENTERED cosine (shows actual discriminative structure)
    header = (f'{"class":20s} | {"type":6s} | {"n":>5s} | '
              f'{"cos_raw":>8s} | {"cos_ctrd":>9s} | {"intra_cos":>10s} | '
              f'{"nearest_ctrd":>30s}')
    lines.append(header)
    lines.append('-' * len(header))

    for cls in known_classes + ['unknown']:
        if cls not in features or len(features[cls]) < 1:
            continue

        n = len(features[cls])
        cls_type = 'base' if cls in IDD_CLASSES_T1 else \
                   'novel' if cls in IDD_CLASSES_T2 else 'unk'

        # cos(prompt, centroid) — raw space
        cos_raw = ''
        if cls in prompt_dict and cls in centroids:
            cos_raw = f'{cosine_sim(prompt_dict[cls], centroids[cls]):.4f}'

        # cos(L2_prompt, centered_centroid) — more meaningful
        cos_ctrd = ''
        if cls in l2_prompts and cls in centered_centroids:
            cos_ctrd = f'{cosine_sim(l2_prompts[cls], centered_centroids[cls]):.4f}'

        # Intra-class mean cosine (in centered space)
        intra = ''
        if cls in centroids and n > 1:
            centered_feats = features[cls].float() - global_mean
            feats_norm = F.normalize(centered_feats, dim=1)
            ctr_norm = F.normalize(centered_centroids[cls].float().unsqueeze(0), dim=1)
            cos_vals = (feats_norm @ ctr_norm.T).squeeze()
            intra = f'{cos_vals.mean().item():.4f}'

        # Nearest other class (in centered space — reveals actual confusion)
        nearest = ''
        if cls in centered_centroids:
            best_name, best_cos = '', -2.0
            for other, other_ctr in centered_centroids.items():
                if other == cls:
                    continue
                c = cosine_sim(centered_centroids[cls], other_ctr)
                if c > best_cos:
                    best_cos = c
                    best_name = other
            if best_name:
                nearest = f'{best_name} ({best_cos:.4f})'

        lines.append(f'{cls:20s} | {cls_type:6s} | {n:5d} | '
                     f'{cos_raw:>8s} | {cos_ctrd:>9s} | {intra:>10s} | '
                     f'{nearest:>30s}')

    lines.append('')

    # A-OSE breakdown from logits
    if 'unknown' in logits_all and len(logits_all['unknown']) > 0:
        lines.append('=' * 80)
        lines.append('A-OSE ANALYSIS (from GT-location logits)')
        lines.append('=' * 80)
        unk_logits = logits_all['unknown']
        num_known = len(known_classes)
        scores = torch.sigmoid(unk_logits[:, :num_known])
        unk_score_idx = num_known
        if unk_logits.shape[1] > unk_score_idx:
            unk_scores = torch.sigmoid(unk_logits[:, unk_score_idx])
        else:
            unk_scores = torch.zeros(len(unk_logits))

        top_known = scores.argmax(dim=1)
        lines.append('')
        lines.append('Per-unknown-GT: which known class has highest score:')
        absorb_count = defaultdict(int)
        for k in top_known.numpy():
            absorb_count[k] += 1
        for idx in sorted(absorb_count.keys(), key=lambda x: -absorb_count[x]):
            lines.append(f'  {known_classes[idx]:25s}: {absorb_count[idx]:5d}')

        # Unknown vs best-known score comparison
        best_known_score = scores.max(dim=1)[0]
        lines.append('')
        lines.append(f'Mean best-known score at unknown GT: {best_known_score.mean().item():.4f}')
        lines.append(f'Mean unknown score at unknown GT:    {unk_scores.mean().item():.4f}')
        lines.append(f'Fraction where known > unknown:      '
                     f'{(best_known_score > unk_scores).float().mean().item():.4f}')

        # Per original unknown class
        lines.append('')
        lines.append('Per original unknown class:')
        unk_meta_list = gt_meta['unknown']
        orig_classes = sorted(set(m['original_cls'] for m in unk_meta_list))
        for orig in orig_classes:
            mask = torch.tensor([m['original_cls'] == orig for m in unk_meta_list])
            n = mask.sum().item()
            if n == 0:
                continue
            bk = best_known_score[mask].mean().item()
            us = unk_scores[mask].mean().item()
            frac = (best_known_score[mask] > unk_scores[mask]).float().mean().item()
            top_cls_idx = top_known[mask].mode()[0].item()
            lines.append(f'  {orig:25s}: n={n:4d}, mean_known={bk:.4f}, '
                         f'mean_unk={us:.4f}, known>unk={frac:.2%}, '
                         f'top_absorber={known_classes[top_cls_idx]}')

    lines.append('')
    lines.append('=' * 80)

    out_path = os.path.join(out_dir, 'analysis.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Saved analysis.txt')
    # Also print to stdout
    print('\n'.join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    known_classes, num_prev, num_cur = get_class_info(args.task)

    if args.output_dir is None:
        args.output_dir = f'visualizations/embed_diag_t{args.task}'
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'per_class'), exist_ok=True)

    cache_path = os.path.join(args.output_dir, 'cached_features.pt')

    # ── Step 1: Extract or load features ──
    if args.cache and os.path.exists(args.cache):
        print(f'Loading cached features from {args.cache}')
        cached = torch.load(args.cache, map_location='cpu')
        features = cached['features']
        logits_all = cached['logits']
        gt_meta = cached['gt_meta']
    else:
        print('Loading model...')
        model, cfg = load_model(args.config, args.checkpoint, args.device)
        print('Extracting features...')
        features, logits_all, gt_meta = extract_features(
            model, cfg, known_classes, args.max_images, args.device)
        # Cache
        torch.save(dict(features=features, logits=logits_all, gt_meta=gt_meta),
                   cache_path)
        print(f'Cached features to {cache_path}')
        # Free GPU
        del model
        torch.cuda.empty_cache()

    # ── Step 2: Load prompt embeddings ──
    print('Loading prompt embeddings...')
    prompt_dict = load_prompt_embeddings(args.checkpoint, known_classes, args.task)

    # ── Step 3: Dimensionality reduction ──
    print(f'Running {args.method.upper()} projection...')
    all_vecs = []
    all_labels = []

    for cls, feats in features.items():
        for v in feats:
            all_vecs.append(v)
            all_labels.append(cls)

    # Add prompt embeddings — L2 normalized, matching BNContrastiveHead.forward()
    # which does w = F.normalize(w, dim=-1, p=2) before the dot product
    for cls, emb in prompt_dict.items():
        all_vecs.append(F.normalize(emb.unsqueeze(0), dim=-1, p=2).squeeze(0))
        all_labels.append(f'PROMPT_{cls}')

    all_vecs = torch.stack(all_vecs)
    coords_all = run_projection(all_vecs, all_labels, args.method)

    # Split coords back
    coords_dict = defaultdict(list)
    prompt_coords = {}
    for i, lbl in enumerate(all_labels):
        if lbl.startswith('PROMPT_'):
            prompt_coords[lbl.replace('PROMPT_', '')] = coords_all[i]
        else:
            coords_dict[lbl].append(coords_all[i])

    for k in coords_dict:
        coords_dict[k] = np.array(coords_dict[k])

    colors = make_colors(known_classes, num_prev)

    # ── Step 4: Generate figures ──
    print('Generating figures...')
    fig1_embedding_overview(coords_dict, prompt_coords, colors,
                            args.output_dir, args.method)
    fig2_per_class(features, prompt_dict, known_classes, colors,
                   args.output_dir, coords_all, all_labels, args.method)
    fig3_aose_confusion(logits_all, gt_meta, known_classes, args.output_dir)
    fig4_score_distributions(logits_all, gt_meta, features, known_classes,
                             num_prev, args.output_dir)
    fig5_cosine_heatmaps(prompt_dict, features, known_classes, args.output_dir)
    fig6_decision_boundary(logits_all, gt_meta, known_classes, args.output_dir)

    # ── Step 5: Write analysis ──
    write_analysis(features, logits_all, gt_meta, prompt_dict, known_classes,
                   num_prev, args.output_dir)

    print(f'\nAll outputs saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
