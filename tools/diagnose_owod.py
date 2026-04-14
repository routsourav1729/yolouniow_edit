"""Universal OWOD diagnostic script for all datasets.

Runs on GPU. Loads T1 and T2 checkpoints, runs forward on val images,
and produces detailed diagnostics:
  1. Embedding analysis (offline)
  2. Model forward diagnostics (score distributions, gatekeeper stats)
  3. Visual centroid vs CLIP text init
  4. T_unk proximity analysis
  5. Score margin at GT locations
  6. Novel class feature compactness

Usage (via sbatch):
  DATASET=IDD TASK=2 python tools/diagnose_owod.py \\
    --config configs/owod_ft/...t2.py \\
    --checkpoint work_dirs/.../best.pth \\
    --t1-checkpoint work_dirs/.../best.pth \\
    --num-images 50
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# ─── constants ────────────────────────────────────────────────────────────
FPN_STRIDES = [8, 16, 32]
AREA_THRESHOLDS = [96**2, 192**2]


# ─── dataset definitions ──────────────────────────────────────────────────
DATASET_INFO = {
    'IDD': {
        'task_list': [0, 8, 14],
        't1_classes': ['car', 'motorcycle', 'rider', 'person',
                       'autorickshaw', 'bicycle', 'traffic sign',
                       'traffic light'],
        't2_novel': ['bus', 'truck', 'tanker_vehicle', 'crane_truck',
                     'street_cart', 'excavator'],
        'remaining_unknown': ['pole', 'animal', 'tractor',
                              'concrete_mixer', 'pull_cart', 'road_roller'],
        'embed_dir': 'embeddings/uniow-idd',
        'embed_prefix': 'idd',
        'ann_dataset': 'IDD',
        'fewshot_dir_key': 'iddsplit',
    },
    'FOOD_VOCCOCO': {
        'task_list': [0, 20, 40],
        't1_classes': [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
        ],
        't2_novel': [
            'truck', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator',
        ],
        'remaining_unknown': [
            'book', 'cup', 'bowl', 'banana', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'knife',
            'spoon', 'fork', 'apple', 'sandwich', 'orange',
            'broccoli', 'wine glass', 'laptop', 'cell phone',
            'scissors',
        ],
        'embed_dir': 'embeddings/uniow-food-voccoco',
        'embed_prefix': 'food_voccoco',
        'ann_dataset': 'FOOD_VOCCOCO',
        'fewshot_dir_key': 'voccocosplit',
    },
    'FOOD_VOC': {
        'task_list': [0, 10, 15],
        't1_classes': [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
        ],
        't2_novel': [
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
        ],
        'remaining_unknown': [
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
        ],
        'embed_dir': 'embeddings/uniow-food-voc',
        'embed_prefix': 'food_voc',
        'ann_dataset': 'FOOD_VOC',
        'fewshot_dir_key': 'vocsplit',
    },
}


def parse_args():
    p = argparse.ArgumentParser(description='OWOD universal diagnostic')
    p.add_argument('--config', required=True, help='T2 config file path')
    p.add_argument('--checkpoint', required=True, help='T2 checkpoint .pth')
    p.add_argument('--t1-checkpoint', default='',
                   help='T1 checkpoint (for visual centroid analysis)')
    p.add_argument('--num-images', type=int, default=0,
                   help='Number of val images (0=all)')
    p.add_argument('--sections', nargs='+', default=['all'],
                   choices=['embedding', 'visual_centroid', 'tunk_proximity',
                            'forward', 'score_margin', 'compactness',
                            'gradient', 'unknown_fate', 'all'],
                   help='Which sections to run (default: all)')
    return p.parse_args()


# ─── helpers ──────────────────────────────────────────────────────────────

def fpn_level_for_area(area):
    if area < AREA_THRESHOLDS[0]:
        return 0
    elif area < AREA_THRESHOLDS[1]:
        return 1
    return 2


def parse_voc_xml(xml_path):
    """Parse VOC XML -> list of dict(name, bbox=[x1,y1,x2,y2])."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        bb = obj.find('bndbox')
        box = [float(bb.find(t).text) for t in ('xmin', 'ymin', 'xmax', 'ymax')]
        objs.append(dict(name=name, bbox=box))
    return objs


def cosine_sim(a, b):
    """Cosine similarity between two 1-D tensors."""
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).float()
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b).float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def build_model(cfg, checkpoint_path):
    """Build model, load checkpoint, return on GPU in eval mode."""
    from mmengine.runner import load_state_dict
    from mmyolo.registry import MODELS
    model = MODELS.build(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    state_dict = {k: v for k, v in state_dict.items()
                  if 'text_model' not in k}
    if 'embeddings' in state_dict:
        state_dict['embeddings'] = model.update_embeddings(
            state_dict['embeddings'])
    load_state_dict(model, state_dict, strict=False)
    model.eval()
    model.cuda()
    return model


def setup_hooks(head_module):
    """Register forward hooks on one2one BN and logit outputs of all FPN levels.
    Uses one2one head (same as inference/eval), NOT one2many.
    Returns (hooked_dict, handle_list)."""
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
        if hasattr(layer, 'norm'):
            handles.append(layer.norm.register_forward_hook(
                make_hook(f'cls_bn_{i}')))
    return hooked, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def extract_at_box(hooked, bbox_scaled, use_bn=True):
    """Extract BN feature + logit at box center from hooked outputs.
    bbox_scaled = (x1, y1, x2, y2) in resized+padded coords.
    Returns (feat_512, logit_K) or (None, None)."""
    x1, y1, x2, y2 = bbox_scaled
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    area = (x2 - x1) * (y2 - y1)
    level = fpn_level_for_area(area)
    stride = FPN_STRIDES[level]

    feat_key = f'cls_bn_{level}' if use_bn else f'cls_pred_{level}'
    logit_key = f'cls_logit_{level}'
    if feat_key not in hooked or logit_key not in hooked:
        return None, None

    feat_map = hooked[feat_key]    # (1, 512, H, W)
    logit_map = hooked[logit_key]  # (1, K, H, W)

    _, C, H, W = feat_map.shape
    gx = max(0, min(int(cx / stride), W - 1))
    gy = max(0, min(int(cy / stride), H - 1))

    return feat_map[0, :, gy, gx], logit_map[0, :, gy, gx]


def load_and_preprocess_image(img_path, pipeline):
    """Load image, run pipeline, return (tensor, data_sample, scale_factor, pad_param)."""
    data = dict(img_path=img_path,
                img_id=os.path.basename(img_path).replace('.jpg', ''),
                instances=[])
    data = pipeline(data)
    img_tensor = data['inputs'].unsqueeze(0).float().cuda() / 255.0
    data_sample = data['data_samples']
    meta = data_sample.metainfo
    scale_factor = np.array(meta.get('scale_factor', [1.0, 1.0]))
    pad_param = meta.get('pad_param', np.zeros(4, dtype=np.float32))
    return img_tensor, data_sample, scale_factor, pad_param


def scale_box(bbox, scale_factor, pad_param):
    """Map original box to resized+padded coords.
    scale_factor = (w_scale, h_scale), pad_param = [top, bottom, left, right]."""
    return (bbox[0] * scale_factor[0] + pad_param[2],
            bbox[1] * scale_factor[1] + pad_param[0],
            bbox[2] * scale_factor[0] + pad_param[2],
            bbox[3] * scale_factor[1] + pad_param[0])


def build_image_pipeline(cfg):
    """Build image transform pipeline from config, skip annotation loading."""
    from mmengine.dataset import Compose
    # Pipeline may be at dataset level or dataset.dataset level
    vd = cfg.val_dataloader.dataset
    if hasattr(vd, 'pipeline'):
        raw_pipeline = vd.pipeline
    elif hasattr(vd, 'dataset') and hasattr(vd.dataset, 'pipeline'):
        raw_pipeline = vd.dataset.pipeline
    else:
        raise ValueError("Cannot find pipeline in val_dataloader config")
    img_pipeline_cfg = []
    for t in raw_pipeline:
        t_type = t.get('type', '')
        if 'LoadAnnotations' in t_type or 'PackDetInputs' in t_type:
            continue
        img_pipeline_cfg.append(t)
    img_pipeline_cfg.append(dict(type='mmdet.PackDetInputs',
                                 meta_keys=('img_id', 'img_path', 'ori_shape',
                                            'img_shape', 'scale_factor',
                                            'pad_param')))
    return Compose(img_pipeline_cfg)


def trigger_hooks(model, img_tensor, data_sample):
    """Run extract_feat + forward_one2one to populate hooks (matches eval)."""
    img_feats, txt_feats = model.extract_feat(img_tensor, [data_sample])
    model.bbox_head.head_module.forward_one2one(img_feats, txt_feats)


def analyze_embeddings(dataset_key, task):
    """Offline embedding analysis (no GPU needed)."""
    info = DATASET_INFO[dataset_key]
    edir = info['embed_dir']
    prefix = info['embed_prefix']

    t1 = np.load(f'{edir}/{prefix}_t1.npy')
    obj = np.load(f'{edir}/object.npy')
    obj_tuned = np.load(f'{edir}/object_tuned.npy')

    print(f"\n{'='*70}")
    print(f"EMBEDDING ANALYSIS — {dataset_key} T{task}")
    print(f"{'='*70}")
    print(f"  T1 embeddings:     shape={t1.shape}, norms=[{np.linalg.norm(t1, axis=1).min():.3f}, {np.linalg.norm(t1, axis=1).max():.3f}]")
    print(f"  object.npy:        shape={obj.shape}, norm={np.linalg.norm(obj):.3f}")
    print(f"  object_tuned.npy:  shape={obj_tuned.shape}, norm={np.linalg.norm(obj_tuned):.3f}")

    if task >= 2:
        t2 = np.load(f'{edir}/{prefix}_t2.npy')
        print(f"  T2 embeddings:     shape={t2.shape}, norms=[{np.linalg.norm(t2, axis=1).min():.3f}, {np.linalg.norm(t2, axis=1).max():.3f}]")

        n_base = len(info['t1_classes'])
        n_novel = len(info['t2_novel'])
        all_classes = info['t1_classes'] + info['t2_novel']

        t2_normed = t2 / np.linalg.norm(t2, axis=1, keepdims=True)

        # Base-to-novel similarity
        cross_sim = t2_normed[:n_base] @ t2_normed[n_base:].T
        print(f"\n  Base → Novel class max cosine similarity:")
        for i in range(n_base):
            max_j = np.argmax(cross_sim[i])
            print(f"    {info['t1_classes'][i]:25s} → {info['t2_novel'][max_j]:25s}: {cross_sim[i, max_j]:.3f}")

        # Novel inter-class similarity
        novel_sim = t2_normed[n_base:] @ t2_normed[n_base:].T
        print(f"\n  Novel class pairwise (top-5 most similar):")
        pairs = []
        for i in range(n_novel):
            for j in range(i + 1, n_novel):
                pairs.append((novel_sim[i, j], info['t2_novel'][i], info['t2_novel'][j]))
        pairs.sort(reverse=True)
        for sim_val, c1, c2 in pairs[:5]:
            print(f"    {c1:25s} <-> {c2:25s}: {sim_val:.3f}")

        # Anchor (object_tuned) similarity to all classes
        obj_tuned_normed = obj_tuned.flatten() / np.linalg.norm(obj_tuned)
        anchor_sim = t2_normed @ obj_tuned_normed
        print(f"\n  Anchor (object_tuned) sim to all known classes:")
        for i, c in enumerate(all_classes):
            print(f"    {c:25s}: {anchor_sim[i]:.3f}")


def run_forward_diagnostics(model, cfg, dataset_key, task, n_imgs):
    """Run model forward on val images and collect score distributions."""
    from mmyolo.registry import DATASETS

    info = DATASET_INFO[dataset_key]
    n_base = len(info['t1_classes'])
    n_novel = len(info['t2_novel']) if task >= 2 else 0
    n_known = n_base + n_novel

    print(f"\n{'='*70}")
    print(f"MODEL FORWARD DIAGNOSTICS — {dataset_key} T{task}")
    print(f"{'='*70}")
    txt_feats = model.embeddings[None].cuda()
    print(f"  Embeddings shape: {txt_feats.shape}")
    norms = txt_feats[0].norm(dim=-1)
    print(f"  Embed norms: known=[{norms[:n_known].min():.3f}, {norms[:n_known].max():.3f}]"
          f"  T_unk={norms[-2]:.3f}  anchor={norms[-1]:.3f}")

    val_dataset = DATASETS.build(cfg.val_dataloader.dataset)
    head = model.bbox_head.head_module
    n_imgs = len(val_dataset) if n_imgs <= 0 else min(n_imgs, len(val_dataset))

    bn_layer = head.one2one_cls_contrasts[0].norm
    rmean = bn_layer.running_mean
    rvar = bn_layer.running_var
    print(f"  BN running_mean norm: {rmean.norm():.2f}  (one2one head)")
    print(f"  BN running_var  mean: {rvar.mean():.4f}")
    print(f"  BN SNR: {rvar.mean().sqrt() / rmean.norm():.4f}")

    all_max_known_prob = []
    all_anchor_prob = []
    all_unk_prob = []
    gk_max_known = []
    gk_anchor = []
    gk_best_cls = []

    print(f"\n  Processing {n_imgs} val images...")
    from mmengine.dataset import pseudo_collate

    with torch.no_grad():
        for idx in range(n_imgs):
            data = val_dataset[idx]
            batch = pseudo_collate([data])
            batch_inputs = batch['inputs']
            if isinstance(batch_inputs, list):
                batch_inputs = torch.stack(batch_inputs)
            batch_inputs = batch_inputs.float().cuda()

            img_feats, txt_feats = model.extract_feat(
                batch_inputs, batch['data_samples'])

            for lvl_idx, img_feat in enumerate(img_feats):
                cls_embed = head.one2one_cls_preds[lvl_idx](img_feat)
                cls_logit = head.one2one_cls_contrasts[lvl_idx](
                    cls_embed, txt_feats)
                b, k, h, w = cls_logit.shape
                flat = cls_logit.permute(0, 2, 3, 1).reshape(-1, k)
                probs = flat.sigmoid()

                known_probs = probs[:, :n_known]
                unk_probs = probs[:, -2]
                anchor_probs = probs[:, -1]
                max_known, best_cls = known_probs.max(dim=1)

                all_max_known_prob.append(max_known.cpu())
                all_anchor_prob.append(anchor_probs.cpu())
                all_unk_prob.append(unk_probs.cpu())

                gk_mask = (anchor_probs > 0.01) & (anchor_probs > max_known)
                if gk_mask.any():
                    gk_max_known.append(max_known[gk_mask].cpu())
                    gk_anchor.append(anchor_probs[gk_mask].cpu())
                    gk_best_cls.append(best_cls[gk_mask].cpu())

            if (idx + 1) % 10 == 0:
                print(f"    ... {idx+1}/{n_imgs}")

    max_known_all = torch.cat(all_max_known_prob)
    anchor_all = torch.cat(all_anchor_prob)
    unk_all = torch.cat(all_unk_prob)

    print(f"\n{'='*70}")
    print(f"SCORE DISTRIBUTION — all {max_known_all.shape[0]} anchors")
    print(f"{'='*70}")
    for name, vals in [('max_known', max_known_all),
                       ('anchor', anchor_all),
                       ('T_unk', unk_all)]:
        v = vals.numpy()
        print(f"  {name:12s}: mean={v.mean():.6f} std={v.std():.6f} "
              f"p50={np.percentile(v, 50):.6f} p95={np.percentile(v, 95):.6f} "
              f"p99={np.percentile(v, 99):.6f}")

    if gk_anchor:
        gk_mk = torch.cat(gk_max_known)
        gk_an = torch.cat(gk_anchor)
        gk_cls = torch.cat(gk_best_cls)
        n_gk = len(gk_mk)
        n_total_anchors = len(max_known_all)

        print(f"\n{'='*70}")
        print(f"GATEKEEPER-PASSING ANCHORS: {n_gk}/{n_total_anchors} "
              f"({100*n_gk/n_total_anchors:.3f}%)")
        print(f"{'='*70}")
        print(f"  max_known: mean={gk_mk.mean():.6f} std={gk_mk.std():.6f}")
        print(f"  anchor:    mean={gk_an.mean():.6f} std={gk_an.std():.6f}")

        ratio = gk_mk / gk_an.clamp(min=1e-8)
        w_r = (1.0 - ratio).clamp(0, 1)
        r = ratio.numpy()
        print(f"\n  Score ratio (max_known/anchor):")
        print(f"    mean={r.mean():.4f} std={r.std():.4f}")
        print(f"    p25={np.percentile(r, 25):.4f} p50={np.percentile(r, 50):.4f} "
              f"p75={np.percentile(r, 75):.4f} p95={np.percentile(r, 95):.4f}")
        print(f"  w_r = 1 - ratio:")
        print(f"    mean={w_r.mean():.4f} std={w_r.std():.4f}")

        all_classes = info['t1_classes'] + (info['t2_novel'] if task >= 2
                                            else [])
        cls_counts = defaultdict(int)
        for c in gk_cls.numpy():
            cls_counts[int(c)] += 1
        print(f"\n  Best known class among GK-passing anchors "
              f"(potential A-OSE sources):")
        sorted_cls = sorted(cls_counts.items(), key=lambda x: -x[1])
        for cls_id, cnt in sorted_cls[:10]:
            cls_name = (all_classes[cls_id] if cls_id < len(all_classes)
                        else f"class_{cls_id}")
            pct = 100 * cnt / n_gk
            print(f"    {cls_name:25s}: {cnt:6d} ({pct:5.1f}%)")
    else:
        print("\n  WARNING: No anchors passed gatekeeper!")


# ══════════════════════════════════════════════════════════════════════════
# Section 3: VISUAL CENTROID vs CLIP TEXT INIT
# ══════════════════════════════════════════════════════════════════════════

def section_visual_centroid(args, cfg, dataset_key, task, pipeline):
    """Extract K-shot BN features using T1 model, compare to T2 prompts."""
    if not args.t1_checkpoint:
        print(f"\n  [SKIP] Visual Centroid: --t1-checkpoint not provided")
        return

    info = DATASET_INFO[dataset_key]
    n_base = len(info['t1_classes'])
    novel_classes = info['t2_novel']

    print(f"\n{'='*70}")
    print(f"VISUAL CENTROID vs CLIP TEXT INIT — {dataset_key} T{task}")
    print(f"{'='*70}")

    # CLIP zero-shot embeddings for novel classes
    edir = info['embed_dir']
    prefix = info['embed_prefix']
    clip_t2 = np.load(f'{edir}/{prefix}_t2.npy')
    clip_novel = clip_t2[n_base:]  # (n_novel, 512)

    # T2 fine-tuned prompt embeddings
    t2_ckpt = torch.load(args.checkpoint, map_location='cpu')
    t2_sd = t2_ckpt.get('state_dict', t2_ckpt)
    t2_embeds = t2_sd['embeddings']
    t2_novel_prompts = t2_embeds[n_base:n_base + len(novel_classes)].numpy()

    # Load fewshot image lists for novel classes
    fewshot_seed = int(os.environ.get('FEWSHOT_SEED', '1'))
    fewshot_k = int(os.environ.get('FEWSHOT_K', '10'))
    fewshot_dir = f"data/OWOD/{info['fewshot_dir_key']}/seed{fewshot_seed}"
    ann_dir = f"data/OWOD/Annotations/{info['ann_dataset']}"
    img_dir = f"data/OWOD/JPEGImages/{info['ann_dataset']}"

    class_images = {}  # cls_name -> [(img_path, [bbox, ...])]
    for cls_name in novel_classes:
        fs_name = cls_name.replace(' ', '_')
        fs_file = os.path.join(fewshot_dir,
                               f'box_{fewshot_k}shot_{fs_name}_train.txt')
        if not os.path.exists(fs_file):
            print(f"  [WARN] Fewshot file not found: {fs_file}")
            continue
        with open(fs_file) as f:
            img_paths = [x.strip() for x in f if x.strip()]
        entries = []
        for ip in img_paths:
            img_id = os.path.basename(ip).replace('.jpg', '').replace(
                '.png', '')
            xml_path = os.path.join(ann_dir, img_id + '.xml')
            real_img_path = os.path.join(img_dir, img_id + '.jpg')
            if not os.path.exists(xml_path) or not os.path.exists(
                    real_img_path):
                continue
            objs = parse_voc_xml(xml_path)
            boxes = [o['bbox'] for o in objs if o['name'] == cls_name]
            if boxes:
                entries.append((real_img_path, boxes))
        class_images[cls_name] = entries

    # Build T1 model and extract features at K-shot locations
    print(f"  Loading T1 model: {args.t1_checkpoint}")
    t1_model = build_model(cfg, args.t1_checkpoint)
    hooked, handles = setup_hooks(t1_model.bbox_head.head_module)

    visual_centroids = {}
    with torch.no_grad():
        for cls_name in novel_classes:
            if cls_name not in class_images or not class_images[cls_name]:
                print(f"  [WARN] No images for {cls_name}")
                continue
            feats = []
            for img_path, boxes in class_images[cls_name]:
                img_tensor, ds, sf, pp = load_and_preprocess_image(
                    img_path, pipeline)
                trigger_hooks(t1_model, img_tensor, ds)
                for box in boxes:
                    sb = scale_box(box, sf, pp)
                    feat_vec, _ = extract_at_box(hooked, sb, use_bn=True)
                    if feat_vec is not None:
                        feats.append(feat_vec.cpu())
            if feats:
                stacked = torch.stack(feats)
                centroid = F.normalize(stacked.mean(dim=0), dim=0)
                visual_centroids[cls_name] = centroid.numpy()

    remove_hooks(handles)
    del t1_model
    torch.cuda.empty_cache()

    # Print table
    print(f"\n  {'Class':25s} | {'cos(T2,visCent)':>15s} | "
          f"{'cos(T2,clipInit)':>16s} | {'cos(visCent,clip)':>17s} | "
          f"{'n_feat':>6s}")
    print(f"  {'-'*25}-+-{'-'*15}-+-{'-'*16}-+-{'-'*17}-+-{'-'*6}")
    for i, cls_name in enumerate(novel_classes):
        t2_vec = t2_novel_prompts[i]
        clip_vec = clip_novel[i]
        if cls_name in visual_centroids:
            vc = visual_centroids[cls_name]
            c_t2_vc = cosine_sim(t2_vec, vc)
            c_t2_clip = cosine_sim(t2_vec, clip_vec)
            c_vc_clip = cosine_sim(vc, clip_vec)
            n = sum(len(b) for _, b in class_images.get(cls_name, []))
            print(f"  {cls_name:25s} | {c_t2_vc:15.4f} | "
                  f"{c_t2_clip:16.4f} | {c_vc_clip:17.4f} | {n:6d}")
        else:
            c_t2_clip = cosine_sim(t2_vec, clip_vec)
            print(f"  {cls_name:25s} | {'N/A':>15s} | "
                  f"{c_t2_clip:16.4f} | {'N/A':>17s} | {0:6d}")


# ══════════════════════════════════════════════════════════════════════════
# Section 4: T_UNK PROXIMITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def section_tunk_proximity(args, dataset_key, task):
    info = DATASET_INFO[dataset_key]
    n_base = len(info['t1_classes'])
    all_classes = info['t1_classes'] + info['t2_novel']

    t2_ckpt = torch.load(args.checkpoint, map_location='cpu')
    t2_sd = t2_ckpt.get('state_dict', t2_ckpt)
    t2_embeds = t2_sd['embeddings']
    t2_tunk = t2_embeds[-2]
    t2_known = t2_embeds[:len(all_classes)]

    print(f"\n{'='*70}")
    print(f"T_UNK PROXIMITY ANALYSIS — {dataset_key} T{task}")
    print(f"{'='*70}")

    sims = []
    for i, cls_name in enumerate(all_classes):
        s = cosine_sim(t2_tunk, t2_known[i])
        cls_type = 'base' if i < n_base else 'novel'
        sims.append((s, cls_name, cls_type))
    sims.sort(reverse=True)

    print(f"\n  T_unk (T2, norm={t2_tunk.norm():.3f}) similarity to known "
          f"classes (sorted):")
    print(f"  {'Class':25s} | {'cos(T_unk,cls)':>14s} | {'type':>6s}")
    print(f"  {'-'*25}-+-{'-'*14}-+-{'-'*6}")
    for s, cls_name, cls_type in sims:
        print(f"  {cls_name:25s} | {s:14.4f} | {cls_type:>6s}")

    if args.t1_checkpoint and os.path.exists(args.t1_checkpoint):
        t1_ckpt = torch.load(args.t1_checkpoint, map_location='cpu')
        t1_sd = t1_ckpt.get('state_dict', t1_ckpt)
        t1_embeds = t1_sd['embeddings']
        t1_tunk = t1_embeds[-2]

        print(f"\n  T_unk T1 (norm={t1_tunk.norm():.3f}) vs "
              f"T2 (norm={t2_tunk.norm():.3f})")
        print(f"  cos(T_unk_T1, T_unk_T2) = {cosine_sim(t1_tunk, t2_tunk):.4f}")

        print(f"\n  cos(T_unk_T1, novel prompts from T2):")
        for i, cls_name in enumerate(info['t2_novel']):
            s = cosine_sim(t1_tunk, t2_known[n_base + i])
            print(f"    {cls_name:25s}: {s:.4f}")

        drift = (t2_tunk - t1_tunk).norm().item()
        print(f"\n  T_unk drift (L2 distance T1->T2): {drift:.4f}")


# ══════════════════════════════════════════════════════════════════════════
# Section 5: SCORE MARGIN AT GT LOCATIONS
# ══════════════════════════════════════════════════════════════════════════

def section_score_margin(model, hooked, pipeline, dataset_key, task, n_imgs):
    """For each GT box in val images, extract scores and analyze margins."""
    info = DATASET_INFO[dataset_key]
    n_base = len(info['t1_classes'])
    n_novel = len(info['t2_novel']) if task >= 2 else 0
    n_known = n_base + n_novel
    all_classes = info['t1_classes'] + (info['t2_novel'] if task >= 2 else [])
    unk_class_names = set(info['remaining_unknown'])
    novel_class_names = set(info['t2_novel'])
    base_class_names = set(info['t1_classes'])

    ann_dir = f"data/OWOD/Annotations/{info['ann_dataset']}"
    img_dir = f"data/OWOD/JPEGImages/{info['ann_dataset']}"

    test_set_file = f"data/OWOD/ImageSets/{info['ann_dataset']}/test.txt"
    with open(test_set_file) as f:
        all_img_ids = [x.strip() for x in f if x.strip()]
    if n_imgs > 0:
        rng = np.random.RandomState(42)
        img_ids = list(rng.choice(all_img_ids,
                                  min(n_imgs, len(all_img_ids)),
                                  replace=False))
    else:
        img_ids = all_img_ids

    unk_gt_records = []
    novel_gt_records = []
    base_gt_records = []
    cls_to_idx = {c: i for i, c in enumerate(all_classes)}

    print(f"\n{'='*70}")
    print(f"SCORE MARGIN AT GT LOCATIONS — {dataset_key} T{task}")
    print(f"{'='*70}")
    print(f"  Processing {len(img_ids)} val images for GT analysis...")

    with torch.no_grad():
        for idx, img_id in enumerate(img_ids):
            xml_path = os.path.join(ann_dir, img_id + '.xml')
            img_path = os.path.join(img_dir, img_id + '.jpg')
            if not os.path.exists(xml_path) or not os.path.exists(img_path):
                continue

            objs = parse_voc_xml(xml_path)
            if not objs:
                continue

            img_tensor, ds, sf, pp = load_and_preprocess_image(img_path, pipeline)
            trigger_hooks(model, img_tensor, ds)

            for obj in objs:
                name, bbox = obj['name'], obj['bbox']
                sb = scale_box(bbox, sf, pp)
                _, logit_vec = extract_at_box(hooked, sb, use_bn=True)
                if logit_vec is None:
                    continue

                scores = logit_vec.sigmoid().cpu()
                known_scores = scores[:n_known]
                anchor_score = scores[-1].item()
                max_known_score, pred_cls_idx = known_scores.max(dim=0)
                max_known_score = max_known_score.item()
                pred_cls_idx = pred_cls_idx.item()

                known_copy = known_scores.clone()
                known_copy[pred_cls_idx] = -1
                second_best = known_copy.max().item()
                margin = max_known_score - second_best

                record = dict(
                    gt_name=name,
                    pred_cls=all_classes[pred_cls_idx],
                    pred_idx=pred_cls_idx,
                    score_pred=max_known_score,
                    score_anchor=anchor_score,
                    margin=margin,
                )

                if name in unk_class_names:
                    unk_gt_records.append(record)
                elif name in novel_class_names:
                    record['gt_idx'] = cls_to_idx.get(name, -1)
                    record['correct'] = (pred_cls_idx == record['gt_idx'])
                    novel_gt_records.append(record)
                elif name in base_class_names:
                    record['gt_idx'] = cls_to_idx.get(name, -1)
                    record['correct'] = (pred_cls_idx == record['gt_idx'])
                    base_gt_records.append(record)

            if (idx + 1) % 10 == 0:
                print(f"    ... {idx+1}/{len(img_ids)}")

    # ─── Unknown GT analysis ───
    n_unk = len(unk_gt_records)
    print(f"\n  UNKNOWN GT BOXES: {n_unk} total")
    if n_unk > 0:
        unk_correct = sum(1 for r in unk_gt_records
                          if r['score_anchor'] > r['score_pred'])
        unk_mis = n_unk - unk_correct
        print(f"    Detected as unknown (anchor>max_known): "
              f"{unk_correct} ({100*unk_correct/n_unk:.1f}%)")
        print(f"    Misclassified as known (A-OSE proxy):   "
              f"{unk_mis} ({100*unk_mis/n_unk:.1f}%)")

        if unk_mis > 0:
            absorb = defaultdict(int)
            m_lo = m_mid = m_hi = 0
            for r in unk_gt_records:
                if r['score_anchor'] <= r['score_pred']:
                    absorb[r['pred_cls']] += 1
                    if r['margin'] < 0.05:
                        m_lo += 1
                    elif r['margin'] < 0.2:
                        m_mid += 1
                    else:
                        m_hi += 1
            print(f"\n    Margin distribution of misclassified:")
            print(f"      margin<0.05 (borderline): "
                  f"{m_lo} ({100*m_lo/unk_mis:.1f}%)")
            print(f"      0.05<=margin<0.2:         "
                  f"{m_mid} ({100*m_mid/unk_mis:.1f}%)")
            print(f"      margin>=0.2 (confident):   "
                  f"{m_hi} ({100*m_hi/unk_mis:.1f}%)")

            print(f"\n    Absorbing known class breakdown:")
            for cls_name, cnt in sorted(absorb.items(),
                                        key=lambda x: -x[1])[:10]:
                ct = 'novel' if cls_name in novel_class_names else 'base'
                print(f"      {cls_name:25s} ({ct:5s}): "
                      f"{cnt:4d} ({100*cnt/unk_mis:.1f}%)")

    # ─── Novel GT analysis ───
    n_novel_gt = len(novel_gt_records)
    print(f"\n  NOVEL GT BOXES: {n_novel_gt} total")
    if n_novel_gt > 0:
        novel_correct = sum(1 for r in novel_gt_records if r['correct'])
        print(f"    Correctly classified: "
              f"{novel_correct} ({100*novel_correct/n_novel_gt:.1f}%)")
        novel_wrong = [r for r in novel_gt_records if not r['correct']]
        if novel_wrong:
            c_base = sum(1 for r in novel_wrong
                         if r['pred_cls'] in base_class_names)
            c_novel = sum(1 for r in novel_wrong
                          if r['pred_cls'] in novel_class_names)
            print(f"    Confused with base:   "
                  f"{c_base} ({100*c_base/n_novel_gt:.1f}%)")
            print(f"    Confused with novel:  "
                  f"{c_novel} ({100*c_novel/n_novel_gt:.1f}%)")
            conf = defaultdict(int)
            for r in novel_wrong:
                conf[r['pred_cls']] += 1
            print(f"\n    Top confusion targets:")
            for cls_name, cnt in sorted(conf.items(),
                                        key=lambda x: -x[1])[:8]:
                print(f"      {cls_name:25s}: {cnt:4d}")

    # ─── Base GT brief ───
    n_base_gt = len(base_gt_records)
    if n_base_gt > 0:
        base_correct = sum(1 for r in base_gt_records if r['correct'])
        print(f"\n  BASE GT BOXES: {n_base_gt} total, correct: "
              f"{base_correct} ({100*base_correct/n_base_gt:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════
# Section 6: NOVEL CLASS FEATURE COMPACTNESS
# ══════════════════════════════════════════════════════════════════════════

def section_feature_compactness(model, hooked, pipeline, dataset_key,
                                task, n_imgs):
    """Intra-class feature compactness and prompt alignment for all classes."""
    info = DATASET_INFO[dataset_key]
    n_base = len(info['t1_classes'])
    n_novel = len(info['t2_novel']) if task >= 2 else 0
    n_known = n_base + n_novel
    all_classes = info['t1_classes'] + (info['t2_novel'] if task >= 2 else [])
    known_set = set(all_classes)

    ann_dir = f"data/OWOD/Annotations/{info['ann_dataset']}"
    img_dir = f"data/OWOD/JPEGImages/{info['ann_dataset']}"

    test_set_file = f"data/OWOD/ImageSets/{info['ann_dataset']}/test.txt"
    with open(test_set_file) as f:
        all_img_ids = [x.strip() for x in f if x.strip()]
    if n_imgs > 0:
        rng = np.random.RandomState(42)
        img_ids = list(rng.choice(all_img_ids,
                                  min(n_imgs, len(all_img_ids)),
                                  replace=False))
    else:
        img_ids = all_img_ids

    class_feats = defaultdict(list)

    print(f"\n{'='*70}")
    print(f"NOVEL CLASS FEATURE COMPACTNESS — {dataset_key} T{task}")
    print(f"{'='*70}")
    print(f"  Extracting BN features at GT locations ({len(img_ids)} images)...")

    with torch.no_grad():
        for idx, img_id in enumerate(img_ids):
            xml_path = os.path.join(ann_dir, img_id + '.xml')
            img_path = os.path.join(img_dir, img_id + '.jpg')
            if not os.path.exists(xml_path) or not os.path.exists(img_path):
                continue
            objs = parse_voc_xml(xml_path)
            if not objs:
                continue

            img_tensor, ds, sf, pp = load_and_preprocess_image(img_path, pipeline)
            trigger_hooks(model, img_tensor, ds)

            for obj in objs:
                if obj['name'] not in known_set:
                    continue
                sb = scale_box(obj['bbox'], sf, pp)
                feat_vec, _ = extract_at_box(hooked, sb, use_bn=True)
                if feat_vec is not None:
                    class_feats[obj['name']].append(feat_vec.cpu())

    t2_prompts = model.embeddings[:n_known].detach().cpu()

    print(f"\n  {'Class':25s} | {'type':>5s} | {'n':>4s} | "
          f"{'intra_cos':>9s} | {'cos(cent,prompt)':>16s}")
    print(f"  {'-'*25}-+-{'-'*5}-+-{'-'*4}-+-{'-'*9}-+-{'-'*16}")

    for cls_idx, cls_name in enumerate(all_classes):
        cls_type = 'base' if cls_idx < n_base else 'novel'
        feats = class_feats.get(cls_name, [])
        n = len(feats)
        if n == 0:
            print(f"  {cls_name:25s} | {cls_type:>5s} | {0:4d} | "
                  f"{'N/A':>9s} | {'N/A':>16s}")
            continue

        stacked = torch.stack(feats)
        normed = F.normalize(stacked, dim=1)
        centroid = F.normalize(stacked.mean(dim=0), dim=0)

        if n >= 2:
            sim_mat = normed @ normed.T
            mask = ~torch.eye(n, dtype=bool)
            intra_cos = sim_mat[mask].mean().item()
        else:
            intra_cos = 1.0

        cos_cp = cosine_sim(centroid, t2_prompts[cls_idx])
        print(f"  {cls_name:25s} | {cls_type:>5s} | {n:4d} | "
              f"{intra_cos:9.4f} | {cos_cp:16.4f}")


# ══════════════════════════════════════════════════════════════════════════
# Section 7: GRADIENT SIGNAL ANALYSIS AT GT LOCATIONS
# ══════════════════════════════════════════════════════════════════════════

def section_gradient_signal(model, hooked, pipeline, dataset_key, task,
                            n_imgs):
    """Analytical gradient magnitude at GT box locations for novel classes."""
    if task < 2:
        return

    info = DATASET_INFO[dataset_key]
    n_base = len(info['t1_classes'])
    n_novel = len(info['t2_novel'])
    n_known = n_base + n_novel
    all_classes = info['t1_classes'] + info['t2_novel']
    novel_class_names = set(info['t2_novel'])
    cls_to_idx = {c: i for i, c in enumerate(all_classes)}

    ann_dir = f"data/OWOD/Annotations/{info['ann_dataset']}"
    img_dir = f"data/OWOD/JPEGImages/{info['ann_dataset']}"

    test_set_file = f"data/OWOD/ImageSets/{info['ann_dataset']}/test.txt"
    with open(test_set_file) as f:
        all_img_ids = [x.strip() for x in f if x.strip()]
    if n_imgs > 0:
        rng = np.random.RandomState(42)
        img_ids = list(rng.choice(all_img_ids,
                                  min(n_imgs, len(all_img_ids)),
                                  replace=False))
    else:
        img_ids = all_img_ids

    # Get logit_scale from BNContrastiveHead (one2one, same as eval)
    contrast_head = model.bbox_head.head_module.one2one_cls_contrasts[0]
    logit_scale_val = contrast_head.logit_scale.exp().item()
    bias_val = contrast_head.bias.item()

    # T_unk index is second-to-last in embedding
    unk_idx = model.embeddings.shape[0] - 2

    # Per-class accumulators
    # For each novel class: list of (pos_grad, mean_neg_known, neg_unk, worst_cls_idx)
    class_records = defaultdict(list)

    print(f"\n{'='*70}")
    print(f"GRADIENT SIGNAL ANALYSIS AT GT LOCATIONS — {dataset_key} T{task}")
    print(f"{'='*70}")
    print(f"  logit_scale = exp({contrast_head.logit_scale.item():.4f}) "
          f"= {logit_scale_val:.4f},  bias = {bias_val:.4f}")
    print(f"  Processing {len(img_ids)} val images...")

    with torch.no_grad():
        for idx, img_id in enumerate(img_ids):
            xml_path = os.path.join(ann_dir, img_id + '.xml')
            img_path = os.path.join(img_dir, img_id + '.jpg')
            if not os.path.exists(xml_path) or not os.path.exists(img_path):
                continue

            objs = parse_voc_xml(xml_path)
            if not objs:
                continue

            # Only process if there are novel GT boxes in this image
            has_novel = any(o['name'] in novel_class_names for o in objs)
            if not has_novel:
                continue

            img_tensor, ds, sf, pp = load_and_preprocess_image(
                img_path, pipeline)
            trigger_hooks(model, img_tensor, ds)

            for obj in objs:
                name = obj['name']
                if name not in novel_class_names:
                    continue

                gt_idx = cls_to_idx.get(name, -1)
                if gt_idx < 0:
                    continue

                sb = scale_box(obj['bbox'], sf, pp)
                feat_vec, logit_vec = extract_at_box(hooked, sb, use_bn=True)
                if feat_vec is None or logit_vec is None:
                    continue

                bn_norm = feat_vec.norm().item()
                probs = logit_vec.sigmoid()

                # Positive gradient: for correct class
                pos_grad = (1.0 - probs[gt_idx].item()) * logit_scale_val * bn_norm

                # Negative gradient: for each wrong known class
                neg_grads_known = []
                worst_neg = 0.0
                worst_cls = -1
                for j in range(n_known):
                    if j == gt_idx:
                        continue
                    ng = probs[j].item() * logit_scale_val * bn_norm
                    neg_grads_known.append(ng)
                    if ng > worst_neg:
                        worst_neg = ng
                        worst_cls = j

                mean_neg_known = (sum(neg_grads_known) / len(neg_grads_known)
                                  if neg_grads_known else 0.0)

                # T_unk negative gradient
                neg_unk = probs[unk_idx].item() * logit_scale_val * bn_norm

                class_records[name].append(dict(
                    pos_grad=pos_grad,
                    mean_neg_known=mean_neg_known,
                    neg_unk=neg_unk,
                    worst_cls=worst_cls,
                ))

            if (idx + 1) % 100 == 0:
                print(f"    ... {idx+1}/{len(img_ids)}")

    # Print table
    print(f"\n  {'Class':25s} | {'n_gt':>5s} | {'pos_grad':>10s} | "
          f"{'neg_known':>10s} | {'neg_unk':>10s} | {'ratio':>7s} | "
          f"{'max_neg_class':>20s}")
    print(f"  {'-'*25}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-"
          f"{'-'*10}-+-{'-'*7}-+-{'-'*20}")

    all_pos = []
    all_neg_known = []
    all_neg_unk = []

    for cls_name in info['t2_novel']:
        recs = class_records.get(cls_name, [])
        n = len(recs)
        if n == 0:
            print(f"  {cls_name:25s} | {0:5d} | {'N/A':>10s} | "
                  f"{'N/A':>10s} | {'N/A':>10s} | {'N/A':>7s} | "
                  f"{'N/A':>20s}")
            continue

        mp = np.mean([r['pos_grad'] for r in recs])
        mn = np.mean([r['mean_neg_known'] for r in recs])
        mu = np.mean([r['neg_unk'] for r in recs])
        ratio = mn / mp if mp > 1e-8 else float('inf')

        all_pos.extend(r['pos_grad'] for r in recs)
        all_neg_known.extend(r['mean_neg_known'] for r in recs)
        all_neg_unk.extend(r['neg_unk'] for r in recs)

        # Find most common worst class
        worst_counts = defaultdict(int)
        for r in recs:
            if r['worst_cls'] >= 0:
                worst_counts[r['worst_cls']] += 1
        if worst_counts:
            wc = max(worst_counts, key=worst_counts.get)
            worst_name = (all_classes[wc] if wc < len(all_classes)
                          else f"cls_{wc}")
        else:
            worst_name = "N/A"

        print(f"  {cls_name:25s} | {n:5d} | {mp:10.4f} | "
              f"{mn:10.4f} | {mu:10.4f} | {ratio:7.4f} | "
              f"{worst_name:>20s}")

    if all_pos:
        total_pos = np.mean(all_pos)
        total_neg = np.mean(all_neg_known)
        total_unk = np.mean(all_neg_unk)
        total_ratio = total_neg / total_pos if total_pos > 1e-8 else float('inf')
        print(f"\n  Summary across all novel GT boxes ({len(all_pos)} total):")
        print(f"    positive grad  = {total_pos:.4f}")
        print(f"    negative grad  = {total_neg:.4f}")
        print(f"    neg_unk grad   = {total_unk:.4f}")
        print(f"    ratio(neg/pos) = {total_ratio:.4f}")


# ══════════════════════════════════════════════════════════════════════════
# Section 8: UNKNOWN FATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def section_unknown_fate(model, cfg, hooked, pipeline, dataset_key, task):
    """Section 8: Unknown Fate Analysis — full test set, single pass.

    Part A: Per-unknown-class score profiles at GT locations (hook-based)
    Part B: Fate decomposition via model inference (predict-based)
    Part C: Full confusion matrix (derived from Part B)
    Part D: WAPR formula what-if simulation (uses Part A scores)
    """
    info = DATASET_INFO[dataset_key]
    n_base = len(info['t1_classes'])
    n_novel = len(info['t2_novel'])
    n_known = n_base + n_novel
    all_known_classes = info['t1_classes'] + info['t2_novel']
    unk_class_names = info['remaining_unknown']
    unk_set = set(unk_class_names)

    ann_dir = f"data/OWOD/Annotations/{info['ann_dataset']}"
    img_dir = f"data/OWOD/JPEGImages/{info['ann_dataset']}"
    test_set_file = f"data/OWOD/ImageSets/{info['ann_dataset']}/test.txt"
    with open(test_set_file) as f:
        img_ids = [x.strip() for x in f if x.strip()]

    class_profiles = defaultdict(list)  # Part A: per-class score dicts
    class_fates = defaultdict(list)     # Part B: per-class fate labels

    print(f"\n{'='*70}")
    print(f"UNKNOWN FATE ANALYSIS — {dataset_key} T{task}")
    print(f"{'='*70}")
    print(f"  Test images: {len(img_ids)} (full set)")
    print(f"  Unknown classes: {unk_class_names}")
    print(f"  n_known={n_known}, T_unk_idx={n_known}")

    # Lower score threshold to catch all predictions
    test_cfg = model.bbox_head.test_cfg
    orig_score_thr = test_cfg.get('score_thr', 0.05)
    test_cfg['score_thr'] = 0.005
    orig_num_classes = getattr(model.bbox_head, 'num_classes', n_known + 2)
    num_test_classes = getattr(model, 'num_test_classes', n_known + 2)

    from mmdet.structures.bbox import bbox_overlaps as compute_iou

    with torch.no_grad():
        for idx, img_id in enumerate(img_ids):
            xml_path = os.path.join(ann_dir, img_id + '.xml')
            img_path = os.path.join(img_dir, img_id + '.jpg')
            if not os.path.exists(xml_path) or not os.path.exists(img_path):
                continue

            objs = parse_voc_xml(xml_path)
            unk_objs = [o for o in objs if o['name'] in unk_set]
            if not unk_objs:
                if (idx + 1) % 100 == 0:
                    print(f"    ... {idx+1}/{len(img_ids)}")
                continue

            img_tensor, ds, sf, pp = load_and_preprocess_image(
                img_path, pipeline)

            # Single backbone pass: extract features once
            img_feats, txt_feats = model.extract_feat(img_tensor, [ds])

            # Part A: run one2one head to populate hooks, extract at GT
            model.bbox_head.head_module.forward_one2one(
                img_feats, txt_feats)
            for obj in unk_objs:
                sb = scale_box(obj['bbox'], sf, pp)
                _, logit_vec = extract_at_box(hooked, sb, use_bn=True)
                if logit_vec is None:
                    continue
                scores = logit_vec.sigmoid().cpu()
                anchor_s = scores[-1].item()
                tunk_s = scores[-2].item()
                known_s = scores[:n_known]
                max_ks, best_ki = known_s.max(dim=0)
                max_ks = max_ks.item()
                best_ki = best_ki.item()
                class_profiles[obj['name']].append(dict(
                    anchor_score=anchor_s,
                    tunk_score=tunk_s,
                    max_known_score=max_ks,
                    best_known_class=best_ki,
                    wapr_ratio=max_ks / max(anchor_s, 1e-8),
                ))

            # Part B: predict using already-computed features (no 2nd backbone)
            model.bbox_head.num_classes = num_test_classes
            results_list = model.bbox_head.predict(
                img_feats, txt_feats, [ds], rescale=True)
            pred = results_list[0]
            pred_bboxes = pred.bboxes
            pred_labels = pred.labels

            gt_boxes = [o['bbox'] for o in unk_objs]
            if pred_bboxes.shape[0] > 0:
                gt_t = torch.tensor(gt_boxes, dtype=torch.float32,
                                    device=pred_bboxes.device)
                ious = compute_iou(pred_bboxes, gt_t)  # (Np, Ng)
                for g, obj in enumerate(unk_objs):
                    max_iou, best_p = ious[:, g].max(dim=0)
                    if max_iou.item() >= 0.5:
                        lbl = pred_labels[best_p].item()
                        if lbl == n_known:
                            class_fates[obj['name']].append('detected_unk')
                        elif lbl < n_known:
                            class_fates[obj['name']].append(lbl)
                        else:
                            class_fates[obj['name']].append('undetected')
                    else:
                        class_fates[obj['name']].append('undetected')
            else:
                for obj in unk_objs:
                    class_fates[obj['name']].append('undetected')

            if (idx + 1) % 100 == 0:
                print(f"    ... {idx+1}/{len(img_ids)}")

    # Restore model state
    test_cfg['score_thr'] = orig_score_thr
    model.bbox_head.num_classes = orig_num_classes

    # ── Part A output: score profiles table ──────────────────────────────
    print(f"\n{'='*70}")
    print(f"UNKNOWN CLASS SCORE PROFILES AT GT LOCATIONS — "
          f"{dataset_key} T{task}")
    print(f"{'='*70}")
    _h_anc = 'anchor(\u03bc\u00b1\u03c3)'
    _h_unk = 'T_unk(\u03bc\u00b1\u03c3)'
    _h_mk  = 'max_known(\u03bc\u00b1\u03c3)'
    _h_rat = 'ratio(\u03bc\u00b1\u03c3)'
    print(f"  {'Class':21s} | {'n_gt':>4s} | {_h_anc:>11s} | "
          f"{_h_unk:>10s} | {_h_mk:>14s} | "
          f"{_h_rat:>10s} | top_known_class")
    print(f"  {'-'*21}-+-{'-'*4}-+-{'-'*11}-+-{'-'*10}-+-"
          f"{'-'*14}-+-{'-'*10}-+-{'-'*20}")

    all_profiles_flat = []
    for cls_name in unk_class_names:
        profs = class_profiles.get(cls_name, [])
        n = len(profs)
        if n == 0:
            print(f"  {cls_name:21s} | {0:4d} | {'N/A':>11s} | "
                  f"{'N/A':>10s} | {'N/A':>14s} | "
                  f"{'N/A':>10s} | N/A")
            continue
        all_profiles_flat.extend(profs)
        anchors = np.array([p['anchor_score'] for p in profs])
        tunks = np.array([p['tunk_score'] for p in profs])
        mk = np.array([p['max_known_score'] for p in profs])
        ratios = np.array([p['wapr_ratio'] for p in profs])
        cls_counts = defaultdict(int)
        for p in profs:
            cls_counts[p['best_known_class']] += 1
        top_i = max(cls_counts, key=cls_counts.get)
        top_n = (all_known_classes[top_i] if top_i < len(all_known_classes)
                 else f"cls_{top_i}")
        top_pct = 100 * cls_counts[top_i] / n
        print(f"  {cls_name:21s} | {n:4d} | "
              f"{anchors.mean():.2f}\u00b1{anchors.std():.2f} | "
              f"{tunks.mean():.2f}\u00b1{tunks.std():.2f} | "
              f"{mk.mean():.2f}\u00b1{mk.std():.2f} | "
              f"{ratios.mean():.2f}\u00b1{ratios.std():.2f} | "
              f"{top_n} ({top_pct:.0f}%)")

    print(f"\n  WAPR ratio percentiles per class:")
    print(f"  {'Class':21s} | {'p25':>6s} | {'p50':>6s} | "
          f"{'p75':>6s} | {'p95':>6s}")
    print(f"  {'-'*21}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for cls_name in unk_class_names:
        profs = class_profiles.get(cls_name, [])
        if not profs:
            continue
        ratios = np.array([p['wapr_ratio'] for p in profs])
        print(f"  {cls_name:21s} | {np.percentile(ratios, 25):6.3f} | "
              f"{np.percentile(ratios, 50):6.3f} | "
              f"{np.percentile(ratios, 75):6.3f} | "
              f"{np.percentile(ratios, 95):6.3f}")

    # ── Part B output: fate decomposition ────────────────────────────────
    print(f"\n{'='*70}")
    print(f"UNKNOWN FATE DECOMPOSITION — {dataset_key} T{task}")
    print(f"{'='*70}")
    print(f"  {'Class':21s} | {'n_gt':>4s} | {'detected_unk':>12s} | "
          f"{'misclassified':>13s} | {'undetected':>10s} | "
          f"{'U-Recall':>8s}")
    print(f"  {'-'*21}-+-{'-'*4}-+-{'-'*12}-+-{'-'*13}-+-"
          f"{'-'*10}-+-{'-'*8}")

    total_n = total_det = total_mis = total_undet = 0
    for cls_name in unk_class_names:
        fates = class_fates.get(cls_name, [])
        n = len(fates)
        if n == 0:
            print(f"  {cls_name:21s} | {0:4d} | {'--':>12s} | "
                  f"{'--':>13s} | {'--':>10s} | {'--':>8s}")
            continue
        det = sum(1 for f in fates if f == 'detected_unk')
        undet = sum(1 for f in fates if f == 'undetected')
        mis = n - det - undet
        total_n += n
        total_det += det
        total_mis += mis
        total_undet += undet
        print(f"  {cls_name:21s} | {n:4d} | "
              f"{det:4d} ({100*det/n:5.1f}%) | "
              f"{mis:4d} ({100*mis/n:5.1f}%) | "
              f"{undet:4d} ({100*undet/n:4.1f}%) | "
              f"{100*det/n:5.1f}%")

    if total_n > 0:
        print(f"  {'TOTAL':21s} | {total_n:4d} | "
              f"{total_det:4d} ({100*total_det/total_n:5.1f}%) | "
              f"{total_mis:4d} ({100*total_mis/total_n:5.1f}%) | "
              f"{total_undet:4d} ({100*total_undet/total_n:4.1f}%) | "
              f"{100*total_det/total_n:5.1f}%")

    # ── Part C output: confusion matrix ──────────────────────────────────
    print(f"\n{'='*70}")
    print(f"UNKNOWN \u2192 KNOWN CONFUSION MATRIX — {dataset_key} T{task}")
    print(f"{'='*70}")
    for cls_name in unk_class_names:
        fates = class_fates.get(cls_name, [])
        n = len(fates)
        if n == 0:
            continue
        confusion = defaultdict(int)
        det_unk = undet = 0
        for f in fates:
            if f == 'detected_unk':
                det_unk += 1
            elif f == 'undetected':
                undet += 1
            else:
                confusion[f] += 1
        sorted_conf = sorted(confusion.items(), key=lambda x: -x[1])
        parts = []
        for ci, cnt in sorted_conf:
            cn = (all_known_classes[ci] if ci < len(all_known_classes)
                  else f"cls_{ci}")
            parts.append(f"{cn}: {cnt} ({100*cnt/n:.1f}%)")
        print(f"\n  {cls_name} (n={n}):")
        if parts:
            print(f"    {', '.join(parts)}")
        print(f"    \u2192 unknown: {det_unk} ({100*det_unk/n:.1f}%), "
              f"\u2192 undetected: {undet} ({100*undet/n:.1f}%)")

    # ── Part D output: WAPR what-if simulation ───────────────────────────
    print(f"\n{'='*70}")
    print(f"WAPR FORMULA WHAT-IF — {dataset_key} T{task}")
    print(f"{'='*70}")
    if not all_profiles_flat:
        print("  No unknown GT score profiles collected.")
        return

    all_ratios_t = torch.tensor(
        [p['wapr_ratio'] for p in all_profiles_flat]).float()

    formulas = [
        ('No WAPR',         lambda r: torch.ones_like(r)),
        ('Current (1-r)',   lambda r: (1.0 - r).clamp(0, 1)),
        ('Sigmoid(10,0.5)', lambda r: 1 - torch.sigmoid(10 * (r - 0.5))),
        ('Hard \u03c4=0.7',      lambda r: (r < 0.7).float()),
        ('Hard \u03c4=0.5',      lambda r: (r < 0.5).float()),
        ('Squared (1-r)\u00b2',  lambda r: (1.0 - r).clamp(0, 1) ** 2),
    ]

    per_cls_r = {}
    active_cls = []
    for c in unk_class_names:
        profs = class_profiles.get(c, [])
        if profs:
            per_cls_r[c] = torch.tensor(
                [p['wapr_ratio'] for p in profs]).float()
            active_cls.append(c)

    hdrs = ' | '.join(f'{c[:7]:>7s}' for c in active_cls)
    print(f"  {'Formula':18s} | {'mean_w_r':>8s} | "
          f"{'%_suppr':>7s} | {hdrs}")
    csep = ' | '.join(['-' * 7] * len(active_cls))
    print(f"  {'-'*18}-+-{'-'*8}-+-{'-'*7}-+-{csep}")

    for fname, func in formulas:
        wr = func(all_ratios_t)
        mwr = wr.mean().item()
        sup = 100 * (wr < 0.5).float().mean().item()
        cvals = []
        for c in active_cls:
            cv = func(per_cls_r[c]).mean().item()
            cvals.append(f'{cv:7.3f}')
        print(f"  {fname:18s} | {mwr:8.3f} | "
              f"{sup:6.1f}% | {' | '.join(cvals)}")


def main():
    args = parse_args()
    sections = set(args.sections)
    dataset_key = os.environ.get('DATASET', 'IDD')
    task = int(os.environ.get('TASK', '2'))

    if dataset_key not in DATASET_INFO:
        print(f"ERROR: Unknown dataset '{dataset_key}'. "
              f"Supported: {list(DATASET_INFO.keys())}")
        sys.exit(1)

    def run(name):
        return name in sections or 'all' in sections

    print(f"Dataset: {dataset_key}, Task: T{task}")
    print(f"Config:  {args.config}")
    print(f"Ckpt:    {args.checkpoint}")
    if args.t1_checkpoint:
        print(f"T1 Ckpt: {args.t1_checkpoint}")
    print(f"Sections: {sorted(sections)}")

    # Phase 1: Embedding analysis (no GPU)
    if run('embedding'):
        analyze_embeddings(dataset_key, task)

    # Import and register modules
    from mmengine.config import Config
    import yolo_world  # noqa
    from mmyolo.utils import register_all_modules
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    cfg.work_dir = '/tmp/diagnose_owod'

    # Build image pipeline once (for GT-level sections)
    pipeline = build_image_pipeline(cfg)

    # Phase 2: Visual centroid (loads T1 model, then frees it)
    if run('visual_centroid') and task >= 2:
        section_visual_centroid(args, cfg, dataset_key, task, pipeline)

    # Phase 3: T_unk proximity (CPU only, reads checkpoints for embeddings)
    if run('tunk_proximity') and task >= 2:
        section_tunk_proximity(args, dataset_key, task)

    # Build T2 model only if needed by remaining phases
    needs_model = any(run(s) for s in ['forward', 'score_margin',
                                        'compactness', 'gradient',
                                        'unknown_fate'])
    if not needs_model:
        print(f"\n{'='*70}")
        print("DIAGNOSTIC COMPLETE")
        print(f"{'='*70}")
        return

    print(f"\n  Loading T2 model: {args.checkpoint}")
    model = build_model(cfg, args.checkpoint)

    # Phase 4: Original forward diagnostics
    if run('forward'):
        run_forward_diagnostics(model, cfg, dataset_key, task,
                                args.num_images)

    # Phase 5-8: Need hooks for GT-level analysis
    needs_hooks = any(run(s) for s in ['score_margin', 'compactness',
                                        'gradient', 'unknown_fate'])
    hooked, handles = None, []
    if needs_hooks:
        hooked, handles = setup_hooks(model.bbox_head.head_module)

    # Phase 5: Score margin at GT locations
    if run('score_margin') and task >= 2:
        section_score_margin(model, hooked, pipeline,
                             dataset_key, task, args.num_images)

    # Phase 6: Feature compactness
    if run('compactness'):
        section_feature_compactness(model, hooked, pipeline,
                                    dataset_key, task, args.num_images)

    # Phase 7: Gradient signal analysis
    if run('gradient') and task >= 2:
        section_gradient_signal(model, hooked, pipeline,
                                dataset_key, task, args.num_images)

    # Phase 8: Unknown fate analysis (always full test set)
    if run('unknown_fate') and task >= 2:
        section_unknown_fate(model, cfg, hooked, pipeline,
                             dataset_key, task)

    if handles:
        remove_hooks(handles)

    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
