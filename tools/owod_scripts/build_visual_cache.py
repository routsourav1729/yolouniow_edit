"""Build a (n_novel, K, D) post-BN visual cache from K-shot GT boxes.

Loads the T2 model, runs each K-shot training image through the head, matches
each novel-class GT to its best anchor (IoU>=0.5) at any FPN level, and stores
that anchor's BN-normalized cls_embed (D=512) in a per-class cache.

The cache is consumed at inference by VisualCache (see
yolo_world/models/dense_heads/visual_cache.py) for logit-space ensembling.
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='T2 config (env vars set externally)')
    p.add_argument('--checkpoint', required=True, help='T2 best checkpoint .pth')
    p.add_argument('--out', required=True, help='output .pt path')
    p.add_argument('--iou-thresh', type=float, default=0.5)
    p.add_argument('--reduce-default', default='mean',
                   choices=['mean', 'topk_mean', 'max'],
                   help='metadata only — not used at build time')
    return p.parse_args()


def box_iou(a, b):
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    a_, b_ = a.unsqueeze(1), b.unsqueeze(0)
    lt = torch.maximum(a_[..., :2], b_[..., :2])
    rb = torch.minimum(a_[..., 2:], b_[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a_[..., 2] - a_[..., 0]) * (a_[..., 3] - a_[..., 1])
    area_b = (b_[..., 2] - b_[..., 0]) * (b_[..., 3] - b_[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_state_dict
    from mmyolo.registry import MODELS, DATASETS
    import mmyolo  # noqa
    import yolo_world  # noqa
    init_default_scope('mmyolo')

    print(f'[init] config:     {args.config}')
    print(f'[init] checkpoint: {args.checkpoint}')
    cfg = Config.fromfile(args.config)

    model = MODELS.build(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    state_dict = {k: v for k, v in state_dict.items() if 'text_model' not in k}
    if 'embeddings' in state_dict:
        ckpt_emb = state_dict['embeddings']
        if ckpt_emb.shape == model.embeddings.shape:
            with torch.no_grad():
                model.embeddings.data.copy_(ckpt_emb)
            state_dict = {k: v for k, v in state_dict.items() if k != 'embeddings'}
        else:
            state_dict['embeddings'] = model.update_embeddings(ckpt_emb)
    load_state_dict(model, state_dict, strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    head = model.bbox_head
    hm = head.head_module
    num_classes = hm.num_classes

    n_base = model.num_prev_classes
    n_novel = num_classes - n_base - 2
    embed_dim = hm.embed_dims
    print(f'[init] n_base={n_base}  n_novel={n_novel}  embed_dim={embed_dim}')

    # Build train (fewshot) dataset and val pipeline (eval-mode forward).
    train_ds = DATASETS.build(cfg.train_dataloader.dataset)
    inner = getattr(train_ds, 'dataset', train_ds)
    class_names = list(inner.CLASS_NAMES)
    novel_names = class_names[n_base:n_base + n_novel]
    print(f'[init] novel classes: {novel_names}')
    print(f'[init] {len(train_ds)} few-shot images')

    val_ds_cfg = cfg.val_dataloader.dataset
    raw_pipeline = (val_ds_cfg.pipeline
                    if hasattr(val_ds_cfg, 'pipeline')
                    else val_ds_cfg.dataset.pipeline)
    img_pipeline = [t for t in raw_pipeline
                    if 'LoadAnnotations' not in t.get('type', '')
                    and 'PackDetInputs' not in t.get('type', '')]
    img_pipeline.append(dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param')))
    pipeline = Compose(img_pipeline)

    cache_lists = {ci: [] for ci in range(n_novel)}  # per-novel-idx feature list

    with torch.no_grad():
        for i in range(len(train_ds)):
            info = train_ds.get_data_info(i)
            instances = info.get('instances', [])
            # Only novel-class GTs in this image (fewshot dataset already
            # filtered: only the class that selected this image is kept).
            novel_gts = [(inst['bbox_label'] - n_base, inst['bbox'])
                         for inst in instances
                         if n_base <= inst['bbox_label'] < n_base + n_novel]
            if not novel_gts:
                continue

            data = pipeline(dict(img_path=info['img_path'],
                                 img_id=info['img_id'],
                                 instances=[]))
            img = data['inputs'].unsqueeze(0).float().to(device) / 255.0
            ds = data['data_samples']
            meta = ds.metainfo
            scale = np.asarray(meta['scale_factor'])
            pad = np.asarray(meta.get('pad_param', np.zeros(4, dtype=np.float32)))
            sx, sy = float(scale[0]), float(scale[1])
            pad_top, pad_left = float(pad[0]), float(pad[2])

            img_feats, txt_feats = model.extract_feat(img, [ds])

            # Manually replicate forward_one2one to capture BN(cls_embed)
            # at every FPN level.
            assert len(img_feats) == hm.num_levels
            bn_list, bbox_list = [], []
            for lvl in range(hm.num_levels):
                feat = img_feats[lvl]
                cls_pred = hm.one2one_cls_preds[lvl]
                reg_pred = hm.one2one_reg_preds[lvl]
                cls_contrast = hm.one2one_cls_contrasts[lvl]
                cls_embed = cls_pred(feat)
                bn_embed = cls_contrast.norm(cls_embed)               # (1,D,Hl,Wl)
                bn_list.append(bn_embed)

                bbox_dist = reg_pred(feat)
                if hm.reg_max > 1:
                    b, _, h, w = feat.shape
                    bbox_dist = bbox_dist.reshape(
                        [-1, 4, hm.reg_max, h * w]).permute(0, 3, 1, 2)
                    bbox_pred = bbox_dist.softmax(3).matmul(
                        hm.one2one_proj.view([-1, 1])).squeeze(-1)
                    bbox_pred = bbox_pred.transpose(1, 2).reshape(b, -1, h, w)
                else:
                    bbox_pred = bbox_dist
                bbox_list.append(bbox_pred)

            featmap_sizes = [(t.shape[-2], t.shape[-1]) for t in bn_list]
            mlvl_priors = head.prior_generator.grid_priors(
                featmap_sizes, dtype=img.dtype, device=device, with_stride=True)
            flat_priors = torch.cat(mlvl_priors, dim=0)
            flat_bn = torch.cat([
                t.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
                for t in bn_list], dim=1)[0]                         # (N, D)
            flat_bbox = torch.cat([
                t.permute(0, 2, 3, 1).reshape(1, -1, 4)
                for t in bbox_list], dim=1)
            anchor_boxes = head.bbox_coder.decode(
                flat_priors[..., :2], flat_bbox,
                flat_priors[:, [2]][..., 0])[0]                      # (N, 4)

            # Build GT boxes in padded-image space.
            gt_boxes = torch.tensor([g[1] for g in novel_gts],
                                    dtype=torch.float32, device=device)
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * sx + pad_left
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * sy + pad_top
            gt_cls = [g[0] for g in novel_gts]

            ious = box_iou(gt_boxes, anchor_boxes)                   # (G, N)
            best_iou, best_idx = ious.max(dim=1)
            for gi in range(len(gt_cls)):
                if float(best_iou[gi]) < args.iou_thresh:
                    continue
                ci = gt_cls[gi]
                cache_lists[ci].append(flat_bn[best_idx[gi]].clone().cpu())

            if (i + 1) % 25 == 0 or (i + 1) == len(train_ds):
                print(f'  processed {i + 1}/{len(train_ds)}')

    # Stack per-class.
    cache_dict = {}
    matched = {}
    for ci in range(n_novel):
        feats = cache_lists[ci]
        matched[ci] = len(feats)
        if feats:
            cache_dict[ci] = torch.stack(feats, dim=0)               # (M_c, D)

    # Sanity report.
    print('\n[cache] per-class match counts:')
    for ci in range(n_novel):
        print(f'  [{ci:2d}] {novel_names[ci]:35s}  M={matched[ci]}')

    print('\n[cache] within-class mean cosine (sanity, want >0.4):')
    norm_per_cls = {}
    for ci, feats in cache_dict.items():
        n = feats.shape[0]
        if n < 2:
            norm_per_cls[ci] = float('nan')
            print(f'  [{ci:2d}] {novel_names[ci]:35s}  N={n}  (insufficient)')
            continue
        v = torch.nn.functional.normalize(feats, dim=-1)
        sims = [float((v[a] * v[b]).sum()) for a, b in combinations(range(n), 2)]
        m = float(np.mean(sims))
        norm_per_cls[ci] = m
        warn = '  <-- LOW' if m < 0.4 else ''
        print(f'  [{ci:2d}] {novel_names[ci]:35s}  N={n}  within={m:.3f}{warn}')

    print('\n[cache] cross-class mean cosine (pairs with within < cross flagged):')
    cls_means = {ci: torch.nn.functional.normalize(feats, dim=-1).mean(dim=0)
                 for ci, feats in cache_dict.items()}
    keys = sorted(cls_means.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            cs = float(torch.nn.functional.cosine_similarity(
                cls_means[a].unsqueeze(0), cls_means[b].unsqueeze(0)))
            wa, wb = norm_per_cls.get(a, 0), norm_per_cls.get(b, 0)
            warn = '  <-- CONFUSER' if cs > min(wa, wb) else ''
            print(f'  ({novel_names[a]:25s}, {novel_names[b]:25s})  '
                  f'cross={cs:.3f}{warn}')

    torch.save({
        'cache': cache_dict,
        'novel_class_names': novel_names,
        'n_base': n_base,
        'n_novel': n_novel,
        'embed_dim': embed_dim,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'reduce_default': args.reduce_default,
        'matched_per_class': matched,
    }, args.out)
    print(f'\n[write] {args.out}')


if __name__ == '__main__':
    main()
