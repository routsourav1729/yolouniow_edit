#!/usr/bin/env python3
"""Anchor-level diagnostics for T_unk vs novel visual-cache calibration.

This does not change predictions. It measures, at the best-IoU anchor for each
GT object, whether a tiny T_unk margin shave could help novel classes and how
often the same condition appears on true unknown classes.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--num-images', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--iou-thr', type=float, default=0.5)
    p.add_argument('--taus', default='0.50,0.55,0.60,0.65')
    p.add_argument('--margins', default='0.0005,0.001,0.002,0.003,0.005')
    return p.parse_args()


def parse_float_list(s):
    return [float(x) for x in s.split(',') if x.strip()]


def box_iou_xyxy(a, b):
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    a_, b_ = a.unsqueeze(1), b.unsqueeze(0)
    lt = torch.maximum(a_[..., :2], b_[..., :2])
    rb = torch.minimum(a_[..., 2:], b_[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a_[..., 2] - a_[..., 0]) * (a_[..., 3] - a_[..., 1])
    area_b = (b_[..., 2] - b_[..., 0]) * (b_[..., 3] - b_[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)


def role(name, base_set, novel_set):
    if name in base_set:
        return 'base'
    if name in novel_set:
        return 'novel'
    return 'unknown'


def pct(n, d):
    return float(n) / float(d) * 100.0 if d else 0.0


def stats(vals):
    if not vals:
        return dict(n=0)
    a = np.asarray(vals, dtype=np.float64)
    out = dict(n=int(a.size), mean=float(a.mean()), p50=float(np.percentile(a, 50)))
    for q in (75, 90, 95):
        out[f'p{q}'] = float(np.percentile(a, q))
    return out


def load_cache(cache_path, device):
    payload = torch.load(cache_path, map_location='cpu')
    n_novel = int(payload['n_novel'])
    embed_dim = int(payload['embed_dim'])
    num_levels = int(payload['num_levels'])
    prototypes, masks = [], []
    for lvl in range(num_levels):
        proto = torch.zeros(n_novel, embed_dim)
        mask = torch.zeros(n_novel, dtype=torch.bool)
        for ci, feats in payload['cache_per_level'].get(lvl, {}).items():
            feats = feats.float()
            if feats.numel():
                proto[int(ci)] = F.normalize(feats.mean(0), dim=0)
                mask[int(ci)] = True
        prototypes.append(proto.to(device))
        masks.append(mask.to(device))
    return dict(prototypes=prototypes, masks=masks,
                novel_names=list(payload.get('novel_class_names', [])))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from analyze_owod_test import build_model_ctx, build_pipeline, parse_voc_xml_full
    from mmengine.registry import init_default_scope
    import mmyolo, yolo_world  # noqa: F401
    init_default_scope('mmyolo')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = build_model_ctx(args.config, args.checkpoint, device)
    model, head, hm = ctx['model'], ctx['head'], ctx['head_module']
    num_prev, known_count = ctx['num_prev'], ctx['known_count']
    base_set, novel_set = ctx['base_set'], ctx['novel_set']
    label_names = ctx['model_label_names']
    cache = load_cache(args.cache, device)

    bn_capture = [None] * hm.num_levels
    hooks = []
    for lvl, contrast in enumerate(hm.one2one_cls_contrasts):
        def make_hook(i):
            def hook(_m, _inp, out):
                bn_capture[i] = out
            return hook
        hooks.append(contrast.norm.register_forward_hook(make_hook(lvl)))

    data_root = ctx['data_root']
    dataset_name = ctx['dataset_name']
    split_path = os.path.join(data_root, 'ImageSets', dataset_name, f'{args.split}.txt')
    with open(split_path) as f:
        ids = [x.strip() for x in f if x.strip()]
    if args.num_images:
        ids = ids[:args.num_images]
    img_dir = os.path.join(data_root, 'JPEGImages', dataset_name)
    ann_dir = os.path.join(data_root, 'Annotations', dataset_name)
    img_ext = '.jpg'
    for ext in ('.jpg', '.jpeg', '.png'):
        if ids and os.path.exists(os.path.join(img_dir, ids[0] + ext)):
            img_ext = ext
            break
    pipeline = build_pipeline(ctx)
    taus = parse_float_list(args.taus)
    margins = parse_float_list(args.margins)

    per_class = defaultdict(lambda: defaultdict(float))
    samples = []

    def load_one(img_id):
        img_path = os.path.join(img_dir, img_id + img_ext)
        ann_path = os.path.join(ann_dir, img_id + '.xml')
        if not os.path.exists(img_path) or not os.path.exists(ann_path):
            return None
        data = pipeline(dict(img_path=img_path, img_id=img_id, instances=[]))
        meta = data['data_samples'].metainfo
        return (img_id, data['inputs'].unsqueeze(0), data['data_samples'],
                meta, np.asarray(meta['scale_factor']),
                np.asarray(meta.get('pad_param', np.zeros(4, dtype=np.float32))),
                ann_path)

    def flush(buf):
        if not buf:
            return
        with torch.no_grad():
            batch_inputs = torch.cat([x[1] for x in buf], 0).float().to(device) / 255.0
            data_samples = [x[2] for x in buf]
            img_feats, txt_feats = model.extract_feat(batch_inputs, data_samples)
            cls_list, bbox_list = hm.forward_one2one(img_feats, txt_feats)
            bsz = batch_inputs.shape[0]
            featmap_sizes = [(t.shape[-2], t.shape[-1]) for t in cls_list]
            priors = head.prior_generator.grid_priors(
                featmap_sizes, dtype=batch_inputs.dtype, device=device, with_stride=True)
            flat_priors = torch.cat(priors, 0)
            flat_logits = torch.cat([
                t.permute(0, 2, 3, 1).reshape(bsz, -1, hm.num_classes)
                for t in cls_list], 1)
            flat_bbox = torch.cat([
                t.permute(0, 2, 3, 1).reshape(bsz, -1, 4)
                for t in bbox_list], 1)
            level_ids = []
            flat_cache_cos = []
            for lvl, t in enumerate(cls_list):
                n = t.shape[-2] * t.shape[-1]
                level_ids.extend([lvl] * n)
                bn = F.normalize(bn_capture[lvl], dim=1)
                proto = cache['prototypes'][lvl]
                mask = cache['masks'][lvl]
                if bool(mask.any()):
                    cos = torch.einsum('bdhw,nd->bnhw', bn, proto[mask]).max(1).values
                else:
                    cos = torch.zeros((bsz, t.shape[-2], t.shape[-1]), device=device)
                flat_cache_cos.append(cos.reshape(bsz, -1))
            level_ids = torch.tensor(level_ids, device=device)
            flat_cache_cos = torch.cat(flat_cache_cos, 1)
            anchor_boxes = head.bbox_coder.decode(
                flat_priors[..., :2], flat_bbox, flat_priors[:, [2]][..., 0])

            for b, (img_id, _inp, _ds, _meta, scale, pad, ann_path) in enumerate(buf):
                gt = [g for g in parse_voc_xml_full(ann_path) if not g[2]]
                if not gt:
                    continue
                gt_boxes = torch.tensor([g[1] for g in gt], dtype=torch.float32, device=device)
                sx, sy = float(scale[0]), float(scale[1])
                pad_top, pad_left = float(pad[0]), float(pad[2])
                gt_pad = gt_boxes.clone()
                gt_pad[:, [0, 2]] = gt_pad[:, [0, 2]] * sx + pad_left
                gt_pad[:, [1, 3]] = gt_pad[:, [1, 3]] * sy + pad_top
                ious = box_iou_xyxy(anchor_boxes[b], gt_pad)
                best_iou, best_anchor = ious.max(0)
                scores = flat_logits[b].sigmoid()
                for gi, (name, _box, _diff) in enumerate(gt):
                    if best_iou[gi].item() < args.iou_thr:
                        continue
                    ai = best_anchor[gi]
                    r = role(name, base_set, novel_set)
                    sc = scores[ai]
                    best_known_val, best_known_idx = sc[:known_count].max(0)
                    best_novel_val, best_novel_rel = sc[num_prev:known_count].max(0)
                    tunk = sc[known_count]
                    margin = (tunk - best_novel_val).item()
                    cos = flat_cache_cos[b, ai].item()
                    winner_idx = int(sc[:-1].argmax().item())
                    winner = 'unknown' if winner_idx == known_count else label_names[winner_idx]
                    pc = per_class[(r, name)]
                    pc['n'] += 1
                    pc['tunk_sum'] += float(tunk)
                    pc['novel_sum'] += float(best_novel_val)
                    pc['margin_sum'] += margin
                    pc['cos_sum'] += cos
                    pc['tunk_wins'] += int(winner_idx == known_count)
                    pc['known_wins'] += int(winner_idx != known_count)
                    pc[f'pred_{winner}'] += 1
                    for tau in taus:
                        if cos >= tau:
                            pc[f'cos_ge_{tau:g}'] += 1
                            if winner_idx == known_count:
                                for m in margins:
                                    if 0.0 < margin <= m:
                                        pc[f'reclaimable_tau{tau:g}_m{m:g}'] += 1
                    samples.append(dict(role=r, cls=name, tunk=float(tunk),
                                        best_novel=float(best_novel_val),
                                        margin=margin, cache_cos=cos,
                                        winner=winner,
                                        best_known=float(best_known_val),
                                        best_iou=float(best_iou[gi]),
                                        level=int(level_ids[ai])))

    batch = []
    for i, img_id in enumerate(ids, 1):
        item = load_one(img_id)
        if item:
            batch.append(item)
        if len(batch) >= args.batch_size:
            flush(batch)
            batch = []
        if i % 500 == 0:
            print(f'  processed {i}/{len(ids)}')
    flush(batch)
    for h in hooks:
        h.remove()

    rows = []
    for (r, name), pc in sorted(per_class.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = int(pc['n'])
        row = dict(role=r, cls=name, n=n,
                   tunk_mean=pc['tunk_sum'] / n,
                   best_novel_mean=pc['novel_sum'] / n,
                   margin_mean=pc['margin_sum'] / n,
                   cache_cos_mean=pc['cos_sum'] / n,
                   tunk_wins=int(pc['tunk_wins']),
                   tunk_win_pct=pct(pc['tunk_wins'], n))
        for tau in taus:
            row[f'cos_ge_{tau:g}_pct'] = pct(pc[f'cos_ge_{tau:g}'], n)
            for m in margins:
                row[f'reclaimable_tau{tau:g}_m{m:g}'] = int(pc[f'reclaimable_tau{tau:g}_m{m:g}'])
                row[f'reclaimable_tau{tau:g}_m{m:g}_pct'] = pct(pc[f'reclaimable_tau{tau:g}_m{m:g}'], n)
        rows.append(row)

    csv_path = os.path.join(args.out_dir, 'tunk_margin_cache_diagnostic.csv')
    fields = sorted({k for row in rows for k in row})
    preferred = ['role', 'cls', 'n', 'tunk_mean', 'best_novel_mean',
                 'margin_mean', 'cache_cos_mean', 'tunk_wins', 'tunk_win_pct']
    fields = preferred + [f for f in fields if f not in preferred]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)

    summary = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'cache': args.cache,
        'num_samples': len(samples),
        'taus': taus,
        'margins': margins,
        'by_role': {},
    }
    for r in ('base', 'novel', 'unknown'):
        vals = [s for s in samples if s['role'] == r]
        summary['by_role'][r] = {
            'n': len(vals),
            'tunk': stats([s['tunk'] for s in vals]),
            'best_novel': stats([s['best_novel'] for s in vals]),
            'margin': stats([s['margin'] for s in vals]),
            'cache_cos': stats([s['cache_cos'] for s in vals]),
            'tunk_wins': sum(1 for s in vals if s['winner'] == 'unknown'),
        }
    json_path = os.path.join(args.out_dir, 'tunk_margin_cache_diagnostic.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('DIAGNOSTIC SUMMARY')
    for r, s in summary['by_role'].items():
        print(f"  {r:7s} N={s['n']:6d} TunkWin={s['tunk_wins']:5d} "
              f"margin_mean={s['margin'].get('mean', 0):+.5f} "
              f"cos_mean={s['cache_cos'].get('mean', 0):.4f}")
    print(f'[write] {csv_path}')
    print(f'[write] {json_path}')


if __name__ == '__main__':
    main()
