#!/usr/bin/env python3
"""Probe GT anchors against cached novel post-BN visual prototypes.

This replaces the older norm probe. It tests the suppression hypothesis:

  If an anchor selected for a true unknown GT has high similarity to the
  K-shot novel visual cache, suppressing T_unk from that evidence would likely
  reduce unknown recall. If true unknown anchors are far from the novel cache,
  cache-gated T_unk suppression is less likely to hurt U-Recall.

For each GT in the test split:
  1. run the one2one head,
  2. pick a matched anchor by IoU plus optional TAL-style alignment,
  3. compare BN(anchor) with every per-level novel prototype separately,
  4. summarize a GT-class x novel-cache-class cosine matrix.
"""
import argparse
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--cache', required=True,
                   help='visual cache built by tools/owod_scripts/build_visual_cache.py')
    p.add_argument('--split', default='test')
    p.add_argument('--out-csv', required=True)
    p.add_argument('--out-matrix', default='',
                   help='Per GT-class x novel-class cosine summary CSV. '
                        'Default: <out-csv stem>_matrix.csv')
    p.add_argument('--out-json', default='')
    p.add_argument('--num-images', type=int, default=0, help='0 = all')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--match-iou', type=float, default=0.9)
    p.add_argument('--selection', default='tunk_tal',
                   choices=['tunk_tal', 'best_iou'],
                   help='How to choose one anchor per GT. tunk_tal uses the '
                        'correct known class for base/novel and T_unk for unknown.')
    p.add_argument('--tal-alpha', type=float, default=1.0)
    p.add_argument('--tal-beta', type=float, default=6.0)
    p.add_argument('--t1-tunk-checkpoint', default='',
                   help='Optional T1 checkpoint; reads its T_unk embedding for comparison.')
    p.add_argument('--t1-unk-index', type=int, default=8,
                   help='IDD T1 T_unk index. Used only with --t1-tunk-checkpoint.')
    p.add_argument('--top-sim-thresholds', default='0.3,0.4,0.5,0.6,0.7',
                   help='Comma-separated thresholds for risk counts.')
    return p.parse_args()


def parse_voc_xml(xml_path):
    out = []
    root = ET.parse(xml_path).getroot()
    for obj in root.findall('object'):
        bb = obj.find('bndbox')
        diff = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        out.append((obj.find('name').text.strip(), [
            float(bb.find('xmin').text) - 1.0,
            float(bb.find('ymin').text) - 1.0,
            float(bb.find('xmax').text) - 1.0,
            float(bb.find('ymax').text) - 1.0], diff))
    return out


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


def percentile_stats(values):
    a = np.asarray(values, dtype=np.float32)
    if a.size == 0:
        return {'N': 0}
    return {
        'N': int(a.size),
        'mean': float(a.mean()),
        'std': float(a.std()),
        'p10': float(np.percentile(a, 10)),
        'p50': float(np.percentile(a, 50)),
        'p90': float(np.percentile(a, 90)),
        'max': float(a.max()),
    }


def safe_col(name):
    return ''.join(ch if ch.isalnum() else '_' for ch in name).strip('_')


def load_cache_prototypes(cache_path, device):
    payload = torch.load(cache_path, map_location='cpu')
    cache_per_level = payload['cache_per_level']
    novel_names = list(payload.get('novel_class_names', []))
    n_novel = int(payload['n_novel'])
    embed_dim = int(payload['embed_dim'])
    num_levels = int(payload['num_levels'])

    prototypes = []
    masks = []
    for lvl in range(num_levels):
        proto = torch.zeros(n_novel, embed_dim, dtype=torch.float32)
        mask = torch.zeros(n_novel, dtype=torch.bool)
        for ci, feats in cache_per_level.get(lvl, {}).items():
            feats = feats.float()
            if feats.numel() == 0:
                continue
            mean = feats.mean(dim=0)
            proto[int(ci)] = F.normalize(mean, dim=0)
            mask[int(ci)] = True
        prototypes.append(proto.to(device))
        masks.append(mask.to(device))
    return {
        'path': cache_path,
        'novel_names': novel_names,
        'n_novel': n_novel,
        'embed_dim': embed_dim,
        'num_levels': num_levels,
        'prototypes': prototypes,
        'masks': masks,
        'matched': payload.get('matched_per_class_per_level', {}),
    }


def load_t1_tunk_embedding(path, unk_idx, device):
    if not path:
        return None
    ckpt = torch.load(path, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    if 'embeddings' not in sd:
        raise KeyError(f'{path} does not contain state_dict["embeddings"]')
    emb = sd['embeddings']
    if emb.dim() == 3 and emb.shape[0] == 1:
        emb = emb[0]
    if unk_idx < 0 or unk_idx >= emb.shape[0]:
        raise IndexError(f'T1 unk index {unk_idx} out of range for embeddings {tuple(emb.shape)}')
    return F.normalize(emb[unk_idx].float().to(device), dim=0)


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    for path in (repo_root, tools_dir):
        if path not in sys.path:
            sys.path.insert(0, path)

    from analyze_owod_test import build_model_ctx, build_pipeline
    from mmengine.registry import init_default_scope
    import mmyolo, yolo_world  # noqa: F401
    init_default_scope('mmyolo')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = build_model_ctx(args.config, args.checkpoint, device)
    model = ctx['model']
    head = ctx['head']
    hm = ctx['head_module']
    num_classes = ctx['num_classes']
    known_count = ctx['known_count']
    unk_idx = known_count
    dataset_name = ctx['dataset_name']

    from yolo_world.datasets import owodb_const as oc
    if dataset_name != 'IDD':
        print(f'[warn] designed for IDD; running with dataset={dataset_name}')
    base_set = set(oc.IDD_T1_CLASS_NAMES) if dataset_name == 'IDD' else ctx['base_set']
    novel_set = set(oc.IDD_T2_CLASS_NAMES) if dataset_name == 'IDD' else ctx['novel_set']
    class_to_idx = {name: i for i, name in enumerate(ctx['all_class_names'][:known_count])}

    def gt_role(name):
        if name in base_set:
            return 'base'
        if name in novel_set:
            return 'novel'
        return 'unknown'

    cache = load_cache_prototypes(args.cache, device)
    if cache['num_levels'] != hm.num_levels:
        raise RuntimeError(f'cache levels {cache["num_levels"]} != model levels {hm.num_levels}')
    if cache['embed_dim'] != hm.embed_dims:
        raise RuntimeError(f'cache dim {cache["embed_dim"]} != model dim {hm.embed_dims}')

    embeddings = model.embeddings.detach()
    if embeddings.dim() == 3 and embeddings.shape[0] == 1:
        embeddings = embeddings[0]
    current_tunk_proto = F.normalize(embeddings[unk_idx].float().to(device), dim=0)
    t1_tunk_proto = load_t1_tunk_embedding(args.t1_tunk_checkpoint, args.t1_unk_index, device)
    if t1_tunk_proto is None:
        t1_tunk_proto = current_tunk_proto
        t1_src = 'current checkpoint'
    else:
        t1_src = args.t1_tunk_checkpoint

    contrasts = hm.one2one_cls_contrasts
    scales = [cc.logit_scale.detach().exp().to(device) for cc in contrasts]
    biases = [cc.bias.detach().to(device) for cc in contrasts]

    bn_capture = [None] * hm.num_levels

    def make_hook(i):
        def hook(_m, _inp, out):
            bn_capture[i] = out
        return hook

    hooks = [contrasts[i].norm.register_forward_hook(make_hook(i))
             for i in range(hm.num_levels)]

    data_root = ctx['data_root']
    split_path = os.path.join(data_root, 'ImageSets', dataset_name, f'{args.split}.txt')
    with open(split_path) as f:
        ids = [line.strip() for line in f if line.strip()]
    n = len(ids) if args.num_images == 0 else min(args.num_images, len(ids))
    ids = ids[:n]

    img_dir = os.path.join(data_root, 'JPEGImages', dataset_name)
    ann_dir = os.path.join(data_root, 'Annotations', dataset_name)
    img_ext = '.jpg'
    for ext in ('.jpg', '.jpeg', '.png'):
        if ids and os.path.exists(os.path.join(img_dir, ids[0] + ext)):
            img_ext = ext
            break

    pipeline = build_pipeline(ctx)
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)

    rows = []
    stats = defaultdict(list)
    per_class = defaultdict(lambda: defaultdict(list))
    pair_cos = defaultdict(lambda: defaultdict(list))
    thresholds = [float(x) for x in args.top_sim_thresholds.split(',') if x.strip()]
    out_matrix = args.out_matrix or os.path.splitext(args.out_csv)[0] + '_matrix.csv'
    os.makedirs(os.path.dirname(out_matrix) or '.', exist_ok=True)
    novel_cos_cols = {
        name: f'cos_to_{safe_col(name)}'
        for name in cache['novel_names']
    }

    print('============================================================')
    print('GT vs NOVEL-CACHE EVIDENCE PROBE')
    print(f'  dataset/split: {dataset_name}/{args.split}  images={n}')
    print(f'  config:        {args.config}')
    print(f'  checkpoint:    {args.checkpoint}')
    print(f'  cache:         {args.cache}')
    print(f'  T2 unk idx:    {unk_idx}  T1 Tunk source: {t1_src}')
    print(f'  match_iou:     {args.match_iou}  selection={args.selection}')
    print(f'  roles:         {len(base_set)} base, {len(novel_set)} novel, other annotation names = unknown')
    print('============================================================')

    def load_one(img_id):
        img_path = os.path.join(img_dir, img_id + img_ext)
        ann_path = os.path.join(ann_dir, img_id + '.xml')
        if not os.path.exists(img_path) or not os.path.exists(ann_path):
            return None
        try:
            data = pipeline(dict(img_path=img_path, img_id=img_id, instances=[]))
        except Exception as exc:
            print(f'  [skip] {img_id}: {exc}')
            return None
        meta = data['data_samples'].metainfo
        return (img_id, data['inputs'].unsqueeze(0), data['data_samples'],
                np.asarray(meta['scale_factor']),
                np.asarray(meta.get('pad_param', np.zeros(4, dtype=np.float32))),
                ann_path)

    def flush(buf):
        if not buf:
            return
        with torch.no_grad():
            batch_inputs = torch.cat([x[1] for x in buf], dim=0).float().to(device) / 255.0
            samples = [x[2] for x in buf]
            img_feats, txt_feats = model.extract_feat(batch_inputs, samples)
            cls_list, bbox_list = hm.forward_one2one(img_feats, txt_feats)
            B = len(buf)

            featmap_sizes = [(t.shape[-2], t.shape[-1]) for t in cls_list]
            mlvl_priors = head.prior_generator.grid_priors(
                featmap_sizes, dtype=batch_inputs.dtype, device=device, with_stride=True)
            flat_priors = torch.cat(mlvl_priors, dim=0)
            flat_logits = torch.cat(
                [t.permute(0, 2, 3, 1).reshape(B, -1, num_classes) for t in cls_list], dim=1)
            flat_bbox = torch.cat(
                [t.permute(0, 2, 3, 1).reshape(B, -1, 4) for t in bbox_list], dim=1)

            level_ranges = []
            start = 0
            flat_bn_parts = []
            for lvl, bn in enumerate(bn_capture):
                count = bn.shape[-2] * bn.shape[-1]
                level_ranges.append((start, start + count))
                start += count
                flat_bn_parts.append(bn.permute(0, 2, 3, 1).reshape(B, count, bn.shape[1]))
            flat_bn = torch.cat(flat_bn_parts, dim=1)
            flat_bn_unit = F.normalize(flat_bn, dim=-1)
            flat_scores = flat_logits.sigmoid()

            for b, (img_id, _inp, _ds, scale, pad, ann_path) in enumerate(buf):
                gt_items = [g for g in parse_voc_xml(ann_path) if not g[2]]
                if not gt_items:
                    continue

                sx, sy = float(scale[0]), float(scale[1])
                pad_top, pad_left = float(pad[0]), float(pad[2])
                gt_pad = torch.tensor([g[1] for g in gt_items],
                                      dtype=torch.float32, device=device)
                gt_pad[:, [0, 2]] = gt_pad[:, [0, 2]] * sx + pad_left
                gt_pad[:, [1, 3]] = gt_pad[:, [1, 3]] * sy + pad_top

                decoded = head.bbox_coder.decode(
                    flat_priors[..., :2],
                    flat_bbox[b:b + 1],
                    flat_priors[:, [2]][..., 0])[0]
                ious = box_iou_xyxy(decoded, gt_pad)

                for gi, (gname, gbox, _diff) in enumerate(gt_items):
                    role = gt_role(gname)
                    iou_a = ious[:, gi]
                    valid = iou_a >= args.match_iou
                    if not valid.any():
                        continue
                    if args.selection == 'best_iou':
                        score_for_select = iou_a.clone()
                    else:
                        corr_idx_for_select = class_to_idx.get(gname, unk_idx)
                        corr_score = flat_scores[b, :, corr_idx_for_select].clamp(min=1e-6)
                        score_for_select = (corr_score ** args.tal_alpha) * (iou_a ** args.tal_beta)
                    score_for_select[~valid] = -1
                    anchor = int(score_for_select.argmax().item())
                    level = 0
                    for lvl, (lo, hi) in enumerate(level_ranges):
                        if lo <= anchor < hi:
                            level = lvl
                            break

                    bn = flat_bn[b, anchor]
                    bn_u = flat_bn_unit[b, anchor]
                    proto = cache['prototypes'][level]
                    mask = cache['masks'][level]
                    if not bool(mask.any()):
                        continue

                    cos = torch.matmul(proto[mask], bn_u)
                    dot = torch.matmul(proto[mask], bn)
                    novel_logits = dot * scales[level] + biases[level]
                    novel_scores = novel_logits.sigmoid()
                    valid_names = [cache['novel_names'][i] if i < len(cache['novel_names']) else f'novel_{i}'
                                   for i, ok in enumerate(mask.detach().cpu().tolist()) if ok]
                    cos_by_name = {
                        valid_names[j]: float(cos[j].item())
                        for j in range(len(valid_names))
                    }

                    best_i = int(cos.argmax().item())
                    best_score_i = int(novel_scores.argmax().item())
                    t2_tunk_logit = flat_logits[b, anchor, unk_idx]
                    t2_tunk_score = t2_tunk_logit.sigmoid()
                    corr_idx = class_to_idx.get(gname, unk_idx)
                    corr_logit = flat_logits[b, anchor, corr_idx]
                    corr_score = corr_logit.sigmoid()
                    t1_dot = torch.dot(bn, t1_tunk_proto)
                    t1_tunk_logit = t1_dot * scales[level] + biases[level]
                    t1_tunk_score = t1_tunk_logit.sigmoid()

                    novel_max_score = novel_scores.max()
                    ratio_vs_t1 = novel_max_score / t1_tunk_score.clamp(min=1e-6)
                    ratio_vs_t2 = novel_max_score / t2_tunk_score.clamp(min=1e-6)
                    box_w = float(gbox[2] - gbox[0])
                    box_h = float(gbox[3] - gbox[1])

                    row = {
                        'img_id': img_id,
                        'gt_class': gname,
                        'role': role,
                        'stride': int(round(float(flat_priors[anchor, 2].item()))),
                        'level': level,
                        'anchor_index': anchor,
                        'iou': float(iou_a[anchor].item()),
                        'bn_norm': float(bn.norm().item()),
                        'novel_mean_cos': float(cos.mean().item()),
                        'novel_max_cos': float(cos.max().item()),
                        'novel_best_cos_class': valid_names[best_i],
                        'novel_mean_score': float(novel_scores.mean().item()),
                        'novel_max_score': float(novel_max_score.item()),
                        'novel_best_score_class': valid_names[best_score_i],
                        't2_tunk_score': float(t2_tunk_score.item()),
                        'correct_score': float(corr_score.item()),
                        't1_tunk_score': float(t1_tunk_score.item()),
                        't2_tunk_logit': float(t2_tunk_logit.item()),
                        'correct_logit': float(corr_logit.item()),
                        't1_tunk_logit': float(t1_tunk_logit.item()),
                        'novel_over_t1_tunk': float(ratio_vs_t1.item()),
                        'novel_over_t2_tunk': float(ratio_vs_t2.item()),
                        'box_w': box_w,
                        'box_h': box_h,
                    }
                    for name, value in cos_by_name.items():
                        row[novel_cos_cols.get(name, f'cos_to_{safe_col(name)}')] = value
                    rows.append(row)
                    for name, value in cos_by_name.items():
                        pair_cos[(role, gname)][name].append(value)
                    for key in ('novel_mean_cos', 'novel_max_cos',
                                'novel_mean_score', 'novel_max_score',
                                't1_tunk_score', 't2_tunk_score',
                                'novel_over_t1_tunk', 'novel_over_t2_tunk',
                                'bn_norm', 'iou'):
                        stats[key].append(row[key])
                        per_class[gname][key].append(row[key])

    prefetch = max(1, args.num_workers * 2)
    batch = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = []
        for i in range(min(prefetch, len(ids))):
            futures.append(pool.submit(load_one, ids[i]))
        next_submit = len(futures)
        done = 0
        while futures:
            result = futures.pop(0).result()
            done += 1
            if next_submit < len(ids):
                futures.append(pool.submit(load_one, ids[next_submit]))
                next_submit += 1
            if result is not None:
                batch.append(result)
            if len(batch) >= args.batch_size:
                flush(batch)
                batch = []
            if done % 500 == 0:
                print(f'  processed {done}/{len(ids)} images, matched_gt={len(rows)}')
    flush(batch)
    for hook in hooks:
        hook.remove()

    fieldnames = [
        'img_id', 'gt_class', 'role', 'stride', 'level', 'anchor_index', 'iou', 'bn_norm',
        'novel_mean_cos', 'novel_max_cos', 'novel_best_cos_class',
        'novel_mean_score', 'novel_max_score', 'novel_best_score_class',
        'correct_score', 't2_tunk_score', 't1_tunk_score',
        'correct_logit', 't2_tunk_logit', 't1_tunk_logit',
        'novel_over_t1_tunk', 'novel_over_t2_tunk', 'box_w', 'box_h',
    ] + [novel_cos_cols[name] for name in cache['novel_names']]
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    matrix_rows = []
    for role, cls_name in sorted(pair_cos.keys()):
        for novel_class in cache['novel_names']:
            values = pair_cos[(role, cls_name)].get(novel_class, [])
            s = percentile_stats(values)
            row = {
                'gt_class': cls_name,
                'role': role,
                'novel_class': novel_class,
                'N': s.get('N', 0),
                'mean_cos': s.get('mean', ''),
                'std_cos': s.get('std', ''),
                'p10_cos': s.get('p10', ''),
                'p50_cos': s.get('p50', ''),
                'p90_cos': s.get('p90', ''),
                'max_cos': s.get('max', ''),
            }
            matrix_rows.append(row)
    with open(out_matrix, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['gt_class', 'role', 'novel_class', 'N', 'mean_cos',
                        'std_cos', 'p10_cos', 'p50_cos', 'p90_cos', 'max_cos'])
        writer.writeheader()
        writer.writerows(matrix_rows)

    summary = {
        'dataset': dataset_name,
        'split': args.split,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'cache': args.cache,
        't1_tunk_checkpoint': args.t1_tunk_checkpoint,
        't1_unk_index': args.t1_unk_index,
        'match_iou': args.match_iou,
        'selection': args.selection,
        'num_rows': len(rows),
        'overall': {k: percentile_stats(v) for k, v in stats.items()},
        'risk_counts': {},
        'per_class': {},
        'class_x_novel_cosine': {},
    }
    max_cos = np.asarray(stats['novel_max_cos'], dtype=np.float32)
    for th in thresholds:
        count = int((max_cos >= th).sum()) if max_cos.size else 0
        summary['risk_counts'][f'novel_max_cos_ge_{th:g}'] = {
            'count': count,
            'frac': float(count / max(1, max_cos.size)),
        }
    for cname, d in sorted(per_class.items()):
        summary['per_class'][cname] = {
            'role': gt_role(cname),
            'metrics': {
                key: percentile_stats(vals) for key, vals in d.items()
            },
        }
    for role, cls_name in sorted(pair_cos.keys()):
        summary['class_x_novel_cosine'][cls_name] = {
            'role': role,
            'novel_classes': {
                novel_class: percentile_stats(vals)
                for novel_class, vals in sorted(pair_cos[(role, cls_name)].items())
            },
        }

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(summary, f, indent=2)

    print('\n' + '=' * 80)
    print('GT vs NOVEL-CACHE SUMMARY')
    print('=' * 80)
    role_counts = defaultdict(int)
    for row in rows:
        role_counts[row['role']] += 1
    print(f'  matched GT anchors: {len(rows)}  '
          f'base={role_counts["base"]} novel={role_counts["novel"]} unknown={role_counts["unknown"]}')
    for key in ('novel_max_cos', 'novel_mean_cos', 'novel_max_score',
                't1_tunk_score', 't2_tunk_score',
                'novel_over_t1_tunk', 'novel_over_t2_tunk'):
        s = summary['overall'].get(key, {'N': 0})
        if s.get('N', 0):
            print(f'  {key:22s} N={s["N"]:5d} mean={s["mean"]:.4f} '
                  f'p50={s["p50"]:.4f} p90={s["p90"]:.4f} max={s["max"]:.4f}')
    for name, item in summary['risk_counts'].items():
        print(f'  {name:26s} {item["count"]:5d} ({item["frac"]:.2%})')
    if pair_cos:
        print('\n  Per GT class: mean cosine to each novel cache')
        header = 'role'.ljust(9) + 'gt_class'.ljust(24) + ''.join(
            n[:13].rjust(14) for n in cache['novel_names'])
        print('  ' + header)
        role_order = {'base': 0, 'novel': 1, 'unknown': 2}
        for role, cls_name in sorted(pair_cos.keys(), key=lambda x: (role_order.get(x[0], 9), x[1])):
            cells = []
            for novel_class in cache['novel_names']:
                vals = pair_cos[(role, cls_name)].get(novel_class, [])
                cells.append(f'{np.mean(vals):14.4f}' if vals else f'{"":14s}')
            print('  ' + role.ljust(9) + cls_name[:24].ljust(24) + ''.join(cells))
    print(f'[write] {args.out_csv}')
    print(f'[write] {out_matrix}')
    if args.out_json:
        print(f'[write] {args.out_json}')


if __name__ == '__main__':
    main()
