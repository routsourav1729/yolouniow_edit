#!/usr/bin/env python3
"""Inference-only T_unk calibration using novel visual-cache evidence.

This is deliberately an experiment, not a model change. For anchors whose
current winner is T_unk, compute similarity to the per-level novel visual cache
and softly lower the T_unk logit:

    gate = sigmoid((best_cos - tau) / temp)
    excess = max(0, Tunk - max_novel_text_logit)
    Tunk' = max(logit(floor_score), Tunk - alpha * gate * excess)

The floor keeps unknown detections alive. The correction is dataset-agnostic:
all novel cache classes participate, and the amount is tied to how much T_unk
currently exceeds the best novel text logit.

For the few-shot calibration regime, the script also supports a score-space
mode:

    excess = max(0, Tunk_score - best_novel_score)
    Tunk_score' = max(floor_score,
                      Tunk_score - gate * min(score_delta, excess + margin_eps))

This keeps the correction in the same tiny probability range as the observed
T_unk scores, and only removes T_unk's small winning margin over the novel
prompts instead of globally penalizing objectness.
"""
import argparse
import copy
import csv
import json
import math
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
    p.add_argument('--conf-thr', type=float, default=0.05)
    p.add_argument('--pred-score-thr', type=float, default=-1.0,
                   help='Override head.test_cfg.score_thr before predict_by_feat. '
                        'Use 0.001 to study low-score unknown recall.')
    p.add_argument('--taus', default='0.50,0.55,0.60')
    p.add_argument('--alphas', default='0.10,0.20,0.30',
                   help='Fraction of the T_unk-vs-best-novel logit excess to remove.')
    p.add_argument('--score-deltas', default='',
                   help='If set, use score-space subtraction with these deltas, e.g. 0.003,0.005,0.008.')
    p.add_argument('--margin-eps', type=float, default=5e-4,
                   help='Tiny extra score margin used by score-space calibration '
                        'to let a close novel score overtake T_unk.')
    p.add_argument('--protect-base', action='store_true',
                   help='Never lower T_unk below the best base-class score. '
                        'This lets novel classes reclaim T_unk, but avoids handing '
                        'unknown objects to base classes.')
    p.add_argument('--temp', type=float, default=0.04)
    p.add_argument('--floor-score', type=float, default=0.05)
    return p.parse_args()


def parse_float_list(s):
    return [float(x) for x in s.split(',') if x.strip()]


def logit(p):
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


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


def class_role(cls, base_set, novel_set):
    if cls in base_set:
        return 'base'
    if cls in novel_set:
        return 'novel'
    return 'unknown'


def load_cache_prototypes(cache_path, device):
    payload = torch.load(cache_path, map_location='cpu')
    cache_per_level = payload['cache_per_level']
    n_novel = int(payload['n_novel'])
    embed_dim = int(payload['embed_dim'])
    num_levels = int(payload['num_levels'])
    novel_names = list(payload.get('novel_class_names', []))
    prototypes, masks = [], []
    for lvl in range(num_levels):
        proto = torch.zeros(n_novel, embed_dim, dtype=torch.float32)
        mask = torch.zeros(n_novel, dtype=torch.bool)
        for ci, feats in cache_per_level.get(lvl, {}).items():
            feats = feats.float()
            if feats.numel() == 0:
                continue
            proto[int(ci)] = F.normalize(feats.mean(dim=0), dim=0)
            mask[int(ci)] = True
        prototypes.append(proto.to(device))
        masks.append(mask.to(device))
    return dict(prototypes=prototypes, masks=masks, novel_names=novel_names,
                n_novel=n_novel, embed_dim=embed_dim, num_levels=num_levels)


def init_accum():
    return {
        'confusion': defaultdict(lambda: defaultdict(int)),
        'false_pos': defaultdict(int),
        'gt_counts': defaultdict(int),
        'u_hit': defaultdict(int),
        'u_total': defaultdict(int),
        'calib': defaultdict(float),
    }


def update_accum(accum, results, buf, ctx, conf_thr):
    base_set = ctx['base_set']
    novel_set = ctx['novel_set']
    model_label_names = ctx['model_label_names']
    unk_idx = ctx['known_count']
    from analyze_owod_test import parse_voc_xml_full

    for pred, (img_id, _inp, _ds, _meta, _scale, _pad, ann_path) in zip(results, buf):
        keep = pred.scores >= conf_thr
        pred_bboxes = pred.bboxes[keep]
        pred_scores = pred.scores[keep]
        pred_labels = pred.labels[keep]
        gt_full = parse_voc_xml_full(ann_path) if os.path.exists(ann_path) else []
        gt_nd = [g for g in gt_full if not g[2]]
        gt_names = [g[0] for g in gt_nd]
        for name in gt_names:
            accum['gt_counts'][name] += 1

        gt_boxes = (torch.tensor([g[1] for g in gt_nd],
                                 dtype=torch.float32,
                                 device=pred_bboxes.device)
                    if gt_nd else torch.zeros((0, 4), device=pred_bboxes.device))
        confusion = accum['confusion']
        if len(pred_bboxes) > 0 and gt_boxes.shape[0] > 0:
            ious = box_iou_xyxy(pred_bboxes, gt_boxes)
            order = pred_scores.argsort(descending=True)
            gt_matched = torch.zeros(len(gt_names), dtype=torch.bool, device=pred_bboxes.device)
            pred_matched = torch.zeros(len(pred_bboxes), dtype=torch.bool, device=pred_bboxes.device)
            labels_np = pred_labels.detach().cpu().numpy()
            for pi in order.tolist():
                row = ious[pi].clone()
                row[gt_matched] = 0
                best_iou, best_gi = row.max(0)
                if best_iou.item() >= 0.5:
                    lbl = int(labels_np[pi])
                    pred_cls = (model_label_names[lbl]
                                if lbl < len(model_label_names)
                                else f'cls_{lbl}')
                    confusion[gt_names[best_gi.item()]][pred_cls] += 1
                    gt_matched[best_gi] = True
                    pred_matched[pi] = True
            for gi, name in enumerate(gt_names):
                if not gt_matched[gi].item():
                    confusion[name]['missed'] += 1
            for pi in (~pred_matched).nonzero(as_tuple=True)[0].tolist():
                lbl = int(labels_np[pi])
                pred_cls = (model_label_names[lbl]
                            if lbl < len(model_label_names)
                            else f'cls_{lbl}')
                accum['false_pos'][pred_cls] += 1
        else:
            for name in gt_names:
                confusion[name]['missed'] += 1
            if len(pred_bboxes) > 0:
                labels_np = pred_labels.detach().cpu().numpy()
                for lbl in labels_np:
                    pred_cls = (model_label_names[int(lbl)]
                                if int(lbl) < len(model_label_names)
                                else f'cls_{int(lbl)}')
                    accum['false_pos'][pred_cls] += 1

        unk_pred_bboxes = pred.bboxes[pred.labels == unk_idx]
        all_gt_boxes = (torch.tensor([g[1] for g in gt_full],
                                     dtype=torch.float32,
                                     device=pred_bboxes.device)
                        if gt_full else torch.zeros((0, 4), device=pred_bboxes.device))
        for gi, (name, _box, diff) in enumerate(gt_full):
            if diff or class_role(name, base_set, novel_set) != 'unknown':
                continue
            accum['u_total'][name] += 1
            if unk_pred_bboxes.shape[0] == 0:
                continue
            if box_iou_xyxy(unk_pred_bboxes, all_gt_boxes[gi:gi + 1]).max().item() >= 0.5:
                accum['u_hit'][name] += 1


def summarize(accum, ctx):
    base_set = ctx['base_set']
    novel_set = ctx['novel_set']
    confusion = accum['confusion']
    gt_counts = accum['gt_counts']
    all_cls = sorted(gt_counts.keys(), key=lambda c: (
        0 if c in base_set else 1 if c in novel_set else 2, c))
    role_totals = defaultdict(int)
    role_correct = defaultdict(int)
    role_unknown = defaultdict(int)
    role_missed = defaultdict(int)
    unknown_as_novel = 0
    unknown_as_base = 0
    base_false_unknown = 0
    novel_false_unknown = 0
    for cls in all_cls:
        role = class_role(cls, base_set, novel_set)
        total = gt_counts[cls]
        role_totals[role] += total
        if role == 'unknown':
            role_correct[role] += confusion[cls].get('unknown', 0)
            unknown_as_novel += sum(confusion[cls].get(n, 0) for n in novel_set)
            unknown_as_base += sum(confusion[cls].get(b, 0) for b in base_set)
        else:
            role_correct[role] += confusion[cls].get(cls, 0)
        role_unknown[role] += confusion[cls].get('unknown', 0)
        role_missed[role] += confusion[cls].get('missed', 0)
        if role == 'base':
            base_false_unknown += confusion[cls].get('unknown', 0)
        if role == 'novel':
            novel_false_unknown += confusion[cls].get('unknown', 0)

    u_total = sum(accum['u_total'].values())
    u_hit = sum(accum['u_hit'].values())
    out = {
        'role_totals': dict(role_totals),
        'role_correct': dict(role_correct),
        'role_unknown_pred': dict(role_unknown),
        'role_missed': dict(role_missed),
        'base_false_unknown': int(base_false_unknown),
        'novel_false_unknown': int(novel_false_unknown),
        'unknown_as_novel': int(unknown_as_novel),
        'unknown_as_base': int(unknown_as_base),
        'u_recall_hit': int(u_hit),
        'u_recall_total': int(u_total),
        'u_recall_pct': float(u_hit / u_total * 100) if u_total else 0.0,
        'per_class_confusion': {k: dict(v) for k, v in confusion.items()},
    }
    for role in ('base', 'novel', 'unknown'):
        total = max(1, role_totals.get(role, 0))
        out[f'{role}_correct_pct'] = role_correct.get(role, 0) / total * 100
        out[f'{role}_unknown_pred_pct'] = role_unknown.get(role, 0) / total * 100
        out[f'{role}_missed_pct'] = role_missed.get(role, 0) / total * 100
    return out


def calibrate_cls_scores(cls_list, bn_capture, cache, num_prev, known_count,
                         params, floor_logit):
    tau = params['tau']
    temp = params['temp']
    mode = params['mode']
    out = []
    total_unknown_winner = 0
    total_calibrated = 0
    gate_sum = 0.0
    for lvl, logits in enumerate(cls_list):
        cal = logits.clone()
        bn = bn_capture[lvl]
        proto = cache['prototypes'][lvl]
        mask = cache['masks'][lvl]
        if not bool(mask.any()):
            out.append(cal)
            continue
        bn_u = F.normalize(bn, dim=1)
        eligible = mask.clone()
        if not bool(eligible.any()):
            out.append(cal)
            continue
        cos = torch.einsum('bdhw,nd->bnhw', bn_u, proto[eligible])
        best = cos.max(dim=1).values

        scores = logits.sigmoid()
        scores_no_anchor = scores.clone()
        scores_no_anchor[:, -1] = -1
        winner = scores_no_anchor.argmax(dim=1)
        unk_winner = winner == known_count
        gate = torch.sigmoid((best - tau) / temp)
        old = cal[:, known_count]

        if mode == 'score_delta':
            score_delta = params['score_delta']
            tunk_score = scores[:, known_count]
            best_novel_score = scores[:, num_prev:known_count].max(dim=1).values
            excess = (tunk_score - best_novel_score).clamp(min=0)
            floor_score = tunk_score.new_tensor(params['floor_score'])
            if params.get('protect_base', False) and num_prev > 0:
                best_base_score = scores[:, :num_prev].max(dim=1).values
                floor_score = torch.maximum(floor_score, best_base_score)
            max_delta = tunk_score.new_tensor(score_delta)
            margin_delta = torch.minimum(max_delta, excess + params['margin_eps'])
            new_score = torch.maximum(
                tunk_score - margin_delta * gate,
                floor_score)
            new = torch.logit(new_score.clamp(1e-6, 1.0 - 1e-6))
            apply = unk_winner & (score_delta > 0) & (excess > 0)
        else:
            alpha = params['alpha']
            tunk_logit = logits[:, known_count]
            best_novel_text_logit = logits[:, num_prev:known_count].max(dim=1).values
            excess = (tunk_logit - best_novel_text_logit).clamp(min=0)
            delta = alpha * gate * excess
            new = torch.maximum(old - delta, old.new_tensor(floor_logit))
            apply = unk_winner & (alpha > 0) & (excess > 0)

        cal[:, known_count] = torch.where(apply, new, old)
        out.append(cal)

        total_unknown_winner += int(unk_winner.sum().item())
        total_calibrated += int((apply & (gate > 0.05)).sum().item())
        gate_sum += float(gate[apply].sum().item()) if apply.any() else 0.0
    return out, {
        'unknown_winner_anchors': total_unknown_winner,
        'calibrated_gate_gt_0.05': total_calibrated,
        'gate_sum': gate_sum,
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from analyze_owod_test import (build_model_ctx, build_pipeline,
                                   parse_voc_xml_full)
    from mmengine.registry import init_default_scope
    import mmyolo, yolo_world  # noqa: F401
    init_default_scope('mmyolo')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = build_model_ctx(args.config, args.checkpoint, device)
    model = ctx['model']
    head = ctx['head']
    hm = ctx['head_module']
    num_prev = ctx['num_prev']
    known_count = ctx['known_count']
    cache = load_cache_prototypes(args.cache, device)
    if cache['num_levels'] != hm.num_levels:
        raise RuntimeError('cache/model level mismatch')
    if args.pred_score_thr >= 0 and hasattr(head, 'test_cfg'):
        old_thr = head.test_cfg.get('score_thr', None)
        head.test_cfg.score_thr = args.pred_score_thr
        print(f'  [predict] head.test_cfg.score_thr: {old_thr} -> {args.pred_score_thr}')

    bn_capture = [None] * hm.num_levels
    hooks = []
    for lvl, contrast in enumerate(hm.one2one_cls_contrasts):
        def make_hook(i):
            def hook(_m, _inp, out):
                bn_capture[i] = out
            return hook
        hooks.append(contrast.norm.register_forward_hook(make_hook(lvl)))

    variants = [('baseline', None)]
    score_deltas = parse_float_list(args.score_deltas)
    if score_deltas:
        for tau in parse_float_list(args.taus):
            for score_delta in score_deltas:
                label = f'tau{tau:g}_sd{score_delta:g}'
                variants.append((label, {
                    'mode': 'score_delta',
                    'tau': tau,
                    'score_delta': score_delta,
                    'margin_eps': args.margin_eps,
                    'temp': args.temp,
                    'floor_score': args.floor_score,
                    'protect_base': args.protect_base,
                }))
    else:
        for tau in parse_float_list(args.taus):
            for alpha in parse_float_list(args.alphas):
                label = f'tau{tau:g}_a{alpha:g}'
                variants.append((label, {
                    'mode': 'excess_frac',
                    'tau': tau,
                    'alpha': alpha,
                    'margin_eps': args.margin_eps,
                    'temp': args.temp,
                    'floor_score': args.floor_score,
                    'protect_base': args.protect_base,
                }))
    accums = {label: init_accum() for label, _ in variants}
    variant_meta = {
        label: ({'baseline': True} if params is None else dict(params))
        for label, params in variants
    }

    data_root = ctx['data_root']
    dataset_name = ctx['dataset_name']
    split_path = os.path.join(data_root, 'ImageSets', dataset_name,
                              f'{args.split}.txt')
    with open(split_path) as f:
        ids = [line.strip() for line in f if line.strip()]
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
    floor_logit = logit(args.floor_score)

    print('============================================================')
    print('TUNK VISUAL-CACHE CALIBRATION SWEEP')
    print(f'  dataset/split: {dataset_name}/{args.split} images={len(ids)}')
    print(f'  checkpoint:    {args.checkpoint}')
    print(f'  cache:         {args.cache}')
    print(f'  variants:      {len(variants)} including baseline')
    print(f'  floor_score:   {args.floor_score}')
    print(f'  pred_score_thr:{args.pred_score_thr}')
    print(f'  margin_eps:    {args.margin_eps}')
    print(f'  protect_base:  {args.protect_base}')
    print(f'  score_deltas:  {score_deltas if score_deltas else None}')
    print(f'  novel cache:   {cache["novel_names"]}')
    print('============================================================')

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
            batch_inputs = torch.cat([x[1] for x in buf], dim=0).float().to(device) / 255.0
            samples = [x[2] for x in buf]
            metas = [x[3] for x in buf]
            img_feats, txt_feats = model.extract_feat(batch_inputs, samples)
            cls_list, bbox_list = hm.forward_one2one(img_feats, txt_feats)
            for label, params in variants:
                if params is None:
                    use_cls = cls_list
                    calib_stats = {}
                else:
                    use_cls, calib_stats = calibrate_cls_scores(
                        cls_list, bn_capture, cache, num_prev, known_count,
                        params, floor_logit)
                    for k, v in calib_stats.items():
                        accums[label]['calib'][k] += v
                results = head.predict_by_feat(
                    use_cls, bbox_list, batch_img_metas=metas,
                    rescale=True, with_nms=True)
                update_accum(accums[label], results, buf, ctx, args.conf_thr)

    batch = []
    for i, img_id in enumerate(ids, 1):
        item = load_one(img_id)
        if item is not None:
            batch.append(item)
        if len(batch) >= args.batch_size:
            flush(batch)
            batch = []
        if i % 500 == 0:
            print(f'  processed {i}/{len(ids)}')
    flush(batch)
    for hook in hooks:
        hook.remove()

    summaries = {}
    rows = []
    baseline = None
    for label, _params in variants:
        s = summarize(accums[label], ctx)
        s['calibration_stats'] = dict(accums[label]['calib'])
        s['params'] = variant_meta[label]
        summaries[label] = s
        if label == 'baseline':
            baseline = s
    for label, _params in variants:
        s = summaries[label]
        row = {
            'label': label,
            **variant_meta[label],
            'u_recall_pct': s['u_recall_pct'],
            'u_recall_hit': s['u_recall_hit'],
            'u_recall_total': s['u_recall_total'],
            'novel_correct_pct': s.get('novel_correct_pct', 0.0),
            'unknown_correct_pct': s.get('unknown_correct_pct', 0.0),
            'base_correct_pct': s.get('base_correct_pct', 0.0),
            'base_false_unknown': s['base_false_unknown'],
            'novel_false_unknown': s['novel_false_unknown'],
            'unknown_as_novel': s['unknown_as_novel'],
            'unknown_as_base': s['unknown_as_base'],
            'base_unknown_pred_pct': s.get('base_unknown_pred_pct', 0.0),
            'novel_unknown_pred_pct': s.get('novel_unknown_pred_pct', 0.0),
            'unknown_unknown_pred_pct': s.get('unknown_unknown_pred_pct', 0.0),
            'unknown_winner_anchors': s['calibration_stats'].get('unknown_winner_anchors', 0),
            'calibrated_gate_gt_0.05': s['calibration_stats'].get('calibrated_gate_gt_0.05', 0),
        }
        if baseline is not None and label != 'baseline':
            row['delta_u_recall_pct'] = row['u_recall_pct'] - baseline['u_recall_pct']
            row['delta_novel_correct_pct'] = row['novel_correct_pct'] - baseline.get('novel_correct_pct', 0.0)
            row['delta_base_false_unknown'] = row['base_false_unknown'] - baseline['base_false_unknown']
            row['delta_novel_false_unknown'] = row['novel_false_unknown'] - baseline['novel_false_unknown']
            row['delta_unknown_as_novel'] = row['unknown_as_novel'] - baseline['unknown_as_novel']
        rows.append(row)

    summary_csv = os.path.join(args.out_dir, 'calibration_sweep.csv')
    fieldnames = sorted({k for row in rows for k in row.keys()})
    preferred = ['label', 'baseline', 'mode', 'tau', 'alpha', 'score_delta',
                 'margin_eps', 'protect_base',
                 'u_recall_pct', 'delta_u_recall_pct',
                 'novel_correct_pct', 'delta_novel_correct_pct',
                 'base_false_unknown', 'delta_base_false_unknown',
                 'novel_false_unknown', 'delta_novel_false_unknown',
                 'unknown_as_novel', 'delta_unknown_as_novel',
                 'base_unknown_pred_pct', 'novel_unknown_pred_pct',
                 'unknown_unknown_pred_pct',
                 'unknown_winner_anchors', 'calibrated_gate_gt_0.05']
    ordered = preferred + [f for f in fieldnames if f not in preferred]
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    summary_json = os.path.join(args.out_dir, 'calibration_sweep.json')
    with open(summary_json, 'w') as f:
        json.dump({
            'config': args.config,
            'checkpoint': args.checkpoint,
            'cache': args.cache,
            'conf_thr': args.conf_thr,
            'floor_score': args.floor_score,
            'pred_score_thr': args.pred_score_thr,
            'protect_base': args.protect_base,
            'summaries': summaries,
        }, f, indent=2)

    print('\nTOP SWEEP ROWS (prefer small U-recall drop, then improve novel correct):')
    sortable = [r for r in rows if r['label'] != 'baseline']
    sortable.sort(key=lambda r: (
        abs(min(0.0, r.get('delta_u_recall_pct', 0.0))),
        -(r.get('delta_novel_correct_pct', 0.0)),
        r.get('unknown_as_novel', 0)))
    for r in sortable[:12]:
        print(f"  {r['label']:18s} "
              f"dNovel={r.get('delta_novel_correct_pct', 0):+.3f} "
              f"dU={r.get('delta_u_recall_pct', 0):+.3f} "
              f"dBaseUnk={r.get('delta_base_false_unknown', 0):+d} "
              f"dNovUnk={r.get('delta_novel_false_unknown', 0):+d} "
              f"dUnkAsNovel={r.get('delta_unknown_as_novel', 0):+d}")
    print(f'[write] {summary_csv}')
    print(f'[write] {summary_json}')


if __name__ == '__main__':
    main()
