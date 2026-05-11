#!/usr/bin/env python3
"""Analyze K-shot post-BN visual-cache geometry.

Reads a cache built by tools/owod_scripts/build_visual_cache.py and reports:
  - within-class support compactness
  - cross-class prototype similarity
  - leave-one-out nearest-prototype accuracy/margins
  - support-level nearest-class confusion
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def stats(vals):
    if not vals:
        return dict(n=0)
    a = np.asarray(vals, dtype=np.float64)
    out = dict(n=int(a.size), mean=float(a.mean()), std=float(a.std()),
               min=float(a.min()), p50=float(np.percentile(a, 50)),
               max=float(a.max()))
    for q in (10, 25, 75, 90):
        out[f'p{q}'] = float(np.percentile(a, q))
    return out


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    payload = torch.load(args.cache, map_location='cpu')
    cache = payload['cache_per_level']
    names = list(payload.get('novel_class_names', []))
    n_novel = int(payload['n_novel'])
    num_levels = int(payload['num_levels'])

    per_class_rows = []
    cross_rows = []
    loo_rows = []
    support_rows = []
    summary = {
        'cache': args.cache,
        'novel_class_names': names,
        'num_levels': num_levels,
        'per_level': {},
    }

    for lvl in range(num_levels):
        feats_by_cls = {}
        protos = {}
        for ci in range(n_novel):
            feats = cache.get(lvl, {}).get(ci)
            if feats is None or feats.numel() == 0:
                continue
            feats = F.normalize(feats.float(), dim=-1)
            feats_by_cls[ci] = feats
            protos[ci] = F.normalize(feats.mean(0), dim=0)

        lvl_summary = {}
        all_support_correct = 0
        all_support_total = 0
        all_margins = []

        for ci, feats in feats_by_cls.items():
            name = names[ci] if ci < len(names) else f'novel_{ci}'
            within = []
            if feats.shape[0] >= 2:
                for a, b in combinations(range(feats.shape[0]), 2):
                    within.append(float((feats[a] * feats[b]).sum()))
            own_proto = protos[ci]
            own_to_proto = [float((f * own_proto).sum()) for f in feats]
            other_proto_scores = []
            for cj, proto in protos.items():
                if cj == ci:
                    continue
                other_proto_scores.extend(float((f * proto).sum()) for f in feats)
            row = {
                'level': lvl,
                'class_idx': ci,
                'class': name,
                'n_support': int(feats.shape[0]),
                'within_mean': stats(within).get('mean', np.nan),
                'within_p10': stats(within).get('p10', np.nan),
                'own_proto_mean': stats(own_to_proto).get('mean', np.nan),
                'best_other_proto_mean': np.nan,
                'own_minus_best_other_mean': np.nan,
            }
            best_other = []
            for f in feats:
                scores = [(cj, float((f * proto).sum()))
                          for cj, proto in protos.items() if cj != ci]
                if scores:
                    best_other.append(max(s for _cj, s in scores))
            if best_other:
                row['best_other_proto_mean'] = float(np.mean(best_other))
                row['own_minus_best_other_mean'] = float(
                    np.mean(np.asarray(own_to_proto) - np.asarray(best_other)))
            per_class_rows.append(row)
            lvl_summary[name] = row

        for a, b in combinations(sorted(protos.keys()), 2):
            an = names[a] if a < len(names) else f'novel_{a}'
            bn = names[b] if b < len(names) else f'novel_{b}'
            sim = float((protos[a] * protos[b]).sum())
            cross_rows.append({
                'level': lvl,
                'class_a': an,
                'class_b': bn,
                'proto_cos': sim,
            })

        # Leave-one-out nearest prototype. For each support, build its own
        # class prototype from the remaining supports, then classify by max cos.
        confusion = defaultdict(lambda: defaultdict(int))
        for ci, feats in feats_by_cls.items():
            true_name = names[ci] if ci < len(names) else f'novel_{ci}'
            for k, f in enumerate(feats):
                proto_items = []
                for cj, other_feats in feats_by_cls.items():
                    if cj == ci:
                        if other_feats.shape[0] < 2:
                            continue
                        train_feats = torch.cat([other_feats[:k], other_feats[k + 1:]], 0)
                    else:
                        train_feats = other_feats
                    proto = F.normalize(train_feats.mean(0), dim=0)
                    proto_items.append((cj, proto))
                scores = [(cj, float((f * proto).sum())) for cj, proto in proto_items]
                if not scores:
                    continue
                scores.sort(key=lambda x: x[1], reverse=True)
                pred_ci, top = scores[0]
                second = scores[1][1] if len(scores) > 1 else -1.0
                pred_name = names[pred_ci] if pred_ci < len(names) else f'novel_{pred_ci}'
                correct = pred_ci == ci
                all_support_total += 1
                all_support_correct += int(correct)
                all_margins.append(top - second)
                confusion[true_name][pred_name] += 1
                support_rows.append({
                    'level': lvl,
                    'true_class': true_name,
                    'pred_class': pred_name,
                    'correct': int(correct),
                    'top_cos': top,
                    'second_cos': second,
                    'margin': top - second,
                })

        for true_name, preds in confusion.items():
            total = sum(preds.values())
            correct = preds.get(true_name, 0)
            row = {
                'level': lvl,
                'class': true_name,
                'loo_total': total,
                'loo_correct': correct,
                'loo_acc': correct / total if total else 0.0,
            }
            for pred_name, count in sorted(preds.items(), key=lambda kv: -kv[1])[:4]:
                row[f'pred_{pred_name}'] = count
            loo_rows.append(row)

        summary['per_level'][lvl] = {
            'n_classes': len(feats_by_cls),
            'n_supports': int(sum(f.shape[0] for f in feats_by_cls.values())),
            'loo_acc': all_support_correct / all_support_total if all_support_total else 0.0,
            'loo_margin': stats(all_margins),
            'classes': lvl_summary,
        }

    def write_csv(name, rows):
        path = os.path.join(args.out_dir, name)
        fields = sorted({k for r in rows for k in r.keys()})
        preferred = ['level', 'class', 'class_idx', 'class_a', 'class_b',
                     'n_support', 'within_mean', 'own_proto_mean',
                     'best_other_proto_mean', 'own_minus_best_other_mean',
                     'proto_cos', 'loo_total', 'loo_correct', 'loo_acc',
                     'true_class', 'pred_class', 'correct', 'margin']
        fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerows(rows)
        print(f'[write] {path}')

    write_csv('per_class_compactness.csv', per_class_rows)
    write_csv('cross_class_prototypes.csv', cross_rows)
    write_csv('loo_confusion.csv', loo_rows)
    write_csv('support_predictions.csv', support_rows)
    json_path = os.path.join(args.out_dir, 'summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[write] {json_path}')

    print('\nVISUAL CACHE SPACE SUMMARY')
    for lvl, s in summary['per_level'].items():
        print(f"  L{lvl}: supports={s['n_supports']} classes={s['n_classes']} "
              f"LOO_acc={s['loo_acc'] * 100:.2f}% "
              f"margin_mean={s['loo_margin'].get('mean', 0):.4f}")
    print('\nMost similar class prototypes:')
    for r in sorted(cross_rows, key=lambda x: -x['proto_cos'])[:12]:
        print(f"  L{r['level']} {r['class_a']:15s} ~ {r['class_b']:15s} cos={r['proto_cos']:.4f}")


if __name__ == '__main__':
    main()
