#!/usr/bin/env python3
"""Probe per-anchor BN(F_v) norms grouped by class role.

Tests the hypothesis: in the score
    s_c(v) = exp(scale) * <BN(F_v), L2(t_c)> + b
           = exp(scale) * ||BN(F_v)|| * cos(BN(F_v), t_c) + b
the radial term ||BN(F_v)|| is class-frequency-dependent and suppresses
novel/unknown scores even when angular alignment is fine.

Per matched GT (TAL alignment, IoU>=match_iou), records:
  bn_norm   = ||BN(F_v)|| at the matched anchor (channel L2-norm)
  cos_corr  = cos(BN(F_v), t_correct)         (Tunk for unknowns)
  cos_unk   = cos(BN(F_v), t_unk)
  raw_corr  = pre-sigmoid logit for correct channel
  raw_unk   = pre-sigmoid logit for Tunk
  stride    = 8 / 16 / 32
  iou       = IoU of matched anchor with GT

Output: CSV (one row per matched GT) + JSON summary.
"""
import argparse
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', required=True, help='test | t2_train_10shot | t1_train | t2_train | ...')
    p.add_argument('--out-csv', required=True)
    p.add_argument('--out-json', default='')
    p.add_argument('--num-images', type=int, default=0, help='0 = all')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=8,
                   help='Prefetch threads for image loading')
    p.add_argument('--match-iou', type=float, default=0.9)
    p.add_argument('--tal-alpha', type=float, default=1.0)
    p.add_argument('--tal-beta',  type=float, default=6.0)
    p.add_argument('--vis', action='store_true',
                   help='Visualization-only mode: collect matched BN vectors and render a '
                        'UMAP projection (Euclidean, raw features). Skips norm CSV/JSON output.')
    p.add_argument('--vis-out', default='',
                   help='Output prefix for --vis (writes <prefix>.npz and <prefix>.png). '
                        'Default: derived from --out-csv.')
    p.add_argument('--vis-max-per-class', type=int, default=2000,
                   help='Cap matched anchors per class before UMAP (memory).')
    p.add_argument('--vis-n-neighbors', type=int, default=30)
    p.add_argument('--vis-min-dist', type=float, default=0.1)
    return p.parse_args()


def parse_voc_xml(xml_path):
    out = []
    for obj in ET.parse(xml_path).getroot().findall('object'):
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


def class_role(cls, base_set, novel_set):
    if cls in base_set: return 'base'
    if cls in novel_set: return 'novel'
    return 'unknown'


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    for p in (repo_root, tools_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Reuse analyze_owod_test for model + pipeline construction
    from analyze_owod_test import build_model_ctx, build_pipeline
    from mmengine.registry import init_default_scope
    import mmyolo, yolo_world  # noqa: F401
    init_default_scope('mmyolo')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = build_model_ctx(args.config, args.checkpoint, device)

    head        = ctx['head']
    head_module = ctx['head_module']
    all_class_names = ctx['all_class_names']
    known_count = ctx['known_count']
    num_classes = ctx['num_classes']     # known + Tunk + Tobj
    unk_idx     = known_count            # Tunk channel index

    # Override base/novel from dataset constants (TASK-independent).
    # IDD T1 has all classes annotated in training data (base+novel+unknown);
    # nuOWODB likewise. Use the per-dataset T1/T2/T3 class lists directly.
    dataset_name = ctx['dataset_name']
    from yolo_world.datasets import owodb_const as _oc
    if dataset_name == 'IDD':
        base_set  = set(_oc.IDD_T1_CLASS_NAMES)
        novel_set = set(_oc.IDD_T2_CLASS_NAMES)
    elif dataset_name == 'nuOWODB':
        base_set  = set(_oc.T1_CLASS_NAMES)
        novel_set = set(_oc.T2_CLASS_NAMES)
    else:
        base_set  = ctx['base_set']
        novel_set = ctx['novel_set']
    print(f'[probe] role-sets  base={len(base_set)}  novel={len(novel_set)}  '
          f'(unknown = anything else in annotations)')

    # ── Hook BN of each cls_contrast (one2one path) ─────────────────────────
    bn_capture = [None] * head_module.num_levels
    def make_hook(i):
        def h(_m, _inp, out):
            bn_capture[i] = out          # (B, C, H, W)
        return h
    contrasts = head_module.one2one_cls_contrasts
    hooks = [contrasts[i].norm.register_forward_hook(make_hook(i))
             for i in range(head_module.num_levels)]

    # Class prototypes (L2-normalized) — extracted directly from model.embeddings
    # Shape: (num_classes, D). embeddings used by the head are the same.
    proto = F.normalize(ctx['model'].embeddings.detach().to(device), dim=-1)
    proto_known = proto[:known_count]    # (K, D)
    proto_unk   = proto[unk_idx:unk_idx+1]  # (1, D)

    # Logit scales / biases per stride (BNContrastiveHead)
    # logit = exp(scale) * <BN(x), L2(w)> + bias
    scales = torch.stack([cc.logit_scale.detach().exp() for cc in contrasts])
    biases = torch.stack([cc.bias.detach() for cc in contrasts])

    # ── Resolve dataset paths ────────────────────────────────────────────────
    data_root    = ctx['data_root']
    split_path = os.path.join(data_root, 'ImageSets', dataset_name, f'{args.split}.txt')
    with open(split_path) as f:
        ids = [l.strip() for l in f if l.strip()]
    img_dir = os.path.join(data_root, 'JPEGImages', dataset_name)
    ann_dir = os.path.join(data_root, 'Annotations', dataset_name)

    img_ext = '.jpg'
    for ext in ('.jpg', '.jpeg', '.png'):
        if os.path.exists(os.path.join(img_dir, ids[0] + ext)):
            img_ext = ext
            break

    pipeline = build_pipeline(ctx)
    n = len(ids) if args.num_images == 0 else min(args.num_images, len(ids))
    print(f'[probe] dataset={dataset_name} split={args.split} N={n} match_iou={args.match_iou}')
    print(f'[probe] known={known_count} ({len(base_set)} base + {len(novel_set)} novel)  Tunk_idx={unk_idx}')

    # ── CSV writer (norm mode only) ──────────────────────────────────────────
    if not args.vis:
        os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
        fcsv = open(args.out_csv, 'w', newline='')
        w = csv.writer(fcsv)
        w.writerow(['img_id', 'gt_class', 'role', 'stride', 'iou',
                    'bn_norm', 'cos_corr', 'cos_unk',
                    'raw_corr', 'raw_unk', 'box_w', 'box_h'])
    else:
        fcsv = None
        w = None

    per_cls_norms = defaultdict(list)
    per_cls_role  = {}
    # vis-mode collectors: full BN vector per matched anchor
    vis_vecs = []        # list of np.ndarray (C,)
    vis_labels = []      # list of class name
    vis_roles = []       # list of role

    batch_buf = []
    def flush(buf):
        if not buf: return
        with torch.no_grad():
            B = len(buf)
            batch_inputs = torch.cat([e[1] for e in buf], dim=0).float().to(device) / 255.0
            data_samples = [e[2] for e in buf]
            img_feats, txt_feats = ctx['model'].extract_feat(batch_inputs, data_samples)
            cls_list, bbox_list = head_module.forward_one2one(img_feats, txt_feats)

            # Build flat priors / boxes / logits / norms
            featmap_sizes = [(t.shape[-2], t.shape[-1]) for t in cls_list]
            mlvl_priors = head.prior_generator.grid_priors(
                featmap_sizes, dtype=batch_inputs.dtype, device=device, with_stride=True)
            flat_priors = torch.cat(mlvl_priors, dim=0)
            flat_logits = torch.cat([t.permute(0,2,3,1).reshape(B,-1,num_classes) for t in cls_list], dim=1)
            flat_bbox   = torch.cat([t.permute(0,2,3,1).reshape(B,-1,4)           for t in bbox_list], dim=1)

            # BN feature per anchor — per stride, then concat in same order as cls_list
            bn_per_stride = []
            for i, bn_t in enumerate(bn_capture):
                # bn_t: (B, C, H, W)  — channel norm gives ||BN(F_v)|| per anchor
                bn_flat = bn_t.permute(0,2,3,1).reshape(B, -1, bn_t.shape[1])  # (B, A_i, C)
                bn_per_stride.append(bn_flat)
            flat_bn = torch.cat(bn_per_stride, dim=1)                          # (B, A, C)
            flat_norm = flat_bn.norm(p=2, dim=-1)                              # (B, A)
            flat_bn_n = F.normalize(flat_bn, dim=-1)                           # for cosine
            cos_all_known = torch.einsum('bac,kc->bak', flat_bn_n, proto_known)  # (B, A, K)
            cos_unk_t     = torch.einsum('bac,kc->bak', flat_bn_n, proto_unk).squeeze(-1)  # (B, A)

            anchor_strides = flat_priors[:, 2]                                  # (A,)

            for b, (img_id, _, _ds, meta, scale, pad, ann_path) in enumerate(buf):
                gt = parse_voc_xml(ann_path) if os.path.exists(ann_path) else []
                gt = [g for g in gt if not g[2]]                                # drop difficult
                if not gt:
                    continue
                sx, sy = float(scale[0]), float(scale[1])
                pad_top, pad_left = float(pad[0]), float(pad[2])
                gt_pad = torch.tensor([g[1] for g in gt], dtype=torch.float32, device=device)
                gt_pad[:, [0,2]] = gt_pad[:, [0,2]] * sx + pad_left
                gt_pad[:, [1,3]] = gt_pad[:, [1,3]] * sy + pad_top

                # Decode per-sample (matches analyzer pattern)
                anchor_boxes_b = head.bbox_coder.decode(
                    flat_priors[..., :2],
                    flat_bbox[b:b+1],
                    flat_priors[:, [2]][..., 0])[0]                             # (A, 4)
                ious = box_iou_xyxy(anchor_boxes_b, gt_pad)                     # (A, G)
                cls_scores = flat_logits[b].sigmoid() if not args.vis else None
                tunk_score = cls_scores[:, unk_idx] if cls_scores is not None else None
                for gi, (gname, gbox, _diff) in enumerate(gt):
                    role = class_role(gname, base_set, novel_set)
                    gt_lbl = all_class_names.index(gname) if gname in all_class_names else -1
                    iou_a = ious[:, gi]
                    if args.vis:
                        # Visualization: pick best-IoU anchor (independent of weights/scores).
                        # Use the same IoU floor for matching consistency.
                        if iou_a.max().item() < args.match_iou:
                            continue
                        a = int(iou_a.argmax().item())
                        per_cls_role[gname] = role
                        if len(per_cls_norms[gname]) >= args.vis_max_per_class:
                            continue
                        vis_vecs.append(flat_bn[b, a].detach().cpu().numpy())
                        vis_labels.append(gname)
                        vis_roles.append(role)
                        per_cls_norms[gname].append(1)
                        continue
                    # Norm-mode TAL match
                    if role == 'unknown':
                        score_a = tunk_score
                        corr_idx = unk_idx
                    elif 0 <= gt_lbl < known_count:
                        score_a = cls_scores[:, gt_lbl]
                        corr_idx = gt_lbl
                    else:
                        continue
                    align = (score_a.clamp(min=1e-6) ** args.tal_alpha) * (iou_a ** args.tal_beta)
                    valid = iou_a >= args.match_iou
                    if not valid.any():
                        continue
                    align[~valid] = -1
                    a = int(align.argmax().item())
                    per_cls_role[gname] = role
                    bn_norm  = float(flat_norm[b, a].item())
                    raw_corr = float(flat_logits[b, a, corr_idx].item())
                    raw_unk  = float(flat_logits[b, a, unk_idx].item())
                    if role == 'unknown':
                        cos_corr = float(cos_unk_t[b, a].item())
                    else:
                        cos_corr = float(cos_all_known[b, a, corr_idx].item())
                    cos_u = float(cos_unk_t[b, a].item())
                    stride = int(round(float(anchor_strides[a].item())))
                    bw = float(gbox[2] - gbox[0])
                    bh = float(gbox[3] - gbox[1])
                    w.writerow([img_id, gname, role, stride, f'{float(iou_a[a]):.4f}',
                                f'{bn_norm:.4f}', f'{cos_corr:.4f}', f'{cos_u:.4f}',
                                f'{raw_corr:.4f}', f'{raw_unk:.4f}',
                                f'{bw:.1f}', f'{bh:.1f}'])
                    per_cls_norms[gname].append(bn_norm)

    def load_one(img_id):
        img_path = os.path.join(img_dir, img_id + img_ext)
        ann_path = os.path.join(ann_dir, img_id + '.xml')
        if not os.path.exists(img_path):
            return None
        try:
            data = pipeline(dict(img_path=img_path, img_id=img_id, instances=[]))
        except Exception as e:
            print(f'  [skip] {img_id}: {e}')
            return None
        meta = data['data_samples'].metainfo
        return (img_id, data['inputs'].unsqueeze(0), data['data_samples'],
                meta, np.asarray(meta['scale_factor']),
                np.asarray(meta.get('pad_param', np.zeros(4, dtype=np.float32))),
                ann_path)

    # Prefetch images with a thread pool while GPU runs inference.
    # Keep 2 batches worth of futures in-flight to hide disk/decode latency.
    prefetch = args.num_workers * 2
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = []
        done = 0
        # Submit first prefetch window
        for i in range(min(prefetch, n)):
            futures.append(pool.submit(load_one, ids[i]))
        next_submit = len(futures)

        while futures:
            result = futures.pop(0).result()
            done += 1
            # Submit next image to keep the window full
            if next_submit < n:
                futures.append(pool.submit(load_one, ids[next_submit]))
                next_submit += 1
            if result is not None:
                batch_buf.append(result)
            if len(batch_buf) == args.batch_size:
                flush(batch_buf); batch_buf = []
            if done % 1000 == 0:
                print(f'  processed {done}/{n}')
    flush(batch_buf)
    if fcsv is not None:
        fcsv.close()
    for h in hooks: h.remove()

    # ── Visualization-only path ──────────────────────────────────────────────
    if args.vis:
        if not vis_vecs:
            print('[vis] no matched anchors — nothing to visualize'); return
        prefix = args.vis_out or os.path.splitext(args.out_csv)[0] + '_umap'
        os.makedirs(os.path.dirname(prefix) or '.', exist_ok=True)
        X = np.stack(vis_vecs).astype(np.float32)
        # Raw BN(F_v), no L2-normalize — UMAP uses both magnitude + direction.
        labels = np.asarray(vis_labels)
        roles  = np.asarray(vis_roles)

        # Raw text embeddings (before L2-norm in head).
        # base norm≈4-5, novel≈1, Tunk≈1. They are 100-400× smaller than visual features
        # (norm≈27-40), so we scale them up into the same norm range before joint UMAP.
        raw_embed = ctx['model'].embeddings.detach().cpu().numpy().astype(np.float32)
        prompt_X_raw = raw_embed[:known_count + 1]                  # base+novel+Tunk
        prompt_names = list(all_class_names[:known_count]) + ['Tunk']
        vis_mean_norm = float(np.linalg.norm(X, axis=1).mean())
        prompt_norm   = float(np.linalg.norm(prompt_X_raw, axis=1).mean()) + 1e-9
        scale_factor  = vis_mean_norm / prompt_norm
        prompt_X = prompt_X_raw * scale_factor   # scale into visual feature norm range
        print(f'[vis] visual mean norm={vis_mean_norm:.1f}  '
              f'prompt mean norm={prompt_norm:.2f}  scale={scale_factor:.1f}')

        npz = prefix + '.npz'
        np.savez_compressed(npz, X=X, labels=labels, roles=roles,
                            prompt_X=prompt_X_raw,
                            prompt_names=np.asarray(prompt_names))
        print(f'[vis] saved vectors: {npz}  anchors={X.shape}  prompts={prompt_X_raw.shape}')
        try:
            import umap  # type: ignore
        except Exception as e:
            print(f'[vis] umap-learn not installed ({e}); '
                  f'Install: pip install umap-learn matplotlib'); return

        # Joint fit: include scaled prompts so they are embedded in the same manifold.
        # Flag them with a sentinel label so we can split after embedding.
        _PROMPT_SENTINEL = '__PROMPT__'
        X_joint      = np.concatenate([X, prompt_X], axis=0)
        labels_joint = np.concatenate([labels,
                                       np.array([_PROMPT_SENTINEL] * len(prompt_names))])

        print(f'[vis] running UMAP (joint)  N={X_joint.shape[0]}  D={X_joint.shape[1]}  '
              f'n_neighbors={args.vis_n_neighbors}  min_dist={args.vis_min_dist}')
        reducer = umap.UMAP(n_components=2, metric='euclidean',
                            n_neighbors=args.vis_n_neighbors,
                            min_dist=args.vis_min_dist,
                            random_state=0, verbose=True)
        emb_joint = reducer.fit_transform(X_joint)
        emb       = emb_joint[:len(X)]
        prompt_emb = emb_joint[len(X):]

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f'[vis] matplotlib unavailable ({e}); embedding saved to npz only'); return

        np.savez_compressed(prefix + '_emb.npz', emb=emb, labels=labels, roles=roles,
                            prompt_emb=prompt_emb,
                            prompt_names=np.asarray(prompt_names))

        # ── Color / marker scheme ────────────────────────────────────────────
        base_classes    = sorted([c for c in set(labels.tolist()) if per_cls_role[c] == 'base'])
        novel_classes   = sorted([c for c in set(labels.tolist()) if per_cls_role[c] == 'novel'])
        unknown_classes = sorted([c for c in set(labels.tolist()) if per_cls_role[c] == 'unknown'])

        def palette(cmap_name, n):
            cm = plt.get_cmap(cmap_name)
            if n <= 1: return [cm(0.55)]
            return [cm(0.2 + 0.65 * i / max(n - 1, 1)) for i in range(n)]

        base_colors  = dict(zip(base_classes,  palette('tab10', len(base_classes))))
        novel_colors = dict(zip(novel_classes, palette('Greens', len(novel_classes))))
        warm_pool    = ['#e6194B', '#f58231', '#ffe119', '#911eb4',
                        '#9A6324', '#fabebe', '#bfef45', '#469990', '#a9a9a9']
        unknown_colors = {c: warm_pool[i % len(warm_pool)]
                          for i, c in enumerate(unknown_classes)}
        color_of   = {**base_colors, **novel_colors, **unknown_colors}
        role_marker = {'base': 'o', 'novel': '^', 'unknown': 'X'}

        # ── Two-panel figure ─────────────────────────────────────────────────
        # Left: all anchors + prompts.  Right: zoom on prompt region with labels.
        fig, axes = plt.subplots(1, 2, figsize=(20, 9),
                                 gridspec_kw={'width_ratios': [3, 2]})
        fig.suptitle(f'BN(F_v) UMAP — {dataset_name}/{args.split}  '
                     f'(N={X.shape[0]}, K={known_count}, '
                     f'ckpt={os.path.basename(args.checkpoint)})', fontsize=11)

        for ax in axes:
            for c in base_classes + novel_classes + unknown_classes:
                m = labels == c
                ax.scatter(emb[m, 0], emb[m, 1],
                           s=8, alpha=0.45, linewidths=0,
                           color=color_of[c],
                           marker=role_marker[per_cls_role[c]],
                           label=f'{c} [{per_cls_role[c]}]' if ax is axes[0] else '_')

            # Prompt stars
            for pi, pname in enumerate(prompt_names):
                role_p = ('base' if pname in base_colors else
                          'novel' if pname in novel_colors else 'unk')
                pcolor = (base_colors.get(pname) or
                          novel_colors.get(pname) or 'black')
                ax.scatter(prompt_emb[pi, 0], prompt_emb[pi, 1],
                           s=280, marker='*', color=pcolor,
                           edgecolor='black', linewidth=0.8, zorder=6)
                if ax is axes[1]:   # labels only on the zoom panel
                    ax.annotate(pname, (prompt_emb[pi, 0], prompt_emb[pi, 1]),
                                fontsize=8, fontweight='bold',
                                xytext=(4, 3), textcoords='offset points', zorder=7)

        # Right panel: zoom to prompt bounding box ± 20% margin
        if len(prompt_emb):
            px, py = prompt_emb[:, 0], prompt_emb[:, 1]
            pw, ph = px.max() - px.min(), py.max() - py.min()
            pad_x, pad_y = max(pw * 0.4, 0.5), max(ph * 0.4, 0.5)
            axes[1].set_xlim(px.min() - pad_x, px.max() + pad_x)
            axes[1].set_ylim(py.min() - pad_y, py.max() + pad_y)
        axes[1].set_title('Zoom: prompt region')

        axes[0].set_title('All anchors + prompts (★)')
        axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')
        axes[1].set_xlabel('UMAP 1')
        axes[0].legend(fontsize=7, loc='center left', bbox_to_anchor=(1.01, 0.5),
                       ncol=1, frameon=False)
        plt.tight_layout()
        png = prefix + '.png'
        plt.savefig(png, dpi=160, bbox_inches='tight')
        print(f'[vis] saved figure: {png}')
        return

    # ── Summary ──────────────────────────────────────────────────────────────
    print('\n' + '=' * 80)
    print(f'BN(F_v) NORM SUMMARY  ({dataset_name} / {args.split})')
    print('=' * 80)
    print(f'  {"class":35s} {"role":7s} {"N":>6s}  {"mean":>8s} {"std":>8s} {"p10":>8s} {"p50":>8s} {"p90":>8s}')
    summary = {}
    for cname in sorted(per_cls_norms.keys(),
                        key=lambda c: (0 if per_cls_role[c]=='base' else
                                       1 if per_cls_role[c]=='novel' else 2, c)):
        a = np.asarray(per_cls_norms[cname])
        summary[cname] = {
            'role': per_cls_role[cname], 'N': int(a.size),
            'mean': float(a.mean()), 'std': float(a.std()),
            'p10': float(np.percentile(a,10)), 'p50': float(np.percentile(a,50)),
            'p90': float(np.percentile(a,90)),
        }
        s = summary[cname]
        print(f'  {cname:35s} {s["role"]:7s} {s["N"]:6d}  '
              f'{s["mean"]:8.3f} {s["std"]:8.3f} {s["p10"]:8.3f} {s["p50"]:8.3f} {s["p90"]:8.3f}')

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump({'dataset': dataset_name, 'split': args.split,
                       'config': args.config, 'checkpoint': args.checkpoint,
                       'match_iou': args.match_iou,
                       'per_class': summary}, f, indent=2)
        print(f'[write] {args.out_json}')
    print(f'[write] {args.out_csv}')


if __name__ == '__main__':
    main()
