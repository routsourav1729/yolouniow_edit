#!/usr/bin/env python3
"""OWOD Test-Set Analysis — Confusion Matrix + Per-Class Anchor Metrics.

Runs model on ALL test-set images for a given dataset/task checkpoint.

DATA FLOW
─────────
  Source                              What it provides
  ──────────────────────────────────  ──────────────────────────────────────
  Config file (--config)              Model architecture, dataset name,
                                      class splits (via env vars $DATASET,
                                      $TASK, $FEWSHOT_K, etc.)
  data/OWOD/ImageSets/{DS}/test.txt   Image IDs for the test split
  data/OWOD/JPEGImages/{DS}/{id}.ext  Test images (jpg/png)
  data/OWOD/Annotations/{DS}/{id}.xml GT bounding boxes + class names
  Checkpoint (--checkpoint)           Trained weights + embeddings
  owodb_const.py                      Full class name list per dataset

ANALYSIS 1 — CONFUSION MATRIX
  For each test image:
    • Run full predict pipeline → bboxes, scores, labels (original coords)
    • Filter predictions by --conf-thr
    • Greedy match predictions to GT (IoU >= 0.5, highest score first)
    • Record: confusion[gt_class][pred_class] += 1
    • Unmatched GT         → confusion[gt_class]["missed"]  += 1
    • Unmatched predictions → false_positive[pred_class]    += 1

ANALYSIS 2 — PER-CLASS ANCHOR METRICS
  For each test image with GT:
    • Raw forward gives per-anchor logits
    • For each GT box, find best-matching anchor (IoU >= 0.5)
    • Record: anchor_score, tunk_score, max_known, ratio, energy, entropy
    • Grouped by GT class name and role (base / novel / unknown)

MODES
  Single checkpoint (existing):
    python analyze_owod_test.py --config C --checkpoint K --out-dir D [...]

  Multi checkpoint (data loads once, N checkpoints share each batch):
    python analyze_owod_test.py --runs-json runs.json [--num-images N] [--conf-thr T] [--batch-size B]
    runs.json = [{"label":"t2_baseline","config":"...","checkpoint":"...","out_dir":"..."},
                 {"label":"t2_wapr","config":"...","checkpoint":"...","out_dir":"..."}]
    All runs must share the same DATASET / TASK env vars (same test split).

OUTPUTS  (per run, under its out_dir)
  confusion_matrix.csv     Row = GT class, Col = predicted class
  false_positives.csv      Unmatched predictions by predicted class
  test_analysis.json       Per-class metrics + confusion + FP data
"""
import argparse
import copy
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import torch


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='OWOD test-set analysis')
    # Single-checkpoint mode
    p.add_argument('--config', default='')
    p.add_argument('--checkpoint', default='')
    p.add_argument('--out-dir', default='')
    # Multi-checkpoint mode
    p.add_argument('--runs-json', default='',
                   help='JSON file with list of {label,config,checkpoint,out_dir}. '
                        'When set, test data loads once; all runs share each batch.')
    # Shared options
    p.add_argument('--num-images', type=int, default=0, help='0 = all test images.')
    p.add_argument('--conf-thr', type=float, default=0.05)
    p.add_argument('--batch-size', type=int, default=64)
    return p.parse_args()


def parse_voc_xml_full(xml_path):
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
    if cls in base_set:
        return 'base'
    if cls in novel_set:
        return 'novel'
    return 'unknown'


def percentiles(arr):
    if len(arr) == 0:
        return {}
    a = np.asarray(arr, dtype=np.float64)
    out = {'count': int(a.size),
           'min': float(a.min()), 'max': float(a.max()),
           'mean': float(a.mean()), 'std': float(a.std())}
    for q in (10, 25, 50, 75, 90, 95, 99):
        out[f'p{q}'] = float(np.percentile(a, q))
    return out


# ── Model loading ─────────────────────────────────────────────────────────────

def build_model_ctx(cfg_path, ckpt_path, device):
    """Load config + checkpoint → model context dict."""
    from mmengine.config import Config
    from mmengine.runner import load_state_dict
    from mmyolo.registry import MODELS

    cfg = Config.fromfile(cfg_path)

    model = MODELS.build(cfg.model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    state_dict = {k: v for k, v in state_dict.items() if 'text_model' not in k}

    if 'embeddings' in state_dict:
        ckpt_emb = state_dict['embeddings']
        if ckpt_emb.shape == model.embeddings.shape:
            with torch.no_grad():
                model.embeddings.data.copy_(ckpt_emb.to(model.embeddings.device))
            print(f'  [model] embeddings from ckpt (shape={tuple(ckpt_emb.shape)})')
            state_dict = {k: v for k, v in state_dict.items() if k != 'embeddings'}
        else:
            print(f'  [warn] embeddings shape mismatch: ckpt={tuple(ckpt_emb.shape)} '
                  f'model={tuple(model.embeddings.shape)} — falling back')
            state_dict['embeddings'] = model.update_embeddings(ckpt_emb)

    import logging
    _mme_logger = logging.getLogger('mmengine')
    _prev_level = _mme_logger.level
    _mme_logger.setLevel(logging.ERROR)   # suppress "missing keys" info-level noise
    load_state_dict(model, state_dict, strict=False)
    _mme_logger.setLevel(_prev_level)
    model.eval().to(device)

    head = model.bbox_head
    head_module = head.head_module
    head.num_classes = model.num_test_classes
    num_classes = head_module.num_classes
    num_prev = model.num_prev_classes
    num_cur = num_classes - num_prev - 2
    known_count = num_prev + num_cur

    # Dataset / class info from config
    dataset_name = str(cfg.owod_dataset)
    data_root = str(getattr(cfg, 'owod_root', 'data/OWOD'))
    test_image_set = str(getattr(cfg, 'test_image_set', 'test'))

    from yolo_world.datasets.owodb_const import VOC_COCO_CLASS_NAMES
    all_class_names = list(VOC_COCO_CLASS_NAMES[dataset_name])
    base_classes = all_class_names[:num_prev]
    novel_classes = all_class_names[num_prev:num_prev + num_cur]

    return {
        'model': model,
        'head': head,
        'head_module': head_module,
        'num_classes': num_classes,
        'num_prev': num_prev,
        'num_cur': num_cur,
        'known_count': known_count,
        'dataset_name': dataset_name,
        'data_root': data_root,
        'test_image_set': test_image_set,
        'all_class_names': all_class_names,
        'base_classes': base_classes,
        'novel_classes': novel_classes,
        'base_set': set(base_classes),
        'novel_set': set(novel_classes),
        'model_label_names': list(all_class_names[:known_count]) + ['unknown'],
        'cfg': cfg,
        'cfg_path': cfg_path,
        'ckpt_path': ckpt_path,
    }


def build_pipeline(ctx):
    """Extract image-only pipeline from config (no LoadAnnotations / PackDetInputs)."""
    from mmengine.dataset import Compose
    cfg = ctx['cfg']
    val_ds_cfg = cfg.val_dataloader.dataset
    raw_pipeline = (val_ds_cfg.pipeline
                    if hasattr(val_ds_cfg, 'pipeline')
                    else val_ds_cfg.dataset.pipeline)
    img_pipeline_cfg = [t for t in raw_pipeline
                        if 'LoadAnnotations' not in t.get('type', '')
                        and 'PackDetInputs' not in t.get('type', '')]
    img_pipeline_cfg.append(dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param')))
    return Compose(img_pipeline_cfg)


# ── Accumulators ──────────────────────────────────────────────────────────────

METRIC_NAMES = ['anchor', 'max_known', 'tunk', 'ratio',
                'energy', 'entropy',
                'tunk_raw', 'tanchor_raw', 'correct_cls_score',
                'correct_cls_rank']


def init_accum():
    return {
        'confusion': defaultdict(lambda: defaultdict(int)),
        'false_pos_by_cls': defaultdict(int),
        'gt_counts': defaultdict(int),
        'gt_counts_all': defaultdict(int),
        'per_cls_metrics': defaultdict(lambda: {m: [] for m in METRIC_NAMES}),
        'u_recall_hit': defaultdict(int),
        'u_recall_total': defaultdict(int),
        'n_proc': 0,
        'n_skip': 0,
    }


def _stride_key(s):
    """Bin a stride float to its integer stride label (8/16/32/…)."""
    return int(round(s))


# ── Batch processing ──────────────────────────────────────────────────────────

def process_batch_with_ctx(buf, ctx, accum, device, conf_thr):
    """Run one preprocessed batch through one model context, updating accum."""
    B = len(buf)
    batch_inputs = torch.cat(
        [entry[1] for entry in buf], dim=0
    ).float().to(device) / 255.0

    data_samples = [entry[2] for entry in buf]
    batch_metas  = [entry[3] for entry in buf]

    model       = ctx['model']
    head        = ctx['head']
    head_module = ctx['head_module']
    num_classes  = ctx['num_classes']
    known_count  = ctx['known_count']
    num_prev     = ctx['num_prev']
    all_class_names  = ctx['all_class_names']
    base_set     = ctx['base_set']
    novel_set    = ctx['novel_set']
    model_label_names = ctx['model_label_names']
    unk_cls_idx  = known_count

    img_feats, txt_feats = model.extract_feat(batch_inputs, data_samples)
    cls_list, bbox_list  = head_module.forward_one2one(img_feats, txt_feats)

    results_list = head.predict_by_feat(
        cls_list, bbox_list,
        batch_img_metas=batch_metas,
        rescale=True,
        with_nms=True)

    featmap_sizes = [(t.shape[-2], t.shape[-1]) for t in cls_list]
    mlvl_priors   = head.prior_generator.grid_priors(
        featmap_sizes, dtype=batch_inputs.dtype,
        device=device, with_stride=True)
    flat_priors = torch.cat(mlvl_priors, dim=0)

    flat_logits_b = torch.cat([
        t.permute(0, 2, 3, 1).reshape(B, -1, num_classes)
        for t in cls_list], dim=1)
    flat_bbox_b = torch.cat([
        t.permute(0, 2, 3, 1).reshape(B, -1, 4)
        for t in bbox_list], dim=1)

    confusion        = accum['confusion']
    false_pos_by_cls = accum['false_pos_by_cls']
    gt_counts        = accum['gt_counts']
    gt_counts_all    = accum['gt_counts_all']
    per_cls_metrics  = accum['per_cls_metrics']
    u_recall_hit     = accum['u_recall_hit']
    u_recall_total   = accum['u_recall_total']

    for b, (img_id, _, ds, meta, scale, pad, ann_path) in enumerate(buf):
        pred       = results_list[b]
        keep_mask  = pred.scores >= conf_thr
        pred_bboxes = pred.bboxes[keep_mask]
        pred_scores = pred.scores[keep_mask]
        pred_labels = pred.labels[keep_mask]

        gt_full   = (parse_voc_xml_full(ann_path)
                     if os.path.exists(ann_path) else [])
        gt_names  = [g[0] for g in gt_full]
        gt_difficult = [g[2] for g in gt_full]
        for gn, diff in zip(gt_names, gt_difficult):
            gt_counts_all[gn] += 1
            if not diff:
                gt_counts[gn] += 1

        gt_boxes_nd = (
            torch.tensor([g[1] for g, d in zip(gt_full, gt_difficult) if not d],
                         dtype=torch.float32, device=device)
            if any(not d for d in gt_difficult)
            else torch.zeros((0, 4), device=device))
        gt_names_nd = [g[0] for g, d in zip(gt_full, gt_difficult) if not d]

        gt_boxes_orig = (
            torch.tensor([g[1] for g in gt_full],
                         dtype=torch.float32, device=device)
            if gt_full else torch.zeros((0, 4), device=device))

        if len(pred_bboxes) > 0 and gt_boxes_nd.shape[0] > 0:
            iou_pg = box_iou_xyxy(pred_bboxes, gt_boxes_nd)  # [P, G]
            # Vectorized greedy matching: for each pred (high→low score),
            # pick best unmatched GT with IoU≥0.5. Avoid per-element .item() calls.
            sorted_pi    = pred_scores.argsort(descending=True)
            gt_matched   = torch.zeros(len(gt_names_nd), dtype=torch.bool, device=device)
            pred_matched = torch.zeros(len(pred_bboxes), dtype=torch.bool, device=device)
            pred_lbls_np = pred_labels.cpu().numpy()
            for pi in sorted_pi.tolist():
                row = iou_pg[pi].clone()
                row[gt_matched] = 0.0
                best_iou_val, best_gi = row.max(0)
                if best_iou_val.item() >= 0.5:
                    lbl      = int(pred_lbls_np[pi])
                    pred_cls = (model_label_names[lbl]
                                if lbl < len(model_label_names)
                                else f'cls_{lbl}')
                    confusion[gt_names_nd[best_gi.item()]][pred_cls] += 1
                    gt_matched[best_gi]  = True
                    pred_matched[pi]     = True
            for gi, gn in enumerate(gt_names_nd):
                if not gt_matched[gi].item():
                    confusion[gn]['missed'] += 1
            for pi in (~pred_matched).nonzero(as_tuple=True)[0].tolist():
                lbl      = int(pred_lbls_np[pi])
                pred_cls = (model_label_names[lbl]
                            if lbl < len(model_label_names)
                            else f'cls_{lbl}')
                false_pos_by_cls[pred_cls] += 1
        elif gt_names_nd:
            for gn in gt_names_nd:
                confusion[gn]['missed'] += 1
        else:
            pred_lbls_np = pred_labels.cpu().numpy()
            for pi in range(len(pred_bboxes)):
                lbl      = int(pred_lbls_np[pi])
                pred_cls = (model_label_names[lbl]
                            if lbl < len(model_label_names)
                            else f'cls_{lbl}')
                false_pos_by_cls[pred_cls] += 1

        # U-Recall
        unk_pred_mask   = pred.labels == unk_cls_idx
        unk_pred_bboxes = pred.bboxes[unk_pred_mask]
        for gi, (gname, gdiff) in enumerate(zip(gt_names, gt_difficult)):
            if gdiff:
                continue
            role = class_role(gname, base_set, novel_set)
            if role != 'unknown':
                continue
            u_recall_total[gname] += 1
            if unk_pred_bboxes.shape[0] > 0:
                gt_box = gt_boxes_orig[gi:gi + 1]
                if box_iou_xyxy(unk_pred_bboxes, gt_box).max().item() >= 0.5:
                    u_recall_hit[gname] += 1

        # Per-class anchor metrics
        if gt_boxes_nd.shape[0] > 0:
            sx, sy        = float(scale[0]), float(scale[1])
            pad_top, pad_left = float(pad[0]), float(pad[2])
            gt_pad        = gt_boxes_nd.clone()
            gt_pad[:, [0, 2]] = gt_pad[:, [0, 2]] * sx + pad_left
            gt_pad[:, [1, 3]] = gt_pad[:, [1, 3]] * sy + pad_top

            flat_logits  = flat_logits_b[b]
            anchor_boxes = head.bbox_coder.decode(
                flat_priors[..., :2],
                flat_bbox_b[b:b + 1],
                flat_priors[:, [2]][..., 0])[0]

            anchor_scores  = flat_logits[:, -1].sigmoid()
            tunk_scores    = flat_logits[:, -2].sigmoid()
            known_logits   = flat_logits[:, :-2]
            known_scores   = known_logits.sigmoid()
            max_known_vals, _ = known_scores.max(dim=1)
            ratio_vals     = max_known_vals / anchor_scores.clamp(min=1e-6)
            energy_vals    = -torch.logsumexp(known_logits, dim=1)
            log_sm         = torch.log_softmax(known_logits, dim=1)
            entropy_vals   = -(log_sm.exp() * log_sm).sum(dim=1)
            all_scores     = flat_logits.sigmoid()

            iou_ag = box_iou_xyxy(anchor_boxes, gt_pad)

            a_np   = anchor_scores.cpu().numpy()
            mk_np  = max_known_vals.cpu().numpy()
            tu_np  = tunk_scores.cpu().numpy()
            rt_np  = ratio_vals.cpu().numpy()
            en_np  = energy_vals.cpu().numpy()
            et_np  = entropy_vals.cpu().numpy()
            tunk_raw_np    = flat_logits[:, -2].cpu().numpy()
            tanchor_raw_np = flat_logits[:, -1].cpu().numpy()
            all_scores_np  = all_scores.cpu().numpy()
            known_logits_np = known_logits.cpu().numpy()

            for gi in range(len(gt_names_nd)):
                best_iou_val, best_idx = iou_ag[:, gi].max(dim=0)
                if best_iou_val.item() >= 0.5:
                    idx   = best_idx.item()
                    gname = gt_names_nd[gi]
                    role  = class_role(gname, base_set, novel_set)
                    gt_lbl = (all_class_names.index(gname)
                              if gname in all_class_names else -1)

                    if role == 'unknown':
                        correct_score = float(tu_np[idx])
                        correct_raw   = float(tunk_raw_np[idx])
                        correct_rank  = int(
                            (all_scores_np[idx] > all_scores_np[idx, known_count])
                            .sum()) + 1
                    elif 0 <= gt_lbl < known_count:
                        correct_score = float(known_scores[idx, gt_lbl].item())
                        correct_raw   = float(known_logits_np[idx, gt_lbl])
                        correct_rank  = int(
                            (all_scores_np[idx] > all_scores_np[idx, gt_lbl])
                            .sum()) + 1
                    else:
                        correct_score = 0.0
                        correct_raw   = 0.0
                        correct_rank  = known_count + 2

                    vals = (a_np[idx], mk_np[idx], tu_np[idx],
                            rt_np[idx], en_np[idx], et_np[idx],
                            float(tunk_raw_np[idx]), float(tanchor_raw_np[idx]),
                            correct_score, float(correct_rank))
                    for k, v in zip(METRIC_NAMES, vals):
                        per_cls_metrics[gname][k].append(float(v))
                    per_cls_metrics[gname].setdefault('_correct_raw', []).append(correct_raw)
                    per_cls_metrics[gname].setdefault('_raw_logits_by_channel', []).append(
                        known_logits_np[idx].tolist())
                    # stride of the best-matching anchor (8 / 16 / 32)
                    per_cls_metrics[gname].setdefault('_anchor_stride', []).append(
                        _stride_key(float(flat_priors[idx, 2].item())))

        accum['n_proc'] += 1
        # Progress print every 500 images (checked per-image so it fires at exactly 500,1000,...)
        # Only print from the first accumulator to avoid duplicate lines in multi-run mode.
        if accum is _primary_accum and accum['n_proc'] % 500 == 0:
            print(f'  processed {accum["n_proc"]}/{_n_imgs_total}  '
                  f'(skipped {accum["n_skip"]})')


# Module-level sentinels set before processing starts (avoids closure over mutable locals)
_primary_accum  = None
_n_imgs_total   = 0


# ── Result saving ─────────────────────────────────────────────────────────────

def save_run_results(accum, ctx, out_dir, conf_thr):
    os.makedirs(out_dir, exist_ok=True)

    all_class_names   = ctx['all_class_names']
    base_set, novel_set = ctx['base_set'], ctx['novel_set']
    known_count       = ctx['known_count']
    model_label_names = ctx['model_label_names']

    confusion        = accum['confusion']
    false_pos_by_cls = accum['false_pos_by_cls']
    gt_counts        = accum['gt_counts']
    gt_counts_all    = accum['gt_counts_all']
    per_cls_metrics  = accum['per_cls_metrics']
    u_recall_hit     = accum['u_recall_hit']
    u_recall_total   = accum['u_recall_total']
    n_proc           = accum['n_proc']
    n_skip           = accum['n_skip']

    u_total_all = sum(u_recall_total.values())
    u_hit_all   = sum(u_recall_hit.values())
    u_recall_compat = u_hit_all / u_total_all * 100 if u_total_all > 0 else 0.0

    print(f'\n[{ctx["label"]}] ' + '=' * 70)
    print(f'U-RECALL: {u_hit_all}/{u_total_all} = {u_recall_compat:.2f}%')
    for cls in sorted(u_recall_total.keys(), key=lambda x: -u_recall_total[x]):
        n, d = u_recall_total[cls], u_recall_hit[cls]
        print(f'  {cls:30s}: {d:5d}/{n:5d} = {d/n*100 if n>0 else 0:.2f}%')

    all_gt_cls = sorted(gt_counts.keys(), key=lambda c: (
        0 if c in base_set else 1 if c in novel_set else 2, c))
    pred_cols  = list(model_label_names) + ['missed']

    print(f'\n[{ctx["label"]}] CONFUSION MATRIX (top-5 per GT class)')
    for gt_cls in all_gt_cls:
        role  = class_role(gt_cls, base_set, novel_set)
        total = gt_counts[gt_cls]
        top   = sorted(confusion[gt_cls].items(), key=lambda x: -x[1])[:5]
        print(f'  {gt_cls:30s} ({role:7s}) N={total:6d}  '
              + ', '.join(f'{c}:{n}' for c, n in top))

    # confusion_matrix.csv
    csv_path = os.path.join(out_dir, 'confusion_matrix.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gt_class', 'role', 'total_gt'] + pred_cols)
        for gt_cls in all_gt_cls:
            role = class_role(gt_cls, base_set, novel_set)
            row  = [gt_cls, role, gt_counts[gt_cls]]
            for pc in pred_cols:
                row.append(confusion[gt_cls].get(pc, 0))
            w.writerow(row)
    print(f'[write] {csv_path}')

    # Per-class anchor metrics
    pc_summary = {}
    for cls_name in all_gt_cls:
        d = per_cls_metrics[cls_name]
        n = len(d['anchor'])
        if n == 0:
            continue
        role = class_role(cls_name, base_set, novel_set)
        s = {m: percentiles(d[m]) for m in METRIC_NAMES}
        ch_stats = {}
        raw_mat = d.get('_raw_logits_by_channel', [])
        if raw_mat:
            arr = np.asarray(raw_mat)
            for ci, cname in enumerate(all_class_names[:known_count]):
                ch_stats[cname] = {
                    'mean': float(arr[:, ci].mean()),
                    'std':  float(arr[:, ci].std()),
                }
        pc_summary[cls_name] = {
            'role': role, 'count': n, 'stats': s,
            'per_channel_logit': ch_stats,
        }
        cr = d.get('_correct_raw', [])
        if cr:
            pc_summary[cls_name]['correct_raw_stats'] = percentiles(cr)

    # ROLE-CORRECT SCORE TABLE
    print(f'\n[{ctx["label"]}] ROLE-CORRECT SCORE TABLE')
    print(f'  {"class":30s} {"role":7s} {"N":>6s}  '
          f'{"correct_sig_μ":>13s}  {"rank_μ":>6s}  {"rank_p50":>8s}')
    for cls_name in all_gt_cls:
        if cls_name not in pc_summary:
            continue
        ps = pc_summary[cls_name]
        s  = ps['stats']
        print(f'  {cls_name:30s} {ps["role"]:7s} {ps["count"]:6d}  '
              f'{s["correct_cls_score"]["mean"]:13.4f}  '
              f'{s["correct_cls_rank"]["mean"]:6.1f}  '
              f'{s["correct_cls_rank"]["p50"]:8.0f}')

    # ── Unknown-class Tobj / Tunk distribution by anchor stride ──────────────
    unk_classes = [c for c in all_gt_cls
                   if class_role(c, base_set, novel_set) == 'unknown'
                   and c in per_cls_metrics]

    unk_stride_data = {}   # cls → stride → {'tobj': [...], 'tunk': [...]}
    all_strides_seen = set()
    for cls_name in unk_classes:
        d = per_cls_metrics[cls_name]
        strides = d.get('_anchor_stride', [])
        tobj    = d.get('anchor', [])
        tunk    = d.get('tunk', [])
        if not strides:
            continue
        by_stride = {}
        for s, to, tu in zip(strides, tobj, tunk):
            by_stride.setdefault(s, {'tobj': [], 'tunk': []})
            by_stride[s]['tobj'].append(to)
            by_stride[s]['tunk'].append(tu)
            all_strides_seen.add(s)
        unk_stride_data[cls_name] = by_stride

    sorted_strides = sorted(all_strides_seen)

    if unk_stride_data:
        print(f'\n[{ctx["label"]}] UNKNOWN CLASS Tobj / Tunk DISTRIBUTION BY ANCHOR STRIDE')
        print(f'  Strides observed: {sorted_strides}')
        hdr = (f'  {"class":35s} {"N":>5s}  '
               + '  '.join(f'stride={s:<2d}  Tobj μ±σ (p50)   Tunk μ±σ (p50)'
                            for s in sorted_strides))
        print(hdr)
        for cls_name in unk_classes:
            if cls_name not in unk_stride_data:
                continue
            by_stride = unk_stride_data[cls_name]
            total = sum(len(v['tobj']) for v in by_stride.values())
            row = f'  {cls_name:35s} {total:>5d}'
            for s in sorted_strides:
                if s not in by_stride:
                    row += '  ' + ' ' * 44
                    continue
                to_arr = np.asarray(by_stride[s]['tobj'])
                tu_arr = np.asarray(by_stride[s]['tunk'])
                to_p50 = float(np.percentile(to_arr, 50))
                tu_p50 = float(np.percentile(tu_arr, 50))
                n = len(to_arr)
                row += (f'  [{n:4d}] '
                        f'Tobj {to_arr.mean():.3f}±{to_arr.std():.3f}({to_p50:.3f})  '
                        f'Tunk {tu_arr.mean():.3f}±{tu_arr.std():.3f}({tu_p50:.3f})')
            print(row)

        # aggregate across all unknown classes
        all_tobj_by_stride = defaultdict(list)
        all_tunk_by_stride = defaultdict(list)
        for cls_name in unk_classes:
            if cls_name not in unk_stride_data:
                continue
            for s, vals in unk_stride_data[cls_name].items():
                all_tobj_by_stride[s].extend(vals['tobj'])
                all_tunk_by_stride[s].extend(vals['tunk'])

        print(f'\n  --- AGGREGATE over all {len(unk_classes)} unknown classes ---')
        for s in sorted_strides:
            if s not in all_tobj_by_stride:
                continue
            to_arr = np.asarray(all_tobj_by_stride[s])
            tu_arr = np.asarray(all_tunk_by_stride[s])
            print(f'  stride={s:<2d}  N={len(to_arr):6d}'
                  f'  Tobj: mean={to_arr.mean():.4f} std={to_arr.std():.4f}'
                  f'  p10={np.percentile(to_arr,10):.4f} p25={np.percentile(to_arr,25):.4f}'
                  f'  p50={np.percentile(to_arr,50):.4f} p75={np.percentile(to_arr,75):.4f}'
                  f'  p90={np.percentile(to_arr,90):.4f}'
                  f'  ||  Tunk: mean={tu_arr.mean():.4f} std={tu_arr.std():.4f}'
                  f'  p50={np.percentile(tu_arr,50):.4f} p90={np.percentile(tu_arr,90):.4f}')

        # Save unknown_tobj_by_stride.csv
        unk_stride_csv = os.path.join(out_dir, 'unknown_tobj_by_stride.csv')
        with open(unk_stride_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['class', 'stride', 'n',
                        'tobj_mean', 'tobj_std',
                        'tobj_p10', 'tobj_p25', 'tobj_p50', 'tobj_p75', 'tobj_p90', 'tobj_p95',
                        'tunk_mean', 'tunk_std',
                        'tunk_p10', 'tunk_p25', 'tunk_p50', 'tunk_p75', 'tunk_p90', 'tunk_p95',
                        'tobj_min', 'tobj_max', 'tunk_min', 'tunk_max'])
            for cls_name in unk_classes + ['__ALL_UNKNOWNS__']:
                if cls_name == '__ALL_UNKNOWNS__':
                    by_stride_iter = {s: {'tobj': all_tobj_by_stride[s],
                                          'tunk': all_tunk_by_stride[s]}
                                      for s in sorted_strides if s in all_tobj_by_stride}
                else:
                    if cls_name not in unk_stride_data:
                        continue
                    by_stride_iter = unk_stride_data[cls_name]
                for s in sorted_strides:
                    if s not in by_stride_iter:
                        continue
                    to_arr = np.asarray(by_stride_iter[s]['tobj'])
                    tu_arr = np.asarray(by_stride_iter[s]['tunk'])
                    w.writerow([
                        cls_name, s, len(to_arr),
                        round(float(to_arr.mean()), 6), round(float(to_arr.std()), 6),
                        *[round(float(np.percentile(to_arr, q)), 6) for q in (10,25,50,75,90,95)],
                        round(float(tu_arr.mean()), 6), round(float(tu_arr.std()), 6),
                        *[round(float(np.percentile(tu_arr, q)), 6) for q in (10,25,50,75,90,95)],
                        round(float(to_arr.min()), 6), round(float(to_arr.max()), 6),
                        round(float(tu_arr.min()), 6), round(float(tu_arr.max()), 6),
                    ])
        print(f'[write] {unk_stride_csv}')

        # Persist in JSON
        unk_stride_json = {}
        for cls_name in unk_classes + ['__ALL_UNKNOWNS__']:
            if cls_name == '__ALL_UNKNOWNS__':
                by_s = {s: {'tobj': all_tobj_by_stride[s], 'tunk': all_tunk_by_stride[s]}
                        for s in sorted_strides if s in all_tobj_by_stride}
            else:
                if cls_name not in unk_stride_data:
                    continue
                by_s = unk_stride_data[cls_name]
            unk_stride_json[cls_name] = {}
            for s, vals in by_s.items():
                to_arr = np.asarray(vals['tobj'])
                tu_arr = np.asarray(vals['tunk'])
                unk_stride_json[cls_name][str(s)] = {
                    'n': len(to_arr),
                    'tobj': {**percentiles(to_arr.tolist())},
                    'tunk': {**percentiles(tu_arr.tolist())},
                }
    else:
        unk_stride_json = {}
        print(f'\n[{ctx["label"]}] No unknown GT objects with matched anchors found '
              f'— skipping Tobj stride analysis.')

    # false_positives.csv
    fp_path = os.path.join(out_dir, 'false_positives.csv')
    with open(fp_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['predicted_class', 'count'])
        for cls_name in sorted(false_pos_by_cls.keys(),
                               key=lambda c: -false_pos_by_cls[c]):
            w.writerow([cls_name, false_pos_by_cls[cls_name]])
    print(f'[write] {fp_path}')

    # test_analysis.json
    json_path = os.path.join(out_dir, 'test_analysis.json')
    out = {
        'label':    ctx.get('label', ''),
        'config':   ctx['cfg_path'],
        'checkpoint': ctx['ckpt_path'],
        'dataset':  ctx['dataset_name'],
        'conf_thr': conf_thr,
        'num_images_processed': n_proc,
        'num_images_skipped':   n_skip,
        'base_classes':  ctx['base_classes'],
        'novel_classes': ctx['novel_classes'],
        'model_label_names': model_label_names,
        'gt_counts':     dict(gt_counts),
        'gt_counts_all': dict(gt_counts_all),
        'confusion':     {k: dict(v) for k, v in confusion.items()},
        'false_positives': dict(false_pos_by_cls),
        'per_class_metrics': pc_summary,
        'u_recall_compat': {
            'total': u_total_all,
            'hit':   u_hit_all,
            'recall_pct': round(u_recall_compat, 4),
            'per_class': {
                cls: {
                    'hit': u_recall_hit[cls],
                    'total': u_recall_total[cls],
                    'recall_pct': round(u_recall_hit[cls] / u_recall_total[cls] * 100, 2)
                    if u_recall_total[cls] > 0 else 0.0,
                }
                for cls in u_recall_total
            },
        },
        'unknown_tobj_by_stride': unk_stride_json,
    }
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[write] {json_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    import mmyolo          # noqa: F401
    import yolo_world      # noqa: F401
    init_default_scope('mmyolo')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Resolve runs list ──────────────────────────────────────────────────
    if args.runs_json:
        with open(args.runs_json) as f:
            runs = json.load(f)
        assert runs, f'runs-json {args.runs_json} is empty'
    else:
        assert args.config and args.checkpoint and args.out_dir, \
            'Provide --config, --checkpoint, --out-dir (or --runs-json for multi mode)'
        runs = [{'label': 'analysis',
                 'config': args.config,
                 'checkpoint': args.checkpoint,
                 'out_dir': args.out_dir}]

    print('=' * 80)
    print(f'OWOD TEST-SET ANALYSIS  —  {len(runs)} run(s)')
    for r in runs:
        print(f'  [{r["label"]}]  config={r["config"]}')
        print(f'           ckpt={r["checkpoint"]}')
        print(f'           out ={r["out_dir"]}')
    print('=' * 80)

    # ── Load all model contexts ────────────────────────────────────────────
    print('\n[init] Loading models...')
    ctxs = []
    for r in runs:
        print(f'  [{r["label"]}] {r["config"]}  +  {r["checkpoint"]}')
        ctx = build_model_ctx(r['config'], r['checkpoint'], device)
        ctx['label']   = r['label']
        ctx['out_dir'] = r['out_dir']
        ctxs.append(ctx)
        print(f'    dataset={ctx["dataset_name"]}  known={ctx["known_count"]} '
              f'(prev={ctx["num_prev"]} cur={ctx["num_cur"]})')

    # Use FIRST run's config for dataset/image setup (all runs must share same dataset)
    primary = ctxs[0]
    dataset_name    = primary['dataset_name']
    data_root       = primary['data_root']
    test_image_set  = primary['test_image_set']

    test_id_path = os.path.join(
        data_root, 'ImageSets', dataset_name, f'{test_image_set}.txt')
    with open(test_id_path) as f:
        test_ids = [l.strip() for l in f if l.strip()]

    img_dir = os.path.join(data_root, 'JPEGImages', dataset_name)
    ann_dir = os.path.join(data_root, 'Annotations', dataset_name)

    img_ext = '.jpg'
    for ext in ('.jpg', '.jpeg', '.png'):
        if os.path.exists(os.path.join(img_dir, test_ids[0] + ext)):
            img_ext = ext
            break

    pipeline = build_pipeline(primary)

    n_imgs = len(test_ids) if args.num_images == 0 else min(args.num_images, len(test_ids))
    print(f'\n[data] {n_imgs} test images from {test_id_path}  (ext={img_ext})')
    print(f'       conf_thr={args.conf_thr}  batch_size={args.batch_size}')
    print(f'       Each batch processed through ALL {len(ctxs)} model(s).')

    # ── Per-run accumulators ───────────────────────────────────────────────
    accums = [init_accum() for _ in ctxs]

    # Set module-level sentinels so process_batch_with_ctx can print progress
    global _primary_accum, _n_imgs_total
    _primary_accum = accums[0]
    _n_imgs_total  = n_imgs

    # ── Main loop: load batch → run ALL models → repeat ───────────────────
    batch_buf = []

    def flush_batch(buf):
        if not buf:
            return
        with torch.no_grad():
            for ctx, accum in zip(ctxs, accums):
                process_batch_with_ctx(buf, ctx, accum, device, args.conf_thr)

    for i in range(n_imgs):
        img_id   = test_ids[i]
        img_path = os.path.join(img_dir, img_id + img_ext)
        ann_path = os.path.join(ann_dir, img_id + '.xml')

        if not os.path.exists(img_path):
            for accum in accums:
                accum['n_skip'] += 1
            continue

        try:
            data = pipeline(dict(img_path=img_path, img_id=img_id, instances=[]))
        except Exception as e:
            print(f'  [skip] {img_id}: {e}')
            for accum in accums:
                accum['n_skip'] += 1
            continue

        batch_buf.append((
            img_id,
            data['inputs'].unsqueeze(0),
            data['data_samples'],
            data['data_samples'].metainfo,
            np.asarray(data['data_samples'].metainfo['scale_factor']),
            np.asarray(data['data_samples'].metainfo.get(
                'pad_param', np.zeros(4, dtype=np.float32))),
            ann_path,
        ))

        if len(batch_buf) == args.batch_size:
            flush_batch(batch_buf)
            batch_buf = []

    flush_batch(batch_buf)

    print(f'\n[done] {accums[0]["n_proc"]}/{n_imgs} images processed, '
          f'{accums[0]["n_skip"]} skipped')

    # ── Save results for each run ──────────────────────────────────────────
    print('\n' + '=' * 80)
    print('SAVING RESULTS')
    print('=' * 80)
    for ctx, accum in zip(ctxs, accums):
        print(f'\n── [{ctx["label"]}]  →  {ctx["out_dir"]}')
        save_run_results(accum, ctx, ctx['out_dir'], args.conf_thr)

    print('\n' + '=' * 80)
    print('ALL ANALYSES COMPLETE')
    for ctx in ctxs:
        print(f'  [{ctx["label"]}]  {ctx["out_dir"]}')
    print('=' * 80)


if __name__ == '__main__':
    main()
