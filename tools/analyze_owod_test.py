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

  Model pipeline                      What it computes
  ──────────────────────────────────  ──────────────────────────────────────
  extract_feat()                      Image features + text embeddings
  forward_one2one()                   Raw per-anchor logits + bbox deltas
  predict_by_feat()                   Final predictions after NMS + unknown_nms

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

OUTPUTS
  {out-dir}/confusion_matrix.csv     Row = GT class, Col = predicted class
  {out-dir}/false_positives.csv      Unmatched predictions by predicted class
  {out-dir}/test_analysis.json       Per-class metrics + confusion + FP data
"""
import argparse
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
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--num-images', type=int, default=0,
                   help='0 = all test images.')
    p.add_argument('--conf-thr', type=float, default=0.05,
                   help='Score threshold for confusion-matrix predictions.')
    p.add_argument('--batch-size', type=int, default=32,
                   help='Inference batch size (default 32, fits L40 48 GB).')
    return p.parse_args()


def parse_voc_xml_full(xml_path):
    """Parse VOC XML → list[(class_name, [x1, y1, x2, y2], difficult_int)]."""
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
    """IoU between two sets of [x1, y1, x2, y2] boxes."""
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_state_dict
    from mmyolo.registry import MODELS          # noqa: F401
    import mmyolo                               # noqa: F401
    import yolo_world                           # noqa: F401
    init_default_scope('mmyolo')

    # ── Load config ───────────────────────────────────────────────────────
    print(f'[init] config:     {args.config}')
    print(f'[init] checkpoint: {args.checkpoint}')
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.out_dir

    # ── Build + load model ────────────────────────────────────────────────
    model = MODELS.build(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    state_dict = {k: v for k, v in state_dict.items()
                  if 'text_model' not in k}

    if 'embeddings' in state_dict:
        ckpt_emb = state_dict['embeddings']
        if ckpt_emb.shape == model.embeddings.shape:
            with torch.no_grad():
                model.embeddings.data.copy_(ckpt_emb.to(model.embeddings.device))
            print(f'[init] embeddings loaded from ckpt directly '
                  f'(shape={tuple(ckpt_emb.shape)})')
            state_dict = {k: v for k, v in state_dict.items()
                          if k != 'embeddings'}
        else:
            print(f'[warn] embeddings shape mismatch: ckpt={tuple(ckpt_emb.shape)} '
                  f'model={tuple(model.embeddings.shape)} — falling back')
            state_dict['embeddings'] = model.update_embeddings(ckpt_emb)

    load_state_dict(model, state_dict, strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    head = model.bbox_head
    head_module = head.head_module
    # Mirror OWODDetector.predict(): set num_classes on the outer head to
    # num_test_classes so predict_by_feat uses the correct class count.
    head.num_classes = model.num_test_classes
    num_classes = head_module.num_classes
    num_prev = model.num_prev_classes
    num_cur = num_classes - num_prev - 2
    known_count = num_prev + num_cur

    # ── Dataset / class info (from config + owodb_const) ──────────────────
    # Extract resolved dataset name and paths from the config.
    dataset_name = str(cfg.owod_dataset)
    data_root = str(getattr(cfg, 'owod_root', 'data/OWOD'))
    test_image_set = str(getattr(cfg, 'test_image_set', 'test'))

    from yolo_world.datasets.owodb_const import VOC_COCO_CLASS_NAMES
    all_class_names = list(VOC_COCO_CLASS_NAMES[dataset_name])

    base_classes = all_class_names[:num_prev]
    novel_classes = all_class_names[num_prev:num_prev + num_cur]
    base_set, novel_set = set(base_classes), set(novel_classes)

    # Model label index → name  (0..known_count-1 known, known_count = T_unk)
    model_label_names = list(all_class_names[:known_count]) + ['unknown']

    print(f'[init] dataset={dataset_name}  K(known)={known_count} '
          f'(prev={num_prev}, cur={num_cur})')
    print(f'[init] base={base_classes}')
    print(f'[init] novel={novel_classes}')

    # Read test split IDs
    test_id_path = os.path.join(
        data_root, 'ImageSets', dataset_name, f'{test_image_set}.txt')
    with open(test_id_path) as f:
        test_ids = [l.strip() for l in f if l.strip()]

    img_dir = os.path.join(data_root, 'JPEGImages', dataset_name)
    ann_dir = os.path.join(data_root, 'Annotations', dataset_name)

    # Auto-detect image extension from first available test image
    img_ext = '.jpg'
    for ext in ('.jpg', '.jpeg', '.png'):
        if os.path.exists(os.path.join(img_dir, test_ids[0] + ext)):
            img_ext = ext
            break
    print(f'[init] {len(test_ids)} test images  (ext={img_ext})  '
          f'conf_thr={args.conf_thr}')

    # ── Build image pipeline (from val config, strip annotations) ─────────
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
    pipeline = Compose(img_pipeline_cfg)

    # ── Accumulators ──────────────────────────────────────────────────────
    metric_names = ['anchor', 'max_known', 'tunk', 'ratio',
                    'energy', 'entropy',
                    'tunk_raw', 'tanchor_raw', 'correct_cls_score',
                    'correct_cls_rank']
    confusion = defaultdict(lambda: defaultdict(int))  # [gt_cls][pred_cls]
    false_pos_by_cls = defaultdict(int)
    gt_counts = defaultdict(int)          # non-difficult GT only (matches evaluator)
    gt_counts_all = defaultdict(int)      # all GT including difficult (for reference)
    per_cls_metrics = defaultdict(lambda: {m: [] for m in metric_names})

    # U-Recall compatible counters (matches evaluator logic exactly):
    #   numerator = unknown GT boxes (non-difficult) that have ANY
    #               surviving unknown prediction at IoU >= 0.5
    #               (unknown predictions bypass score threshold, same as evaluator)
    #   denominator = non-difficult GT boxes whose class is unknown
    unk_cls_idx = known_count  # label index of 'unknown' in model_label_names
    u_recall_hit = defaultdict(int)    # per-original-class unknown GT boxes recalled
    u_recall_total = defaultdict(int)  # per-original-class unknown GT total (non-diff)

    n_imgs = len(test_ids) if args.num_images == 0 else min(
        args.num_images, len(test_ids))
    print(f'\n[forward] processing {n_imgs} test images  '
          f'(batch_size={args.batch_size}) ...')

    n_proc = 0
    n_skip = 0

    # ── per-batch forward + analysis ──────────────────────────────────────
    # Each entry in buf: (img_id, raw_cpu_tensor[1,C,H,W], ds, meta,
    #                     scale_np, pad_np, ann_path)
    # raw_cpu_tensor is uint8 on CPU; normalisation happens inside here.
    def process_batch(buf):
        nonlocal n_proc
        B = len(buf)

        # Stack into a single GPU batch and normalise
        batch_inputs = torch.cat(
            [entry[1] for entry in buf], dim=0
        ).float().to(device) / 255.0                             # [B, C, H, W]

        data_samples = [entry[2] for entry in buf]
        batch_metas  = [entry[3] for entry in buf]

        # Single batched forward — both backbone and head run once
        # for the whole batch.  forward_one2one uses the one-to-one head
        # only (the one-to-many head is training-only).
        img_feats, txt_feats = model.extract_feat(batch_inputs, data_samples)
        cls_list, bbox_list  = head_module.forward_one2one(img_feats, txt_feats)

        # ── Analysis 1: NMS predictions ───────────────────────────────────
        # predict_by_feat is on the OUTER head (YOLOv10WorldHead).
        # It zeros anchor slot (-1), applies sigmoid + filter_scores_and_topk
        # (multi_label path), decodes bboxes, runs class-specific NMS
        # (IoU=0.7), then _unknown_post_process (IoU≥0.99 suppression).
        # Returns one InstanceData per image (original-image coords because
        # rescale=True).
        results_list = head.predict_by_feat(
            cls_list, bbox_list,
            batch_img_metas=batch_metas,
            rescale=True,
            with_nms=True)                                       # list[B]

        # ── Analysis 2: shared anchor priors (same spatial layout all imgs) ─
        featmap_sizes = [(t.shape[-2], t.shape[-1]) for t in cls_list]
        mlvl_priors   = head.prior_generator.grid_priors(
            featmap_sizes, dtype=batch_inputs.dtype,
            device=device, with_stride=True)
        flat_priors   = torch.cat(mlvl_priors, dim=0)           # [N_anc, 4]

        # Batched logits and bbox deltas — index [b] per image below
        flat_logits_b = torch.cat([
            t.permute(0, 2, 3, 1).reshape(B, -1, num_classes)
            for t in cls_list], dim=1)                           # [B, N_anc, C]
        flat_bbox_b   = torch.cat([
            t.permute(0, 2, 3, 1).reshape(B, -1, 4)
            for t in bbox_list], dim=1)                          # [B, N_anc, 4]

        for b, (img_id, _, ds, meta, scale, pad, ann_path) in enumerate(buf):
            # ── Confusion matrix ──────────────────────────────────────────
            pred       = results_list[b]
            keep_mask  = pred.scores >= args.conf_thr
            pred_bboxes = pred.bboxes[keep_mask]                 # orig-img coords
            pred_scores = pred.scores[keep_mask]
            pred_labels = pred.labels[keep_mask]

            gt_full   = (parse_voc_xml_full(ann_path)
                         if os.path.exists(ann_path) else [])
            gt_names  = [g[0] for g in gt_full]
            gt_difficult = [g[2] for g in gt_full]  # 0 or 1
            for gn, diff in zip(gt_names, gt_difficult):
                gt_counts_all[gn] += 1
                if not diff:
                    gt_counts[gn] += 1  # only non-difficult, matches voc_eval npos

            # Non-difficult GT boxes only for confusion matrix (matches evaluator)
            gt_boxes_nd = (
                torch.tensor([g[1] for g, d in zip(gt_full, gt_difficult) if not d],
                             dtype=torch.float32, device=device)
                if any(not d for d in gt_difficult)
                else torch.zeros((0, 4), device=device))
            gt_names_nd = [g[0] for g, d in zip(gt_full, gt_difficult) if not d]

            # All GT boxes (including difficult) for U-Recall — evaluator
            # also computes unknown_class_recs from all GT mapped to 'unknown'
            # but excludes difficult ones from being recalled (det flag).
            # We pre-separate for clarity.
            gt_boxes_orig = (
                torch.tensor([g[1] for g in gt_full],
                             dtype=torch.float32, device=device)
                if gt_full else torch.zeros((0, 4), device=device))

            # ── Confusion matrix on non-difficult GT ──────────────────────────────
            # Greedy match: highest-scoring prediction (above conf_thr) → GT.
            # Uses only non-difficult GT boxes, matching evaluator's npos denominator.
            if len(pred_bboxes) > 0 and gt_boxes_nd.shape[0] > 0:
                iou_pg      = box_iou_xyxy(pred_bboxes, gt_boxes_nd)
                matched_gt  = set()
                matched_pred = set()
                for pi in pred_scores.argsort(descending=True).tolist():
                    best_iou, best_gi = 0.0, -1
                    for gi in range(len(gt_names_nd)):
                        if gi in matched_gt:
                            continue
                        v = iou_pg[pi, gi].item()
                        if v >= 0.5 and v > best_iou:
                            best_iou, best_gi = v, gi
                    if best_gi >= 0:
                        lbl      = pred_labels[pi].item()
                        pred_cls = (model_label_names[lbl]
                                    if lbl < len(model_label_names)
                                    else f'cls_{lbl}')
                        confusion[gt_names_nd[best_gi]][pred_cls] += 1
                        matched_gt.add(best_gi)
                        matched_pred.add(pi)
                for gi in range(len(gt_names_nd)):
                    if gi not in matched_gt:
                        confusion[gt_names_nd[gi]]['missed'] += 1
                for pi in range(len(pred_bboxes)):
                    if pi not in matched_pred:
                        lbl      = pred_labels[pi].item()
                        pred_cls = (model_label_names[lbl]
                                    if lbl < len(model_label_names)
                                    else f'cls_{lbl}')
                        false_pos_by_cls[pred_cls] += 1
            elif gt_names_nd:
                for gn in gt_names_nd:
                    confusion[gn]['missed'] += 1
            else:
                for pi in range(len(pred_bboxes)):
                    lbl      = pred_labels[pi].item()
                    pred_cls = (model_label_names[lbl]
                                if lbl < len(model_label_names)
                                else f'cls_{lbl}')
                    false_pos_by_cls[pred_cls] += 1

            # ── U-Recall compatible counter ────────────────────────────────────────
            # Mirrors evaluator: for each non-difficult unknown GT box, check if
            # ANY surviving unknown prediction (no score threshold — evaluator keeps
            # all unknown preds regardless of score) has IoU >= 0.5.
            # This is separate from the top-1 confusion matrix assignment.
            unk_pred_mask = pred.labels == unk_cls_idx          # all unknown preds
            unk_pred_bboxes = pred.bboxes[unk_pred_mask]        # no conf_thr filter!

            for gi, (gname, gdiff) in enumerate(zip(gt_names, gt_difficult)):
                if gdiff:
                    continue  # skip difficult, same as evaluator
                role = class_role(gname, base_set, novel_set)
                if role != 'unknown':
                    continue  # only track unknown-class GT
                u_recall_total[gname] += 1
                if unk_pred_bboxes.shape[0] > 0:
                    gt_box = gt_boxes_orig[gi:gi + 1]
                    ious_unk = box_iou_xyxy(unk_pred_bboxes, gt_box)
                    if ious_unk.max().item() >= 0.5:
                        u_recall_hit[gname] += 1

            # ── Per-class anchor metrics ──────────────────────────────────────────
            # Uses non-difficult GT only for consistency with confusion matrix.
            if gt_boxes_nd.shape[0] > 0:
                sx, sy         = float(scale[0]), float(scale[1])
                pad_top, pad_left = float(pad[0]), float(pad[2])
                gt_pad         = gt_boxes_nd.clone()
                gt_pad[:, [0, 2]] = gt_pad[:, [0, 2]] * sx + pad_left
                gt_pad[:, [1, 3]] = gt_pad[:, [1, 3]] * sy + pad_top

                # Per-image slice of the batched tensors
                flat_logits  = flat_logits_b[b]                  # [N_anc, C]
                anchor_boxes = head.bbox_coder.decode(
                    flat_priors[..., :2],
                    flat_bbox_b[b:b + 1],                        # [1, N_anc, 4]
                    flat_priors[:, [2]][..., 0])[0]              # [N_anc, 4]

                anchor_scores  = flat_logits[:, -1].sigmoid()
                tunk_scores    = flat_logits[:, -2].sigmoid()
                known_logits   = flat_logits[:, :-2]
                known_scores   = known_logits.sigmoid()
                max_known_vals, _ = known_scores.max(dim=1)
                ratio_vals     = max_known_vals / anchor_scores.clamp(min=1e-6)
                energy_vals    = -torch.logsumexp(known_logits, dim=1)
                log_sm         = torch.log_softmax(known_logits, dim=1)
                entropy_vals   = -(log_sm.exp() * log_sm).sum(dim=1)
                # all K+2 sigmoid scores for rank computation
                all_scores     = flat_logits.sigmoid()           # [N_anc, K+2]

                iou_ag = box_iou_xyxy(anchor_boxes, gt_pad)

                a_np   = anchor_scores.cpu().numpy()
                mk_np  = max_known_vals.cpu().numpy()
                tu_np  = tunk_scores.cpu().numpy()
                rt_np  = ratio_vals.cpu().numpy()
                en_np  = energy_vals.cpu().numpy()
                et_np  = entropy_vals.cpu().numpy()
                tunk_raw_np    = flat_logits[:, -2].cpu().numpy()
                tanchor_raw_np = flat_logits[:, -1].cpu().numpy()
                all_scores_np  = all_scores.cpu().numpy()        # [N_anc, K+2]

                # per-class logit distributions: store per-known-class raw logit
                # keyed by class name so we can compute mean±std per GT class
                known_logits_np = known_logits.cpu().numpy()     # [N_anc, K]

                for gi in range(len(gt_names_nd)):
                    best_iou_val, best_idx = iou_ag[:, gi].max(dim=0)
                    if best_iou_val.item() >= 0.5:
                        idx   = best_idx.item()
                        gname = gt_names_nd[gi]
                        role  = class_role(gname, base_set, novel_set)
                        gt_lbl = (all_class_names.index(gname)
                                  if gname in all_class_names else -1)

                        # correct_score: what the model *should* fire for this GT
                        #   base/novel  → sigmoid of that class's known channel
                        #   unknown     → T_unk sigmoid (channel K, index -2)
                        if role == 'unknown':
                            correct_score = float(tu_np[idx])
                            correct_raw   = float(tunk_raw_np[idx])
                            # rank of T_unk channel (index known_count) among all K+2
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
                                float(tunk_raw_np[idx]),
                                float(tanchor_raw_np[idx]),
                                correct_score, float(correct_rank))
                        for k, v in zip(metric_names, vals):
                            per_cls_metrics[gname][k].append(float(v))
                        per_cls_metrics[gname].setdefault('_correct_raw', []).append(correct_raw)
                        # per-channel raw logit for overlap analysis
                        per_cls_metrics[gname].setdefault('_raw_logits_by_channel', []).append(
                            known_logits_np[idx].tolist())

            n_proc += 1
            if n_proc % 500 == 0:
                print(f'  processed {n_proc}/{n_imgs}  (skipped {n_skip})')

    # ── Main loop: accumulate into batches, flush when full ───────────────
    batch_buf = []
    with torch.no_grad():
        for i in range(n_imgs):
            img_id   = test_ids[i]
            img_path = os.path.join(img_dir, img_id + img_ext)
            ann_path = os.path.join(ann_dir, img_id + '.xml')

            if not os.path.exists(img_path):
                n_skip += 1
                continue

            try:
                data = pipeline(dict(img_path=img_path, img_id=img_id,
                                     instances=[]))
            except Exception as e:
                print(f'  [skip] {img_id}: {e}')
                n_skip += 1
                continue

            # Keep on CPU until we stack the batch — avoids per-image
            # .to(device) overhead and lets the GPU do one big transfer.
            batch_buf.append((
                img_id,
                data['inputs'].unsqueeze(0),                     # [1,C,H,W] uint8 CPU
                data['data_samples'],
                data['data_samples'].metainfo,
                np.asarray(data['data_samples'].metainfo['scale_factor']),
                np.asarray(data['data_samples'].metainfo.get(
                    'pad_param', np.zeros(4, dtype=np.float32))),
                ann_path,
            ))

            if len(batch_buf) == args.batch_size:
                process_batch(batch_buf)
                batch_buf = []

        if batch_buf:                                            # flush remainder
            process_batch(batch_buf)
            batch_buf = []

    print(f'  processed {n_proc}/{n_imgs}  (skipped {n_skip})')
    print(f'\n[done] {n_proc} images processed, {n_skip} skipped')

    # ══════════════════════════════════════════════════════════════════════
    # Output
    # ══════════════════════════════════════════════════════════════════════

    # ── U-Recall compatible summary ───────────────────────────────────────
    # Matches evaluator logic: non-difficult unknown GT, any unknown prediction
    # (no score threshold), IoU >= 0.5.
    u_total_all = sum(u_recall_total.values())
    u_hit_all   = sum(u_recall_hit.values())
    u_recall_compat = u_hit_all / u_total_all * 100 if u_total_all > 0 else 0.0
    print('\n' + '=' * 80)
    print('U-RECALL COMPATIBLE  (any unknown pred, no score thr, non-difficult GT)')
    print(f'  Should match evaluator U-Recall (expected ~79% for VOC-COCO wapr)')
    print('=' * 80)
    for cls in sorted(u_recall_total.keys(), key=lambda x: -u_recall_total[x]):
        n, d = u_recall_total[cls], u_recall_hit[cls]
        r = d / n * 100 if n > 0 else 0.0
        print(f'  {cls:30s}: {d:5d}/{n:5d} = {r:6.2f}%')
    print(f'  {"-" * 44}')
    print(f'  {"TOTAL":30s}: {u_hit_all:5d}/{u_total_all:5d} = {u_recall_compat:6.2f}%')
    print('=' * 80)

    # Order GT classes: base → novel → unknown, alphabetical within role
    all_gt_cls = sorted(gt_counts.keys(), key=lambda c: (
        0 if c in base_set else 1 if c in novel_set else 2, c))

    # Columns for confusion matrix: model class names + "missed"
    pred_cols = list(model_label_names) + ['missed']

    # ── Print confusion matrix summary ────────────────────────────────────
    print('\n' + '=' * 80)
    print('CONFUSION MATRIX  (top-5 predicted per GT class)')
    print('=' * 80)
    for gt_cls in all_gt_cls:
        role = class_role(gt_cls, base_set, novel_set)
        total = gt_counts[gt_cls]
        entries = sorted(confusion[gt_cls].items(), key=lambda x: -x[1])
        top = entries[:5]
        top_str = ', '.join(f'{c}:{n}' for c, n in top)
        print(f'  {gt_cls:30s} ({role:7s}) N={total:6d}  {top_str}')

    # ── Write confusion_matrix.csv ────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, 'confusion_matrix.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gt_class', 'role', 'total_gt'] + pred_cols)
        for gt_cls in all_gt_cls:
            role = class_role(gt_cls, base_set, novel_set)
            row = [gt_cls, role, gt_counts[gt_cls]]
            for pc in pred_cols:
                row.append(confusion[gt_cls].get(pc, 0))
            w.writerow(row)
    print(f'\n[write] {csv_path}')

    # ── Print per-class anchor metrics ────────────────────────────────────
    print('\n' + '=' * 80)
    print('PER-CLASS ANCHOR METRICS  (best anchor per GT, IoU >= 0.5)')
    print('=' * 80)
    pc_summary = {}
    for cls_name in all_gt_cls:
        d = per_cls_metrics[cls_name]
        n = len(d['anchor'])
        if n == 0:
            continue
        role = class_role(cls_name, base_set, novel_set)
        s = {m: percentiles(d[m]) for m in metric_names}
        # per-channel raw logit mean±std across all known classes
        ch_stats = {}
        raw_mat = d.get('_raw_logits_by_channel', [])
        if raw_mat:
            arr = np.asarray(raw_mat)          # [N_gt, K]
            for ci, cname in enumerate(all_class_names[:known_count]):
                ch_stats[cname] = {
                    'mean': float(arr[:, ci].mean()),
                    'std':  float(arr[:, ci].std()),
                }
        pc_summary[cls_name] = {
            'role': role, 'count': n, 'stats': s,
            'per_channel_logit': ch_stats,
        }
        print(f'  {cls_name:30s} ({role:7s}) N={n:5d} '
              f'anchor[μ={s["anchor"]["mean"]:.3f} σ={s["anchor"]["std"]:.3f}] '
              f'maxK[μ={s["max_known"]["mean"]:.3f}] '
              f'Tunk[μ={s["tunk"]["mean"]:.3f}] tunk_raw[μ={s["tunk_raw"]["mean"]:.2f}] '
              f'tanchor_raw[μ={s["tanchor_raw"]["mean"]:.2f}] '
              f'correct_score[μ={s["correct_cls_score"]["mean"]:.3f}] '
              f'rank[μ={s["correct_cls_rank"]["mean"]:.1f} '
              f'p50={s["correct_cls_rank"]["p50"]:.0f}] '
              f'ratio[μ={s["ratio"]["mean"]:.3f}] '
              f'energy[μ={s["energy"]["mean"]:.3f}]')
        # store correct_raw in summary too
        cr = d.get('_correct_raw', [])
        if cr:
            pc_summary[cls_name]['correct_raw_stats'] = percentiles(cr)

    # ── Print ROLE-CORRECT SCORE TABLE ───────────────────────────────────
    # base/novel → correct class channel score + raw logit
    # unknown    → T_unk score + raw logit
    # This directly answers: how well does the model score the RIGHT channel?
    print('\n' + '=' * 80)
    print('ROLE-CORRECT SCORE TABLE')
    print('  base/novel: correct_class_channel sigmoid + raw_logit')
    print('  unknown:    T_unk sigmoid + raw_logit')
    print(f'  {"class":30s} {"role":7s} {"N":>6s}  '
          f'{"correct_sig_μ":>13s}  {"correct_sig_p50":>15s}  '
          f'{"correct_raw_μ":>13s}  {"rank_μ":>6s}  {"rank_p50":>8s}')
    print('=' * 80)
    for cls_name in all_gt_cls:
        if cls_name not in pc_summary:
            continue
        ps = pc_summary[cls_name]
        s  = ps['stats']
        cr = ps.get('correct_raw_stats', {})
        role = ps['role']
        n    = ps['count']
        sig_mean  = s['correct_cls_score']['mean']
        sig_p50   = s['correct_cls_score']['p50']
        raw_mean  = cr.get('mean', float('nan'))
        rank_mean = s['correct_cls_rank']['mean']
        rank_p50  = s['correct_cls_rank']['p50']
        print(f'  {cls_name:30s} {role:7s} {n:6d}  '
              f'{sig_mean:13.4f}  {sig_p50:15.4f}  '
              f'{raw_mean:13.2f}  {rank_mean:6.1f}  {rank_p50:8.0f}')

    # ── Print false positives ─────────────────────────────────────────────
    print('\n' + '=' * 80)
    print('FALSE POSITIVES  (predictions with no GT match, by predicted class)')
    print('=' * 80)
    for cls_name in sorted(false_pos_by_cls.keys(),
                           key=lambda c: -false_pos_by_cls[c]):
        print(f'  {cls_name:30s} {false_pos_by_cls[cls_name]:6d}')

    # ── Write false_positives.csv ─────────────────────────────────────────
    fp_path = os.path.join(args.out_dir, 'false_positives.csv')
    with open(fp_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['predicted_class', 'count'])
        for cls_name in sorted(false_pos_by_cls.keys(),
                               key=lambda c: -false_pos_by_cls[c]):
            w.writerow([cls_name, false_pos_by_cls[cls_name]])
    print(f'[write] {fp_path}')

    # ── Write test_analysis.json ──────────────────────────────────────────
    json_path = os.path.join(args.out_dir, 'test_analysis.json')
    out = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'dataset': dataset_name,
        'conf_thr': args.conf_thr,
        'num_images_processed': n_proc,
        'num_images_skipped': n_skip,
        'base_classes': base_classes,
        'novel_classes': novel_classes,
        'model_label_names': model_label_names,
        'gt_counts': dict(gt_counts),             # non-difficult only
        'gt_counts_all': dict(gt_counts_all),     # including difficult
        'confusion': {k: dict(v) for k, v in confusion.items()},
        'false_positives': dict(false_pos_by_cls),
        'per_class_metrics': pc_summary,
        'u_recall_compat': {
            'total': u_total_all,
            'hit': u_hit_all,
            'recall_pct': round(u_recall_compat, 4),
            'per_class': {
                cls: {'hit': u_recall_hit[cls], 'total': u_recall_total[cls],
                      'recall_pct': round(u_recall_hit[cls] / u_recall_total[cls] * 100, 2)
                      if u_recall_total[cls] > 0 else 0.0}
                for cls in u_recall_total
            },
        },
    }
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[write] {json_path}')


if __name__ == '__main__':
    main()
