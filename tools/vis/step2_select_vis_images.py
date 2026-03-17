"""
Step 2 (one-time): Run per-image inference on GT-candidate images and
select the top-N images best suited for open-world detection visualization.

Scoring per image
-----------------
  +3.0  for each unknown prediction that IoU-matches a novel T2 GT box
         (model correctly detected novel object as "unknown")
  +0.5  for each known-class true-positive (IoU > 0.5 with known GT)
  +0.1  for each extra unknown prediction (may show open-world sensitivity)
  ×penalty if total detections > --max-total-preds  (avoid cluttered images)

The scoring rewards images where the model both (a) fires on novel objects
and (b) correctly detects known objects — making them ideal for side-by-side
comparison with other models.

Usage
-----
    # set required env vars first
    export DATASET=IDD TASK=1 THRESHOLD=0.05

    python tools/vis/step2_select_vis_images.py \\
        --config   configs/owod_ft/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod_idd.py \\
        --checkpoint  work_dirs/.../best_owod_Both_epoch_XX.pth \\
        --candidates  vis_output/candidates.txt \\
        --data-root   data/OWOD \\
        --dataset IDD --task 1 \\
        --top-n 30 \\
        --output  vis_output/selected_images.txt
"""

import os
import sys
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict

import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

# ── Class definitions ─────────────────────────────────────────────────────────
DATASET_SPLITS = {
    'IDD': {
        'T1': ['car', 'motorcycle', 'rider', 'person',
               'autorickshaw', 'bicycle', 'traffic sign', 'traffic light'],
        'T2': ['bus', 'truck', 'tanker_vehicle', 'crane_truck',
               'street_cart', 'excavator'],
        'test': 'test',
    },
    'MOWODB': {
        'T1': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
               'sofa', 'train', 'tvmonitor'],
        'T2': ['truck', 'traffic light', 'fire hydrant', 'stop sign',
               'parking meter', 'bench', 'elephant', 'bear', 'zebra',
               'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator'],
        'test': 'all_task_test',
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description='Step 2: inference-based image selection for visualization')
    p.add_argument('--config',      required=True,
                   help='OWOD config file (e.g. configs/owod_ft/...idd.py)')
    p.add_argument('--checkpoint',  required=True,
                   help='Trained checkpoint path (best_owod_Both_*.pth)')
    p.add_argument('--candidates',  default='vis_output/candidates.txt',
                   help='Output of step1 (one image ID per line)')
    p.add_argument('--use-full-testset', action='store_true',
                   help='Ignore --candidates and process full test image set.')
    p.add_argument('--image-set', default='test',
                   help='Image set stem under ImageSets/{dataset}/ when using full test set.')
    p.add_argument('--data-root',   default='data/OWOD')
    p.add_argument('--dataset',     default='IDD',
                   choices=list(DATASET_SPLITS))
    p.add_argument('--task',        type=int, default=1)
    p.add_argument('--threshold',   type=float, default=0.05,
                   help='Confidence threshold for known-class predictions')
    p.add_argument('--unk-threshold', type=float, default=0.15,
                   help='Confidence threshold for unknown-class predictions')
    p.add_argument('--iou-match',   type=float, default=0.30,
                   help='IoU threshold to match unknown pred → novel GT')
    p.add_argument('--top-n',       type=int, default=30,
                   help='Number of images to select')
    p.add_argument('--target-class', default='',
                   help='Optional GT class to prioritize (e.g. animal).')
    p.add_argument('--min-box-size', type=int, default=0,
                   help='If > 0, GT boxes smaller than this (w/h) are ignored for filtering.')
    p.add_argument('--min-target-count', type=int, default=0,
                   help='Require at least this many large target GT boxes.')
    p.add_argument('--require-novel-gt', action='store_true',
                   help='Require at least one large novel GT box per selected image.')
    p.add_argument('--max-total-preds', type=int, default=25,
                   help='Density cap: images with more total preds are penalised')
    p.add_argument('--output',      default='vis_output/selected_images.txt')
    p.add_argument('--device',      default='cuda:0')
    p.add_argument('--amp',         action='store_true',
                   help='Use automatic mixed precision')
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_iou(box, boxes):
    """IoU of one box [x1,y1,x2,y2] against an (N,4) array."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=float)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    a_box   = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    a_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * \
              np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = a_box + a_boxes - inter
    return inter / (union + 1e-6)


def parse_gt(xml_path, known_classes, novel_classes,
             min_box_size=0, target_class=''):
    """Return known/novel GT and counts after optional min-size filtering."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    known_boxes, novel_boxes = [], []
    target_count = 0
    other_count = 0
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        diff = obj.find('difficult')
        if diff is not None and int(diff.text):
            continue
        bb = obj.find('bndbox')
        box = [int(bb.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        if min_box_size > 0:
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            if bw < min_box_size or bh < min_box_size:
                continue
        if target_class and name == target_class:
            target_count += 1
        if name in known_classes:
            known_boxes.append(box)
        elif name in novel_classes:
            novel_boxes.append((name, box))
        else:
            other_count += 1
    return (np.array(known_boxes, dtype=float).reshape(-1, 4),
            novel_boxes,
            target_count,
            other_count)


def run_inference(model, img_path, test_pipeline, device, use_amp):
    """Return (bboxes (N,4), scores (N,), labels (N,)) as numpy arrays."""
    # texts are ignored by OWODDetector (uses self.embeddings) but
    # PackDetInputs may expect the key to exist.
    n = model.num_test_classes
    texts = [[f'c{i}'] for i in range(n - 1)] + [['unknown']]

    data_info = dict(img_id=0, img_path=img_path, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(
        inputs=data_info['inputs'].unsqueeze(0).to(device),
        data_samples=[data_info['data_samples']],
    )
    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]

    pred = output.pred_instances
    return (pred.bboxes.cpu().numpy(),
            pred.scores.cpu().numpy(),
            pred.labels.cpu().numpy())


def score_image(bboxes, scores, labels,
                known_gt_arr, novel_gt_list,
                threshold, unk_threshold, iou_match,
                max_total, unknown_class_idx):
    """
    Return (score, novel_matched, known_tps) for one image.
    """
    known_mask = (labels < unknown_class_idx) & (scores > threshold)
    unk_mask   = (labels == unknown_class_idx) & (scores > unk_threshold)

    known_preds = bboxes[known_mask]
    unk_preds   = bboxes[unk_mask]
    unk_scores  = scores[unk_mask]

    total = len(known_preds) + len(unk_preds)
    if total == 0:
        return 0.0, 0, 0

    # ── Match unknown predictions to novel GT boxes ───────────────────────────
    novel_gt_arr = np.array([b for _, b in novel_gt_list],
                             dtype=float).reshape(-1, 4)
    novel_matched  = 0
    already_hit    = set()
    for ub in unk_preds:
        ious = compute_iou(ub, novel_gt_arr)
        if len(ious) > 0 and ious.max() > iou_match:
            best = int(ious.argmax())
            if best not in already_hit:
                novel_matched += 1
                already_hit.add(best)

    # ── Known-class TPs ───────────────────────────────────────────────────────
    known_tps = 0
    if len(known_gt_arr) > 0:
        for kb in known_preds:
            ious = compute_iou(kb, known_gt_arr)
            if len(ious) > 0 and ious.max() > 0.5:
                known_tps += 1

    # ── Composite score ───────────────────────────────────────────────────────
    extra_unk = len(unk_preds) - novel_matched
    score = novel_matched * 3.0 + known_tps * 0.5 + extra_unk * 0.1

    # Density penalty
    if total > max_total:
        score *= max(0.35, 1.0 - (total - max_total) * 0.03)

    return score, novel_matched, known_tps


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── Env vars so the mmengine config template resolves correctly ───────────
    os.environ.setdefault('DATASET',   args.dataset)
    os.environ.setdefault('TASK',      str(args.task))
    os.environ.setdefault('THRESHOLD', str(args.threshold))
    os.environ.setdefault('SAVE',      'False')
    os.environ.setdefault('FEWSHOT_DIR', '')
    os.environ.setdefault('FEWSHOT_K', '0')
    os.environ.setdefault('FEWSHOT_SEED', '1')

    splits        = DATASET_SPLITS[args.dataset]
    known_classes = set(splits['T1'])
    novel_classes = set(splits['T2'])

    # Unknown class index:
    #   owod_cfg.num_classes = PREV + CUR + 1  → for IDD T1: 0+8+1 = 9
    #   unknown_idx          = num_classes - 1 → 8
    n_known         = len(splits['T1'])
    unknown_class_idx = n_known  # 8 for IDD T1

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'[Step 2] Loading config:     {args.config}')
    print(f'[Step 2] Loading checkpoint: {args.checkpoint}')
    cfg   = Config.fromfile(args.config)
    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)
    model.eval()

    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline     = Compose(test_pipeline_cfg)

    # ── Read candidate list ───────────────────────────────────────────────────
    if args.use_full_testset:
        image_set_file = os.path.join(
            args.data_root, 'ImageSets', args.dataset, f'{args.image_set}.txt')
        with open(image_set_file) as f:
            candidate_ids = [l.strip() for l in f if l.strip()]
        print(f'[Step 2] Full-test mode: {len(candidate_ids)} images from '
              f'{args.dataset}/{args.image_set}.')
    else:
        with open(args.candidates) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        candidate_ids = [l.split()[0] for l in lines]
        print(f'[Step 2] {len(candidate_ids)} candidates to process ...')
    if args.target_class:
        print(f'[Step 2] Target filter: class={args.target_class}, '
              f'min_target_count={args.min_target_count}, '
              f'min_box_size={args.min_box_size}, '
              f'require_novel_gt={args.require_novel_gt}\n')
    else:
        print()

    anno_dir = os.path.join(args.data_root, 'Annotations', args.dataset)
    img_dir  = os.path.join(args.data_root, 'JPEGImages',  args.dataset)

    records = []

    for i, img_id in enumerate(candidate_ids, 1):
        img_path = os.path.join(img_dir, img_id + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, img_id + '.png')
        xml_path = os.path.join(anno_dir, img_id + '.xml')

        if not os.path.exists(img_path):
            print(f'  [WARN] Image not found:      {img_id}')
            continue
        if not os.path.exists(xml_path):
            print(f'  [WARN] Annotation not found: {img_id}')
            continue

        try:
            known_gt, novel_gt, target_count, other_gt = parse_gt(
                xml_path, known_classes, novel_classes,
                min_box_size=args.min_box_size,
                target_class=args.target_class)

            if args.target_class and target_count < args.min_target_count:
                continue
            if args.require_novel_gt and len(novel_gt) == 0:
                continue

            bboxes, scores, labels = run_inference(
                model, img_path, test_pipeline, args.device, args.amp)
        except Exception as e:
            print(f'  [WARN] {img_id}: {e}')
            continue

        sc, n_novel, n_ktp = score_image(
            bboxes, scores, labels, known_gt, novel_gt,
            args.threshold, args.unk_threshold, args.iou_match,
            args.max_total_preds, unknown_class_idx,
        )

        # Additional boost for target-rich and unknown-rich scenes.
        if args.target_class:
            unk_mask = (labels == unknown_class_idx) & (scores > args.unk_threshold)
            sc += target_count * 4.0
            sc += len(novel_gt) * 0.8
            sc += int(unk_mask.sum()) * 0.15
            sc += other_gt * 0.2

        records.append({
            'img_id':       img_id,
            'score':        sc,
            'novel_matched': n_novel,
            'known_tps':    n_ktp,
            'target_count': target_count,
            'novel_gt':     len(novel_gt),
        })

        if i % 20 == 0:
            print(f'  [{i:>4d}/{len(candidate_ids)}] processed ...')

    records.sort(key=lambda x: -x['score'])
    top = records[:args.top_n]

    print(f'\n[Step 2] Top {len(top)} selected images:')
    print(f'  {"img_id":>14s}  {"score":>6s}  novel_matched  known_tps  target  novel_gt')
    print(f'  {"-"*14}  {"-"*6}  {"-"*13}  {"-"*9}  {"-"*6}  {"-"*8}')
    for r in top:
        print(f"  {r['img_id']:>14s}  {r['score']:6.2f}  "
              f"{r['novel_matched']:>13d}  {r['known_tps']:>9d}  "
              f"{r['target_count']:>6d}  {r['novel_gt']:>8d}")

    with open(args.output, 'w') as f:
        f.write(f'# Dataset: {args.dataset}  Task: {args.task}\n')
        f.write(f'# Checkpoint: {args.checkpoint}\n')
        f.write('# img_id  score  novel_matched  known_tps  target_count  novel_gt\n')
        for r in top:
            f.write(f"{r['img_id']}  {r['score']:.2f}  "
                f"{r['novel_matched']}  {r['known_tps']}  "
                f"{r['target_count']}  {r['novel_gt']}\n")

    print(f'\n[Step 2] Selected images saved to: {args.output}')


if __name__ == '__main__':
    main()
