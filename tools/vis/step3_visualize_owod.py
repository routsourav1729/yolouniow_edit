"""
Step 3 (one-time): Generate color-coded open-world detection visualizations.

Color scheme (predictions)
--------------------------
  GREEN  (gradient, one shade per known class)
      → known T1 class predictions (model correctly labels them)
  ORANGE (gradient, one shade per novel class)
      → unknown predictions that spatially overlap (IoU ≥ --iou-match) with
        a novel T2 GT box  →  model found the novel object as "unknown"
  RED
      → unknown predictions with no novel GT overlap  →  spurious / truly
        open-world unknowns

GT ground-truth boxes (dashed outlines, drawn for reference)
  - Dashed bright-green  → known GT objects
  - Dashed bright-orange → novel GT objects

Outputs
-------
  {output_dir}/{img_id}.jpg     — annotated image for each selected image
  {output_dir}/image_list.txt   — plain list of saved filenames +
                                   metadata (checkpoint, colours, etc.)

Usage
-----
    export DATASET=IDD TASK=1 THRESHOLD=0.05

    python tools/vis/step3_visualize_owod.py \\
        --config   configs/owod_ft/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod_idd.py \\
        --checkpoint  work_dirs/.../best_owod_Both_epoch_XX.pth \\
        --selected-images  vis_output/selected_images.txt \\
        --data-root   data/OWOD \\
        --dataset IDD --task 1 \\
        --output-dir  visualizations/idd_t1_vis
"""

import os
import sys
import argparse
import numpy as np
import cv2
import xml.etree.ElementTree as ET

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
    },
}

# ── Colour palettes (BGR for OpenCV) ─────────────────────────────────────────
# Ten green shades — one per known class slot (up to 20 classes)
KNOWN_COLORS_BGR = [
    (0,   210,  30),   # vivid green
    (40,  215,  60),   # lime green
    (0,   185, 110),   # spring green
    (100, 205,  40),   # yellow-green
    (0,   160, 130),   # cyan-green
    (55,  200,  90),   # medium sea green
    (30,  175,  30),   # forest green
    (110, 220,  70),   # yellow-chartreuse
    (0,   215, 155),   # medium aquamarine
    (80,  200,   0),   # olive-green
    (0,   230,  80),   # emerald
    (60,  195, 120),   # jade
    (20,  170,  60),   # dark spring green
    (140, 225,  90),   # light yellow-green
    (0,   190, 170),   # teal
    (70,  210, 130),   # aqua-green
    (30,  155,  50),   # deep green
    (120, 215, 100),   # pastel green
    (0,   175,  90),   # malachite
    (50,  185, 140),   # medium teal
]

# Six orange shades — one per novel class slot
NOVEL_COLORS_BGR = [
    (0,  140, 255),   # orange
    (0,  165, 255),   # standard orange
    (0,  120, 220),   # dark orange
    (20, 155, 255),   # light orange
    (0,  180, 255),   # golden orange
    (10, 130, 240),   # deep orange
]

# Red for pure-unknown predictions
PURE_UNK_COLOR_BGR = (50, 50, 220)   # red in BGR

# Dashed-outline colours for GT reference boxes
GT_KNOWN_COLOR_BGR = (0, 255,   0)   # bright green
GT_NOVEL_COLOR_BGR = (0, 165, 255)   # orange


# ── Argument parser ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Step 3: colour-coded OWOD visualization')
    p.add_argument('--config',          required=True)
    p.add_argument('--checkpoint',      required=True)
    p.add_argument('--selected-images', default='vis_output/selected_images.txt')
    p.add_argument('--data-root',       default='data/OWOD')
    p.add_argument('--dataset',         default='IDD',
                   choices=list(DATASET_SPLITS))
    p.add_argument('--task',            type=int, default=1)
    p.add_argument('--threshold',       type=float, default=0.05,
                   help='Known-class confidence threshold')
    p.add_argument('--unk-threshold',   type=float, default=0.15,
                   help='Unknown-class confidence threshold')
    p.add_argument('--iou-match',       type=float, default=0.30,
                   help='IoU to match unknown pred → novel GT')
    p.add_argument('--output-dir',      default='visualizations/idd_t1_vis')
    p.add_argument('--no-gt',           action='store_true',
                   help='Omit dashed GT reference boxes')
    p.add_argument('--no-legend',       action='store_true',
                   help='Omit the legend overlay')
    p.add_argument('--box-thickness',   type=int,   default=2)
    p.add_argument('--font-scale',      type=float, default=0.42)
    p.add_argument('--device',          default='cuda:0')
    p.add_argument('--amp',             action='store_true')
    return p.parse_args()


# ── Drawing utilities ─────────────────────────────────────────────────────────

def draw_dashed_rect(img, x1, y1, x2, y2, color, thickness=1, dash=9):
    """Draw a dashed-line rectangle (GT reference)."""
    for ex in [x1, x2]:
        y, draw = y1, True
        while y < y2:
            ye = min(y + dash, y2)
            if draw:
                cv2.line(img, (ex, y), (ex, ye), color, thickness)
            y, draw = ye, not draw
    for ey in [y1, y2]:
        x, draw = x1, True
        while x < x2:
            xe = min(x + dash, x2)
            if draw:
                cv2.line(img, (x, ey), (xe, ey), color, thickness)
            x, draw = xe, not draw


def draw_pred_box(img, box, label_text, color, thickness, font_scale):
    """Solid rectangle + semi-transparent fill + label banner above box."""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    # Semi-transparent fill
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

    # Solid border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Label
    if label_text:
        (tw, th), base = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        lby = max(y1 - 2, th + base + 3)
        cv2.rectangle(img,
                      (x1, lby - th - base - 2),
                      (x1 + tw + 5, lby + 2),
                      color, -1)
        cv2.putText(img, label_text,
                    (x1 + 2, lby),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)


def draw_legend(img, known_classes, novel_classes, font_scale=0.38):
    """Render a compact colour-key in the bottom-right corner."""
    items = (
        [(cls, KNOWN_COLORS_BGR[i % len(KNOWN_COLORS_BGR)], 'known')
         for i, cls in enumerate(known_classes)] +
        [(cls, NOVEL_COLORS_BGR[i % len(NOVEL_COLORS_BGR)], 'novel')
         for i, cls in enumerate(novel_classes)] +
        [('unknown (no GT match)', PURE_UNK_COLOR_BGR, 'unk')]
    )

    pad, bw, bh = 5, 12, 12
    line_h = 17
    max_tw  = max(
        cv2.getTextSize(n, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        for n, _, _ in items
    )
    leg_w = pad + bw + 5 + max_tw + pad
    leg_h = pad + len(items) * line_h + pad

    H, W  = img.shape[:2]
    x0    = max(W - leg_w - 8, 0)
    y0    = max(H - leg_h - 8, 0)

    # Dark semi-transparent background
    ov = img.copy()
    cv2.rectangle(ov, (x0, y0), (x0 + leg_w, y0 + leg_h), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.72, img, 0.28, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + leg_w, y0 + leg_h), (130, 130, 130), 1)

    for idx, (name, color, _) in enumerate(items):
        y = y0 + pad + idx * line_h
        # Colour swatch
        cv2.rectangle(img, (x0 + pad, y + 1),
                      (x0 + pad + bw, y + bh + 1), color, -1)
        cv2.rectangle(img, (x0 + pad, y + 1),
                      (x0 + pad + bw, y + bh + 1), (180, 180, 180), 1)
        # Text
        cv2.putText(img, name,
                    (x0 + pad + bw + 4, y + bh),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (230, 230, 230), 1, cv2.LINE_AA)


# ── Inference helper ──────────────────────────────────────────────────────────

def run_inference(model, img_path, test_pipeline, device, use_amp):
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


def compute_iou(box, boxes):
    if len(boxes) == 0:
        return np.zeros(0, dtype=float)
    x1 = np.maximum(box[0], boxes[:, 0]);  y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]);  y2 = np.minimum(box[3], boxes[:, 3])
    inter  = np.maximum(0., x2-x1) * np.maximum(0., y2-y1)
    a_box  = max(0., (box[2]-box[0]) * (box[3]-box[1]))
    a_rest = np.maximum(0., boxes[:,2]-boxes[:,0]) * np.maximum(0., boxes[:,3]-boxes[:,1])
    return inter / (a_box + a_rest - inter + 1e-6)


def parse_gt_boxes(xml_path, known_classes, novel_classes):
    """Return (known [(name,box)], novel [(name,box)], other [(name,box)])."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    known, novel, other = [], [], []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        diff = obj.find('difficult')
        if diff is not None and int(diff.text):
            continue
        bb   = obj.find('bndbox')
        box  = [int(bb.find(k).text) for k in ('xmin','ymin','xmax','ymax')]
        if name in known_classes:
            known.append((name, box))
        elif name in novel_classes:
            novel.append((name, box))
        else:
            other.append((name, box))
    return known, novel, other


# ── Core visualize-one-image ──────────────────────────────────────────────────

def visualize_image(img_path, xml_path,
                    model, test_pipeline, device, use_amp,
                    known_classes, novel_classes,
                    threshold, unk_threshold, iou_match,
                    box_thickness, font_scale,
                    draw_gt, draw_legend_flag):
    """
    Returns (annotated_bgr_image, stats_dict).
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {img_path}')

    known_set  = set(known_classes)
    novel_set  = set(novel_classes)

    # GT boxes
    known_gt, novel_gt, _ = parse_gt_boxes(xml_path, known_set, novel_set)
    novel_gt_arr   = np.array([b for _, b in novel_gt],
                               dtype=float).reshape(-1, 4)
    novel_gt_names = [n for n, _ in novel_gt]

    # Inference
    bboxes, scores, labels = run_inference(
        model, img_path, test_pipeline, device, use_amp)

    n_known       = len(known_classes)
    unknown_idx   = n_known              # IDD T1: 8 → unknown_idx = 8

    # Classify predictions
    known_preds, novel_preds, pure_unk_preds = [], [], []
    novel_pred_names = []
    matched_novel_gt = set()

    for box, score, label in zip(bboxes, scores, labels):
        if label > unknown_idx:          # anchor class — skip
            continue
        if label == unknown_idx:
            if score < unk_threshold:
                continue
            ious = compute_iou(box, novel_gt_arr)
            if len(ious) > 0 and ious.max() > iou_match:
                best = int(ious.argmax())
                novel_preds.append((box, score))
                novel_pred_names.append(novel_gt_names[best])
                matched_novel_gt.add(best)
            else:
                pure_unk_preds.append((box, score))
        else:
            if score < threshold:
                continue
            known_preds.append((box, score, int(label)))

    # ── 1. GT dashed outlines ─────────────────────────────────────────────────
    if draw_gt:
        for _, box in known_gt:
            draw_dashed_rect(img, *box, GT_KNOWN_COLOR_BGR, 1)
        for _, box in novel_gt:
            draw_dashed_rect(img, *box, GT_NOVEL_COLOR_BGR, 1)

    # ── 2. Prediction boxes (back-to-front: known → novel → pure-unknown) ─────
    for box, score, label_idx in known_preds:
        cls_name = known_classes[label_idx]
        color    = KNOWN_COLORS_BGR[label_idx % len(KNOWN_COLORS_BGR)]
        draw_pred_box(img, box, f'{cls_name} {score:.2f}',
                      color, box_thickness, font_scale)

    for (box, score), matched_name in zip(novel_preds, novel_pred_names):
        ni    = novel_classes.index(matched_name) \
                if matched_name in novel_classes else 0
        color = NOVEL_COLORS_BGR[ni % len(NOVEL_COLORS_BGR)]
        draw_pred_box(img, box, f'[{matched_name}] {score:.2f}',
                      color, box_thickness, font_scale)

    for box, score in pure_unk_preds:
        draw_pred_box(img, box, f'unknown {score:.2f}',
                      PURE_UNK_COLOR_BGR, box_thickness, font_scale)

    # ── 3. Legend ─────────────────────────────────────────────────────────────
    if draw_legend_flag:
        draw_legend(img, known_classes, novel_classes, font_scale)

    stats = {
        'n_known_pred':    len(known_preds),
        'n_novel_matched': len(novel_preds),
        'n_pure_unk':      len(pure_unk_preds),
        'n_novel_gt':      len(novel_gt),
        'n_known_gt':      len(known_gt),
    }
    return img, stats


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Ensure mmengine config templates always resolve in batch/local runs.
    os.environ.setdefault('DATASET', args.dataset)
    os.environ.setdefault('TASK', str(args.task))
    os.environ.setdefault('THRESHOLD', str(args.threshold))
    os.environ.setdefault('SAVE', 'False')
    os.environ.setdefault('FEWSHOT_DIR', '')
    os.environ.setdefault('FEWSHOT_K', '0')
    os.environ.setdefault('FEWSHOT_SEED', '1')

    os.makedirs(args.output_dir, exist_ok=True)

    # Env vars for mmengine config interpolation
    os.environ.setdefault('DATASET',   args.dataset)
    os.environ.setdefault('TASK',      str(args.task))
    os.environ.setdefault('THRESHOLD', str(args.threshold))
    os.environ.setdefault('SAVE',      'False')

    splits        = DATASET_SPLITS[args.dataset]
    known_classes = splits['T1']
    novel_classes = splits['T2']

    # Load model
    print(f'[Step 3] Loading config:     {args.config}')
    print(f'[Step 3] Loading checkpoint: {args.checkpoint}')
    cfg   = Config.fromfile(args.config)
    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)
    model.eval()

    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline     = Compose(test_pipeline_cfg)

    # Read selected images
    with open(args.selected_images) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    image_ids = [l.split()[0] for l in lines]
    print(f'[Step 3] Visualizing {len(image_ids)} images ...\n')

    anno_dir = os.path.join(args.data_root, 'Annotations', args.dataset)
    img_dir  = os.path.join(args.data_root, 'JPEGImages',  args.dataset)

    saved_names = []

    for i, img_id in enumerate(image_ids, 1):
        img_path = os.path.join(img_dir, img_id + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, img_id + '.png')
        xml_path = os.path.join(anno_dir, img_id + '.xml')

        if not os.path.exists(img_path):
            print(f'  [WARN] Image not found: {img_id}')
            continue
        if not os.path.exists(xml_path):
            print(f'  [WARN] Annotation missing: {img_id}')
            continue

        try:
            annotated, stats = visualize_image(
                img_path=img_path,
                xml_path=xml_path,
                model=model,
                test_pipeline=test_pipeline,
                device=args.device,
                use_amp=args.amp,
                known_classes=known_classes,
                novel_classes=novel_classes,
                threshold=args.threshold,
                unk_threshold=args.unk_threshold,
                iou_match=args.iou_match,
                box_thickness=args.box_thickness,
                font_scale=args.font_scale,
                draw_gt=not args.no_gt,
                draw_legend_flag=not args.no_legend,
            )
        except Exception as e:
            import traceback
            print(f'  [ERROR] {img_id}: {e}')
            traceback.print_exc()
            continue

        out_name = f'{img_id}.jpg'
        cv2.imwrite(os.path.join(args.output_dir, out_name), annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_names.append(out_name)

        print(f'  [{i:>3d}/{len(image_ids)}] {img_id:>14s}  '
              f'known={stats["n_known_pred"]:2d}  '
              f'novel_match={stats["n_novel_matched"]:2d}  '
              f'pure_unk={stats["n_pure_unk"]:2d}  '
              f'(GT: known={stats["n_known_gt"]}, novel={stats["n_novel_gt"]})')

    # ── Save image list ───────────────────────────────────────────────────────
    list_path = os.path.join(args.output_dir, 'image_list.txt')
    with open(list_path, 'w') as f:
        f.write(f'# YOLO-UniOW Visualization — {args.dataset} Task {args.task}\n')
        f.write(f'# Checkpoint : {args.checkpoint}\n')
        f.write(f'# Thresholds : known={args.threshold}  unknown={args.unk_threshold}'
                f'  iou_match={args.iou_match}\n')
        f.write(f'# Colour key : green=known  orange=novel-matched-unknown  '
                f'red=pure-unknown\n')
        f.write(f'# GT dashed  : green=known-GT  orange=novel-GT\n')
        f.write(f'# Total      : {len(saved_names)} images\n')
        f.write('#\n')
        for name in saved_names:
            f.write(name + '\n')

    print(f'\n[Step 3] Done! {len(saved_names)} images → {args.output_dir}')
    print(f'         Image list → {list_path}')


if __name__ == '__main__':
    main()
