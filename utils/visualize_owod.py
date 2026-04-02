"""
Side-by-side GT vs Prediction visualisation for OWOD.

Colour scheme
─────────────
  Predictions (right panel):
    • UNKNOWN  → bright RED     with "unknown" label
    • NOVEL    → YELLOW         with class name
    • KNOWN    → green/teal     (1 colour per class) with class name

  Ground Truth (left panel):
    • UNKNOWN  → bright RED     with class name
    • NOVEL    → YELLOW/ORANGE  with class name
    • KNOWN    → green shades   with class name

Usage (from YOLO-UniOW root, on a GPU node):
    python utils/visualize_owod.py \\
        --config configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_wapr.py \\
        --checkpoint work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task2_10shot_wapr/best_owod_Both_epoch_40.pth \\
        --image-ids 080222 080221 080223 010153 000090

Or via sbatch:
    sbatch scripts/visualize_owod.sbatch
"""
import os
import sys
import argparse
import numpy as np
import cv2
import xml.etree.ElementTree as ET

import torch
from mmengine.config import Config, ConfigDict
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint
from mmengine.runner.amp import autocast
from mmdet.apis import init_detector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

# ── IDD class splits ─────────────────────────────────────────────────────────
IDD_KNOWN_T1 = ['car', 'motorcycle', 'rider', 'person',
                'autorickshaw', 'bicycle', 'traffic sign', 'traffic light']
IDD_NOVEL_T2 = ['bus', 'truck', 'tanker_vehicle', 'crane_truck',
                'street_cart', 'excavator']
IDD_UNKNOWN  = ['pole', 'animal', 'tractor', 'concrete_mixer',
                'pull_cart', 'road_roller']

# ── Colour palette (BGR for OpenCV) ──────────────────────────────────────────
# Unknown → bright red
UNK_COLOR = (60, 60, 230)
# Novel → yellow / gold shades
NOVEL_COLORS = [
    (0, 230, 255),    # yellow
    (0, 200, 255),    # gold
    (30, 220, 240),   # amber-yellow
    (0, 210, 230),    # dark gold
    (50, 240, 255),   # light yellow
    (20, 190, 220),   # honey
]
# Known → green/teal shades (1 per class)
KNOWN_COLORS = [
    (80, 200, 80),    # car         - green
    (100, 190, 50),   # motorcycle  - olive-green
    (120, 210, 90),   # rider       - lime
    (60, 180, 120),   # person      - spring green
    (90, 195, 140),   # autorickshaw- jade
    (50, 170, 60),    # bicycle     - forest
    (140, 215, 100),  # traffic sign- chartreuse
    (70, 200, 160),   # traffic light-teal
]


def parse_args():
    p = argparse.ArgumentParser(description='Side-by-side OWOD visualisation')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image-ids', nargs='+', default=None,
                   help='Image IDs to visualize (e.g. 080222 080221)')
    p.add_argument('--image-ids-file', default=None,
                   help='Text file with one image ID per line')
    p.add_argument('--data-root', default='data/OWOD')
    p.add_argument('--dataset', default='IDD')
    p.add_argument('--task', type=int, default=2)
    p.add_argument('--threshold', type=float, default=0.05)
    p.add_argument('--unk-threshold', type=float, default=0.15)
    p.add_argument('--output-dir', default='visualizations/inference')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--box-thickness', type=int, default=2)
    p.add_argument('--font-scale', type=float, default=0.45)
    return p.parse_args()


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_box(img, box, label_text, color, thickness=2, font_scale=0.45):
    """Solid box + semi-transparent fill + label above."""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    # Semi-transparent fill
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
    # Border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    # Label banner
    if label_text:
        (tw, th), base = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        lby = max(y1 - 2, th + base + 3)
        cv2.rectangle(img, (x1, lby - th - base - 3),
                      (x1 + tw + 6, lby + 2), color, -1)
        cv2.putText(img, label_text, (x1 + 2, lby),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)


def get_gt_color(cls_name, known_classes, novel_classes):
    """Return (color_bgr, category_str) for a GT class name."""
    if cls_name in known_classes:
        idx = known_classes.index(cls_name)
        return KNOWN_COLORS[idx % len(KNOWN_COLORS)], 'known'
    elif cls_name in novel_classes:
        idx = novel_classes.index(cls_name)
        return NOVEL_COLORS[idx % len(NOVEL_COLORS)], 'novel'
    else:
        return UNK_COLOR, 'unknown'


# ── GT parsing ────────────────────────────────────────────────────────────────

def parse_gt(xml_path):
    """Returns list of (class_name, [x1,y1,x2,y2])."""
    tree = ET.parse(xml_path)
    objects = []
    for obj in tree.getroot().findall('object'):
        name = (obj.findtext('name') or '').strip()
        bb = obj.find('bndbox')
        if bb is None:
            continue
        box = [int(float(bb.findtext(k, 0))) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        objects.append((name, box))
    return objects


# ── Inference ─────────────────────────────────────────────────────────────────

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


# ── Draw one panel ────────────────────────────────────────────────────────────

def draw_gt_panel(img, gt_objects, known_classes, novel_classes, thickness, font_scale):
    """Draw all GT boxes colour-coded on img (modifies in-place)."""
    for cls_name, box in gt_objects:
        color, cat = get_gt_color(cls_name, known_classes, novel_classes)
        draw_box(img, box, cls_name, color, thickness, font_scale)
    return img


def draw_pred_panel(img, bboxes, scores, labels,
                    known_classes, novel_classes,
                    threshold, unk_threshold,
                    thickness, font_scale):
    """Draw prediction boxes on img (modifies in-place)."""
    n_known = len(known_classes) + len(novel_classes)  # all known at task2
    unknown_idx = n_known  # last index = unknown

    for box, score, label in zip(bboxes, scores, labels):
        label = int(label)
        if label < 0 or label > unknown_idx:
            continue

        if label == unknown_idx:
            if score < unk_threshold:
                continue
            draw_box(img, box, f'unknown {score:.2f}', UNK_COLOR,
                     thickness, font_scale)
        else:
            if score < threshold:
                continue
            # Known T1 classes: indices 0..7, Novel T2: 8..13
            if label < len(known_classes):
                cls_name = known_classes[label]
                color = KNOWN_COLORS[label % len(KNOWN_COLORS)]
            else:
                novel_idx = label - len(known_classes)
                cls_name = novel_classes[novel_idx] if novel_idx < len(novel_classes) else f'cls{label}'
                color = NOVEL_COLORS[novel_idx % len(NOVEL_COLORS)]
            draw_box(img, box, f'{cls_name} {score:.2f}', color,
                     thickness, font_scale)
    return img


def add_title(img, text, bg_color=(40, 40, 40)):
    """Add a title bar at the top of an image."""
    H, W = img.shape[:2]
    bar_h = 32
    bar = np.full((bar_h, W, 3), bg_color, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(bar, text, ((W - tw) // 2, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Set env vars for mmengine config interpolation
    os.environ.setdefault('DATASET', args.dataset)
    os.environ.setdefault('TASK', str(args.task))
    os.environ.setdefault('THRESHOLD', str(args.threshold))
    os.environ.setdefault('SAVE', 'False')
    os.environ.setdefault('FEWSHOT_DIR', 'data/OWOD/iddsplit')
    os.environ.setdefault('FEWSHOT_K', '10')
    os.environ.setdefault('FEWSHOT_SEED', '1')
    os.environ.setdefault('TRAINING_STRATEGY', '0')
    os.environ.setdefault('IMAGESET', 'train')
    os.environ.setdefault('ANALYZE', '0')

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve image IDs
    image_ids = []
    if args.image_ids:
        image_ids = args.image_ids
    elif args.image_ids_file:
        with open(args.image_ids_file) as f:
            image_ids = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    else:
        # Default: top-20 diverse candidates (1 per drive sequence, all joint novel+unknown)
        image_ids = [
            '080222', '010153', '000090', '110116', '019177',
            '091755', '031829', '098971', '111444', '103257',
            '076054', '061422', '013714', '106652', '094380',
            '100740', '009980', '020747', '085851', '109357',
        ]

    known_classes = list(IDD_KNOWN_T1)
    novel_classes = list(IDD_NOVEL_T2)

    ann_dir = os.path.join(args.data_root, 'Annotations', args.dataset)
    img_dir = os.path.join(args.data_root, 'JPEGImages', args.dataset)

    # Load model — init_detector handles all registry setup via custom_imports
    print(f'Loading model...', flush=True)
    print(f'  Config:     {args.config}', flush=True)
    print(f'  Checkpoint: {args.checkpoint}', flush=True)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()
    cfg = model.cfg

    # Build test pipeline directly — standard YOLOv10 pipeline
    img_scale = cfg.get('img_scale', (640, 640))
    test_pipeline = Compose([
        dict(type='LoadImageFromFile'),
        dict(type='YOLOv5KeepRatioResize', scale=img_scale),
        dict(type='LetterResize', scale=img_scale, allow_scale_up=False,
             pad_val=dict(img=114)),
        dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
        dict(type='mmdet.PackDetInputs',
             meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                        'scale_factor', 'pad_param')),
    ])
    print(f'  Model loaded on {args.device}\n', flush=True)

    print(f'Visualising {len(image_ids)} images → {args.output_dir}/\n', flush=True)

    saved = []
    for i, img_id in enumerate(image_ids, 1):
        img_path = os.path.join(img_dir, img_id + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, img_id + '.png')
        xml_path = os.path.join(ann_dir, img_id + '.xml')

        if not os.path.exists(img_path) or not os.path.exists(xml_path):
            print(f'  [{i}/{len(image_ids)}] {img_id} — SKIP (missing file)',
                  flush=True)
            continue

        # Read original image
        orig = cv2.imread(img_path)
        if orig is None:
            print(f'  [{i}/{len(image_ids)}] {img_id} — SKIP (cannot read)',
                  flush=True)
            continue

        # GT panel (left)
        gt_img = orig.copy()
        gt_objects = parse_gt(xml_path)
        draw_gt_panel(gt_img, gt_objects, known_classes, novel_classes,
                      args.box_thickness, args.font_scale)
        gt_img = add_title(gt_img, 'Ground Truth')

        # Prediction panel (right)
        pred_img = orig.copy()
        bboxes, scores, labels = run_inference(
            model, img_path, test_pipeline, args.device, args.amp)
        draw_pred_panel(pred_img, bboxes, scores, labels,
                        known_classes, novel_classes,
                        args.threshold, args.unk_threshold,
                        args.box_thickness, args.font_scale)
        pred_img = add_title(pred_img, 'Prediction (YOLO-UniOW WAPR)')

        # Side-by-side: 3px white separator
        H = gt_img.shape[0]
        sep = np.full((H, 3, 3), 255, dtype=np.uint8)
        combined = np.hstack([gt_img, sep, pred_img])

        out_path = os.path.join(args.output_dir, f'{img_id}.jpg')
        cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(img_id)

        # Count stats for progress
        n_known = sum(1 for b, s, l in zip(bboxes, scores, labels)
                      if int(l) < len(known_classes) + len(novel_classes) and s >= args.threshold)
        n_unk = sum(1 for b, s, l in zip(bboxes, scores, labels)
                    if int(l) == len(known_classes) + len(novel_classes) and s >= args.unk_threshold)
        n_gt_novel = sum(1 for n, _ in gt_objects if n in novel_classes)
        n_gt_unk = sum(1 for n, _ in gt_objects if n in IDD_UNKNOWN)

        print(f'  [{i}/{len(image_ids)}] {img_id}  '
              f'pred: known={n_known} unk={n_unk}  '
              f'GT: novel={n_gt_novel} unk={n_gt_unk}  ✓', flush=True)

    print(f'\nDone! {len(saved)} images saved to {args.output_dir}/', flush=True)
    print(f'\nColour key:')
    print(f'  RED    = unknown (both GT & pred)')
    print(f'  YELLOW = novel T2 classes')
    print(f'  GREEN  = known T1 classes')


if __name__ == '__main__':
    main()
