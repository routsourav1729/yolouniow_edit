"""
Step 1 (one-time): Scan IDD / MOWODB test-set GT annotations.

Finds images that are rich in NOVEL objects (T2 classes) with bounding
boxes >= min_size × min_size pixels.  Priority is given to large / rare /
interesting categories: tanker_vehicle, crane_truck, excavator (IDD) or
elephant, bear, zebra, giraffe (MOWODB).

Outputs
-------
vis_output/candidates.txt   — ranked list: img_id  score  n_known  n_novel  n_other  novel_counts

Usage
-----
    # from YOLO-UniOW root
    python tools/vis/step1_find_gt_candidates.py \\
        --data-root data/OWOD \\
        --dataset IDD \\
        --task 1 \\
        --min-size 100 \\
        --top-n 200 \\
        --output vis_output/candidates.txt
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

# ── Per-dataset class splits and priority weights ─────────────────────────────
DATASET_CLASSES = {
    'IDD': {
        'T1': [
            'car', 'motorcycle', 'rider', 'person',
            'autorickshaw', 'bicycle', 'traffic sign', 'traffic light',
        ],
        'T2': [
            'bus', 'truck', 'tanker_vehicle', 'crane_truck',
            'street_cart', 'excavator',
        ],
        # Higher weight → prioritised in ranking
        'priority': {
            'tanker_vehicle': 2.5,
            'crane_truck':    2.5,
            'excavator':      2.2,
            'street_cart':    1.8,
            'truck':          1.3,
            'bus':            1.3,
        },
    },
    'MOWODB': {
        'T1': [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
            'sofa', 'train', 'tvmonitor',
        ],
        'T2': [
            'truck', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator',
        ],
        'priority': {
            'elephant': 2.5,
            'bear':     2.5,
            'zebra':    2.2,
            'giraffe':  2.2,
            'truck':    1.3,
        },
    },
    'SOWODB': {
        'T1': [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bus', 'car',
            'cat', 'cow', 'dog', 'horse', 'motorbike', 'sheep', 'train',
            'elephant', 'bear', 'zebra', 'giraffe', 'truck', 'person',
        ],
        'T2': [
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
            'bench', 'chair', 'diningtable', 'pottedplant', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'bed', 'toilet', 'sofa',
        ],
        'priority': {
            'elephant': 2.0,
            'bear':     2.0,
            'zebra':    1.8,
            'giraffe':  1.8,
        },
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description='Step 1: find GT-annotation-rich candidate images')
    p.add_argument('--data-root', default='data/OWOD',
                   help='Root of the OWOD data directory')
    p.add_argument('--dataset', default='IDD',
                   choices=list(DATASET_CLASSES),
                   help='Dataset name')
    p.add_argument('--task', type=int, default=1,
                   help='Task number (known = T1..T{task}); T2 treated as novel')
    p.add_argument('--image-set', default='test',
                   help='Image-set file stem under ImageSets/{dataset}/')
    p.add_argument('--min-size', type=int, default=100,
                   help='Both width AND height of GT box must be >= this value')
    p.add_argument('--top-n', type=int, default=200,
                   help='How many candidate images to output')
    p.add_argument('--max-density', type=int, default=30,
                   help='Images with more total GT boxes than this are penalised')
    p.add_argument('--target-class', default='',
                   help='Optional class to prioritize (e.g., animal).')
    p.add_argument('--min-target-count', type=int, default=0,
                   help='Require at least this many target-class boxes.')
    p.add_argument('--require-novel', action='store_true',
                   help='Require at least one large novel box in selected image.')
    p.add_argument('--target-weight', type=float, default=6.0,
                   help='Score weight per large target-class box.')
    p.add_argument('--novel-weight', type=float, default=1.0,
                   help='Global multiplier for weighted novel score.')
    p.add_argument('--other-weight', type=float, default=0.35,
                   help='Score weight per large other/unknown box.')
    p.add_argument('--known-weight', type=float, default=0.25,
                   help='Score weight per large known box.')
    p.add_argument('--output', default='vis_output/candidates.txt')
    return p.parse_args()


def bbox_wh(bbox):
    """Return (width, height) from [xmin, ymin, xmax, ymax]."""
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    info = DATASET_CLASSES[args.dataset]
    known_classes = set(info['T1'])   # T1 (and earlier tasks if task > 1)
    novel_classes = set(info['T2'])   # T2 novel (future task → unknown during T1 eval)
    priority = info['priority']
    target_class = args.target_class.strip()

    anno_dir = os.path.join(args.data_root, 'Annotations', args.dataset)
    image_set_file = os.path.join(
        args.data_root, 'ImageSets', args.dataset,
        f'{args.image_set}.txt')

    if not os.path.exists(image_set_file):
        raise FileNotFoundError(f'Image-set file not found: {image_set_file}')

    with open(image_set_file) as f:
        image_ids = [line.strip() for line in f if line.strip()]

    print(f'[Step 1] Scanning {len(image_ids)} images in '
          f'{args.dataset}/{args.image_set} '
          f'(min box size = {args.min_size}px) ...')
    if target_class:
        print(f'[Step 1] Focus mode: target_class={target_class}, '
              f'min_target_count={args.min_target_count}, '
              f'require_novel={args.require_novel}')

    records = []
    missing_xml = 0

    for img_id in image_ids:
        xml_path = os.path.join(anno_dir, img_id + '.xml')
        if not os.path.exists(xml_path):
            missing_xml += 1
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        known_boxes, novel_boxes, other_boxes, target_boxes = [], [], [], []

        for obj in root.findall('object'):
            name = obj.find('name').text.strip()
            diff_tag = obj.find('difficult')
            if diff_tag is not None and int(diff_tag.text):
                continue  # skip difficult

            bb = obj.find('bndbox')
            bbox = [
                int(bb.find('xmin').text),
                int(bb.find('ymin').text),
                int(bb.find('xmax').text),
                int(bb.find('ymax').text),
            ]
            w, h = bbox_wh(bbox)
            large_enough = (w >= args.min_size and h >= args.min_size)

            if name in known_classes:
                if large_enough:
                    known_boxes.append((name, bbox))
            elif name in novel_classes:
                if large_enough:
                    novel_boxes.append((name, bbox))
            else:
                if large_enough:
                    other_boxes.append((name, bbox))

            if target_class and large_enough and name == target_class:
                target_boxes.append((name, bbox))

        if target_class:
            if len(target_boxes) < args.min_target_count:
                continue
            if args.require_novel and not novel_boxes:
                continue
        else:
            # Default behavior: keep only images with large novel GT objects
            if not novel_boxes:
                continue

        # Weighted score: each novel box contributes its priority weight.
        novel_score = sum(priority.get(n, 1.0) for n, _ in novel_boxes)

        if target_class:
            score = (
                len(target_boxes) * args.target_weight +
                novel_score * args.novel_weight +
                len(other_boxes) * args.other_weight +
                len(known_boxes) * args.known_weight
            )
        else:
            # Mild bonus for having some known objects (helps visual context)
            balance_bonus = min(len(known_boxes), 5) * 0.25
            score = novel_score + balance_bonus

        # Density penalty (don't want cluttered images)
        total_gt = len(known_boxes) + len(novel_boxes) + len(other_boxes)
        if total_gt > args.max_density:
            penalty = 1.0 - (total_gt - args.max_density) * 0.025
            score *= max(0.4, penalty)

        # Count per novel class (for display and future reference)
        novel_counts = defaultdict(int)
        for n, _ in novel_boxes:
            novel_counts[n] += 1

        records.append({
            'img_id':      img_id,
            'score':       score,
            'n_known':     len(known_boxes),
            'n_novel':     len(novel_boxes),
            'n_other':     len(other_boxes),
            'n_target':    len(target_boxes),
            'novel_counts': dict(novel_counts),
        })

    records.sort(key=lambda x: -x['score'])
    top = records[:args.top_n]

    if target_class:
        print(f'[Step 1] {len(records)} images match target constraints '
              f'(target={target_class}, min_size={args.min_size}px). '
              f'Saving top {len(top)}.')
    else:
        print(f'[Step 1] {len(records)} images contain large novel GT objects '
              f'(>= {args.min_size}px). Saving top {len(top)}.')
    if missing_xml:
        print(f'[Step 1] WARNING: {missing_xml} XML annotation files not found.')

    with open(args.output, 'w') as f:
        f.write(f'# Dataset: {args.dataset}  Task: {args.task}  '
                f'min_size: {args.min_size}\n')
        if target_class:
            f.write(f'# target_class: {target_class}  min_target_count: {args.min_target_count}  '
                    f'require_novel: {args.require_novel}\n')
        f.write('# img_id  score  n_target  n_known  n_novel  n_other  novel_class_counts\n')
        for r in top:
            novel_str = ' '.join(
                f'{k}:{v}' for k, v in sorted(r['novel_counts'].items()))
            f.write(
                f"{r['img_id']}  {r['score']:.2f}  "
                f"{r['n_target']}  {r['n_known']}  {r['n_novel']}  {r['n_other']}  "
                f"{novel_str}\n"
            )

    print(f'[Step 1] Candidates saved to: {args.output}')

    # ── Console summary of top-20 ─────────────────────────────────────────────
    print('\nTop 20 candidates:')
    for r in top[:20]:
        nc = ', '.join(f'{k}:{v}' for k, v in sorted(r['novel_counts'].items()))
        print(
            f"  {r['img_id']:>14s}  score={r['score']:5.2f}  "
            f"target={r['n_target']}  known={r['n_known']}  novel={r['n_novel']}  ({nc})"
        )


if __name__ == '__main__':
    main()
