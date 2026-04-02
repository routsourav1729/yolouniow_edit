"""
Temporary script: fewshot annotation stats for IDD splits.

For a given seed and shot count, reads all box_{shot}shot_{class}_train.txt
files, then for each image listed in those files parses the corresponding
annotation XML.

Logic (mirrors yolo_world_owod.py _parse_xml_for_class usage):
  - image in box_10shot_car_train.txt  →  "car" objects in that image are ANNOTATED
    BUT ONLY if "car" is a KNOWN class for the given task.
  - Objects of UNKNOWN classes are always DISCARDED regardless of which shot file
    listed the image — unknown classes have no annotation supervision.
  - every OTHER object in that same image is also DISCARDED

Output: per-class counts of annotated vs discarded objects across all shot images.
"""
import os
import glob
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict


# ── IDD task class definitions (from embedding_diagnostic.py) ─────────────────
IDD_CLASSES_T1 = ['car', 'motorcycle', 'rider', 'person',
                  'autorickshaw', 'bicycle', 'traffic sign', 'traffic light']
IDD_CLASSES_T2 = ['bus', 'truck', 'tanker_vehicle', 'crane_truck',
                  'street_cart', 'excavator']
IDD_UNKNOWNS   = ['pole', 'animal', 'tractor', 'concrete_mixer',
                  'pull_cart', 'road_roller']

KNOWN_BY_TASK = {
    1: set(IDD_CLASSES_T1),
    2: set(IDD_CLASSES_T1) | set(IDD_CLASSES_T2),
}


# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='FewShot annotation stats')
    p.add_argument('--shot',      type=int, default=10,
                   help='Shot count (e.g. 10 reads all box_10shot_*_train.txt)')
    p.add_argument('--seed',      type=int, default=1,
                   help='Seed number (1-10)')
    p.add_argument('--task',      type=int, default=2, choices=[1, 2],
                   help='IDD OWOD task (1 or 2) — determines known vs unknown classes')
    p.add_argument('--split-dir', default='data/OWOD/iddsplit',
                   help='Path to iddsplit dir containing seed{N}/ folders')
    p.add_argument('--ann-dir',   default='data/OWOD/Annotations/IDD',
                   help='Path to Annotations/IDD dir containing .xml files')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
def extract_class_from_filename(fname: str) -> str:
    """box_10shot_traffic light_train.txt  →  'traffic light'"""
    base = os.path.basename(fname)            # box_10shot_car_train.txt
    base = base.replace('_train.txt', '')     # box_10shot_car
    # strip 'box_{K}shot_' prefix
    parts = base.split('_', 2)               # ['box', '10shot', 'car']
    return parts[2]                           # 'car'  (works for multi-word too)


def load_shot_files(split_dir: str, seed: int, shot: int):
    """
    Returns dict: image_id -> set of annotated class names
    i.e. the classes for which this image was selected as a support image.
    """
    seed_dir = os.path.join(split_dir, f'seed{seed}')
    pattern  = os.path.join(seed_dir, f'box_{shot}shot_*_train.txt')
    files    = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f'No files matching {pattern}\n'
            f'  seed_dir exists: {os.path.isdir(seed_dir)}')

    print(f'[INFO] Found {len(files)} shot files for {shot}-shot seed{seed}:')
    for f in files:
        print(f'       {os.path.basename(f)}')

    # image_id -> set of classes this image is a support image for
    img_to_annotated_classes: dict[str, set[str]] = defaultdict(set)

    for fpath in files:
        cls_name = extract_class_from_filename(fpath)
        with open(fpath) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                # line: data/OWOD/JPEGImages/IDD/011777.jpg
                img_id = os.path.splitext(os.path.basename(line))[0]
                img_to_annotated_classes[img_id].add(cls_name)

    return img_to_annotated_classes


def count_objects_in_xml(xml_path: str):
    """Returns list of class names (one per object) from an annotation XML."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        print(f'[WARN] Could not parse {xml_path}')
        return []
    root = tree.getroot()
    names = []
    for obj in root.iter('object'):
        name_el = obj.find('name')
        if name_el is not None and name_el.text:
            names.append(name_el.text.strip())
    return names


# ──────────────────────────────────────────────────────────────────────────────
def compute_stats(img_to_annotated: dict, ann_dir: str, known_classes: set):
    """
    Returns two dicts (annotated_counts, discarded_counts): str -> int
    Keys are class names.

    An object is ANNOTATED only if:
      1. Its class is in known_classes (not an unknown class), AND
      2. Its class matches the class for which this image was selected.
    Everything else is DISCARDED.
    """
    annotated_counts  = defaultdict(int)
    discarded_counts  = defaultdict(int)

    missing_xml = 0
    for img_id, selected_classes in img_to_annotated.items():
        xml_path = os.path.join(ann_dir, img_id + '.xml')
        if not os.path.exists(xml_path):
            missing_xml += 1
            continue

        # Only classes that are both selected AND known count as annotated
        annotated_classes = selected_classes & known_classes

        obj_names = count_objects_in_xml(xml_path)
        for obj_cls in obj_names:
            if obj_cls in annotated_classes:
                annotated_counts[obj_cls] += 1
            else:
                discarded_counts[obj_cls] += 1

    if missing_xml:
        print(f'[WARN] {missing_xml} images had no corresponding XML')

    return annotated_counts, discarded_counts


def print_table(annotated_counts: dict, discarded_counts: dict):
    all_classes = sorted(
        set(annotated_counts.keys()) | set(discarded_counts.keys()))

    col_w  = max(len(c) for c in all_classes) + 2
    header = f"{'CLASS':<{col_w}} {'ANNOTATED':>12} {'DISCARDED':>12} {'TOTAL':>10} {'DISCARD%':>10}"
    print('\n' + '=' * len(header))
    print(header)
    print('=' * len(header))

    total_ann  = 0
    total_dis  = 0
    for cls in all_classes:
        ann = annotated_counts.get(cls, 0)
        dis = discarded_counts.get(cls, 0)
        tot = ann + dis
        pct = f'{100*dis/tot:.1f}' if tot > 0 else '-'
        print(f'{cls:<{col_w}} {ann:>12,} {dis:>12,} {tot:>10,} {pct:>9}%')
        total_ann += ann
        total_dis += dis

    total_tot = total_ann + total_dis
    total_pct = f'{100*total_dis/total_tot:.1f}' if total_tot > 0 else '-'
    print('-' * len(header))
    print(f'{"TOTAL":<{col_w}} {total_ann:>12,} {total_dis:>12,} {total_tot:>10,} {total_pct:>9}%')
    print('=' * len(header))
    print()
    print('NOTE: counts are over the set of images in the shot files only.')
    print('      Objects in images NOT listed in any shot file are excluded.')


# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    known_classes = KNOWN_BY_TASK[args.task]
    unknown_classes = set(IDD_UNKNOWNS)

    print(f'\n[CONFIG] seed={args.seed}  shot={args.shot}  task={args.task}')
    print(f'         split-dir : {args.split_dir}')
    print(f'         ann-dir   : {args.ann_dir}')
    print(f'         known     : {sorted(known_classes)}')
    print(f'         unknown   : {sorted(unknown_classes)}  ← always discarded\n')

    img_to_annotated = load_shot_files(args.split_dir, args.seed, args.shot)
    print(f'\n[INFO] Total unique images across all shot files: '
          f'{len(img_to_annotated)}\n')

    annotated_counts, discarded_counts = compute_stats(
        img_to_annotated, args.ann_dir, known_classes)

    print_table(annotated_counts, discarded_counts)


if __name__ == '__main__':
    main()
