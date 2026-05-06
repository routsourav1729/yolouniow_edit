#!/usr/bin/env python3
"""
nuOWODB TFA-style Few-Shot Sampling  (mirrors IDD prepare_idd_few_shot_sampling.py)

Protocol:
  - Source data : T2_train images (t2_train.txt) — contains both T1 and T2 class objects
  - BASE classes  = T1 (10 vehicle classes)  → k-shot object sampling
  - NOVEL classes = T2 (7 pedestrian classes) → k-shot object sampling
  - UNKNOWN       = T3 (6 obstacle classes)   → excluded from sampling

Outputs (under --output-dir):
  {k}shot/seed{s}/base/   : XMLs for k-shot BASE sampling  (full XMLs, no filtering)
  {k}shot/seed{s}/fewshot/: XMLs for k-shot NOVEL sampling (full XMLs, no filtering)

  Also writes YOLO-UniOW style .txt files:
  nuowodb_fewshot_split/seed{s}/box_{k}shot_{cls}_train.txt

Usage (run from YOLO-UniOW root):
    python tools/dataset_converters/prepare_nuowodb_tfa_sampling.py \
        --owod-root data/OWOD \
        --output-dir data/nuowodb_fewshot \
        --seeds 1 11 \
        --shots 1 2 3 5 10 20 30
"""
import argparse
import copy
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime

# nuOWODB class definitions (must match owodb_const.py)
T1_CLASSES = [
    'vehicle.bicycle',
    'vehicle.motorcycle',
    'vehicle.car',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.truck',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'vehicle.construction',
    'vehicle.trailer',
]

T2_CLASSES = [
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.wheelchair',
    'human.pedestrian.stroller',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer',
    'human.pedestrian.construction_worker',
]

T3_CLASSES = [
    'movable_object.barrier',
    'movable_object.trafficcone',
    'movable_object.pushable_pullable',
    'movable_object.debris',
    'static_object.bicycle_rack',
    'animal',
]

ALL_CLASSES = T1_CLASSES + T2_CLASSES + T3_CLASSES
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}

BASE_CLASSES = T1_CLASSES
NOVEL_CLASSES = T2_CLASSES


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_t2_train_ids(owod_root, dataset='nuOWODB'):
    path = os.path.join(owod_root, 'ImageSets', dataset, 't2_train.txt')
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def scan_annotations(ann_dir, img_ids, target_classes):
    """
    For each image in img_ids, record which target classes appear and how many objects.

    Returns:
        class_to_xmls  : {cls: [xml_path, ...]}  (xml may appear multiple times per cls)
        img_class_objs : {xml_path: {cls: count}}
    """
    target_set = set(target_classes)
    class_to_xmls = defaultdict(list)
    img_class_objs = {}

    missing = 0
    for img_id in img_ids:
        xml_path = os.path.join(ann_dir, f'{img_id}.xml')
        if not os.path.exists(xml_path):
            missing += 1
            continue
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            continue

        cls_counts = defaultdict(int)
        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls in target_set:
                cls_counts[cls] += 1

        if cls_counts:
            img_class_objs[xml_path] = dict(cls_counts)
            for cls in cls_counts:
                class_to_xmls[cls].append(xml_path)

    if missing:
        print(f"  WARNING: {missing} annotation files missing")
    return class_to_xmls, img_class_objs


def print_stats(class_to_xmls, target_classes, tag):
    print(f"\nPer-class image counts [{tag}]:")
    for cls in target_classes:
        print(f"  {cls:<45}: {len(class_to_xmls.get(cls, [])):>6} images")


# ---------------------------------------------------------------------------
# Incremental k-shot sampling  (mirrors IDD approach)
# ---------------------------------------------------------------------------

def sample_k_shot_incremental(class_to_xmls, img_class_objs, target_classes, shots, seed):
    """
    Object-level incremental sampling: 30-shot ⊇ 20-shot ⊇ 10-shot ⊇ ...

    Returns: {cls: {shot: [xml_paths]}}
    """
    rng = random.Random(seed)
    result = {cls: {} for cls in target_classes}

    for cls in target_classes:
        available = list(class_to_xmls.get(cls, []))
        if not available:
            print(f"  WARNING: No images for {cls}")
            continue

        shuffled = available.copy()
        rng.shuffle(shuffled)

        selected = []
        for j, shot in enumerate(sorted(shots)):
            needed = shots[j] - shots[j - 1] if j > 0 else shot
            obj_count = 0
            for xml_path in shuffled:
                if xml_path in selected:
                    continue
                n = img_class_objs.get(xml_path, {}).get(cls, 0)
                if n > 0:
                    selected.append(xml_path)
                    obj_count += n
                    if obj_count >= needed:
                        break
            result[cls][shot] = copy.deepcopy(selected)

    return result


# ---------------------------------------------------------------------------
# Copy XML files (no filtering — full annotations like IDD)
# ---------------------------------------------------------------------------

def create_fewshot_xmls(sampled, shot, seed, category, target_classes, output_dir):
    out = os.path.join(output_dir, f'{shot}shot', f'seed{seed}', category)
    os.makedirs(out, exist_ok=True)

    all_xmls = set()
    for cls in target_classes:
        if cls in sampled and shot in sampled[cls]:
            all_xmls.update(sampled[cls][shot])

    copied = 0
    for xml_path in all_xmls:
        dst = os.path.join(out, os.path.basename(xml_path))
        if not os.path.exists(dst):
            shutil.copy2(xml_path, dst)
            copied += 1

    return len(all_xmls), copied


# ---------------------------------------------------------------------------
# YOLO-UniOW .txt files (image-ID based)
# ---------------------------------------------------------------------------

def write_yolo_txt_splits(sampled, shots, seeds_range, target_classes, txt_output_dir):
    for seed in seeds_range:
        seed_dir = os.path.join(txt_output_dir, f'seed{seed}')
        os.makedirs(seed_dir, exist_ok=True)

        for cls in target_classes:
            if cls not in sampled[seed]:
                continue
            for shot in shots:
                if shot not in sampled[seed][cls]:
                    continue
                ids = [os.path.basename(p).replace('.xml', '') for p in sampled[seed][cls][shot]]
                ids = list(dict.fromkeys(ids))  # deduplicate preserving order
                fname = f'box_{shot}shot_{cls}_train.txt'
                with open(os.path.join(seed_dir, fname), 'w') as f:
                    f.write('\n'.join(ids) + '\n')


# ---------------------------------------------------------------------------
# TFA COCO JSON
# ---------------------------------------------------------------------------

def create_tfa_json(sampled, shot, seed, target_classes, category_tag, all_classes_map, output_dir):
    """
    TFA-filtered: each image only keeps annotations for target_classes.
    """
    out_dir = os.path.join(output_dir, f'{shot}shot', f'seed{seed}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{category_tag}_{shot}shot_seed{seed}_TFA.json')

    categories = [{'id': idx, 'name': cls, 'supercategory': 'object'}
                  for idx, cls in enumerate(ALL_CLASSES)]
    target_set = set(target_classes)

    # Collect all unique XMLs for this shot
    all_xmls = set()
    for cls in target_classes:
        if cls in sampled and shot in sampled[cls]:
            all_xmls.update(sampled[cls][shot])

    images = []
    annotations = []
    img_id = 1
    ann_id = 1

    for xml_path in sorted(all_xmls):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception:
            continue

        size = root.find('size')
        if size is None:
            continue
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        fname = root.find('filename').text

        obj_anns = []
        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls not in target_set:
                continue
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            obj_anns.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': all_classes_map[cls],
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'area': (x2 - x1) * (y2 - y1),
                'iscrowd': 0,
            })
            ann_id += 1

        if not obj_anns:
            continue

        images.append({'id': img_id, 'file_name': fname, 'width': w, 'height': h})
        annotations.extend(obj_anns)
        img_id += 1

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
        'info': {
            'description': f'nuOWODB TFA {category_tag} {shot}shot seed{seed}',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'seed': seed,
            'protocol': 'TFA per-class filtering',
        },
    }

    with open(out_path, 'w') as f:
        json.dump(coco, f)

    cls_counts = Counter(a['category_id'] for a in annotations)
    return out_path, len(images), len(annotations), cls_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--owod-root', default='data/OWOD')
    p.add_argument('--output-dir', default='data/nuowodb_fewshot',
                   help='Root output for XML dirs + TFA JSONs')
    p.add_argument('--txt-dir', default=None,
                   help='Root for YOLO-UniOW .txt files (default: output-dir/nuowodb_fewshot_split)')
    p.add_argument('--seeds', type=int, nargs=2, default=[1, 11],
                   help='Seed range [start, end)')
    p.add_argument('--shots', type=int, nargs='+', default=[1, 2, 3, 5, 10, 20, 30])
    return p.parse_args()


def main():
    args = parse_args()
    if args.txt_dir is None:
        args.txt_dir = os.path.join(args.output_dir, 'nuowodb_fewshot_split')

    shots = sorted(args.shots)
    seeds = range(args.seeds[0], args.seeds[1])

    ann_dir = os.path.join(args.owod_root, 'Annotations', 'nuOWODB')
    dataset = 'nuOWODB'

    print("=" * 70)
    print("nuOWODB TFA Few-Shot Sampling")
    print(f"  OWOD root  : {args.owod_root}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  TXT dir    : {args.txt_dir}")
    print(f"  Seeds      : {list(seeds)}")
    print(f"  Shots      : {shots}")
    print(f"  BASE (T1)  : {len(BASE_CLASSES)} classes")
    print(f"  NOVEL (T2) : {len(NOVEL_CLASSES)} classes")
    print("=" * 70)

    # Load T2 train IDs (covers both T1+T2 era images)
    t2_ids = load_t2_train_ids(args.owod_root, dataset)
    print(f"\nT2 train images: {len(t2_ids)}")

    # Scan annotations for BASE classes
    print("\n[1/4] Scanning for BASE (T1) class annotations...")
    base_class_to_xmls, base_img_objs = scan_annotations(ann_dir, t2_ids, BASE_CLASSES)
    print_stats(base_class_to_xmls, BASE_CLASSES, 'BASE')

    # Scan annotations for NOVEL classes
    print("\n[2/4] Scanning for NOVEL (T2) class annotations...")
    novel_class_to_xmls, novel_img_objs = scan_annotations(ann_dir, t2_ids, NOVEL_CLASSES)
    print_stats(novel_class_to_xmls, NOVEL_CLASSES, 'NOVEL')

    # Per-seed sampling
    print("\n[3/4] Sampling per seed...")
    base_sampled_all = {}
    novel_sampled_all = {}

    for seed in seeds:
        base_sampled = sample_k_shot_incremental(
            base_class_to_xmls, base_img_objs, BASE_CLASSES, shots, seed)
        novel_sampled = sample_k_shot_incremental(
            novel_class_to_xmls, novel_img_objs, NOVEL_CLASSES, shots, seed)
        base_sampled_all[seed] = base_sampled
        novel_sampled_all[seed] = novel_sampled

        for shot in shots:
            # XML copies
            b_total, b_copied = create_fewshot_xmls(
                base_sampled, shot, seed, 'base', BASE_CLASSES, args.output_dir)
            n_total, n_copied = create_fewshot_xmls(
                novel_sampled, shot, seed, 'fewshot', NOVEL_CLASSES, args.output_dir)

            # TFA JSONs
            _, n_imgs_b, n_anns_b, _ = create_tfa_json(
                base_sampled, shot, seed, BASE_CLASSES, 'base', CLASS_TO_ID, args.output_dir)
            _, n_imgs_n, n_anns_n, _ = create_tfa_json(
                novel_sampled, shot, seed, NOVEL_CLASSES, 'novel', CLASS_TO_ID, args.output_dir)

            print(f"  seed{seed} {shot:>2}shot | "
                  f"base: {b_total} imgs {n_anns_b} anns | "
                  f"novel: {n_total} imgs {n_anns_n} anns")

    # YOLO-UniOW .txt files
    print("\n[4/4] Writing YOLO-UniOW .txt split files...")
    write_yolo_txt_splits(base_sampled_all, shots, seeds, BASE_CLASSES, args.txt_dir)
    write_yolo_txt_splits(novel_sampled_all, shots, seeds, NOVEL_CLASSES, args.txt_dir)

    print("\n" + "=" * 70)
    print("DONE!")
    print(f"\nOutput structure:")
    print(f"  {args.output_dir}/")
    print(f"    {{k}}shot/seed{{s}}/base/     — BASE class sampled XMLs (full annotations)")
    print(f"    {{k}}shot/seed{{s}}/fewshot/  — NOVEL class sampled XMLs (full annotations)")
    print(f"    {{k}}shot/seed{{s}}/base_{{k}}shot_seed{{s}}_TFA.json   — TFA-filtered COCO JSON")
    print(f"    {{k}}shot/seed{{s}}/novel_{{k}}shot_seed{{s}}_TFA.json  — TFA-filtered COCO JSON")
    print(f"\n  {args.txt_dir}/")
    print(f"    seed{{s}}/box_{{k}}shot_{{cls}}_train.txt  — YOLO-UniOW split files")
    print(f"\nT1 training: use t1_train.txt with ALL T1 class annotations (no fewshot needed)")
    print(f"T2 fewshot : DATASET=nuOWODB TASK=2 FEWSHOT_K=10 FEWSHOT_SEED=1 ...")


if __name__ == '__main__':
    main()
