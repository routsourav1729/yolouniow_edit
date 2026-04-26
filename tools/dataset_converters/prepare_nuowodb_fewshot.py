#!/usr/bin/env python3
"""
Prepare TFA-style few-shot splits for nuOWODB.

Setup:
  - BASE classes  = T1 (10 vehicle classes)  → full training data
  - NOVEL classes = T2 (7 pedestrian classes) → k-shot sampling
  - UNKNOWN       = T3 (6 obstacle classes)   → not trained, unknown at eval
  - Test set      = existing OW test.txt

This script generates:
  data/OWOD/nuowodb_fewshot_split/seed{i}/box_{k}shot_{cls}_train.txt

Each file contains image IDs (one per line) that have at least one instance of
that class. The existing YOLO-UniOW fewshot loading code
(owodb.py _extract_fns_fewshot) reads these files directly.

Usage:
    cd YOLO-UniOW
    python tools/dataset_converters/prepare_nuowodb_fewshot.py \
        --owod-root data/OWOD \
        --seeds 1 10 \
        --shots 1 2 3 5 10

Then run training with:
    DATASET=nuOWODB TASK=2 FEWSHOT_K=10 FEWSHOT_SEED=1 \
    FEWSHOT_DIR=data/OWOD/nuowodb_fewshot_split \
    ./tools/dist_train_owod.sh <config> <gpus> --amp
"""
import argparse
import copy
import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--owod-root', default='data/OWOD',
                        help='OWOD data root')
    parser.add_argument('--seeds', type=int, nargs=2, default=[1, 10],
                        help='Range of seeds [start, end)')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 3, 5, 10],
                        help='Shot values to generate')
    parser.add_argument('--output-dir', default=None,
                        help='Output dir (default: {owod-root}/nuowodb_fewshot_split)')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.owod_root, 'nuowodb_fewshot_split')
    return args


def load_annotations(owod_root, dataset='nuOWODB'):
    """Load all annotations and build per-class index.

    Returns:
        data_per_cat: {class_name: [img_id, ...]}
        img_instances: {img_id: {class_name: count}}
    """
    ann_dir = os.path.join(owod_root, 'Annotations', dataset)

    # Read T2 training image IDs (these are the images that include T2-era data)
    # T2 train = T1 train + new T2 images. We only want images that actually
    # contain T2 (novel) class objects.
    t2_train_path = os.path.join(owod_root, 'ImageSets', dataset, 't2_train.txt')
    with open(t2_train_path) as f:
        t2_train_ids = set(line.strip() for line in f if line.strip())

    # Also get T1 train IDs for base class sampling
    t1_train_path = os.path.join(owod_root, 'ImageSets', dataset, 't1_train.txt')
    with open(t1_train_path) as f:
        t1_train_ids = set(line.strip() for line in f if line.strip())

    # Images that are new in T2 (not in T1) — these are the primary source
    # for novel class sampling. But we also check T1 images in case some
    # T2 classes appear there (they'd be unknown in T1 context).
    all_train_ids = t2_train_ids  # Use all T2 train data for sampling

    data_per_cat = defaultdict(list)  # class_name -> [img_id, ...]
    img_instances = {}  # img_id -> {class_name: count}

    missing_ann = 0
    for img_id in sorted(all_train_ids):
        ann_path = os.path.join(ann_dir, f"{img_id}.xml")
        if not os.path.exists(ann_path):
            missing_ann += 1
            continue

        tree = ET.parse(ann_path)
        cls_counts = defaultdict(int)
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls in ALL_CLASSES:
                cls_counts[cls] += 1

        if cls_counts:
            img_instances[img_id] = dict(cls_counts)
            for cls in cls_counts:
                data_per_cat[cls].append(img_id)

    print(f"Loaded annotations from {len(img_instances)} images "
          f"({missing_ann} missing annotation files)")
    print(f"\nPer-class image counts:")
    for cls in T1_CLASSES + T2_CLASSES + T3_CLASSES:
        tag = 'BASE' if cls in T1_CLASSES else ('NOVEL' if cls in T2_CLASSES else 'UNK')
        count = len(data_per_cat.get(cls, []))
        print(f"  [{tag:5s}] {cls}: {count} images")

    return data_per_cat, img_instances


def generate_fewshot_splits(args, data_per_cat, img_instances):
    """Generate k-shot splits for novel (T2) classes."""
    shots = sorted(args.shots)
    novel_classes = T2_CLASSES

    for seed_idx in range(args.seeds[0], args.seeds[1]):
        seed_dir = os.path.join(args.output_dir, f'seed{seed_idx}')
        os.makedirs(seed_dir, exist_ok=True)

        rng = random.Random(seed_idx)

        for cls in novel_classes:
            available = data_per_cat.get(cls, [])
            if not available:
                print(f"WARNING: No images for {cls}, skipping")
                continue

            # Shuffle with seed
            shuffled = list(available)
            rng.shuffle(shuffled)

            # For each shot value, select images that contain enough instances
            selected_ids = []
            for shot in shots:
                # We need 'shot' total instances of this class
                # Greedily pick images until we have enough
                total_instances = 0
                for img_id in shuffled:
                    if img_id in selected_ids:
                        continue
                    n_inst = img_instances[img_id].get(cls, 0)
                    if n_inst > 0:
                        selected_ids.append(img_id)
                        total_instances += n_inst
                        if total_instances >= shot:
                            break

                # Write the file
                fname = f'box_{shot}shot_{cls}_train.txt'
                fpath = os.path.join(seed_dir, fname)
                with open(fpath, 'w') as f:
                    f.write('\n'.join(selected_ids) + '\n')

        print(f"Seed {seed_idx}: wrote {len(novel_classes)} classes x "
              f"{len(shots)} shots to {seed_dir}")


def main():
    args = parse_args()
    print("=" * 60)
    print("nuOWODB Few-Shot Split Generator")
    print(f"  OWOD root: {args.owod_root}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Seeds:     {args.seeds[0]} to {args.seeds[1]-1}")
    print(f"  Shots:     {args.shots}")
    print(f"  BASE (T1): {len(T1_CLASSES)} classes (full training)")
    print(f"  NOVEL (T2): {len(T2_CLASSES)} classes (few-shot)")
    print(f"  UNKNOWN (T3): {len(T3_CLASSES)} classes (not trained)")
    print("=" * 60)

    # Load annotations
    data_per_cat, img_instances = load_annotations(args.owod_root)

    # Generate splits
    print("\n" + "=" * 60)
    print("Generating few-shot splits...")
    generate_fewshot_splits(args, data_per_cat, img_instances)

    # Print usage instructions
    print("\n" + "=" * 60)
    print("DONE! To use with YOLO-UniOW:")
    print(f"  DATASET=nuOWODB TASK=2 FEWSHOT_K=10 FEWSHOT_SEED=1 \\")
    print(f"  FEWSHOT_DIR={args.output_dir} \\")
    print(f"  ./tools/dist_train_owod.sh <config> <gpus> --amp")
    print()
    print("For TFA-style training:")
    print("  1. Train base (T1) normally:  DATASET=nuOWODB TASK=1 ...")
    print("  2. Fine-tune novel (T2) with: DATASET=nuOWODB TASK=2 FEWSHOT_K=10 ...")
    print("  3. Evaluate on test:          DATASET=nuOWODB TASK=2 ...")


if __name__ == '__main__':
    main()
