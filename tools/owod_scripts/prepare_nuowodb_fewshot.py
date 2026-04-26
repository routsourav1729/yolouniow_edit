"""
TFA-style few-shot split generator for nu-OWODB.
Adapted from CED-FOOD/prepare_voc_few_shot.py — same structure as prepare_idd_fewshot.py.

Covers ALL t2_known classes = T1 base (10 vehicle) + T2 novel (7 pedestrian).
Sampling pool: t2_train.txt only — this is the task-2 training set which already
contains both base and novel class images.

Only images that physically exist in JPEGImages/nuOWODB/ are included.

Output: data/OWOD/nuowosplit/seed{i}/box_{k}shot_{cls}_train.txt
  Each line: data/OWOD/JPEGImages/nuOWODB/{timestamp}.jpg
  (OWODDataset._extract_fns_fewshot parses the last path component as image ID)

Usage (from YOLO-UniOW root):
    python tools/owod_scripts/prepare_nuowodb_fewshot.py
    python tools/owod_scripts/prepare_nuowodb_fewshot.py --seeds 1 11 --shots 1 2 3 5 10
"""

import argparse
import copy
import os
import random
import xml.etree.ElementTree as ET

# ALL t2_known classes = T1 base (10) + T2 novel (7)
# Must match CLASS_NAMES[:17] in owodb_const.py exactly
T2_KNOWN_CLASSES = [
    # T1 base — vehicles
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
    # T2 novel — pedestrians
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.wheelchair',
    'human.pedestrian.stroller',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer',
    'human.pedestrian.construction_worker',
]

OWOD_ROOT = 'data/OWOD'
DATASET = 'nuOWODB'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs=2, default=[1, 11],
                        help='Seed range [start, end) e.g. 1 11 gives seeds 1..10')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 3, 5, 10],
                        help='Shot counts to generate')
    parser.add_argument('--owod-root', default=OWOD_ROOT)
    return parser.parse_args()


def generate_seeds(args):
    owod_root = args.owod_root
    shots = sorted(args.shots)

    ann_dir  = os.path.join(owod_root, 'Annotations', DATASET)
    jpeg_dir = os.path.join(owod_root, 'JPEGImages',  DATASET)
    split_dir = os.path.join(owod_root, 'nuowosplit')

    # --- Sampling pool: t2_train only, existence-filtered ---
    t2_file = os.path.join(owod_root, 'ImageSets', DATASET, 't2_train.txt')
    with open(t2_file) as f:
        all_ids = [l.strip() for l in f if l.strip()]

    fileids = [img_id for img_id in all_ids
               if os.path.exists(os.path.join(jpeg_dir, img_id + '.jpg'))]
    print(f't2_train: {len(all_ids)} total, {len(fileids)} with existing images')

    # --- Build data_per_cat: annotation path list per class (same as food/IDD) ---
    data_per_cat = {c: [] for c in T2_KNOWN_CLASSES}

    print('Scanning annotation XMLs...')
    for i, img_id in enumerate(fileids):
        if i % 5000 == 0:
            print(f'  {i}/{len(fileids)}')
        anno_file = os.path.join(ann_dir, img_id + '.xml')
        if not os.path.exists(anno_file):
            continue
        try:
            tree = ET.parse(anno_file)
            clses = set(obj.find('name').text for obj in tree.findall('object'))
            for cls in clses:
                if cls in data_per_cat:
                    data_per_cat[cls].append(anno_file)
        except Exception:
            continue

    print('\nClass distribution:')
    for cls in T2_KNOWN_CLASSES:
        print(f'  {cls}: {len(data_per_cat[cls])} images')

    # --- TFA cumulative k-shot sampling (exact CED-FOOD logic) ---
    result = {cls: {} for cls in T2_KNOWN_CLASSES}

    for seed in range(args.seeds[0], args.seeds[1]):
        random.seed(seed)
        save_dir = os.path.join(split_dir, f'seed{seed}')
        os.makedirs(save_dir, exist_ok=True)

        for cls in T2_KNOWN_CLASSES:
            cls_pool = data_per_cat[cls]
            if not cls_pool:
                print(f'[WARN] No images for {cls}')
                continue

            c_data = []  # accumulates image paths (cumulative across shots)

            for j, shot in enumerate(shots):
                diff_shot = shots[j] - shots[j - 1] if j != 0 else shots[0]

                if len(cls_pool) < diff_shot:
                    shots_c = cls_pool[:]
                else:
                    shots_c = random.sample(cls_pool, diff_shot)

                num_objs = 0
                for anno_path in shots_c:
                    img_id = os.path.splitext(os.path.basename(anno_path))[0]
                    img_path = f'{OWOD_ROOT}/JPEGImages/{DATASET}/{img_id}.jpg'
                    if img_path in c_data:
                        continue
                    try:
                        tree = ET.parse(anno_path)
                        c_data.append(img_path)
                        for obj in tree.findall('object'):
                            if obj.find('name').text == cls:
                                num_objs += 1
                        if num_objs >= diff_shot:
                            break
                    except Exception:
                        continue

                result[cls][shot] = copy.deepcopy(c_data)

        # Write files
        for cls in T2_KNOWN_CLASSES:
            for shot in shots:
                paths = result[cls].get(shot, [])
                fname = f'box_{shot}shot_{cls}_train.txt'
                with open(os.path.join(save_dir, fname), 'w') as fp:
                    fp.write('\n'.join(paths) + '\n')

        n = len(T2_KNOWN_CLASSES) * len(shots)
        print(f'[seed {seed}] {n} files → {save_dir}')

    total = len(T2_KNOWN_CLASSES) * len(shots) * (args.seeds[1] - args.seeds[0])
    print(f'\nDone. {total} files under {split_dir}')
    print(f'Format: {OWOD_ROOT}/JPEGImages/{DATASET}/{{timestamp}}.jpg')


if __name__ == '__main__':
    args = parse_args()
    generate_seeds(args)
