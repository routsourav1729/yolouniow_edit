#!/usr/bin/env python3
"""
TFA-style few-shot sampling for nuOWODB — following the IDD protocol.

Protocol:
  - Source data  : t2_train.txt images (contains both T1 and new T2 images)
  - BASE classes : T1_CLASSES (10 vehicle classes)   → k-shot sampled for base fine-tune
  - NOVEL classes: T2_CLASSES (7 pedestrian classes) → k-shot sampled for novel fine-tune
  - UNKNOWN      : T3_CLASSES (not sampled)
  - Test set     : test.txt  (T1 base + T2 novel + T3 unknown)

Outputs per seed:
  nuowodb_fewshot_tfa/seed{s}/
    box_{k}shot_{cls}_train.txt          # per-class image-ID lists (YOLO-UniOW style)
    {k}shot_base_train.json              # TFA COCO JSON, base classes only
    {k}shot_novel_train.json             # TFA COCO JSON, novel classes only

TFA COCO JSON format: each annotation has ONLY the target class (TFA-filtered).
Images may appear multiple times across classes.

Usage:
    cd YOLO-UniOW
    python tools/dataset_converters/prepare_nuowodb_tfa.py \
        --owod-root data/OWOD \
        --seeds 1 11 \
        --shots 1 2 3 5 10 20 30

Then train T1 (base) normally:
    DATASET=nuOWODB TASK=1 ./tools/dist_train_owod.sh <cfg> <gpus> --amp

Then fine-tune T2 (novel) with TFA:
    DATASET=nuOWODB TASK=2 FEWSHOT_K=10 FEWSHOT_SEED=1 \\
    FEWSHOT_DIR=data/OWOD/nuowodb_fewshot_tfa \\
    ./tools/dist_train_owod.sh <cfg> <gpus> --amp
"""
import argparse
import json
import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy


# nuOWODB class definitions (must match owodb_const.py)
T1_CLASSES = [
    "vehicle.bicycle",
    "vehicle.motorcycle",
    "vehicle.car",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.truck",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.construction",
    "vehicle.trailer",
]

T2_CLASSES = [
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.wheelchair",
    "human.pedestrian.stroller",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.construction_worker",
]

T3_CLASSES = [
    "movable_object.barrier",
    "movable_object.trafficcone",
    "movable_object.pushable_pullable",
    "movable_object.debris",
    "static_object.bicycle_rack",
    "animal",
]

ALL_CLASSES = T1_CLASSES + T2_CLASSES + T3_CLASSES
CLS_TO_ID = {c: i for i, c in enumerate(ALL_CLASSES)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--owod-root", default="data/OWOD")
    p.add_argument("--seeds", type=int, nargs=2, default=[1, 11],
                   help="Seed range [start, end)")
    p.add_argument("--shots", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 30])
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.owod_root, "nuowodb_fewshot_tfa")
    return args


def load_t2_annotations(owod_root):
    """
    Scan all t2_train images and build per-class object index.

    Returns:
        data_per_cls  : {cls: [(img_id, obj_idx, bbox, width, height), ...]}
        img_meta      : {img_id: {cls: [(obj_idx, bbox), ...], 'w': w, 'h': h}}
    """
    ann_dir = os.path.join(owod_root, "Annotations", "nuOWODB")

    with open(os.path.join(owod_root, "ImageSets", "nuOWODB", "t2_train.txt")) as f:
        t2_ids = [l.strip() for l in f if l.strip()]

    data_per_cls = defaultdict(list)
    img_meta = {}

    missing = 0
    for img_id in t2_ids:
        path = os.path.join(ann_dir, f"{img_id}.xml")
        if not os.path.exists(path):
            missing += 1
            continue

        tree = ET.parse(path)
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        per_cls = defaultdict(list)
        for idx, obj in enumerate(root.findall("object")):
            cls = obj.find("name").text
            if cls not in ALL_CLASSES:
                continue
            bb = obj.find("bndbox")
            bbox = [float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]
            # skip degenerate boxes
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            per_cls[cls].append((idx, bbox))

        if not per_cls:
            continue

        img_meta[img_id] = {"cls": dict(per_cls), "w": w, "h": h}
        for cls, objs in per_cls.items():
            for obj_idx, bbox in objs:
                data_per_cls[cls].append((img_id, obj_idx, bbox, w, h))

    print(f"Loaded {len(img_meta)} images ({missing} missing annotations)")
    print("\nPer-class object counts (from t2_train):")
    for cls in T1_CLASSES + T2_CLASSES:
        tag = "BASE" if cls in T1_CLASSES else "NOVEL"
        print(f"  [{tag:5s}] {cls}: {len(data_per_cls.get(cls, []))} objects")
    return data_per_cls, img_meta


def sample_k_shot_incremental(data_per_cls, target_classes, shots, rng):
    """
    Sample k-shot objects per class incrementally.
    Returns {cls: {shot: [img_id, ...]}} — image IDs (may overlap across classes).
    """
    shots = sorted(shots)
    result = {}

    for cls in target_classes:
        available = list(data_per_cls.get(cls, []))
        if not available:
            print(f"  WARNING: 0 objects for {cls}")
            result[cls] = {s: [] for s in shots}
            continue

        objs = available.copy()
        rng.shuffle(objs)

        selected_ids = []   # ordered list of image IDs chosen so far
        selected_set = set()
        result[cls] = {}
        total_objs = 0

        for shot in shots:
            target = shot - (shots[shots.index(shot) - 1] if shots.index(shot) > 0 else 0)
            added = 0
            for img_id, obj_idx, bbox, w, h in objs:
                if img_id in selected_set:
                    continue
                # count how many objects of this class in this image
                n = sum(1 for (iid, _, _, _, _) in objs if iid == img_id)
                selected_ids.append(img_id)
                selected_set.add(img_id)
                added += n
                total_objs += n
                if added >= target:
                    break
            result[cls][shot] = list(selected_ids)

    return result


def write_txt_splits(sampled, output_dir, seed_idx):
    """Write per-class image ID .txt files (YOLO-UniOW format)."""
    seed_dir = os.path.join(output_dir, f"seed{seed_idx}")
    os.makedirs(seed_dir, exist_ok=True)

    for cls, shots_dict in sampled.items():
        for shot, img_ids in shots_dict.items():
            fname = f"box_{shot}shot_{cls}_train.txt"
            with open(os.path.join(seed_dir, fname), "w") as f:
                f.write("\n".join(img_ids) + ("\n" if img_ids else ""))


def build_coco_json(sampled_classes, target_classes, data_per_cls, img_meta, shot):
    """
    Build a TFA-filtered COCO JSON.
    Each annotation entry contains ONLY the target class objects
    (standard TFA fine-tuning: novel-only or base-only).
    """
    images_out = []
    annotations_out = []
    img_id_map = {}
    img_counter = 1
    ann_counter = 1

    # Collect all (img_id, cls, bbox) triples for this shot
    entries = []
    for cls in target_classes:
        img_ids = sampled_classes.get(cls, {}).get(shot, [])
        for img_id in img_ids:
            meta = img_meta.get(img_id)
            if meta is None:
                continue
            for obj_idx, bbox in meta["cls"].get(cls, []):
                entries.append((img_id, cls, bbox, meta["w"], meta["h"]))

    # Deduplicate images
    seen_imgs = {}
    for img_id, cls, bbox, w, h in entries:
        if img_id not in seen_imgs:
            seen_imgs[img_id] = img_counter
            images_out.append({
                "id": img_counter,
                "file_name": f"{img_id}.jpg",
                "width": w,
                "height": h,
            })
            img_counter += 1

        x1, y1, x2, y2 = bbox
        annotations_out.append({
            "id": ann_counter,
            "image_id": seen_imgs[img_id],
            "category_id": CLS_TO_ID[cls],
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area": (x2 - x1) * (y2 - y1),
            "iscrowd": 0,
        })
        ann_counter += 1

    categories = [{"id": i, "name": c, "supercategory": "object"}
                  for i, c in enumerate(ALL_CLASSES)]

    return {"images": images_out, "annotations": annotations_out, "categories": categories}


def main():
    args = parse_args()

    print("=" * 65)
    print("nuOWODB TFA Few-Shot Sampling")
    print(f"  OWOD root : {args.owod_root}")
    print(f"  Output    : {args.output_dir}")
    print(f"  Seeds     : {args.seeds[0]} – {args.seeds[1]-1}")
    print(f"  Shots     : {args.shots}")
    print(f"  BASE (T1) : {len(T1_CLASSES)} classes")
    print(f"  NOVEL (T2): {len(T2_CLASSES)} classes")
    print("=" * 65)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading T2-train annotations...")
    data_per_cls, img_meta = load_t2_annotations(args.owod_root)

    for seed_idx in range(args.seeds[0], args.seeds[1]):
        print(f"\n{'='*65}")
        print(f"Seed {seed_idx}")
        rng = random.Random(seed_idx)

        # Sample base (T1) classes
        rng_base = random.Random(seed_idx * 1000 + 1)
        base_sampled = sample_k_shot_incremental(data_per_cls, T1_CLASSES, args.shots, rng_base)

        # Sample novel (T2) classes
        rng_novel = random.Random(seed_idx * 1000 + 2)
        novel_sampled = sample_k_shot_incremental(data_per_cls, T2_CLASSES, args.shots, rng_novel)

        # Write .txt splits (YOLO-UniOW format) — one file per class per shot
        write_txt_splits(base_sampled, args.output_dir, seed_idx)
        write_txt_splits(novel_sampled, args.output_dir, seed_idx)

        # Write TFA COCO JSONs per shot
        seed_dir = os.path.join(args.output_dir, f"seed{seed_idx}")
        for shot in sorted(args.shots):
            # Base JSON
            base_json = build_coco_json(base_sampled, T1_CLASSES, data_per_cls, img_meta, shot)
            base_path = os.path.join(seed_dir, f"{shot}shot_base_train.json")
            with open(base_path, "w") as f:
                json.dump(base_json, f)

            # Novel JSON
            novel_json = build_coco_json(novel_sampled, T2_CLASSES, data_per_cls, img_meta, shot)
            novel_path = os.path.join(seed_dir, f"{shot}shot_novel_train.json")
            with open(novel_path, "w") as f:
                json.dump(novel_json, f)

            n_base_imgs = len(base_json["images"])
            n_base_anns = len(base_json["annotations"])
            n_novel_imgs = len(novel_json["images"])
            n_novel_anns = len(novel_json["annotations"])
            print(f"  {shot:2d}-shot  base: {n_base_imgs:4d} imgs {n_base_anns:5d} anns | "
                  f"novel: {n_novel_imgs:4d} imgs {n_novel_anns:5d} anns")

    print("\n" + "=" * 65)
    print("DONE")
    print(f"\nOutput layout:")
    print(f"  {args.output_dir}/")
    print(f"    seed{{s}}/")
    print(f"      box_{{k}}shot_{{cls}}_train.txt  (YOLO-UniOW format)")
    print(f"      {{k}}shot_base_train.json        (TFA COCO JSON, T1 classes)")
    print(f"      {{k}}shot_novel_train.json       (TFA COCO JSON, T2 classes)")
    print()
    print("T1 base training (no fewshot):")
    print("  DATASET=nuOWODB TASK=1 ./tools/dist_train_owod.sh <cfg> <gpus> --amp")
    print()
    print("T2 novel fine-tune (TFA, YOLO-UniOW):")
    print("  DATASET=nuOWODB TASK=2 FEWSHOT_K=10 FEWSHOT_SEED=1 \\")
    print(f"  FEWSHOT_DIR={args.output_dir} \\")
    print("  ./tools/dist_train_owod.sh <cfg> <gpus> --amp")


if __name__ == "__main__":
    main()
