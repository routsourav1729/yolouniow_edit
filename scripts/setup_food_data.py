#!/usr/bin/env python3
"""Set up FOOD VOC10-5-5 and FOOD VOC-COCO data directories for YOLO-UniOW.

Creates:
  data/OWOD/ImageSets/FOOD_VOC/   (t1_known, t2_known, t1_train, t2_train, test, t_all_known)
  data/OWOD/ImageSets/FOOD_VOCCOCO/ (same)
  data/OWOD/JPEGImages/FOOD_VOC   -> symlink to REPO/food/datasets VOC images
  data/OWOD/JPEGImages/FOOD_VOCCOCO -> symlink to REPO/food/datasets voc_coco images
  data/OWOD/Annotations/FOOD_VOC  -> symlink
  data/OWOD/Annotations/FOOD_VOCCOCO -> symlink

Fewshot splits (already exist as symlinks):
  data/OWOD/vocsplit/      -> REPO/food/datasets/vocsplit
  data/OWOD/voccocosplit/  -> REPO/food/datasets/voccocosplit

Prerequisite: REPO/food datasets must already be prepared
  (prepare_food_voc_coco.sh already run).
"""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

FOOD_ROOT = Path('/home/agipml/sourav.rout/ALL_FILES/REPO/food')
OWOD_ROOT = Path('/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW/data/OWOD')

VOC07 = FOOD_ROOT / 'datasets' / 'VOC2007'
VOC12 = FOOD_ROOT / 'datasets' / 'VOC2012'
VOCCOCO = FOOD_ROOT / 'datasets' / 'voc_coco'

# -----------------------------------------------------------------------
# FOOD VOC10-5-5 Split 1 class definitions
# -----------------------------------------------------------------------
FOOD_VOC_BASE = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
]
FOOD_VOC_NOVEL = [
    "diningtable", "dog", "horse", "motorbike", "person",
]
FOOD_VOC_UNKNOWN = [
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
FOOD_VOC_ALL = FOOD_VOC_BASE + FOOD_VOC_NOVEL + FOOD_VOC_UNKNOWN

# -----------------------------------------------------------------------
# FOOD VOC-COCO class definitions
# -----------------------------------------------------------------------
FOOD_VOCCOCO_BASE = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
FOOD_VOCCOCO_NOVEL = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator",
]
FOOD_VOCCOCO_UNKNOWN = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl",
]
FOOD_VOCCOCO_ALL = FOOD_VOCCOCO_BASE + FOOD_VOCCOCO_NOVEL + FOOD_VOCCOCO_UNKNOWN


def write_txt(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"  Wrote {path} ({len(lines)} lines)")


def read_imageset(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def get_voc_image_ids_with_classes(voc_dir, imageset_file, target_classes):
    """Get image IDs that contain at least one annotation from target_classes."""
    ids = read_imageset(imageset_file)
    filtered = []
    for img_id in ids:
        anno_file = voc_dir / 'Annotations' / f'{img_id}.xml'
        if not anno_file.exists():
            continue
        tree = ET.parse(str(anno_file))
        for obj in tree.findall('object'):
            cls = obj.find('name').text
            if cls in target_classes:
                filtered.append(img_id)
                break
    return filtered


def symlink_force(target, link_name):
    """Create symlink, removing existing if needed."""
    link_name = Path(link_name)
    if link_name.exists() or link_name.is_symlink():
        link_name.unlink()
    link_name.symlink_to(target)
    print(f"  Symlink: {link_name} -> {target}")


def setup_food_voc():
    """Set up FOOD VOC10-5-5 Split 1."""
    print("=" * 60)
    print("Setting up FOOD_VOC (VOC10-5-5 Split 1)")
    print("=" * 60)

    dataset = "FOOD_VOC"
    imgsets_dir = OWOD_ROOT / 'ImageSets' / dataset
    os.makedirs(imgsets_dir, exist_ok=True)

    # --- t1_known.txt: 10 base class names ---
    write_txt(imgsets_dir / 't1_known.txt', FOOD_VOC_BASE)

    # --- t2_known.txt: 15 known class names (base + novel) ---
    write_txt(imgsets_dir / 't2_known.txt', FOOD_VOC_BASE + FOOD_VOC_NOVEL)

    # --- t_all_known.txt: all 20 VOC classes (for full eval) ---
    write_txt(imgsets_dir / 't_all_known.txt', FOOD_VOC_ALL)

    # --- t1_train.txt: VOC07+12 trainval with at least one base-class annotation ---
    # FOOD base training uses voc_2007_trainval_base1 + voc_2012_trainval_base1
    # which filter to only images containing base classes
    print("  Scanning VOC07 trainval for base-class images...")
    voc07_train = get_voc_image_ids_with_classes(
        VOC07, VOC07 / 'ImageSets' / 'Main' / 'trainval.txt', set(FOOD_VOC_BASE))
    print(f"    VOC07: {len(voc07_train)} images")

    print("  Scanning VOC12 trainval for base-class images...")
    voc12_train = get_voc_image_ids_with_classes(
        VOC12, VOC12 / 'ImageSets' / 'Main' / 'trainval.txt', set(FOOD_VOC_BASE))
    print(f"    VOC12: {len(voc12_train)} images")

    all_train = voc07_train + voc12_train
    write_txt(imgsets_dir / 't1_train.txt', all_train)

    # --- t2_train.txt: same images (fewshot handles novel data separately) ---
    # In FOOD's protocol, T2 finetuning uses fewshot files, not a separate imageset.
    # But the OWODDataset._extract_fns_fewshot() ignores t2_train.txt when
    # fewshot_dir is set. We still create it for completeness.
    write_txt(imgsets_dir / 't2_train.txt', all_train)

    # --- test.txt: VOC07 test (all 21 classes including unknown for open-set eval) ---
    # FOOD evaluates on voc_2007_test_all1 which reads VOC2007 test split
    voc07_test = read_imageset(VOC07 / 'ImageSets' / 'Main' / 'test.txt')
    write_txt(imgsets_dir / 'test.txt', voc07_test)

    # --- Symlinks for images and annotations ---
    # FOOD_VOC images come from both VOC2007 and VOC2012.
    # OWODDataset expects: JPEGImages/FOOD_VOC/{id}.jpg and Annotations/FOOD_VOC/{id}.xml
    # Since IDs can be from either year (6-digit for VOC07, 2009_XXXXXX for VOC12),
    # we create a merged directory with symlinks.
    jpeg_dir = OWOD_ROOT / 'JPEGImages' / dataset
    anno_dir = OWOD_ROOT / 'Annotations' / dataset
    os.makedirs(jpeg_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)

    # Symlink ALL VOC07+12 trainval+test images — not just the base-filtered set.
    # Fewshot files reference VOC12 images with only novel classes (no base class),
    # which are excluded from t1_train.txt but still needed at T2 load time.
    voc07_all = read_imageset(VOC07 / 'ImageSets' / 'Main' / 'trainval.txt') + \
                read_imageset(VOC07 / 'ImageSets' / 'Main' / 'test.txt')
    voc12_all = read_imageset(VOC12 / 'ImageSets' / 'Main' / 'trainval.txt')
    all_ids = set(all_train + voc07_test + voc07_all + voc12_all)
    print(f"  Creating symlinks for {len(all_ids)} unique images (full VOC07+12)...")
    created = 0
    for img_id in all_ids:
        # Determine source directory
        if '_' in img_id:
            src_dir = VOC12
        else:
            src_dir = VOC07

        src_img = src_dir / 'JPEGImages' / f'{img_id}.jpg'
        src_ann = src_dir / 'Annotations' / f'{img_id}.xml'
        dst_img = jpeg_dir / f'{img_id}.jpg'
        dst_ann = anno_dir / f'{img_id}.xml'

        if src_img.exists() and not dst_img.exists():
            dst_img.symlink_to(src_img)
            created += 1
        if src_ann.exists() and not dst_ann.exists():
            dst_ann.symlink_to(src_ann)

    print(f"  Created {created} new image symlinks")
    print(f"  Total images: {len(list(jpeg_dir.glob('*.jpg')))}")
    print(f"  Total annotations: {len(list(anno_dir.glob('*.xml')))}")


def setup_food_voccoco():
    """Set up FOOD VOC-COCO."""
    print()
    print("=" * 60)
    print("Setting up FOOD_VOCCOCO (VOC-COCO)")
    print("=" * 60)

    dataset = "FOOD_VOCCOCO"
    imgsets_dir = OWOD_ROOT / 'ImageSets' / dataset
    os.makedirs(imgsets_dir, exist_ok=True)

    # --- t1_known.txt: 20 VOC base class names ---
    write_txt(imgsets_dir / 't1_known.txt', FOOD_VOCCOCO_BASE)

    # --- t2_known.txt: 40 known class names (base + novel) ---
    write_txt(imgsets_dir / 't2_known.txt', FOOD_VOCCOCO_BASE + FOOD_VOCCOCO_NOVEL)

    # --- t_all_known.txt: all 80 COCO classes ---
    write_txt(imgsets_dir / 't_all_known.txt', FOOD_VOCCOCO_ALL)

    # --- t1_train.txt: VOC07 trainval + VOC12 trainval ---
    # CED-FOOD VOC-COCO base config: TRAIN: ('voc_2007_trainval', 'voc_2012_trainval',)
    # These are detectron2 NATIVE registrations → VOC2007/trainval (5011) + VOC2012/trainval (11540)
    # All 20 VOC classes are base → every image has annotations → no filtering by FILTER_EMPTY_ANNOTATIONS.
    # Total = 16,551.
    # NOTE: do NOT confuse with 'voc_2007_train1' (2,501, voc07train.txt) from register_all_voc_coco()
    #       which is a DIFFERENT dataset name not used in the base config.
    print("  Reading VOC imagesets for base training...")
    voc07_trainval = read_imageset(VOC07 / 'ImageSets' / 'Main' / 'trainval.txt')
    voc12_trainval = read_imageset(VOC12 / 'ImageSets' / 'Main' / 'trainval.txt')
    print(f"    VOC07 trainval: {len(voc07_trainval)} images")
    print(f"    VOC12 trainval: {len(voc12_trainval)} images")
    train_ids = voc07_trainval + voc12_trainval
    write_txt(imgsets_dir / 't1_train.txt', train_ids)

    # --- t2_train.txt: same (fewshot handles novel via voccocosplit) ---
    write_txt(imgsets_dir / 't2_train.txt', train_ids)

    # --- test.txt: COCO val2017 (voc_coco_test in CED-FOOD) ---
    voccoco_test_file = VOCCOCO / 'ImageSets' / 'Main' / 'instances_val2017.txt'
    if voccoco_test_file.exists():
        test_ids = read_imageset(voccoco_test_file)
        print(f"  Test set (COCO val2017): {len(test_ids)} images")
    else:
        print("  WARNING: voc_coco test imageset not found!")
        test_ids = []
    write_txt(imgsets_dir / 'test.txt', test_ids)

    # --- Create image/annotation dir symlinks ---
    # voc_coco is now complete (COCO train+val + VOC07+12) — just symlink dirs.
    jpeg_dir = OWOD_ROOT / 'JPEGImages' / dataset
    anno_dir = OWOD_ROOT / 'Annotations' / dataset

    for d in [jpeg_dir, anno_dir]:
        if d.is_dir() and not d.is_symlink():
            import shutil
            shutil.rmtree(d)
            print(f"  Removed old real dir: {d}")
        elif d.is_symlink():
            d.unlink()

    jpeg_dir.symlink_to(VOCCOCO / 'JPEGImages')
    anno_dir.symlink_to(VOCCOCO / 'Annotations')
    print(f"  JPEGImages -> {os.readlink(jpeg_dir)}")
    print(f"  Annotations -> {os.readlink(anno_dir)}")

    # Verify all train IDs resolve
    missing_train = [i for i in train_ids if not (jpeg_dir / f'{i}.jpg').exists()]
    if missing_train:
        print(f"  WARNING: {len(missing_train)} t1_train IDs missing images!")
        print(f"    First 5: {missing_train[:5]}")
    else:
        print(f"  ✓ All {len(train_ids)} t1_train images verified")

    # Verify all test IDs resolve
    missing_test = [i for i in test_ids if not (jpeg_dir / f'{i}.jpg').exists()]
    if missing_test:
        print(f"  WARNING: {len(missing_test)} test IDs missing images!")
        print(f"    First 5: {missing_test[:5]}")
    else:
        print(f"  ✓ All {len(test_ids)} test images verified")


def setup_fewshot_symlinks():
    """Ensure vocsplit and voccocosplit are symlinked into data/OWOD/."""
    print()
    print("=" * 60)
    print("Verifying fewshot split symlinks")
    print("=" * 60)

    for name, src in [
        ('vocsplit', FOOD_ROOT / 'datasets' / 'vocsplit'),
        ('voccocosplit', FOOD_ROOT / 'datasets' / 'voccocosplit'),
    ]:
        dst = OWOD_ROOT / name
        if dst.is_symlink() or dst.exists():
            print(f"  {name}: already exists -> {os.readlink(dst) if dst.is_symlink() else 'dir'}")
        else:
            if src.exists():
                symlink_force(src, dst)
            else:
                print(f"  WARNING: source {src} does not exist!")


def main():
    setup_food_voc()
    setup_food_voccoco()
    setup_fewshot_symlinks()

    print()
    print("=" * 60)
    print("DONE — Data directories ready for FOOD_VOC and FOOD_VOCCOCO")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Generate embeddings:")
    print("     python tools/owod_scripts/extract_embeddings.py --dataset FOOD_VOC --task 1")
    print("     python tools/owod_scripts/extract_embeddings.py --dataset FOOD_VOCCOCO --task 1")
    print("  2. Train T1 base:")
    print("     DATASET=FOOD_VOC TASK=1 ./tools/dist_train_owod.sh ...")
    print("  3. Train T2 fewshot:")
    print("     DATASET=FOOD_VOC TASK=2 FEWSHOT_K=10 FEWSHOT_DIR=data/OWOD/vocsplit ...")


if __name__ == '__main__':
    main()
