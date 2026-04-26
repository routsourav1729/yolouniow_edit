#!/usr/bin/env python3
"""
Setup nuOWODB data for YOLO-UniOW.

This script:
1. Extracts images from nuimages-v1.0-all-samples.tgz
2. Creates timestamp-named symlinks in JPEGImages/nuOWODB/
3. Symlinks annotations from the existing HONDA/ovow conversion
4. Verifies everything matches the ImageSets

Usage:
    python tools/dataset_converters/setup_nuowodb.py \
        --nuscenes-root /path/to/nuscenes \
        --annotations-src /path/to/existing/Annotations \
        --owod-root data/OWOD

The script expects:
  - nuscenes-root/nuimages-v1.0-all-samples.tgz (OR already extracted samples/)
  - nuscenes-root/v1.0-train/sample_data.json
  - nuscenes-root/v1.0-val/sample_data.json
  - annotations-src/*.xml  (timestamp-named VOC XMLs)
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict


def build_timestamp_to_path(nuscenes_root):
    """Build mapping: timestamp -> relative image path in samples/.

    The nuImages images are named like:
      samples/CAM_FRONT/n008-...__CAM_FRONT__1533151603512404.jpg
    The timestamp is the last numeric part before .jpg.
    """
    ts_map = {}  # timestamp_str -> full_path

    # Build from sample_data.json (both train and val)
    for split in ['v1.0-train', 'v1.0-val']:
        sd_path = os.path.join(nuscenes_root, split, 'sample_data.json')
        if not os.path.exists(sd_path):
            print(f"WARNING: {sd_path} not found, skipping")
            continue
        with open(sd_path) as f:
            sample_data = json.load(f)
        for entry in sample_data:
            fname = entry['filename']  # e.g. samples/CAM_FRONT/n008-...__CAM_FRONT__1533151603512404.jpg
            basename = os.path.basename(fname)
            # Extract timestamp: last part after __ and before .jpg
            parts = basename.rsplit('__', 1)
            if len(parts) == 2:
                ts = parts[1].replace('.jpg', '')
            else:
                ts = basename.replace('.jpg', '')
            ts_map[ts] = fname

    print(f"Built timestamp mapping: {len(ts_map)} entries")
    return ts_map


def get_needed_ids(imagesets_dir):
    """Read all ImageSet files and return set of needed image IDs."""
    needed = set()
    for txt_file in os.listdir(imagesets_dir):
        if txt_file.endswith('.txt') and not txt_file.endswith('_known.txt'):
            fpath = os.path.join(imagesets_dir, txt_file)
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        needed.add(line)
    print(f"Total unique image IDs needed from ImageSets: {len(needed)}")
    return needed


def extract_tarball(nuscenes_root):
    """Extract the samples tarball if not already extracted."""
    tarball = os.path.join(nuscenes_root, 'nuimages-v1.0-all-samples.tgz')
    samples_dir = os.path.join(nuscenes_root, 'samples')

    # Check if already extracted by seeing if CAM_FRONT has files
    cam_front = os.path.join(samples_dir, 'CAM_FRONT')
    if os.path.isdir(cam_front) and len(os.listdir(cam_front)) > 0:
        print("Images already extracted, skipping extraction")
        return True

    if not os.path.exists(tarball):
        print(f"ERROR: Tarball not found: {tarball}")
        return False

    print(f"Extracting {tarball} (this may take a while, ~16GB compressed)...")
    ret = subprocess.run(
        ['tar', 'xzf', tarball, '-C', nuscenes_root],
        capture_output=False
    )
    if ret.returncode != 0:
        print("ERROR: Extraction failed!")
        return False
    print("Extraction complete")
    return True


def setup_images(nuscenes_root, ts_map, needed_ids, jpeg_dir):
    """Create symlinks: jpeg_dir/{timestamp}.jpg -> actual image path."""
    os.makedirs(jpeg_dir, exist_ok=True)
    created = 0
    missing = 0

    for ts in needed_ids:
        link_path = os.path.join(jpeg_dir, f"{ts}.jpg")
        if os.path.exists(link_path):
            created += 1
            continue

        rel_path = ts_map.get(ts)
        if rel_path is None:
            missing += 1
            if missing <= 10:
                print(f"WARNING: No image found for timestamp {ts}")
            continue

        src_path = os.path.join(nuscenes_root, rel_path)
        if not os.path.exists(src_path):
            missing += 1
            if missing <= 10:
                print(f"WARNING: Image file not found: {src_path}")
            continue

        os.symlink(os.path.abspath(src_path), link_path)
        created += 1

    print(f"Images: {created} linked, {missing} missing")
    return missing


def setup_annotations(annotations_src, annotations_dst, needed_ids):
    """Symlink annotations from source to destination."""
    os.makedirs(annotations_dst, exist_ok=True)
    created = 0
    missing = 0

    for ts in needed_ids:
        link_path = os.path.join(annotations_dst, f"{ts}.xml")
        if os.path.exists(link_path):
            created += 1
            continue

        src_path = os.path.join(annotations_src, f"{ts}.xml")
        if not os.path.exists(src_path):
            missing += 1
            if missing <= 10:
                print(f"WARNING: Annotation not found: {src_path}")
            continue

        os.symlink(os.path.abspath(src_path), link_path)
        created += 1

    print(f"Annotations: {created} linked, {missing} missing")
    return missing


def verify(jpeg_dir, annotations_dir, imagesets_dir):
    """Verify all ImageSet entries have corresponding images and annotations."""
    errors = 0
    for txt_file in sorted(os.listdir(imagesets_dir)):
        if not txt_file.endswith('.txt') or txt_file.endswith('_known.txt'):
            continue
        fpath = os.path.join(imagesets_dir, txt_file)
        with open(fpath) as f:
            ids = [l.strip() for l in f if l.strip()]
        img_ok = sum(1 for i in ids if os.path.exists(os.path.join(jpeg_dir, f"{i}.jpg")))
        ann_ok = sum(1 for i in ids if os.path.exists(os.path.join(annotations_dir, f"{i}.xml")))
        print(f"  {txt_file}: {len(ids)} entries, {img_ok} images, {ann_ok} annotations")
        if img_ok < len(ids) or ann_ok < len(ids):
            errors += len(ids) - min(img_ok, ann_ok)
    return errors


def main():
    parser = argparse.ArgumentParser(description='Setup nuOWODB for YOLO-UniOW')
    parser.add_argument('--nuscenes-root',
                        default='/home/agipml/sourav.rout/ALL_FILES/HONDA/ovow/data/nuscenes',
                        help='Path to nuscenes data (with tgz and metadata)')
    parser.add_argument('--annotations-src',
                        default='/home/agipml/sourav.rout/ALL_FILES/HONDA/ovow/datasets/nu-owodb/Annotations',
                        help='Path to existing converted VOC annotations')
    parser.add_argument('--owod-root',
                        default='data/OWOD',
                        help='Path to OWOD data root in YOLO-UniOW')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip tarball extraction (if images already extracted)')
    args = parser.parse_args()

    imagesets_dir = os.path.join(args.owod_root, 'ImageSets', 'nuOWODB')
    jpeg_dir = os.path.join(args.owod_root, 'JPEGImages', 'nuOWODB')
    annotations_dir = os.path.join(args.owod_root, 'Annotations', 'nuOWODB')

    if not os.path.isdir(imagesets_dir):
        print(f"ERROR: ImageSets not found: {imagesets_dir}")
        sys.exit(1)

    # Step 1: Get needed image IDs
    print("=" * 60)
    print("Step 1: Reading ImageSets")
    needed_ids = get_needed_ids(imagesets_dir)

    # Step 2: Build timestamp -> image path mapping
    print("=" * 60)
    print("Step 2: Building timestamp-to-image mapping")
    ts_map = build_timestamp_to_path(args.nuscenes_root)

    # Step 3: Extract images if needed
    if not args.skip_extract:
        print("=" * 60)
        print("Step 3: Extracting images")
        if not extract_tarball(args.nuscenes_root):
            print("Failed to extract. Use --skip-extract if already extracted elsewhere.")
            sys.exit(1)
    else:
        print("Step 3: Skipping extraction")

    # Step 4: Create image symlinks
    print("=" * 60)
    print("Step 4: Setting up image symlinks")
    img_missing = setup_images(args.nuscenes_root, ts_map, needed_ids, jpeg_dir)

    # Step 5: Create annotation symlinks
    print("=" * 60)
    print("Step 5: Setting up annotation symlinks")
    ann_missing = setup_annotations(args.annotations_src, annotations_dir, needed_ids)

    # Step 6: Verify
    print("=" * 60)
    print("Step 6: Verification")
    errors = verify(jpeg_dir, annotations_dir, imagesets_dir)

    print("=" * 60)
    if errors == 0 and img_missing == 0 and ann_missing == 0:
        print("SUCCESS: nuOWODB setup complete!")
    else:
        print(f"DONE with issues: {img_missing} missing images, {ann_missing} missing annotations")

    # Print class name info
    print("\n" + "=" * 60)
    print("Class names in annotations (nuScenes format):")
    print("  T1 (Base/Vehicles):     vehicle.bicycle, vehicle.car, ...")
    print("  T2 (Novel/Pedestrians): human.pedestrian.adult, ...")
    print("  T3 (Unknown/Obstacles): movable_object.barrier, ...")
    print("\nThese match owodb_const.py VOC_COCO_CLASS_NAMES['nuOWODB']")


if __name__ == '__main__':
    main()
