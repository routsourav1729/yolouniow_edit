#!/usr/bin/env python3
"""
Set up JPEGImages symlinks for nuOWODB.

Reads ImageSets to find needed timestamp IDs, builds a timestamp->file mapping
from sample_data.json, then creates symlinks in JPEGImages/nuOWODB/.

Usage:
    python tools/dataset_converters/setup_nuowodb_images.py \
        --nuimages-root data/new \
        --owod-root data/OWOD
"""
import argparse
import json
import os
import sys
from collections import defaultdict


def build_timestamp_map(nuimages_root):
    ts_map = {}
    for split in ["v1.0-train", "v1.0-val"]:
        sd_path = os.path.join(nuimages_root, split, "sample_data.json")
        if not os.path.exists(sd_path):
            print(f"WARNING: {sd_path} not found")
            continue
        with open(sd_path) as f:
            data = json.load(f)
        for entry in data:
            fname = entry["filename"]  # e.g. samples/CAM_FRONT/n008-....__CAM_FRONT__1533...jpg
            basename = os.path.basename(fname)
            parts = basename.rsplit("__", 1)
            ts = parts[1].replace(".jpg", "") if len(parts) == 2 else basename.replace(".jpg", "")
            ts_map[ts] = fname
    print(f"Timestamp map: {len(ts_map)} entries")
    return ts_map


def get_all_needed_ids(imagesets_dir):
    needed = set()
    for txt in os.listdir(imagesets_dir):
        if txt.endswith(".txt") and not txt.endswith("_known.txt"):
            with open(os.path.join(imagesets_dir, txt)) as f:
                for line in f:
                    l = line.strip()
                    if l:
                        needed.add(l)
    print(f"Unique IDs from ImageSets: {len(needed)}")
    return needed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuimages-root", default="data/new")
    parser.add_argument("--owod-root", default="data/OWOD")
    parser.add_argument("--dataset", default="nuOWODB")
    args = parser.parse_args()

    imagesets_dir = os.path.join(args.owod_root, "ImageSets", args.dataset)
    jpeg_dir = os.path.join(args.owod_root, "JPEGImages", args.dataset)

    if not os.path.isdir(imagesets_dir):
        print(f"ERROR: {imagesets_dir} not found")
        sys.exit(1)

    os.makedirs(jpeg_dir, exist_ok=True)

    needed = get_all_needed_ids(imagesets_dir)
    ts_map = build_timestamp_map(args.nuimages_root)

    created = 0
    missing_ts = 0
    missing_file = 0

    for ts in sorted(needed):
        link_path = os.path.join(jpeg_dir, f"{ts}.jpg")
        if os.path.lexists(link_path):
            created += 1
            continue

        rel = ts_map.get(ts)
        if rel is None:
            missing_ts += 1
            if missing_ts <= 5:
                print(f"  WARNING: No ts mapping for {ts}")
            continue

        src = os.path.join(args.nuimages_root, rel)
        if not os.path.exists(src):
            missing_file += 1
            if missing_file <= 5:
                print(f"  WARNING: File missing: {src}")
            continue

        os.symlink(os.path.abspath(src), link_path)
        created += 1

    print(f"\nImage symlinks: created={created}, missing_ts={missing_ts}, missing_file={missing_file}")

    # Verify per ImageSet
    print("\nVerification per ImageSet:")
    for txt in sorted(os.listdir(imagesets_dir)):
        if not txt.endswith(".txt") or txt.endswith("_known.txt"):
            continue
        with open(os.path.join(imagesets_dir, txt)) as f:
            ids = [l.strip() for l in f if l.strip()]
        ok = sum(1 for i in ids if os.path.lexists(os.path.join(jpeg_dir, f"{i}.jpg")))
        print(f"  {txt}: {ok}/{len(ids)}")


if __name__ == "__main__":
    main()
