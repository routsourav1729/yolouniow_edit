#!/usr/bin/env python3
"""
Convert nuImages annotations (both train + val) to Pascal VOC XML format.

Processes v1.0-train and v1.0-val splits, outputs one XML per image.
Image names are the basename of the filename in sample_data.json,
with the timestamp as the stem (matching setup_nuowodb.py convention).

Usage:
    python tools/dataset_converters/nuimages_to_voc_fresh.py \
        --nuimages-root data/new \
        --output-dir data/new/Annotations
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict


def load_category_mapping(json_file):
    with open(json_file) as f:
        categories = json.load(f)
    return {cat["token"]: cat["name"] for cat in categories}


def load_image_metadata(json_file):
    with open(json_file) as f:
        data = json.load(f)
    # token -> {filename, width, height}
    return {entry["token"]: entry for entry in data}


def write_voc_xml(xml_path, image_name, width, height, objects):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "nuOWODB"
    ET.SubElement(root, "filename").text = image_name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"

    for obj in objects:
        obj_elem = ET.SubElement(root, "object")
        ET.SubElement(obj_elem, "name").text = obj["class_name"]
        ET.SubElement(obj_elem, "pose").text = "Unspecified"
        ET.SubElement(obj_elem, "truncated").text = "0"
        ET.SubElement(obj_elem, "difficult").text = "0"
        bndbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(obj["xmin"]))
        ET.SubElement(bndbox, "ymin").text = str(int(obj["ymin"]))
        ET.SubElement(bndbox, "xmax").text = str(int(obj["xmax"]))
        ET.SubElement(bndbox, "ymax").text = str(int(obj["ymax"]))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="unicode", xml_declaration=False)


def process_split(split_dir, output_dir):
    cat_map = load_category_mapping(os.path.join(split_dir, "category.json"))
    img_map = load_image_metadata(os.path.join(split_dir, "sample_data.json"))

    with open(os.path.join(split_dir, "object_ann.json")) as f:
        annotations = json.load(f)

    # Group annotations by image token
    by_image = defaultdict(list)
    for ann in annotations:
        by_image[ann["sample_data_token"]].append(ann)

    written = 0
    skipped = 0
    for img_token, anns in by_image.items():
        img_info = img_map.get(img_token)
        if not img_info:
            skipped += 1
            continue

        fname = os.path.basename(img_info["filename"])
        # Extract timestamp: last part after __ before .jpg
        parts = fname.rsplit("__", 1)
        stem = parts[1].replace(".jpg", "") if len(parts) == 2 else fname.replace(".jpg", "")

        xml_path = os.path.join(output_dir, f"{stem}.xml")

        objects = []
        for ann in anns:
            bbox = ann["bbox"]  # [xmin, ymin, xmax, ymax]
            cls = cat_map.get(ann["category_token"], "Unknown")
            # Skip annotations with invalid boxes
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            objects.append({
                "class_name": cls,
                "xmin": bbox[0],
                "ymin": bbox[1],
                "xmax": bbox[2],
                "ymax": bbox[3],
            })

        if not objects:
            continue

        write_voc_xml(xml_path, fname, img_info["width"], img_info["height"], objects)
        written += 1

    return written, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuimages-root", default="data/new",
                        help="Path containing v1.0-train/ and v1.0-val/")
    parser.add_argument("--output-dir", default="data/new/Annotations",
                        help="Where to write VOC XMLs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_written = 0
    for split in ["v1.0-train", "v1.0-val"]:
        split_dir = os.path.join(args.nuimages_root, split)
        if not os.path.isdir(split_dir):
            print(f"WARNING: {split_dir} not found, skipping")
            continue
        print(f"Processing {split}...")
        written, skipped = process_split(split_dir, args.output_dir)
        print(f"  Written: {written}, skipped (no img meta): {skipped}")
        total_written += written

    print(f"\nTotal XMLs written: {total_written}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
