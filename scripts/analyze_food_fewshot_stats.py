#!/usr/bin/env python3
"""Analyze FOOD few-shot splits using the current YOLO-UniOW T2 logic.

For each target class, this script reads the corresponding
``box_{k}shot_{class}_train.txt`` file, parses the referenced VOC-style XMLs,
counts class-specific instances, and reproduces the per-class k-instance cap
used by ``OWODDataset._cap_fewshot_instances``.
"""

from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


DATASET_CONFIGS = {
    "FOOD_VOC": {
        "split_dir": Path("data/OWOD/vocsplit"),
        "annotations_dir": Path("data/OWOD/Annotations/FOOD_VOC"),
        "novel_classes": [
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
        ],
    },
    "FOOD_VOCCOCO": {
        "split_dir": Path("data/OWOD/voccocosplit"),
        "annotations_dir": Path("data/OWOD/Annotations/FOOD_VOCCOCO"),
        "novel_classes": [
            "truck",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS),
        required=True,
        help="Dataset to analyze",
    )
    parser.add_argument("--shot", type=int, default=10, help="Few-shot k")
    parser.add_argument("--seed", type=int, default=1, help="Few-shot seed")
    return parser.parse_args()


def read_split_image_ids(split_file: Path) -> list[str]:
    image_ids = []
    with split_file.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            image_ids.append(Path(line).stem)
    return image_ids


def parse_class_instances(xml_path: Path, class_name: str) -> int:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    count = 0
    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is not None and name_node.text == class_name:
            count += 1
    return count


def analyze_dataset(dataset: str, shot: int, seed: int) -> None:
    config = DATASET_CONFIGS[dataset]
    split_seed_dir = config["split_dir"] / f"seed{seed}"
    annotations_dir = config["annotations_dir"]
    novel_classes = config["novel_classes"]

    print(f"Dataset:      {dataset}")
    print(f"Shot:         {shot}")
    print(f"Seed:         {seed}")
    print(f"Split dir:    {split_seed_dir}")
    print(f"Annotations:  {annotations_dir}")
    print(f"Novel classes used by T2: {len(novel_classes)}")
    print()

    rows = []
    all_listed_ids = []
    selected_after_cap_ids = set()
    total_raw_instances = 0
    total_capped_instances = 0

    for class_name in novel_classes:
        split_file = split_seed_dir / f"box_{shot}shot_{class_name}_train.txt"
        if not split_file.exists():
            rows.append((class_name, 0, 0, 0, 0, 0, 0))
            continue

        listed_ids = read_split_image_ids(split_file)
        all_listed_ids.extend(listed_ids)

        existing_xmls = []
        missing_xml = 0
        instance_entries = []

        for image_id in listed_ids:
            xml_path = annotations_dir / f"{image_id}.xml"
            if not xml_path.exists():
                missing_xml += 1
                continue
            existing_xmls.append(image_id)
            instance_count = parse_class_instances(xml_path, class_name)
            for instance_index in range(instance_count):
                instance_entries.append((image_id, instance_index))

        raw_instances = len(instance_entries)
        rng = random.Random(seed)
        if raw_instances > shot:
            selected_entries = rng.sample(instance_entries, shot)
        else:
            selected_entries = instance_entries

        capped_instances = len(selected_entries)
        selected_ids = {image_id for image_id, _ in selected_entries}
        selected_after_cap_ids.update(selected_ids)

        total_raw_instances += raw_instances
        total_capped_instances += capped_instances

        rows.append(
            (
                class_name,
                len(listed_ids),
                len(set(listed_ids)),
                len(set(existing_xmls)),
                missing_xml,
                raw_instances,
                capped_instances,
                len(selected_ids),
            )
        )

    print(
        f"{'Class':<18} {'Lines':>5} {'Uniq':>5} {'XMLs':>5} {'Miss':>5} "
        f"{'RawInst':>8} {'CapInst':>8} {'CapImgs':>7}"
    )
    print("-" * 76)
    for row in rows:
        class_name, line_count, uniq_ids, xml_count, missing_xml, raw_instances, capped_instances, capped_images = row
        print(
            f"{class_name:<18} {line_count:>5} {uniq_ids:>5} {xml_count:>5} {missing_xml:>5} "
            f"{raw_instances:>8} {capped_instances:>8} {capped_images:>7}"
        )

    print("-" * 76)
    print(f"Listed image refs:           {len(all_listed_ids)}")
    print(f"Unique listed images:        {len(set(all_listed_ids))}")
    print(f"Total raw target instances:  {total_raw_instances}")
    print(f"Total capped instances:      {total_capped_instances}")
    print(f"Unique images after cap:     {len(selected_after_cap_ids)}")


def main() -> None:
    args = parse_args()
    analyze_dataset(args.dataset, args.shot, args.seed)


if __name__ == "__main__":
    main()