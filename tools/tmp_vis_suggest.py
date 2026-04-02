"""
Suggest best IDD test images for visualisation (fast version).

Speed strategy:
  1. grep the annotation dir for files containing any target class name  →  tiny candidate set
  2. Only parse XML for those candidate files  →  no wasted work
  3. Progress reported at each grep + parse step

Ranking:
  - NOVEL  (T2): bus/truck/tanker_vehicle/crane_truck/street_cart/excavator  area ≥ 80×80
  - UNKNOWN    : pole/tractor/concrete_mixer/pull_cart/road_roller            area ≥ 80×80
                 animal                                                       area ≥ 60×60
  - ★ BONUS    : +8 pts when image has BOTH novel AND unknown (best for joint vis)

Usage (from YOLO-UniOW root):
    python tools/tmp_vis_suggest.py
    python tools/tmp_vis_suggest.py --top-n 20 --min-novel 1
"""
import os
import sys
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

# ── IDD T2 class taxonomy ─────────────────────────────────────────────────────
NOVEL_CLASSES_T2 = {
    'bus':            1.2,
    'truck':          1.2,
    'tanker_vehicle': 2.5,
    'crane_truck':    2.5,
    'street_cart':    1.8,
    'excavator':      2.2,
}
UNKNOWN_CLASSES = {
    'animal':         2.0,   # relaxed area threshold
    'pole':           1.0,
    'tractor':        1.8,
    'concrete_mixer': 2.0,
    'pull_cart':      1.5,
    'road_roller':    2.0,
}
ANIMAL_AREA_THR  = 60 * 60
DEFAULT_AREA_THR = 80 * 80

ALL_TARGET = set(NOVEL_CLASSES_T2) | set(UNKNOWN_CLASSES)

# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Suggest best IDD test images for vis')
    p.add_argument('--data-root', default='data/OWOD')
    p.add_argument('--dataset',   default='IDD')
    p.add_argument('--image-set', default='test')
    p.add_argument('--top-n',     type=int,   default=20)
    p.add_argument('--min-novel', type=int,   default=0)
    p.add_argument('--min-unk',   type=int,   default=0)
    p.add_argument('--novel-w',   type=float, default=2.0)
    p.add_argument('--unk-w',     type=float, default=1.8)
    p.add_argument('--bonus',     type=float, default=8.0)
    p.add_argument('--no-bonus',  action='store_true')
    return p.parse_args()


# ── Fast candidate discovery: scan all test XMLs directly ─────────────────────
def find_best_per_sequence(ann_dir, test_ids, novel_w, unk_w, bonus, no_bonus, min_novel, min_unk):
    """
    Group test IDs by 3-digit prefix (= video sequence).
    For each sequence, scan ALL its test XMLs and keep the best-scoring image.
    Only scans test-set XMLs (21K), not the full 117K annotation dir.
    Returns list of best records, one per sequence.
    """
    from collections import defaultdict as _dd

    # Group test IDs by sequence prefix
    seq_to_ids = _dd(list)
    for img_id in test_ids:
        seq_to_ids[img_id[:3]].append(img_id)

    n_seqs = len(seq_to_ids)
    print(f'[1/2] Scanning {len(test_ids)} test XMLs across {n_seqs} sequences ...',
          flush=True)

    best_per_seq = {}  # prefix → best record
    done = 0
    for prefix, ids in sorted(seq_to_ids.items()):
        seq_best = None
        for img_id in ids:
            xml_path = os.path.join(ann_dir, img_id + '.xml')
            if not os.path.exists(xml_path):
                continue
            novel_det, unk_det = scan_xml(xml_path)
            n_nov = sum(novel_det.values())
            n_unk = sum(unk_det.values())
            if n_nov < min_novel or n_unk < min_unk:
                continue
            if n_nov == 0 and n_unk == 0:
                continue
            s = score_image(novel_det, unk_det, novel_w, unk_w, bonus, no_bonus)
            rec = (s, img_id, n_nov, n_unk, novel_det, unk_det)
            if seq_best is None or s > seq_best[0]:
                seq_best = rec
        if seq_best is not None:
            best_per_seq[prefix] = seq_best
        done += 1
        if done % 20 == 0 or done == n_seqs:
            print(f'      {done}/{n_seqs} sequences scanned, '
                  f'{len(best_per_seq)} with qualifying images', flush=True)

    return list(best_per_seq.values())


# ── XML parsing ───────────────────────────────────────────────────────────────
def box_area(obj_elem):
    bndbox = obj_elem.find('bndbox')
    if bndbox is None:
        return 0
    try:
        xmin = float(bndbox.findtext('xmin', 0))
        ymin = float(bndbox.findtext('ymin', 0))
        xmax = float(bndbox.findtext('xmax', 0))
        ymax = float(bndbox.findtext('ymax', 0))
        return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
    except ValueError:
        return 0


def scan_xml(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return {}, {}

    novel_det = defaultdict(int)
    unk_det   = defaultdict(int)

    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip()
        area = box_area(obj)
        if name in NOVEL_CLASSES_T2 and area >= DEFAULT_AREA_THR:
            novel_det[name] += 1
        elif name in UNKNOWN_CLASSES:
            thr = ANIMAL_AREA_THR if name == 'animal' else DEFAULT_AREA_THR
            if area >= thr:
                unk_det[name] += 1

    return dict(novel_det), dict(unk_det)


def score_image(novel_det, unk_det, novel_w, unk_w, bonus, no_bonus):
    s  = sum(novel_w * NOVEL_CLASSES_T2[c] * n for c, n in novel_det.items())
    s += sum(unk_w   * UNKNOWN_CLASSES[c]   * n for c, n in unk_det.items())
    if not no_bonus and novel_det and unk_det:
        s += bonus
    return s


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    ann_dir      = os.path.join(args.data_root, 'Annotations', args.dataset)
    imageset_path = os.path.join(
        args.data_root, 'ImageSets', args.dataset, f'{args.image_set}.txt')

    print(f'[0/2] Loading test image list from {imageset_path} ...', flush=True)
    with open(imageset_path) as f:
        test_ids = [l.strip() for l in f if l.strip()]
    print(f'      → {len(test_ids)} test images total', flush=True)

    # Step 1: scan test XMLs grouped by sequence, one best per sequence
    seq_reps = find_best_per_sequence(
        ann_dir, test_ids,
        args.novel_w, args.unk_w, args.bonus, args.no_bonus,
        args.min_novel, args.min_unk,
    )

    if not seq_reps:
        print('No qualifying images found — check ann_dir path.', flush=True)
        sys.exit(1)

    n_joint = sum(1 for r in seq_reps if r[4] and r[5])
    print(f'\n[2/2] Diversity-bucketed selection ...', flush=True)
    print(f'      → {len(seq_reps)} sequences with qualifying images '
          f'({n_joint} joint novel+unknown)', flush=True)

    # Sort and separate
    seq_reps.sort(key=lambda x: x[0], reverse=True)
    joint      = [r for r in seq_reps if r[4] and r[5]]
    novel_only = [r for r in seq_reps if r[4] and not r[5]]
    unk_only   = [r for r in seq_reps if not r[4] and r[5]]

    # Fill top-n: joint first, then alternate novel/unk
    top = []
    seen_prefix = set()

    def add_pool(pool):
        for r in pool:
            px = r[1][:3]
            if px not in seen_prefix and len(top) < args.top_n:
                top.append(r)
                seen_prefix.add(px)

    add_pool(joint)
    ni, ui = 0, 0
    while len(top) < args.top_n:
        added = False
        if ni < len(novel_only):
            r = novel_only[ni]; ni += 1
            px = r[1][:3]
            if px not in seen_prefix:
                top.append(r); seen_prefix.add(px); added = True
        if len(top) < args.top_n and ui < len(unk_only):
            r = unk_only[ui]; ui += 1
            px = r[1][:3]
            if px not in seen_prefix:
                top.append(r); seen_prefix.add(px); added = True
        if not added:
            break

    # ── Output table ──────────────────────────────────────────────────────────
    W = 90
    print('\n' + '=' * W)
    print(f'  TOP {args.top_n} VISUALISATION CANDIDATES  (IDD Task-2 test set)')
    print(f'  1 image per drive sequence (3-digit prefix) — guaranteed diversity')
    print(f'  Area thresholds: novel/unk ≥80×80 px²,  animal ≥60×60 px²')
    if not args.no_bonus:
        print(f'  ★ Joint novel+unknown bonus: +{args.bonus:.0f} pts')
    print('=' * W)
    print(f'{"#":>3} {"IMAGE ID":<14} {"SEQ":>4} {"SCORE":>6} {"NOV":>4} {"UNK":>4}  DETAIL')
    print('-' * W)

    for rank, (s, img_id, n_nov, n_unk, nd, ud) in enumerate(top, 1):
        tag = '★' if nd and ud else ' '
        parts = []
        if nd:
            parts.append('nov:[' + ', '.join(f'{c}×{v}' for c, v in sorted(nd.items())) + ']')
        if ud:
            parts.append('unk:[' + ', '.join(f'{c}×{v}' for c, v in sorted(ud.items())) + ']')
        seq = img_id[:3]
        print(f'{rank:>3} {img_id:<14} {seq:>4} {s:>6.1f} {n_nov:>4} {n_unk:>4} {tag} {" ".join(parts)}')

    print('=' * W)
    print(f'\n★ = both novel AND unknown present  |  NOV/UNK = large-box counts\n')

    nboth = sum(1 for r in top if r[4] and r[5])
    print(f'Breakdown in top-{args.top_n}: {nboth} joint ★, '
          f'{sum(1 for r in top if r[4] and not r[5])} novel-only, '
          f'{sum(1 for r in top if not r[4] and r[5])} unk-only')

    print('\n--- Image IDs (copy-paste) ---')
    for _, img_id, *_ in top:
        print(img_id)


if __name__ == '__main__':
    main()

