#!/usr/bin/env python3
"""Analyze T1 pseudo-unknown mining quality for OWOD driving datasets.

The mined candidates mirror the T1 training rule: one2one anchors with
Tobj above a threshold and IoU below the known-GT threshold. By default the
script also excludes TAL foreground anchors, matching the training head's
``fg_mask_pre_prior == 0`` guard.

After mining, candidates are matched against the raw XML annotations and
summarized as:
  * T2 novel classes,
  * raw XML classes outside T1/T2 (auto unknown/open-set classes), and
  * background / non-target matches.
"""

import argparse
import csv
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from array import array
from collections import defaultdict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


METRICS = (
    'tobj',
    'tunk',
    'max_known_iou',
    'best_full_iou',
    'best_target_iou',
    'max_known_score',
    'tunk_minus_tobj',
    'tunk_over_tobj',
)

DEFAULT_SWEEP_THRESHOLDS = '0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30'


@contextmanager
def nullcontext():
    yield


def parse_args():
    p = argparse.ArgumentParser(
        description='Mine T1 pseudo unknowns and classify them with raw XML.')
    p.add_argument('--dataset', default=os.environ.get('DATASET', 'IDD'),
                   choices=['IDD', 'nuOWODB'],
                   help='Dataset to analyze. nuOWODB is the nuScenes OWOD split.')
    p.add_argument(
        '--config',
        default='',
        help='Config path. If omitted, a T1 config is selected from --dataset.')
    p.add_argument('--checkpoint', default='',
                   help='T1 checkpoint. If omitted, the newest T1 best*.pth is used.')
    p.add_argument('--image-set', default='train',
                   help='OWOD image set name. "train" resolves to t1_train.txt.')
    p.add_argument('--split-file', default='',
                   help='Optional explicit split file, one image id per line.')
    p.add_argument('--out-dir',
                   default='',
                   help='Output directory. If omitted, one is derived from dataset/thresholds.')
    p.add_argument('--num-images', type=int, default=0, help='0 = all images.')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--preload-data', action='store_true',
                   help='Preload transformed images and parsed XML annotations before inference.')
    p.add_argument('--no-preload-data', dest='preload_data', action='store_false')
    p.add_argument('--cache-device', choices=['cpu', 'cuda'], default='cpu',
                   help='Where to keep preloaded transformed inputs. Full IDD fits CPU RAM on DGX; CUDA is for small num-images runs.')
    p.add_argument('--pin-cache-memory', action='store_true',
                   help='Pin CPU cached image tensors for faster non-blocking H2D copies.')
    p.add_argument('--progress-interval', type=int, default=10000)
    p.add_argument('--tobj-thr', type=float, default=0.01)
    p.add_argument('--sweep-thresholds', default=DEFAULT_SWEEP_THRESHOLDS,
                   help='Comma-separated Tobj thresholds for retention/purity sweep.')
    p.add_argument('--known-iou-thr', type=float, default=0.5)
    p.add_argument('--match-iou', type=float, default=0.5)
    p.add_argument('--include-difficult', action='store_true')
    p.add_argument('--no-exclude-tal-fg', dest='exclude_tal_fg',
                   action='store_false',
                   help='Disable the TAL foreground exclusion used by training.')
    p.set_defaults(exclude_tal_fg=True)
    p.add_argument('--max-candidates-per-image', type=int, default=0,
                   help='Keep only the top-K Tobj candidates per image. 0 = no cap.')
    p.add_argument('--candidate-csv-max', type=int, default=200000,
                   help='Write at most this many candidate rows. 0 disables row CSV.')
    p.add_argument('--unknown-classes', default='',
                   help='Comma-separated raw XML classes to count as unknown. '
                        'Default: all non-T1/non-T2 XML classes seen.')
    p.add_argument('--amp', action='store_true')
    return p.parse_args()


def repo_paths():
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, '..', '..'))
    tools_dir = os.path.join(repo_root, 'tools')
    for path in (repo_root, tools_dir):
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo_root, tools_dir


def default_config_for_dataset(dataset):
    if dataset == 'IDD':
        return 'configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py'
    if dataset == 'nuOWODB':
        return 'configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod.py'
    raise ValueError(f'Unsupported dataset: {dataset}')


def resolve_checkpoint(repo_root, dataset, ckpt_arg):
    if ckpt_arg:
        return ckpt_arg
    dataset_lower = dataset.lower()
    pattern = os.path.join(
        repo_root,
        'work_dirs',
        f'yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_{dataset_lower}_train_task1',
        'best*.pth')
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f'Could not find a T1 {dataset} best checkpoint. Pass --checkpoint.')
    return os.path.relpath(candidates[0], repo_root)


def parse_thresholds(spec, min_threshold):
    vals = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    vals.append(float(min_threshold))
    vals = sorted({round(v, 10) for v in vals if v >= min_threshold - 1e-12})
    return vals


def parse_voc_xml(xml_path, include_difficult=False):
    objects = []
    root = ET.parse(xml_path).getroot()
    for obj in root.findall('object'):
        difficult = int(obj.findtext('difficult', '0') or 0)
        if difficult and not include_difficult:
            continue
        bb = obj.find('bndbox')
        if bb is None:
            continue
        name = obj.findtext('name', '').strip()
        box = [
            float(bb.findtext('xmin')) - 1.0,
            float(bb.findtext('ymin')) - 1.0,
            float(bb.findtext('xmax')) - 1.0,
            float(bb.findtext('ymax')) - 1.0,
        ]
        objects.append({'name': name, 'bbox': box, 'difficult': difficult})
    return objects


def box_iou_xyxy(a, b):
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    a_ = a.unsqueeze(1)
    b_ = b.unsqueeze(0)
    lt = torch.maximum(a_[..., :2], b_[..., :2])
    rb = torch.minimum(a_[..., 2:], b_[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a_[..., 2] - a_[..., 0]).clamp(min=0) * (
        a_[..., 3] - a_[..., 1]).clamp(min=0)
    area_b = (b_[..., 2] - b_[..., 0]).clamp(min=0) * (
        b_[..., 3] - b_[..., 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-9)


def percentile_stats(values):
    if len(values) == 0:
        return {'count': 0}
    a = np.asarray(values, dtype=np.float64)
    return {
        'count': int(a.size),
        'mean': float(a.mean()),
        'std': float(a.std()),
        'min': float(a.min()),
        'p25': float(np.percentile(a, 25)),
        'p50': float(np.percentile(a, 50)),
        'p75': float(np.percentile(a, 75)),
        'p90': float(np.percentile(a, 90)),
        'max': float(a.max()),
    }


def safe_float(x):
    if isinstance(x, torch.Tensor):
        x = x.item()
    x = float(x)
    if np.isfinite(x):
        return x
    return 0.0


def resolve_split_file(data_root, dataset_name, task_num, image_set, split_file):
    if split_file:
        return split_file
    is_task_file = image_set.startswith('t') and len(image_set) > 1 and image_set[1].isdigit()
    if 'test' in image_set or is_task_file:
        filename = f'{image_set}.txt'
    else:
        filename = f't{task_num}_{image_set}.txt'
    return os.path.join(data_root, 'ImageSets', dataset_name, filename)


def find_image_ext(img_dir, ids):
    for img_id in ids[:100]:
        for ext in ('.jpg', '.jpeg', '.png'):
            if os.path.exists(os.path.join(img_dir, img_id + ext)):
                return ext
    return '.jpg'


def make_gt_tensors(known_boxes, known_labels, batch_size, device, dtype):
    max_count = max([int(x.shape[0]) for x in known_boxes] + [1])
    gt_labels = torch.zeros(batch_size, max_count, 1, device=device, dtype=dtype)
    gt_bboxes = torch.zeros(batch_size, max_count, 4, device=device, dtype=dtype)
    pad_flag = torch.zeros(batch_size, max_count, 1, device=device, dtype=dtype)
    for b in range(batch_size):
        n = int(known_boxes[b].shape[0])
        if n == 0:
            continue
        gt_bboxes[b, :n] = known_boxes[b].to(device=device, dtype=dtype)
        gt_labels[b, :n, 0] = torch.as_tensor(
            known_labels[b], device=device, dtype=dtype)
        pad_flag[b, :n, 0] = 1.0
    return gt_labels, gt_bboxes, pad_flag


def init_group_store():
    return {metric: array('f') for metric in METRICS}


def add_group(groups, key, row):
    bucket = groups[key]
    for metric in METRICS:
        bucket[metric].append(float(row[metric]))


def summarize_groups(groups, role_for_key=None):
    out = {}
    for key in sorted(groups):
        item = {metric: percentile_stats(groups[key][metric])
                for metric in METRICS}
        item['count'] = item['tobj']['count']
        if role_for_key is not None:
            item['role'] = role_for_key.get(key, '')
        out[key] = item
    return out


def write_summary_csv(path, summary, include_role=True):
    fieldnames = ['group']
    if include_role:
        fieldnames.append('role')
    fieldnames.append('count')
    for metric in METRICS:
        for stat in ('mean', 'std', 'min', 'p25', 'p50', 'p75', 'p90', 'max'):
            fieldnames.append(f'{metric}_{stat}')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group, stats in summary.items():
            row = {'group': group, 'count': stats.get('count', 0)}
            if include_role:
                row['role'] = stats.get('role', '')
            for metric in METRICS:
                metric_stats = stats.get(metric, {})
                for stat in ('mean', 'std', 'min', 'p25', 'p50', 'p75', 'p90', 'max'):
                    val = metric_stats.get(stat, '')
                    row[f'{metric}_{stat}'] = val
            writer.writerow(row)


def write_threshold_sweep_summary(path, thresholds, threshold_by_role, role_totals):
    base_total = sum(role_totals.values())
    base_useful = role_totals.get('novel', 0) + role_totals.get('unknown', 0)
    fieldnames = [
        'tobj_threshold', 'total', 'background', 'novel', 'unknown', 'useful',
        'useful_pct', 'background_pct',
        'background_rejection_pct', 'novel_retention_pct',
        'unknown_retention_pct', 'useful_retention_pct',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for threshold in thresholds:
            counts = threshold_by_role[threshold]
            background = int(counts.get('background', 0))
            novel = int(counts.get('novel', 0))
            unknown = int(counts.get('unknown', 0))
            useful = novel + unknown
            total = background + useful
            row = {
                'tobj_threshold': threshold,
                'total': total,
                'background': background,
                'novel': novel,
                'unknown': unknown,
                'useful': useful,
                'useful_pct': useful / max(1, total) * 100.0,
                'background_pct': background / max(1, total) * 100.0,
                'background_rejection_pct': (
                    1.0 - background / max(1, role_totals.get('background', 0))
                ) * 100.0,
                'novel_retention_pct': (
                    novel / max(1, role_totals.get('novel', 0)) * 100.0
                ),
                'unknown_retention_pct': (
                    unknown / max(1, role_totals.get('unknown', 0)) * 100.0
                ),
                'useful_retention_pct': useful / max(1, base_useful) * 100.0,
            }
            if base_total == 0:
                row['background_rejection_pct'] = 0.0
            writer.writerow(row)


def write_threshold_sweep_by_class(
        path, thresholds, threshold_by_class, class_totals, role_for_class):
    fieldnames = [
        'tobj_threshold', 'group', 'role', 'count',
        'class_retention_pct', 'share_at_threshold_pct',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        groups = sorted(class_totals)
        for threshold in thresholds:
            total_at_threshold = sum(threshold_by_class[threshold].values())
            for group in groups:
                count = int(threshold_by_class[threshold].get(group, 0))
                row = {
                    'tobj_threshold': threshold,
                    'group': group,
                    'role': role_for_class.get(group, ''),
                    'count': count,
                    'class_retention_pct': (
                        count / max(1, class_totals.get(group, 0)) * 100.0
                    ),
                    'share_at_threshold_pct': (
                        count / max(1, total_at_threshold) * 100.0
                    ),
                }
                writer.writerow(row)


def main():
    args = parse_args()
    repo_root, _tools_dir = repo_paths()
    os.chdir(repo_root)
    os.environ.setdefault('DATASET', args.dataset)
    os.environ.setdefault('TASK', '1')
    if not args.config:
        args.config = default_config_for_dataset(args.dataset)
    if not args.out_dir:
        tag_tobj = str(args.tobj_thr).replace('.', 'p')
        tag_iou = str(args.known_iou_thr).replace('.', 'p')
        args.out_dir = os.path.join(
            'probe_out', 'unknown_analysis_t1',
            f'{args.dataset.lower()}_t1_{args.image_set}_tobj{tag_tobj}_knowniou{tag_iou}')
        if args.max_candidates_per_image > 0:
            args.out_dir += f'_top{args.max_candidates_per_image}'
    sweep_thresholds = parse_thresholds(args.sweep_thresholds, args.tobj_thr)

    from analyze_owod_test import build_model_ctx, build_pipeline
    from mmengine.registry import init_default_scope
    import mmyolo  # noqa: F401
    import yolo_world  # noqa: F401

    init_default_scope('mmyolo')

    ckpt_path = resolve_checkpoint(repo_root, args.dataset, args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = build_model_ctx(args.config, ckpt_path, device)
    if ctx['dataset_name'] != args.dataset:
        raise RuntimeError(
            f'Config resolved dataset={ctx["dataset_name"]}, expected {args.dataset}')

    cfg_task_list = list(ctx['cfg'].owod_settings[ctx['dataset_name']]['task_list'])
    t1_known_count = int(cfg_task_list[1])
    t2_known_count = int(cfg_task_list[2]) if len(cfg_task_list) > 2 else t1_known_count
    if int(ctx['known_count']) != t1_known_count:
        raise RuntimeError(
            f'Expected T1 known_count={t1_known_count}, '
            f'got {ctx["known_count"]}. Use a TASK=1/T1 config.')

    model = ctx['model']
    head = ctx['head']
    hm = ctx['head_module']
    num_classes = int(ctx['num_classes'])
    known_count = int(ctx['known_count'])
    unk_idx = known_count
    tobj_idx = num_classes - 1
    all_class_names = list(ctx['all_class_names'])
    base_classes = all_class_names[:t1_known_count]
    novel_classes = all_class_names[t1_known_count:t2_known_count]
    base_set = set(base_classes)
    novel_set = set(novel_classes)
    all_known_names = list(all_class_names[:known_count])
    class_to_idx = {name: i for i, name in enumerate(all_known_names)}

    if args.unknown_classes:
        unknown_set = {x.strip() for x in args.unknown_classes.split(',')
                       if x.strip()}
        auto_unknown = False
    else:
        unknown_set = set()
        auto_unknown = True

    data_root = ctx['data_root']
    dataset_name = ctx['dataset_name']
    split_file = resolve_split_file(
        data_root, dataset_name, 1, args.image_set, args.split_file)
    with open(split_file) as f:
        ids = [line.strip() for line in f if line.strip()]
    if args.num_images > 0:
        ids = ids[:args.num_images]

    img_dir = os.path.join(data_root, 'JPEGImages', dataset_name)
    ann_dir = os.path.join(data_root, 'Annotations', dataset_name)
    img_ext = find_image_ext(img_dir, ids)
    pipeline = build_pipeline(ctx)
    os.makedirs(args.out_dir, exist_ok=True)

    by_class = defaultdict(init_group_store)
    by_role = defaultdict(init_group_store)
    by_class_stride = defaultdict(init_group_store)
    threshold_by_role = {thr: defaultdict(int) for thr in sweep_thresholds}
    threshold_by_class = {thr: defaultdict(int) for thr in sweep_thresholds}
    role_for_class = {}
    candidate_counts = defaultdict(int)
    best_xml_counts = defaultdict(int)
    raw_unknown_xml_counts = defaultdict(int)
    raw_novel_xml_counts = defaultdict(int)
    processed = 0
    skipped = 0
    candidate_csv_rows = 0
    for name in novel_classes:
        _ = by_class[name]
        role_for_class[name] = 'novel'
    _ = by_class['__BACKGROUND__']
    role_for_class['__BACKGROUND__'] = 'background'
    for role in ('novel', 'unknown', 'background'):
        _ = by_role[role]

    cand_csv_path = os.path.join(args.out_dir, 'candidate_details.csv')
    cand_csv = None
    cand_writer = None
    if args.candidate_csv_max > 0:
        cand_csv = open(cand_csv_path, 'w', newline='')
        cand_writer = csv.DictWriter(
            cand_csv,
            fieldnames=[
                'img_id', 'anchor_index', 'level', 'stride',
                'x1', 'y1', 'x2', 'y2',
                'matched_role', 'matched_class', 'best_xml_class',
                'tobj', 'tunk', 'max_known_iou',
                'best_full_iou', 'best_target_iou', 'max_known_score',
                'tunk_minus_tobj', 'tunk_over_tobj',
                'top_non_anchor_class', 'top_non_anchor_score',
            ])
        cand_writer.writeheader()

    print('=' * 80)
    print(f'{args.dataset} T1 PSEUDO-UNKNOWN ANALYSIS')
    print(f'  config:      {args.config}')
    print(f'  checkpoint:  {ckpt_path}')
    print(f'  split:       {split_file} ({len(ids)} images)')
    print(f'  out_dir:     {args.out_dir}')
    print(f'  criteria:    max_iou_known < {args.known_iou_thr:g}, '
          f'Tobj >= {args.tobj_thr:g}, '
          f'exclude_tal_fg={args.exclude_tal_fg}')
    print(f'  classify:    XML IoU >= {args.match_iou:g}; '
          f'{len(novel_classes)} novel classes; '
          f'unknown={"auto" if auto_unknown else sorted(unknown_set)}')
    print(f'  cache:       preload_data={args.preload_data}, '
          f'cache_device={args.cache_device}, pin={args.pin_cache_memory}, '
          f'progress_interval={args.progress_interval}')
    print(f'  sweep Tobj:  {sweep_thresholds}')
    print('=' * 80)

    def transform_boxes_to_pad(gt_items, scale, pad, out_device=None):
        if not gt_items:
            return torch.zeros((0, 4), dtype=torch.float32,
                               device=out_device or 'cpu')
        boxes = torch.tensor([g['bbox'] for g in gt_items],
                             dtype=torch.float32, device=out_device or 'cpu')
        sx, sy = float(scale[0]), float(scale[1])
        pad_top, pad_left = float(pad[0]), float(pad[2])
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * sx + pad_left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * sy + pad_top
        return boxes

    def boxes_to_original(boxes, scale, pad, ori_shape):
        out = boxes.clone()
        sx, sy = float(scale[0]), float(scale[1])
        pad_top, pad_left = float(pad[0]), float(pad[2])
        out[:, [0, 2]] = (out[:, [0, 2]] - pad_left) / max(sx, 1e-12)
        out[:, [1, 3]] = (out[:, [1, 3]] - pad_top) / max(sy, 1e-12)
        h, w = int(ori_shape[0]), int(ori_shape[1])
        out[:, 0::2].clamp_(0, w)
        out[:, 1::2].clamp_(0, h)
        return out

    def class_role(name):
        if name in novel_set:
            return 'novel'
        if name in base_set:
            return 'base'
        if auto_unknown:
            return 'unknown'
        if name in unknown_set:
            return 'unknown'
        return 'background'

    def final_group_for_match(best_name, best_iou):
        if best_iou < args.match_iou or not best_name:
            return '__BACKGROUND__', 'background'
        role = class_role(best_name)
        if role == 'novel':
            return best_name, 'novel'
        if role == 'unknown':
            if auto_unknown:
                unknown_set.add(best_name)
            return best_name, 'unknown'
        return '__BACKGROUND__', 'background'

    def load_one(img_id):
        img_path = os.path.join(img_dir, img_id + img_ext)
        ann_path = os.path.join(ann_dir, img_id + '.xml')
        if not os.path.exists(img_path) or not os.path.exists(ann_path):
            return None
        try:
            data = pipeline(dict(img_path=img_path, img_id=img_id, instances=[]))
            gt_full = parse_voc_xml(ann_path, args.include_difficult)
        except Exception as exc:
            print(f'  [skip] {img_id}: {exc}')
            return None
        meta = data['data_samples'].metainfo
        scale = np.asarray(meta['scale_factor'], dtype=np.float32)
        pad = np.asarray(meta.get('pad_param', np.zeros(4, dtype=np.float32)),
                         dtype=np.float32)
        inputs = data['inputs'].unsqueeze(0)
        if args.preload_data and args.cache_device == 'cuda':
            inputs = inputs.to(device, non_blocking=True)
        elif args.preload_data and args.pin_cache_memory and device.type == 'cuda':
            try:
                inputs = inputs.pin_memory()
            except RuntimeError as exc:
                if not getattr(load_one, '_pin_warned', False):
                    print(f'  [warn] pin_memory failed; continuing with regular CPU cache: {exc}')
                    load_one._pin_warned = True
        known_items = [g for g in gt_full if g['name'] in base_set]
        full_boxes = (torch.tensor([g['bbox'] for g in gt_full],
                                   dtype=torch.float32)
                      if gt_full else torch.zeros((0, 4), dtype=torch.float32))
        return {
            'img_id': img_id,
            'inputs': inputs,
            'data_sample': data['data_samples'],
            'scale': scale,
            'pad': pad,
            'ori_shape': meta['ori_shape'],
            'gt_full': gt_full,
            'known_boxes_pad': transform_boxes_to_pad(known_items, scale, pad),
            'known_labels': [
                class_to_idx[g['name']] for g in known_items
                if g['name'] in class_to_idx
            ],
            'full_boxes_orig': full_boxes,
            'full_names': [g['name'] for g in gt_full],
        }

    def flush(buf):
        nonlocal processed, candidate_csv_rows
        if not buf:
            return

        B = len(buf)
        batch_inputs = torch.cat([x['inputs'] for x in buf], dim=0)
        batch_inputs = batch_inputs.float().to(device, non_blocking=True) / 255.0
        samples = [x['data_sample'] for x in buf]

        known_boxes_pad = []
        known_labels = []
        full_boxes_orig = []
        full_names = []
        for item in buf:
            for gt in item['gt_full']:
                if gt['name'] in novel_set:
                    raw_novel_xml_counts[gt['name']] += 1
                elif gt['name'] not in base_set:
                    raw_unknown_xml_counts[gt['name']] += 1
                    if auto_unknown:
                        unknown_set.add(gt['name'])
                        _ = by_class[gt['name']]
                        role_for_class.setdefault(gt['name'], 'unknown')
            known_boxes_pad.append(item['known_boxes_pad'])
            known_labels.append(item['known_labels'])
            full_boxes_orig.append(item['full_boxes_orig'].to(
                device, non_blocking=True))
            full_names.append(item['full_names'])

        with torch.no_grad():
            autocast = (torch.cuda.amp.autocast(enabled=True)
                        if args.amp and device.type == 'cuda'
                        else nullcontext())
            with autocast:
                img_feats, txt_feats = model.extract_feat(batch_inputs, samples)
                cls_list, bbox_list = hm.forward_one2one(img_feats, txt_feats)

        featmap_sizes = [(t.shape[-2], t.shape[-1]) for t in cls_list]
        mlvl_priors = head.prior_generator.grid_priors(
            featmap_sizes, dtype=batch_inputs.dtype, device=device, with_stride=True)
        flat_priors = torch.cat(mlvl_priors, dim=0)
        flat_logits = torch.cat([
            t.permute(0, 2, 3, 1).reshape(B, -1, num_classes)
            for t in cls_list
        ], dim=1)
        flat_bbox = torch.cat([
            t.permute(0, 2, 3, 1).reshape(B, -1, 4)
            for t in bbox_list
        ], dim=1)
        decoded_pad = head.bbox_coder.decode(
            flat_priors[..., :2],
            flat_bbox,
            flat_priors[:, [2]][..., 0])
        flat_scores = flat_logits.sigmoid()

        level_ids = []
        for level, pred in enumerate(cls_list):
            n = pred.shape[-2] * pred.shape[-1]
            level_ids.append(torch.full((n,), level, device=device, dtype=torch.long))
        level_ids = torch.cat(level_ids, dim=0)
        strides = flat_priors[:, 2].round().long()

        tal_fg = torch.zeros(flat_scores.shape[:2], dtype=torch.bool, device=device)
        if args.exclude_tal_fg and any(int(x.shape[0]) for x in known_boxes_pad):
            gt_labels, gt_bboxes, pad_flag = make_gt_tensors(
                known_boxes_pad, known_labels, B, device, flat_logits.dtype)
            assigned = head.one2one_assigner(
                decoded_pad.detach().type(gt_bboxes.dtype),
                flat_scores.detach(),
                flat_priors,
                gt_labels,
                gt_bboxes,
                pad_flag)
            tal_fg = assigned['fg_mask_pre_prior'].bool()

        for b, item in enumerate(buf):
            processed += 1
            known_boxes = known_boxes_pad[b].to(device, non_blocking=True)
            if known_boxes.numel() > 0:
                max_known_iou = box_iou_xyxy(decoded_pad[b], known_boxes).amax(dim=1)
            else:
                max_known_iou = torch.zeros(decoded_pad.shape[1], device=device)

            keep = ((max_known_iou < args.known_iou_thr)
                    & (flat_scores[b, :, tobj_idx] >= args.tobj_thr))
            if args.exclude_tal_fg:
                keep = keep & ~tal_fg[b]
            keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
            if keep_idx.numel() == 0:
                continue
            if args.max_candidates_per_image > 0 and keep_idx.numel() > args.max_candidates_per_image:
                top = flat_scores[b, keep_idx, tobj_idx].topk(
                    args.max_candidates_per_image).indices
                keep_idx = keep_idx[top]

            cand_boxes_orig = boxes_to_original(
                decoded_pad[b, keep_idx], item['scale'], item['pad'], item['ori_shape'])
            full_boxes = full_boxes_orig[b]
            if full_boxes.numel() > 0:
                ious_full = box_iou_xyxy(cand_boxes_orig, full_boxes)
                best_full_iou, best_full_gi = ious_full.max(dim=1)
            else:
                best_full_iou = torch.zeros(keep_idx.numel(), device=device)
                best_full_gi = torch.zeros(keep_idx.numel(), dtype=torch.long, device=device)

            target_mask = [
                (name in novel_set) or
                (name not in base_set and (auto_unknown or name in unknown_set))
                for name in full_names[b]
            ]
            if any(target_mask):
                target_boxes = full_boxes[torch.tensor(target_mask, device=device)]
                target_names = [n for n, ok in zip(full_names[b], target_mask) if ok]
                ious_target = box_iou_xyxy(cand_boxes_orig, target_boxes)
                best_target_iou, best_target_gi = ious_target.max(dim=1)
            else:
                target_names = []
                best_target_iou = torch.zeros(keep_idx.numel(), device=device)
                best_target_gi = torch.zeros(keep_idx.numel(), dtype=torch.long, device=device)

            non_anchor_scores = flat_scores[b, keep_idx, :tobj_idx]
            top_scores, top_idxs = non_anchor_scores.max(dim=1)
            max_known_scores = flat_scores[b, keep_idx, :known_count].amax(dim=1)

            for j, anchor_idx in enumerate(keep_idx.tolist()):
                best_iou = safe_float(best_full_iou[j])
                if full_names[b]:
                    best_xml_name = full_names[b][int(best_full_gi[j].item())]
                else:
                    best_xml_name = ''
                if target_names:
                    target_iou = safe_float(best_target_iou[j])
                    target_name = target_names[int(best_target_gi[j].item())]
                else:
                    target_iou = 0.0
                    target_name = ''

                if target_iou >= args.match_iou:
                    matched_name, matched_role = final_group_for_match(
                        target_name, target_iou)
                else:
                    matched_name, matched_role = '__BACKGROUND__', 'background'

                tobj = safe_float(flat_scores[b, anchor_idx, tobj_idx])
                tunk = safe_float(flat_scores[b, anchor_idx, unk_idx])
                row = {
                    'tobj': tobj,
                    'tunk': tunk,
                    'max_known_iou': safe_float(max_known_iou[anchor_idx]),
                    'best_full_iou': best_iou,
                    'best_target_iou': target_iou,
                    'max_known_score': safe_float(max_known_scores[j]),
                    'tunk_minus_tobj': tunk - tobj,
                    'tunk_over_tobj': tunk / max(tobj, 1e-12),
                }

                stride = int(strides[anchor_idx].item())
                class_stride_key = f'{matched_name}|stride{stride}'
                add_group(by_class, matched_name, row)
                add_group(by_role, matched_role, row)
                add_group(by_class_stride, class_stride_key, row)
                role_for_class.setdefault(matched_name, matched_role)
                role_for_class.setdefault(class_stride_key, matched_role)
                candidate_counts[matched_role] += 1
                for threshold in sweep_thresholds:
                    if tobj >= threshold:
                        threshold_by_role[threshold][matched_role] += 1
                        threshold_by_class[threshold][matched_name] += 1
                if best_xml_name:
                    best_xml_counts[best_xml_name] += 1

                if cand_writer is not None and candidate_csv_rows < args.candidate_csv_max:
                    box = cand_boxes_orig[j].detach().cpu().tolist()
                    top_idx = int(top_idxs[j].item())
                    if top_idx < known_count:
                        top_name = all_known_names[top_idx]
                    elif top_idx == unk_idx:
                        top_name = 'unknown'
                    else:
                        top_name = f'cls_{top_idx}'
                    cand_writer.writerow({
                        'img_id': item['img_id'],
                        'anchor_index': anchor_idx,
                        'level': int(level_ids[anchor_idx].item()),
                        'stride': stride,
                        'x1': round(float(box[0]), 3),
                        'y1': round(float(box[1]), 3),
                        'x2': round(float(box[2]), 3),
                        'y2': round(float(box[3]), 3),
                        'matched_role': matched_role,
                        'matched_class': matched_name,
                        'best_xml_class': best_xml_name,
                        'tobj': row['tobj'],
                        'tunk': row['tunk'],
                        'max_known_iou': row['max_known_iou'],
                        'best_full_iou': row['best_full_iou'],
                        'best_target_iou': row['best_target_iou'],
                        'max_known_score': row['max_known_score'],
                        'tunk_minus_tobj': row['tunk_minus_tobj'],
                        'tunk_over_tobj': row['tunk_over_tobj'],
                        'top_non_anchor_class': top_name,
                        'top_non_anchor_score': safe_float(top_scores[j]),
                    })
                    candidate_csv_rows += 1

            if args.progress_interval > 0 and processed % args.progress_interval == 0:
                total = sum(candidate_counts.values())
                print(f'  processed {processed}/{len(ids)} images; '
                      f'candidates={total}; roles={dict(candidate_counts)}')

    def iter_loaded_items():
        nonlocal skipped
        prefetch = max(1, args.num_workers * 2)
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            futures = []
            for i in range(min(prefetch, len(ids))):
                futures.append(pool.submit(load_one, ids[i]))
            next_submit = len(futures)
            done = 0
            while futures:
                result = futures.pop(0).result()
                done += 1
                if next_submit < len(ids):
                    futures.append(pool.submit(load_one, ids[next_submit]))
                    next_submit += 1
                if result is None:
                    skipped += 1
                else:
                    yield result
                if (args.progress_interval > 0
                        and done % args.progress_interval == 0):
                    label = 'cached' if args.preload_data else 'loaded'
                    print(f'  {label} {done}/{len(ids)} images '
                          f'(processed={processed}, skipped={skipped})')

    def flush_iter(items):
        batch = []
        for item in items:
            batch.append(item)
            if len(batch) >= args.batch_size:
                flush(batch)
                batch = []
        flush(batch)

    if args.preload_data:
        print(f'[cache] Preloading {len(ids)} transformed images/XML annotations '
              f'to {args.cache_device}.')
        cached_items = list(iter_loaded_items())
        print(f'[cache] Ready: {len(cached_items)} items cached, skipped={skipped}.')
        flush_iter(cached_items)
    else:
        flush_iter(iter_loaded_items())

    if cand_csv is not None:
        cand_csv.close()

    class_summary = summarize_groups(by_class, role_for_class)
    role_summary = summarize_groups(by_role)
    class_stride_summary = summarize_groups(by_class_stride, role_for_class)
    for role, stats in role_summary.items():
        stats['role'] = role

    class_csv = os.path.join(args.out_dir, 'summary_by_class.csv')
    role_csv = os.path.join(args.out_dir, 'summary_by_role.csv')
    class_stride_csv = os.path.join(args.out_dir, 'summary_by_class_stride.csv')
    threshold_summary_csv = os.path.join(args.out_dir, 'threshold_sweep_summary.csv')
    threshold_class_csv = os.path.join(args.out_dir, 'threshold_sweep_by_class.csv')
    write_summary_csv(class_csv, class_summary, include_role=True)
    write_summary_csv(role_csv, role_summary, include_role=False)
    write_summary_csv(class_stride_csv, class_stride_summary, include_role=True)
    class_totals = {
        group: int(stats.get('count', 0))
        for group, stats in class_summary.items()
    }
    write_threshold_sweep_summary(
        threshold_summary_csv, sweep_thresholds, threshold_by_role,
        dict(candidate_counts))
    write_threshold_sweep_by_class(
        threshold_class_csv, sweep_thresholds, threshold_by_class,
        class_totals, role_for_class)

    summary = {
        'dataset': args.dataset,
        'config': args.config,
        'checkpoint': ckpt_path,
        'split_file': split_file,
        'num_images_requested': len(ids),
        'num_images_processed': processed,
        'num_images_skipped': skipped,
        'criteria': {
            'tobj_thr': args.tobj_thr,
            'known_iou_thr': args.known_iou_thr,
            'match_iou': args.match_iou,
            'exclude_tal_fg': args.exclude_tal_fg,
            'max_candidates_per_image': args.max_candidates_per_image,
            'preload_data': args.preload_data,
            'cache_device': args.cache_device,
            'pin_cache_memory': args.pin_cache_memory,
            'progress_interval': args.progress_interval,
            'sweep_thresholds': sweep_thresholds,
        },
        't1_known_classes': base_classes,
        't2_novel_classes': novel_classes,
        'unknown_classes_seen': sorted(unknown_set),
        'candidate_counts_by_role': dict(candidate_counts),
        'raw_novel_xml_counts': dict(sorted(
            raw_novel_xml_counts.items(), key=lambda x: (-x[1], x[0]))),
        'raw_unknown_xml_counts': dict(sorted(
            raw_unknown_xml_counts.items(), key=lambda x: (-x[1], x[0]))),
        'best_xml_class_counts': dict(sorted(
            best_xml_counts.items(), key=lambda x: (-x[1], x[0]))),
        'candidate_csv': cand_csv_path if args.candidate_csv_max > 0 else '',
        'candidate_csv_rows': candidate_csv_rows,
        'summary_by_class': class_summary,
        'summary_by_role': role_summary,
        'summary_by_class_stride': class_stride_summary,
        'threshold_sweep_by_role': {
            str(threshold): dict(counts)
            for threshold, counts in threshold_by_role.items()
        },
        'threshold_sweep_by_class': {
            str(threshold): dict(counts)
            for threshold, counts in threshold_by_class.items()
        },
    }
    json_path = os.path.join(args.out_dir, 'summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    total_candidates = sum(candidate_counts.values())
    print('\n' + '=' * 80)
    print('PSEUDO-UNKNOWN QUALITY SUMMARY')
    print(f'  processed images: {processed}/{len(ids)}  skipped={skipped}')
    print(f'  mined candidates: {total_candidates}')
    for role in ('novel', 'unknown', 'background'):
        count = candidate_counts.get(role, 0)
        pct = count / max(1, total_candidates) * 100
        print(f'  {role:10s}: {count:10d} ({pct:6.2f}%)')
    print(f'  auto unknown classes seen: {len(unknown_set)} {sorted(unknown_set)}')
    print(f'[write] {json_path}')
    print(f'[write] {class_csv}')
    print(f'[write] {role_csv}')
    print(f'[write] {class_stride_csv}')
    print(f'[write] {threshold_summary_csv}')
    print(f'[write] {threshold_class_csv}')
    if args.candidate_csv_max > 0:
        print(f'[write] {cand_csv_path} ({candidate_csv_rows} rows max)')
    print('=' * 80)


if __name__ == '__main__':
    main()
