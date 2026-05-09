"""Text-Visual alignment probe for YOLO-UniOW OWOD.

Two modes:
  Normal  (default): multi-prompt per class, leakage diagnostics, full CSV outputs.
  Compare (--compare): raw-CLIP class-name embeddings vs model.embeddings (T2-tuned
           params from checkpoint). Single forward pass, compare.csv + compare.txt only.
           Multi-prompt encoding is NOT done in this mode.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import torch
import yaml


DATASET_CONFIGS = {
    'idd':      'configs/test/idd_prompt_eval.yaml',
    'nuimages': 'configs/test/nuimages_prompt_eval.yaml',
}

# Base (T1) / novel (T2) splits for --compare grouping.
TASK_SPLITS = {
    'idd': {
        'T1': ['car', 'motorcycle', 'rider', 'person', 'autorickshaw',
               'bicycle', 'traffic sign', 'traffic light'],
        'T2': ['bus', 'truck', 'tanker_vehicle', 'crane_truck',
               'street_cart', 'excavator'],
    },
    'nuimages': {
        'T1': ['vehicle.bicycle', 'vehicle.motorcycle', 'vehicle.car',
               'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.truck',
               'vehicle.emergency.ambulance', 'vehicle.emergency.police',
               'vehicle.construction', 'vehicle.trailer'],
        'T2': ['human.pedestrian.adult', 'human.pedestrian.child',
               'human.pedestrian.wheelchair', 'human.pedestrian.stroller',
               'human.pedestrian.personal_mobility',
               'human.pedestrian.police_officer',
               'human.pedestrian.construction_worker'],
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--config', help='Path to probe YAML config')
    g.add_argument('--dataset', choices=list(DATASET_CONFIGS),
                   help='Shorthand dataset name (selects default config)')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--compare', action='store_true',
                   help='Compare raw-CLIP zero-shot (class name) vs T2-tuned '
                        'model.embeddings. Skips multi-prompt encoding entirely.')
    args = p.parse_args()
    if args.dataset:
        args.config = DATASET_CONFIGS[args.dataset]
    return args


def detect_dataset_key(model_config_path):
    s = model_config_path.lower()
    if 'idd' in s:
        return 'idd'
    if 'nuowodb' in s or 'nuimages' in s:
        return 'nuimages'
    return None


def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_').lower()[:60]


def normalize_prompts(p):
    if isinstance(p, str):
        return [s.strip() for s in p.split(',') if s.strip()]
    return [s.strip() for s in p if s and s.strip()]


def box_iou_xyxy(a, b):
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def parse_voc_xml(xml_path):
    out = []
    for obj in ET.parse(xml_path).getroot().findall('object'):
        bb = obj.find('bndbox')
        out.append((obj.find('name').text.strip(), [
            float(bb.find('xmin').text) - 1.0,
            float(bb.find('ymin').text) - 1.0,
            float(bb.find('xmax').text) - 1.0,
            float(bb.find('ymax').text) - 1.0]))
    return out


# ─── Embedding cache ───────────────────────────────────────────────────────
def encode_prompts_cached(model, prompts, cache_dir, device):
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, 'manifest.json')
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    out = [None] * len(prompts)
    to_encode = []
    for i, prompt in enumerate(prompts):
        h = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]
        fname = f'{sanitize(prompt)}__{h}.npy'
        fpath = os.path.join(cache_dir, fname)
        entry = manifest.get(prompt)
        if entry and entry.get('sha') == h and os.path.exists(fpath):
            out[i] = np.load(fpath)
        else:
            to_encode.append((i, prompt, fname, fpath, h))

    if to_encode:
        print(f'[encode] {len(to_encode)}/{len(prompts)} new prompt(s)')
        texts = [[p] for _, p, _, _, _ in to_encode]
        with torch.no_grad():
            txt = model.backbone.forward_text(texts)
        txt = torch.nn.functional.normalize(txt, dim=-1, p=2)
        txt = txt.squeeze(1).cpu().float().numpy()
        for k, (i, prompt, fname, fpath, h) in enumerate(to_encode):
            np.save(fpath, txt[k])
            manifest[prompt] = {'sha': h, 'file': fname}
            out[i] = txt[k]
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    arr = np.stack(out)
    t = torch.from_numpy(arr).float().to(device)
    return torch.nn.functional.normalize(t, dim=-1, p=2)


# ─── Model + text encoder load ─────────────────────────────────────────────
def build_model(model_cfg_path, ckpt_path, te_pretrain_ckpt,
                te_model_name, te_use_lora, device):
    from mmengine.config import Config
    from mmengine.runner import load_state_dict
    from mmyolo.registry import MODELS

    cfg = Config.fromfile(model_cfg_path)
    cfg.model.backbone.with_text_model = True
    cfg.model.backbone.text_model = dict(
        type='HuggingCLIPLanguageBackbone',
        model_name=te_model_name,
        training_use_lora=te_use_lora,
    )
    model = MODELS.build(cfg.model)

    pre = torch.load(te_pretrain_ckpt, map_location='cpu')
    pre_sd = pre.get('state_dict', pre)
    te_sd = {k: v for k, v in pre_sd.items()
             if k.startswith('backbone.text_model.')}
    if not te_sd:
        raise RuntimeError(f'No text_model.* keys in {te_pretrain_ckpt}')
    load_state_dict(model, te_sd, strict=False)
    print(f'[init] text_model: {len(te_sd)} keys from pretrain ckpt')

    ft = torch.load(ckpt_path, map_location='cpu')
    ft_sd = ft.get('state_dict', ft)
    ft_sd = {k: v for k, v in ft_sd.items() if 'text_model' not in k}
    if 'embeddings' in ft_sd and ft_sd['embeddings'].shape == model.embeddings.shape:
        with torch.no_grad():
            model.embeddings.data.copy_(ft_sd['embeddings'])
        ft_sd = {k: v for k, v in ft_sd.items() if k != 'embeddings'}
        print('[init] embeddings: copied from finetune ckpt')
    load_state_dict(model, ft_sd, strict=False)
    print(f'[init] finetune: {len(ft_sd)} keys')
    return model.eval().to(device), cfg


# ─── Probe Dataset (parallel image loading) ────────────────────────────────
class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, ds, pipeline):
        self.ds = ds
        self.pipeline = pipeline

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        info = self.ds.get_data_info(idx)
        img_id, img_path = info['img_id'], info['img_path']
        data = self.pipeline(dict(img_path=img_path, img_id=img_id, instances=[]))
        meta = data['data_samples'].metainfo
        return {
            'img': data['inputs'],
            'img_id': img_id,
            'scale': np.asarray(meta['scale_factor'], dtype=np.float32),
            'pad': np.asarray(meta.get('pad_param', np.zeros(4, dtype=np.float32)),
                              dtype=np.float32),
        }


def collate_probe(batch):
    return {
        'img': torch.stack([b['img'] for b in batch]),
        'img_id': [b['img_id'] for b in batch],
        'scale': [b['scale'] for b in batch],
        'pad': [b['pad'] for b in batch],
    }


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from mmengine.dataset import Compose
    from mmengine.registry import init_default_scope
    from mmyolo.registry import DATASETS
    import mmyolo  # noqa
    import yolo_world  # noqa
    init_default_scope('mmyolo')

    with open(args.config) as f:
        pcfg = yaml.safe_load(f)

    tag = pcfg['experiment_tag']
    out_dir = args.out_dir or f'results/probe/{tag}'
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = f'embeddings/test_prompts/{tag}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(pcfg.get('amp', True)) and device.type == 'cuda'
    bs = int(pcfg.get('batch_size', 8))
    nw = int(pcfg.get('num_workers', 8))

    model, cfg = build_model(
        pcfg['model_config'], pcfg['checkpoint'],
        pcfg['text_encoder_pretrain_ckpt'],
        pcfg['text_encoder_model_name'],
        pcfg['text_encoder_use_lora'], device)
    head, head_module = model.bbox_head, model.bbox_head.head_module

    # ── Parse test_classes ───────────────────────────────────────────────
    raw_classes = pcfg.get('test_classes') or pcfg.get('test_prompts')
    if not raw_classes:
        raise ValueError('YAML must define test_classes.')

    class_entries = []
    flat_prompts = []
    prompt_to_flat = {}

    for entry in raw_classes:
        prompts = normalize_prompts(entry.get('prompts') or entry.get('prompt'))
        if not prompts:
            continue
        gt_alias = entry.get('gt_alias') or [prompts[0]]
        if isinstance(gt_alias, str):
            gt_alias = [gt_alias]
        name = entry.get('name') or (gt_alias[0] if gt_alias else prompts[0])
        idx_list = []
        for p in prompts:
            if p not in prompt_to_flat:
                prompt_to_flat[p] = len(flat_prompts)
                flat_prompts.append(p)
            idx_list.append(prompt_to_flat[p])
        class_entries.append(dict(
            name=name, prompts=prompts, gt_alias=gt_alias,
            prompt_indices=idx_list,
            zs_prompt=entry.get('zs_prompt') or None))

    N_C = len(class_entries)

    # gt_class_name -> test_class index
    gt_to_class = {}
    for ci, ce in enumerate(class_entries):
        for a in ce['gt_alias']:
            gt_to_class[a] = ci

    # ── Build dataset + loader (shared by both modes) ────────────────────
    split = pcfg.get('split', 'test')
    if split == 'train':
        ds_cfg = cfg.train_dataloader.dataset
    elif split == 'test':
        ds_cfg = cfg.test_dataloader.dataset
    else:
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")
    ds = DATASETS.build(ds_cfg)
    inner = getattr(ds, 'dataset', ds)
    imgid2ann = inner.imgid2annotations
    print(f'[init] split={split}  N_images={len(ds)}  N_classes={N_C}')

    test_ds_cfg = cfg.test_dataloader.dataset
    raw_pipeline = (test_ds_cfg.pipeline
                    if hasattr(test_ds_cfg, 'pipeline')
                    else test_ds_cfg.dataset.pipeline)
    img_pipeline_cfg = [t for t in raw_pipeline
                        if 'LoadAnnotations' not in t.get('type', '')
                        and 'PackDetInputs' not in t.get('type', '')]
    img_pipeline_cfg.append(dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param')))
    pipeline = Compose(img_pipeline_cfg)

    n_req = len(ds) if pcfg.get('num_images', 0) == 0 else min(
        pcfg['num_images'], len(ds))
    # Pre-filter: skip indices whose image file is missing on disk.
    # nuImages few-shot training points at filenames that live in the symlink
    # backup dir but not the active JPEGImages tree; without this the loader
    # raises FileNotFoundError mid-batch and aborts the run.
    indices = []
    n_skipped = 0
    for i in range(n_req):
        try:
            p = ds.get_data_info(i)['img_path']
        except Exception:
            n_skipped += 1
            continue
        if os.path.exists(p):
            indices.append(i)
        else:
            n_skipped += 1
    if n_skipped:
        print(f'[init] skipped {n_skipped}/{n_req} images with missing files on disk')
    n_imgs = len(indices)

    class SubsetDS(torch.utils.data.Dataset):
        def __init__(self, base, idxs): self.base, self.idxs = base, idxs
        def __len__(self): return len(self.idxs)
        def __getitem__(self, i): return self.base[self.idxs[i]]

    probe_ds = SubsetDS(ProbeDataset(ds, pipeline), indices)
    loader = torch.utils.data.DataLoader(
        probe_ds, batch_size=bs, num_workers=nw, shuffle=False,
        collate_fn=collate_probe, pin_memory=True, persistent_workers=(nw > 0))

    iou_thr = float(pcfg.get('iou_threshold', 0.5))
    print(f'[forward] bs={bs} workers={nw} amp={use_amp} IoU>={iou_thr}')

    # ═══════════════════════════════════════════════════════════════════
    # COMPARE MODE: zero-shot class name vs T2-tuned model.embeddings
    # ═══════════════════════════════════════════════════════════════════
    if args.compare:
        from yolo_world.models.backbones.mm_backbone import HuggingCLIPLanguageBackbone

        # ── T2-tuned embeddings: pull directly from model.embeddings ────
        owod_classes = list(inner.metainfo.get('classes', []))
        alias_to_emb_idx = {name: i for i, name in enumerate(owod_classes)}
        print(f'[compare] OWOD class list ({len(owod_classes)}): {owod_classes}')

        t2_emb_rows = []
        for ce in class_entries:
            idx = None
            for alias in ce['gt_alias']:
                idx = alias_to_emb_idx.get(alias)
                if idx is not None:
                    break
            t2_emb_rows.append(idx)

        missing = [class_entries[i]['name'] for i, idx in enumerate(t2_emb_rows) if idx is None]
        if missing:
            print(f'[compare] WARNING: no embedding index for: {missing}')

        with torch.no_grad():
            t2_emb = model.embeddings.detach().float()          # [N_owod, D]
            t2_rows_list = []
            for idx in t2_emb_rows:
                if idx is not None:
                    row = torch.nn.functional.normalize(t2_emb[idx], dim=-1, p=2)
                else:
                    row = torch.zeros(t2_emb.shape[1], device=device)
                t2_rows_list.append(row)
            t2_txt = torch.stack(t2_rows_list).to(device)       # [N_C, D]

        # ── Zero-shot embeddings: raw CLIP, single class name ───────────
        # Use explicit zs_prompt if set in YAML (for classes where name.replace('_',' ')
        # would give a nonsensical or over-specific string, e.g. "bus bendy" → "articulated bus").
        zs_prompts = [ce['zs_prompt'] or ce['name'].replace('_', ' ')
                      for ce in class_entries]
        print(f'[compare] zero-shot prompts: {zs_prompts}')

        # NOTE: HuggingCLIPLanguageBackbone overrides .train() without returning
        # self, so .eval() returns None → can't chain .to(device). Call them
        # in sequence on the instance instead.
        zs_te = HuggingCLIPLanguageBackbone(
            model_name=pcfg['text_encoder_model_name'],
            frozen_modules=['all'],
            training_use_lora=False,
        )
        zs_te.eval()
        zs_te.to(device)
        texts = [[p] for p in zs_prompts]
        with torch.no_grad():
            zs_raw = zs_te(texts)                                # [N_C, 1, D]
        zs_txt = torch.nn.functional.normalize(
            zs_raw.squeeze(1).float(), dim=-1, p=2).to(device)  # [N_C, D]
        del zs_te
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # ── Forward: both stacks in one pass [zs || t2] ─────────────────
        full_txt = torch.cat([zs_txt, t2_txt], dim=0)           # [2*N_C, D]
        full_txt_b = full_txt.unsqueeze(0)

        MAX_GT = 200
        gt_index_cmp = {}
        def gt_idx_cmp(name):
            if name not in gt_index_cmp:
                gt_index_cmp[name] = len(gt_index_cmp)
            return gt_index_cmp[name]

        gt_count_cmp = np.zeros(MAX_GT, dtype=np.int64)
        class_win_zs = np.zeros((N_C, MAX_GT), dtype=np.int64)
        class_win_t2 = np.zeros((N_C, MAX_GT), dtype=np.int64)

        t0 = time.time()
        n_done = 0
        with torch.no_grad():
            for batch in loader:
                imgs = batch['img'].to(device, non_blocking=True).float() / 255.0
                B = imgs.shape[0]
                txt_in = full_txt_b.expand(B, -1, -1)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    img_raw = model.backbone.forward_image(imgs)
                    img_feats = model.neck(img_raw) if model.with_neck else img_raw
                    cls_list, bbox_list = head_module.forward_one2one(img_feats, txt_in)

                featmap_sizes = [(t.shape[-2], t.shape[-1]) for t in cls_list]
                mlvl_priors = head.prior_generator.grid_priors(
                    featmap_sizes, dtype=torch.float32,
                    device=device, with_stride=True)
                flat_priors = torch.cat(mlvl_priors, dim=0)
                flat_logits = torch.cat([
                    t.permute(0, 2, 3, 1).reshape(B, -1, 2 * N_C).float()
                    for t in cls_list], dim=1)
                flat_bbox = torch.cat([
                    t.permute(0, 2, 3, 1).reshape(B, -1, 4).float()
                    for t in bbox_list], dim=1)
                anchor_boxes = head.bbox_coder.decode(
                    flat_priors[..., :2], flat_bbox, flat_priors[:, [2]][..., 0])

                all_scores = flat_logits.sigmoid()
                zs_scores = all_scores[..., :N_C]                # [B, N_a, N_C]
                t2_scores = all_scores[..., N_C:]                # [B, N_a, N_C]
                zs_argmax = zs_scores.argmax(dim=2)              # [B, N_a]
                t2_argmax = t2_scores.argmax(dim=2)

                for bi in range(B):
                    img_id = batch['img_id'][bi]
                    gt_full = parse_voc_xml(imgid2ann[img_id])
                    if not gt_full:
                        continue
                    gt_orig = torch.tensor([g[1] for g in gt_full],
                                           dtype=torch.float32, device=device)
                    sx, sy = float(batch['scale'][bi][0]), float(batch['scale'][bi][1])
                    pad_top  = float(batch['pad'][bi][0])
                    pad_left = float(batch['pad'][bi][2])
                    gt_boxes = gt_orig.clone()
                    gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * sx + pad_left
                    gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * sy + pad_top
                    gt_names = [g[0] for g in gt_full]

                    ious = box_iou_xyxy(anchor_boxes[bi], gt_boxes)
                    max_iou, best_gt = ious.max(dim=1)
                    keep = max_iou >= iou_thr
                    if not keep.any():
                        continue
                    kept_idx = keep.nonzero(as_tuple=True)[0]
                    kept_zs_am = zs_argmax[bi][kept_idx].cpu().numpy()
                    kept_t2_am = t2_argmax[bi][kept_idx].cpu().numpy()
                    kept_gt   = best_gt[kept_idx].cpu().numpy()

                    for j, gi in enumerate(kept_gt):
                        g   = gt_names[int(gi)]
                        g_i = gt_idx_cmp(g)
                        if g_i >= MAX_GT:
                            continue
                        gt_count_cmp[g_i] += 1
                        class_win_zs[kept_zs_am[j], g_i] += 1
                        class_win_t2[kept_t2_am[j], g_i] += 1

                n_done += B
                if n_done % (bs * 25) == 0 or n_done >= n_imgs:
                    dt = time.time() - t0
                    ips = n_done / max(dt, 1e-6)
                    eta = (n_imgs - n_done) / max(ips, 1e-6)
                    print(f'  processed {n_done}/{n_imgs}  '
                          f'({ips:.1f} img/s, eta {eta:.0f}s)')

        # ── Compute per-class metrics ────────────────────────────────────
        n_gt_seen = len(gt_index_cmp)
        p_cnt     = gt_count_cmp[:n_gt_seen]
        cwz       = class_win_zs[:, :n_gt_seen]
        cwt       = class_win_t2[:, :n_gt_seen]

        class_on_mask = np.zeros((N_C, n_gt_seen), dtype=bool)
        for ci, ce in enumerate(class_entries):
            for alias in ce['gt_alias']:
                if alias in gt_index_cmp and gt_index_cmp[alias] < n_gt_seen:
                    class_on_mask[ci, gt_index_cmp[alias]] = True

        n_anch = np.zeros(N_C, dtype=np.int64)
        for ci, ce in enumerate(class_entries):
            for alias in ce['gt_alias']:
                if alias in gt_index_cmp and gt_index_cmp[alias] < n_gt_seen:
                    n_anch[ci] += int(p_cnt[gt_index_cmp[alias]])

        def _cmetrics(cw):
            on_w  = (cw *  class_on_mask).sum(axis=1).astype(np.int64)
            off_w = (cw * (~class_on_mask)).sum(axis=1).astype(np.int64)
            tot   = on_w + off_w
            steal = off_w / np.maximum(tot, 1).astype(float)
            win   = on_w  / np.maximum(n_anch, 1).astype(float)
            return on_w, off_w, steal, win

        on_z, off_z, steal_z, win_z = _cmetrics(cwz)
        on_t, off_t, steal_t, win_t = _cmetrics(cwt)

        ds_key   = args.dataset or detect_dataset_key(pcfg['model_config'])
        split_map = TASK_SPLITS.get(ds_key)
        def task_of(ce):
            if not split_map: return ''
            ali = set(ce['gt_alias'])
            if ali & set(split_map['T1']): return 'T1'
            if ali & set(split_map['T2']): return 'T2'
            return ''

        # ── Write compare.csv ────────────────────────────────────────────
        cmp_header = ['class', 'task', 'n_anchors', 'zs_prompt',
                      't2_emb_idx', 'has_t2_emb',
                      'winner_zs', 'winner_t2', 'd_winner',
                      'steal_zs',  'steal_t2',  'd_steal',
                      'on_wins_zs', 'off_wins_zs',
                      'on_wins_t2', 'off_wins_t2']
        cmp_rows = []
        for ci, ce in enumerate(class_entries):
            cmp_rows.append([
                ce['name'], task_of(ce), int(n_anch[ci]),
                zs_prompts[ci],
                t2_emb_rows[ci] if t2_emb_rows[ci] is not None else '',
                t2_emb_rows[ci] is not None,
                f'{win_z[ci]:.4f}', f'{win_t[ci]:.4f}',
                f'{win_t[ci]-win_z[ci]:+.4f}',
                f'{steal_z[ci]:.4f}', f'{steal_t[ci]:.4f}',
                f'{steal_t[ci]-steal_z[ci]:+.4f}',
                int(on_z[ci]), int(off_z[ci]),
                int(on_t[ci]), int(off_t[ci]),
            ])
        with open(os.path.join(out_dir, 'compare.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(cmp_header)
            for r in cmp_rows:
                w.writerow(r)

        # ── Print + save comparison table ────────────────────────────────
        lines = [
            f'=== COMPARE  v={tag}  N_imgs={n_imgs}  N_classes={N_C} ===\n',
            'Zero-shot  = raw CLIP (openai/clip-vit-base-patch32) + class name only.',
            'T2-tuned   = model.embeddings from T2 checkpoint (trained nn.Parameter).',
            'Δ = T2tuned − zeroshot.  d_steal < 0 = tuning reduced stealing (good).',
            '              d_winner > 0 = tuning improved cross-class discrimination.\n',
        ]
        col_fmt = (f'  {"class":22s} {"task":4s} {"n":>8s}  '
                   f'{"win_zs":>7s} {"win_t2":>7s} {"d_win":>7s}   '
                   f'{"steal_zs":>9s} {"steal_t2":>9s} {"d_steal":>9s}')

        for grp, label in [('T1', 'base'), ('T2', 'novel'), ('', 'unlabeled')]:
            members = [(ci, ce) for ci, ce in enumerate(class_entries)
                       if task_of(ce) == grp]
            if not members:
                continue
            lines.append(f'── {grp or "??"} ({label}) ──')
            lines.append(col_fmt)
            for ci, ce in members:
                lines.append(
                    f'  {ce["name"][:22]:22s} {task_of(ce):4s} '
                    f'{int(n_anch[ci]):8d}  '
                    f'{win_z[ci]:7.3f} {win_t[ci]:7.3f} '
                    f'{win_t[ci]-win_z[ci]:+7.3f}   '
                    f'{steal_z[ci]:9.3f} {steal_t[ci]:9.3f} '
                    f'{steal_t[ci]-steal_z[ci]:+9.3f}')
            mwz = float(np.mean([win_z[ci]   for ci, _ in members]))
            mwt = float(np.mean([win_t[ci]   for ci, _ in members]))
            msz = float(np.mean([steal_z[ci] for ci, _ in members]))
            mst = float(np.mean([steal_t[ci] for ci, _ in members]))
            lines.append(
                f'  {"GROUP MEAN":22s} {grp:4s} {"":>8s}  '
                f'{mwz:7.3f} {mwt:7.3f} {mwt-mwz:+7.3f}   '
                f'{msz:9.3f} {mst:9.3f} {mst-msz:+9.3f}')
            lines.append('')

        txt = '\n'.join(lines)
        print('\n' + txt)
        with open(os.path.join(out_dir, 'compare.txt'), 'w') as f:
            f.write(txt + '\n')
        print(f'\n[write] {out_dir}/{{compare.csv,compare.txt}}')
        print(f'\nCOMPARE FINISHED at {time.strftime("%c")}')
        return

    # ═══════════════════════════════════════════════════════════════════
    # NORMAL MODE: multi-prompt leakage diagnostics
    # ═══════════════════════════════════════════════════════════════════
    txt_feats = encode_prompts_cached(model, flat_prompts, cache_dir, device)
    N_flat, D = txt_feats.shape
    print(f'[init] unique_prompts={N_flat}  dim={D}')

    class_flat_idx = [torch.tensor(ce['prompt_indices'], dtype=torch.long,
                                   device=device) for ce in class_entries]

    scores_per_gt     = defaultdict(list)
    sub_scores_per_gt = defaultdict(list)

    gt_index = {}
    def gt_idx_of(name):
        if name not in gt_index:
            gt_index[name] = len(gt_index)
        return gt_index[name]

    MAX_GT = 200
    prompt_sum_on_gt  = np.zeros((N_flat, MAX_GT), dtype=np.float64)
    gt_count          = np.zeros(MAX_GT, dtype=np.int64)
    global_win_count  = np.zeros((N_flat, MAX_GT), dtype=np.int64)

    prompt_owner = np.full(N_flat, -1, dtype=np.int64)
    for ci, ce in enumerate(class_entries):
        for fi in ce['prompt_indices']:
            if prompt_owner[fi] == -1:
                prompt_owner[fi] = ci

    # Per-prompt all-vs-all AUROC (matches selection objective): histogram of
    # sigmoid scores on (own-class anchors) vs (all-other-class anchors).
    AUROC_BINS = 200
    own_hist = np.zeros((N_flat, AUROC_BINS), dtype=np.int64)
    off_hist = np.zeros((N_flat, AUROC_BINS), dtype=np.int64)

    txt_b = txt_feats.unsqueeze(0)  # [1, N_flat, D]

    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch['img'].to(device, non_blocking=True).float() / 255.0
            B = imgs.shape[0]
            txt_in = txt_b.expand(B, -1, -1)

            with torch.cuda.amp.autocast(enabled=use_amp):
                img_raw = model.backbone.forward_image(imgs)
                img_feats = model.neck(img_raw) if model.with_neck else img_raw
                cls_list, bbox_list = head_module.forward_one2one(img_feats, txt_in)

            featmap_sizes = [(t.shape[-2], t.shape[-1]) for t in cls_list]
            mlvl_priors = head.prior_generator.grid_priors(
                featmap_sizes, dtype=torch.float32,
                device=device, with_stride=True)
            flat_priors = torch.cat(mlvl_priors, dim=0)
            flat_logits = torch.cat([
                t.permute(0, 2, 3, 1).reshape(B, -1, N_flat).float()
                for t in cls_list], dim=1)
            flat_bbox = torch.cat([
                t.permute(0, 2, 3, 1).reshape(B, -1, 4).float()
                for t in bbox_list], dim=1)
            anchor_boxes = head.bbox_coder.decode(
                flat_priors[..., :2], flat_bbox, flat_priors[:, [2]][..., 0])
            scores = flat_logits.sigmoid()                       # [B, N_a, N_flat]

            # Per-class score = MAX over that class's prompts (not sum).
            # Sum biased winner-rate toward classes with more prompts (3-prompt
            # class could reach 3.0 vs 1-prompt class capped at 1.0). Max is
            # scale-invariant across class prompt counts and matches what the
            # head would do at inference if you took the strongest prompt.
            class_scores = torch.stack([
                scores.index_select(2, idx).amax(dim=2) for idx in class_flat_idx
            ], dim=2)                                            # [B, N_a, N_C]

            for bi in range(B):
                img_id = batch['img_id'][bi]
                gt_full = parse_voc_xml(imgid2ann[img_id])
                if not gt_full:
                    continue
                gt_orig = torch.tensor([g[1] for g in gt_full],
                                       dtype=torch.float32, device=device)
                sx, sy = float(batch['scale'][bi][0]), float(batch['scale'][bi][1])
                pad_top  = float(batch['pad'][bi][0])
                pad_left = float(batch['pad'][bi][2])
                gt_boxes = gt_orig.clone()
                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * sx + pad_left
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * sy + pad_top
                gt_names = [g[0] for g in gt_full]

                ious = box_iou_xyxy(anchor_boxes[bi], gt_boxes)
                max_iou, best_gt = ious.max(dim=1)
                keep = max_iou >= iou_thr
                if not keep.any():
                    continue
                kept_idx  = keep.nonzero(as_tuple=True)[0]
                kept_class = class_scores[bi][kept_idx].cpu().numpy()   # [N_k, N_C]
                kept_flat  = scores[bi][kept_idx].cpu().numpy()         # [N_k, N_flat]
                kept_gt    = best_gt[kept_idx].cpu().numpy()
                kept_argmax_flat = kept_flat.argmax(axis=1)

                for j, gi in enumerate(kept_gt):
                    g = gt_names[int(gi)]
                    scores_per_gt[g].append(kept_class[j])
                    ci = gt_to_class.get(g)
                    if ci is not None and len(class_entries[ci]['prompt_indices']) > 1:
                        ce = class_entries[ci]
                        sub_scores_per_gt[g].append(kept_flat[j, ce['prompt_indices']])
                    g_i = gt_idx_of(g)
                    if g_i >= MAX_GT:
                        continue
                    prompt_sum_on_gt[:, g_i] += kept_flat[j]
                    gt_count[g_i] += 1
                    global_win_count[kept_argmax_flat[j], g_i] += 1

                target_ci = np.array(
                    [gt_to_class.get(gt_names[int(g)], -1) for g in kept_gt],
                    dtype=np.int64)
                bins = np.clip((kept_flat * AUROC_BINS).astype(np.int64),
                               0, AUROC_BINS - 1)
                own_match = (target_ci[:, None] == prompt_owner[None, :])
                fi_idx = np.broadcast_to(
                    np.arange(N_flat, dtype=np.int64)[None, :], bins.shape)
                fi_flat = fi_idx.ravel()
                b_flat = bins.ravel()
                np.add.at(own_hist,
                          (fi_flat[own_match.ravel()],
                           b_flat[own_match.ravel()]), 1)
                off_mask = ((~own_match) & (target_ci[:, None] >= 0)).ravel()
                np.add.at(off_hist,
                          (fi_flat[off_mask], b_flat[off_mask]), 1)

            n_done += B
            if n_done % (bs * 25) == 0 or n_done >= n_imgs:
                dt = time.time() - t0
                ips = n_done / max(dt, 1e-6)
                eta = (n_imgs - n_done) / max(ips, 1e-6)
                print(f'  processed {n_done}/{n_imgs}  '
                      f'({ips:.1f} img/s, eta {eta:.0f}s)')

    # ── Aggregate ────────────────────────────────────────────────────────
    gt_classes  = sorted(scores_per_gt.keys())
    class_names = [ce['name'] for ce in class_entries]
    score_mat   = np.zeros((len(gt_classes), N_C), dtype=np.float64)
    confusion   = np.zeros((len(gt_classes), N_C), dtype=np.int64)
    counts      = np.zeros(len(gt_classes), dtype=np.int64)
    winner_rate = {}
    margins     = {}

    for gi, g in enumerate(gt_classes):
        arr = np.stack(scores_per_gt[g])
        counts[gi] = arr.shape[0]
        score_mat[gi] = arr.mean(axis=0)
        argmax = arr.argmax(axis=1)
        for a in argmax:
            confusion[gi, a] += 1
        ci = gt_to_class.get(g)
        if ci is not None:
            winner_rate[g] = float((argmax == ci).mean())
            mask = np.ones(N_C, dtype=bool); mask[ci] = False
            best_other = arr[:, mask].max(axis=1) if mask.any() else np.zeros(arr.shape[0])
            margins[g] = float((arr[:, ci] - best_other).mean())
        else:
            winner_rate[g] = None
            margins[g] = None

    within_class = {}
    for ce in class_entries:
        if len(ce['prompts']) <= 1:
            continue
        bucket = []
        for a in ce['gt_alias']:
            bucket.extend(sub_scores_per_gt.get(a, []))
        if not bucket:
            within_class[ce['name']] = {p: None for p in ce['prompts']}
            continue
        sub_arr = np.stack(bucket)
        argmax = sub_arr.argmax(axis=1)
        within_class[ce['name']] = dict(
            N=int(sub_arr.shape[0]),
            winner_rate={p: float((argmax == k).mean()) for k, p in enumerate(ce['prompts'])},
            mean_score={p: float(sub_arr[:, k].mean()) for k, p in enumerate(ce['prompts'])})

    # ── Per-prompt leakage diagnostics ───────────────────────────────────
    n_gt_seen = len(gt_index)
    inv_gt_index = {v: k for k, v in gt_index.items()}
    seen_gt_names = [inv_gt_index[i] for i in range(n_gt_seen)]
    p_sum  = prompt_sum_on_gt[:, :n_gt_seen]
    p_cnt  = gt_count[:n_gt_seen]
    p_wins = global_win_count[:, :n_gt_seen]
    safe_cnt = np.maximum(p_cnt, 1)
    mean_pxg = p_sum / safe_cnt[None, :]

    on_mask = np.zeros((N_flat, n_gt_seen), dtype=bool)
    for fi in range(N_flat):
        ci = prompt_owner[fi]
        if ci < 0:
            continue
        for alias in class_entries[ci]['gt_alias']:
            if alias in gt_index and gt_index[alias] < n_gt_seen:
                on_mask[fi, gt_index[alias]] = True

    eps = 1e-9
    on_n    = (on_mask  * p_cnt[None, :]).sum(axis=1)
    off_n   = ((~on_mask) * p_cnt[None, :]).sum(axis=1)
    on_sum  = (p_sum *  on_mask).sum(axis=1)
    off_sum = (p_sum * (~on_mask)).sum(axis=1)
    mean_on  = on_sum  / np.maximum(on_n,  1)
    mean_off = off_sum / np.maximum(off_n, 1)
    specificity = mean_on / (mean_off + eps)

    on_wins   = (p_wins *  on_mask).sum(axis=1)
    off_wins  = (p_wins * (~on_mask)).sum(axis=1)
    total_wins = on_wins + off_wins
    steal_rate = off_wins / np.maximum(total_wins, 1)
    net_useful = on_wins.astype(np.int64) - off_wins.astype(np.int64)

    # All-vs-all AUROC per prompt (own-class anchors vs all-other-class anchors).
    # Computed from binned score histograms accumulated during the forward pass.
    auroc_per_prompt = np.full(N_flat, np.nan, dtype=np.float64)
    n_own_per_prompt = own_hist.sum(axis=1)
    n_off_per_prompt = off_hist.sum(axis=1)
    for fi in range(N_flat):
        n_own = int(n_own_per_prompt[fi]); n_off = int(n_off_per_prompt[fi])
        if n_own == 0 or n_off == 0:
            continue
        off_lt = np.cumsum(off_hist[fi]) - off_hist[fi]
        auroc_per_prompt[fi] = ((own_hist[fi] * (off_lt + 0.5 * off_hist[fi])).sum()
                                / (n_own * n_off))

    flat_within_wr = np.full(N_flat, np.nan, dtype=np.float64)
    for ce in class_entries:
        info = within_class.get(ce['name'])
        if not info or info.get('N') is None:
            continue
        for p, fi in zip(ce['prompts'], ce['prompt_indices']):
            wr = info['winner_rate'].get(p)
            if wr is not None:
                flat_within_wr[fi] = wr

    # ── Write outputs ────────────────────────────────────────────────────
    def write_csv(path, rows, header):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(header)
            for r in rows: w.writerow(r)

    header = ['gt_class', 'count'] + class_names
    write_csv(os.path.join(out_dir, 'score_matrix.csv'),
              [[g, int(counts[i])] + [f'{v:.4f}' for v in score_mat[i]]
               for i, g in enumerate(gt_classes)], header)
    write_csv(os.path.join(out_dir, 'confusion_argmax.csv'),
              [[g, int(counts[i])] + [int(v) for v in confusion[i]]
               for i, g in enumerate(gt_classes)], header)

    wc_rows = [['test_class', 'N', 'prompt', 'winner_rate', 'mean_score']]
    for cname, d in within_class.items():
        if d.get('N') is None:
            for p in next((ce['prompts'] for ce in class_entries if ce['name']==cname), []):
                wc_rows.append([cname, 0, p, '', ''])
            continue
        for p, wr in d['winner_rate'].items():
            wc_rows.append([cname, d['N'], p, f'{wr:.4f}', f"{d['mean_score'][p]:.4f}"])
    with open(os.path.join(out_dir, 'within_class_winners.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        for r in wc_rows: w.writerow(r)

    pp_header = ['prompt', 'parent_class', 'N_on', 'N_off',
                 'mean_on', 'mean_off', 'specificity', 'auroc',
                 'within_class_winner', 'global_wins_total',
                 'on_target_wins', 'off_target_wins', 'steal_rate', 'net_useful']
    pp_rows = [pp_header]
    rank_idx = np.argsort(-np.where(np.isnan(auroc_per_prompt),
                                    -np.inf, auroc_per_prompt))
    for fi in rank_idx:
        parent = class_entries[prompt_owner[fi]]['name'] if prompt_owner[fi] >= 0 else ''
        wr = flat_within_wr[fi]
        au = auroc_per_prompt[fi]
        pp_rows.append([
            flat_prompts[fi], parent,
            int(on_n[fi]), int(off_n[fi]),
            f'{mean_on[fi]:.4f}', f'{mean_off[fi]:.4f}',
            f'{specificity[fi]:.3f}',
            f'{au:.4f}' if not np.isnan(au) else '',
            f'{wr:.4f}' if not np.isnan(wr) else '',
            int(total_wins[fi]), int(on_wins[fi]), int(off_wins[fi]),
            f'{steal_rate[fi]:.4f}', int(net_useful[fi]),
        ])
    with open(os.path.join(out_dir, 'prompt_diagnostics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        for r in pp_rows: w.writerow(r)

    sm_header = ['victim_gt'] + [
        f'{flat_prompts[fi]}__{class_entries[prompt_owner[fi]]["name"] if prompt_owner[fi]>=0 else ""}'
        for fi in range(N_flat)]
    sm_rows = [sm_header]
    for gi, gname in enumerate(seen_gt_names):
        row = [gname]
        for fi in range(N_flat):
            ci = prompt_owner[fi]
            is_own = (ci >= 0 and gname in class_entries[ci]['gt_alias'])
            row.append(int(p_wins[fi, gi]) if not is_own else 0)
        sm_rows.append(row)
    with open(os.path.join(out_dir, 'stealing_matrix.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        for r in sm_rows: w.writerow(r)

    pxg_header = ['prompt', 'parent_class'] + seen_gt_names
    pxg_rows = [pxg_header]
    for fi in range(N_flat):
        parent = class_entries[prompt_owner[fi]]['name'] if prompt_owner[fi] >= 0 else ''
        pxg_rows.append([flat_prompts[fi], parent] +
                        [f'{mean_pxg[fi, gi]:.4f}' for gi in range(n_gt_seen)])
    with open(os.path.join(out_dir, 'prompt_x_gt_mean_score.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        for r in pxg_rows: w.writerow(r)

    summary = {
        'experiment_tag': tag,
        'config': args.config,
        'checkpoint': pcfg['checkpoint'],
        'num_images_processed': n_imgs,
        'iou_threshold': iou_thr,
        'split': split,
        'test_classes': [{'name': ce['name'], 'gt_alias': ce['gt_alias'],
                          'prompts': ce['prompts']} for ce in class_entries],
        'gt_classes_seen': gt_classes,
        'counts_per_gt': {g: int(counts[i]) for i, g in enumerate(gt_classes)},
        'cross_class_winner_rate': winner_rate,
        'cross_class_mean_margin': margins,
        'within_class': within_class,
        'per_prompt_auroc': {
            flat_prompts[fi]: (None if np.isnan(auroc_per_prompt[fi])
                               else float(auroc_per_prompt[fi]))
            for fi in range(N_flat)},
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    lines = [f'=== probe v={tag}  N_imgs={n_imgs}  N_classes={N_C}  '
             f'N_unique_prompts={N_flat} ===\n']
    lines.append('Cross-class winner rate (correct class = highest summed sigmoid):')
    for g in gt_classes:
        wr = winner_rate[g]; mg = margins[g]
        lines.append(f'  {g:25s} N={counts[gt_classes.index(g)]:7d} '
                     f'winner={"  n/a" if wr is None else f"{wr:.3f}"}  '
                     f'margin={"  n/a" if mg is None else f"{mg:+.3f}"}')

    if within_class:
        lines.append('\nWithin-class winner rate (which sub-prompt wins per anchor):')
        for cname, d in within_class.items():
            if d.get('N') is None:
                lines.append(f'  {cname:25s} (no GT samples)')
                continue
            lines.append(f'  {cname:25s} N={d["N"]:7d}')
            for p, wr in d['winner_rate'].items():
                lines.append(f'      {p:30s} winner={wr:.3f}  mean={d["mean_score"][p]:.4f}')

    lines.append('\nPer-prompt all-vs-all AUROC (own-class vs all-other anchors). '
                 'AUROC=0.5 random; >0.7 useful; >0.8 strong. Sorted by AUROC:')
    lines.append(f'  {"prompt":35s} {"parent":18s} '
                 f'{"AUROC":>6s} {"spec":>6s} {"within":>7s} {"on_wins":>9s} '
                 f'{"off_wins":>9s} {"steal%":>7s} {"net":>9s}')
    for fi in rank_idx:
        parent = class_entries[prompt_owner[fi]]['name'] if prompt_owner[fi] >= 0 else ''
        wr = flat_within_wr[fi]
        au = auroc_per_prompt[fi]
        lines.append(f'  {flat_prompts[fi][:35]:35s} {parent[:18]:18s} '
                     f'{"  -  " if np.isnan(au) else f"{au:.3f}":>6s} '
                     f'{specificity[fi]:6.2f} '
                     f'{"   - " if np.isnan(wr) else f"{wr:.3f}":>7s} '
                     f'{int(on_wins[fi]):9d} {int(off_wins[fi]):9d} '
                     f'{steal_rate[fi]*100:6.1f}% {int(net_useful[fi]):9d}')

    lines.append('\nTop-3 stealers per GT class:')
    for gi, gname in enumerate(seen_gt_names):
        if p_cnt[gi] < 50:
            continue
        steal_vec = p_wins[:, gi].copy().astype(np.float64)
        for fi in range(N_flat):
            ci = prompt_owner[fi]
            if ci >= 0 and gname in class_entries[ci]['gt_alias']:
                steal_vec[fi] = -1
        order = np.argsort(-steal_vec)[:3]
        items = []
        for fi in order:
            if steal_vec[fi] <= 0:
                continue
            parent = class_entries[prompt_owner[fi]]['name'] if prompt_owner[fi] >= 0 else ''
            items.append(f'{flat_prompts[fi]}({parent})={int(steal_vec[fi])}')
        if items:
            lines.append(f'  {gname:30s} N={int(p_cnt[gi]):7d}  ' + '  '.join(items))

    lines.append('\nMean score matrix (rows=GT, cols=test_class):')
    lines.append('GT \\ class'.ljust(25) + ' '.join(f'{n[:10]:>10s}' for n in class_names))
    for i, g in enumerate(gt_classes):
        lines.append(g.ljust(25) + ' '.join(f'{v:10.3f}' for v in score_mat[i]))

    txt = '\n'.join(lines)
    print('\n' + txt)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(txt + '\n')

    print(f'\n[write] {out_dir}/'
          '{score_matrix.csv,confusion_argmax.csv,within_class_winners.csv,'
          'prompt_diagnostics.csv,stealing_matrix.csv,prompt_x_gt_mean_score.csv,'
          'summary.json,summary.txt}')


if __name__ == '__main__':
    main()
