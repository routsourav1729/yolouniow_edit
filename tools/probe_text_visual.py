"""Text-Visual alignment probe for YOLO-UniOW OWOD (multi-prompt, batched).

Each test class can carry MULTIPLE prompts. The cross-class score for an
anchor against test class C is sum_{p in C.prompts}(sigmoid(BN(Fvis) · E_p)).
Within-class winner rate tracks which sub-prompt of C scores highest for
anchors matched to a GT in C.gt_alias.

Inference is batched (DataLoader workers + GPU batch size + AMP fp16).
Embeddings are encoded via the model's own LoRA-finetuned CLIP text encoder
(loaded from pretrain ckpt) and cached per experiment_tag.
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

def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--config', help='Path to probe YAML config')
    g.add_argument('--dataset', choices=list(DATASET_CONFIGS),
                   help='Shorthand dataset name (selects default config)')
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()
    if args.dataset:
        args.config = DATASET_CONFIGS[args.dataset]
    return args


def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_').lower()[:60]


def normalize_prompts(p):
    """Accept either a list or a comma-separated string."""
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


# ─── Embedding cache (per-prompt; shared across classes) ───────────────────
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
            'img': data['inputs'],            # uint8 CHW tensor
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

    # ── Parse test_classes (each entry: name + prompts list + gt_alias) ──
    raw_classes = pcfg.get('test_classes') or pcfg.get('test_prompts')
    if not raw_classes:
        raise ValueError('YAML must define test_classes (or legacy test_prompts).')

    class_entries = []      # list of dicts: {name, prompts, gt_alias, prompt_indices}
    flat_prompts = []       # all unique prompts (order-preserving), shared cache
    prompt_to_flat = {}

    for entry in raw_classes:
        prompts = normalize_prompts(entry.get('prompts') or entry.get('prompt'))
        if not prompts:
            continue
        gt_alias = entry.get('gt_alias') or [prompts[0]]
        if isinstance(gt_alias, str):
            gt_alias = [gt_alias]
        name = entry.get('name') or gt_alias[0] if gt_alias else prompts[0]
        idx_list = []
        for p in prompts:
            if p not in prompt_to_flat:
                prompt_to_flat[p] = len(flat_prompts)
                flat_prompts.append(p)
            idx_list.append(prompt_to_flat[p])
        class_entries.append(dict(
            name=name, prompts=prompts, gt_alias=gt_alias,
            prompt_indices=idx_list))

    txt_feats = encode_prompts_cached(model, flat_prompts, cache_dir, device)
    N_flat, D = txt_feats.shape
    N_C = len(class_entries)
    print(f'[init] classes={N_C}  unique_prompts={N_flat}  dim={D}')

    # gt_class_name -> test_class index whose gt_alias contains it
    gt_to_class = {}
    for ci, ce in enumerate(class_entries):
        for a in ce['gt_alias']:
            gt_to_class[a] = ci

    # Pre-compute per-class flat indices on device
    class_flat_idx = [torch.tensor(ce['prompt_indices'], dtype=torch.long,
                                   device=device) for ce in class_entries]

    # ── Build dataset ────────────────────────────────────────────────────
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
    print(f'[init] split={split}  N_images={len(ds)}')

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

    n_imgs = len(ds) if pcfg.get('num_images', 0) == 0 else min(
        pcfg['num_images'], len(ds))
    indices = list(range(n_imgs))

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

    # gt_class -> list of [N_C] class-summed sigmoid score vectors
    scores_per_gt = defaultdict(list)
    # gt_class -> list of arrays (one per matched anchor) of sigmoid scores
    # restricted to that GT's test_class sub-prompts (used for within-class)
    sub_scores_per_gt = defaultdict(list)

    # ── Streaming per-prompt diagnostics ────────────────────────────────
    # gt_class index assignment (extended dynamically as we see GTs)
    gt_index = {}                                                      # gt_name -> idx
    def gt_idx_of(name):
        if name not in gt_index:
            gt_index[name] = len(gt_index)
        return gt_index[name]

    # All structures grow as we see new GT classes; allocate generously
    # then trim after processing.
    MAX_GT = 200
    prompt_sum_on_gt = np.zeros((N_flat, MAX_GT), dtype=np.float64)    # mean score per (prompt, gt)
    gt_count = np.zeros(MAX_GT, dtype=np.int64)                        # anchors per GT class
    global_win_count = np.zeros((N_flat, MAX_GT), dtype=np.int64)      # global argmax wins per (prompt, gt)

    # Also: which test_class entry "owns" each unique prompt (for parent_class / on_target masks)
    prompt_owner = np.full(N_flat, -1, dtype=np.int64)                 # flat prompt -> class entry idx
    for ci, ce in enumerate(class_entries):
        for fi in ce['prompt_indices']:
            if prompt_owner[fi] == -1:
                prompt_owner[fi] = ci

    txt_feats_b = txt_feats.unsqueeze(0)  # [1, N_flat, D]

    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch['img'].to(device, non_blocking=True).float() / 255.0
            B = imgs.shape[0]
            txt_in = txt_feats_b.expand(B, -1, -1)

            with torch.cuda.amp.autocast(enabled=use_amp):
                img_raw = model.backbone.forward_image(imgs)
                img_feats = model.neck(img_raw) if model.with_neck else img_raw
                cls_list, bbox_list = head_module.forward_one2one(img_feats, txt_in)

            featmap_sizes = [(t.shape[-2], t.shape[-1]) for t in cls_list]
            mlvl_priors = head.prior_generator.grid_priors(
                featmap_sizes, dtype=torch.float32,
                device=device, with_stride=True)
            flat_priors = torch.cat(mlvl_priors, dim=0)
            # [B, N_a, N_flat]
            flat_logits = torch.cat([
                t.permute(0, 2, 3, 1).reshape(B, -1, N_flat).float()
                for t in cls_list], dim=1)
            flat_bbox = torch.cat([
                t.permute(0, 2, 3, 1).reshape(B, -1, 4).float()
                for t in bbox_list], dim=1)
            anchor_boxes = head.bbox_coder.decode(
                flat_priors[..., :2], flat_bbox, flat_priors[:, [2]][..., 0])
            scores = flat_logits.sigmoid()                              # [B, N_a, N_flat]

            # Class-summed scores: [B, N_a, N_C]
            class_scores = torch.stack([
                scores.index_select(2, idx).sum(dim=2) for idx in class_flat_idx
            ], dim=2)

            for bi in range(B):
                img_id = batch['img_id'][bi]
                gt_full = parse_voc_xml(imgid2ann[img_id])
                if not gt_full:
                    continue
                gt_orig = torch.tensor([g[1] for g in gt_full],
                                       dtype=torch.float32, device=device)
                sx, sy = float(batch['scale'][bi][0]), float(batch['scale'][bi][1])
                pad_top, pad_left = float(batch['pad'][bi][0]), float(batch['pad'][bi][2])
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
                kept_class = class_scores[bi][kept_idx].cpu().numpy()    # [N_k, N_C]
                kept_flat = scores[bi][kept_idx].cpu().numpy()           # [N_k, N_flat]
                kept_gt = best_gt[kept_idx].cpu().numpy()

                # Global argmax across ALL prompts (for stealing detection)
                kept_argmax_flat = kept_flat.argmax(axis=1)              # [N_k]

                for j, gi in enumerate(kept_gt):
                    g = gt_names[int(gi)]
                    scores_per_gt[g].append(kept_class[j])
                    ci = gt_to_class.get(g)
                    if ci is not None and len(class_entries[ci]['prompt_indices']) > 1:
                        ce = class_entries[ci]
                        sub = kept_flat[j, ce['prompt_indices']]
                        sub_scores_per_gt[g].append(sub)

                    # Streaming per-prompt aggregations
                    g_i = gt_idx_of(g)
                    if g_i >= MAX_GT:
                        continue
                    prompt_sum_on_gt[:, g_i] += kept_flat[j]
                    gt_count[g_i] += 1
                    global_win_count[kept_argmax_flat[j], g_i] += 1

            n_done += B
            if n_done % (bs * 25) == 0 or n_done >= n_imgs:
                dt = time.time() - t0
                ips = n_done / max(dt, 1e-6)
                eta = (n_imgs - n_done) / max(ips, 1e-6)
                print(f'  processed {n_done}/{n_imgs}  '
                      f'({ips:.1f} img/s, eta {eta:.0f}s)')

    # ── Aggregate ────────────────────────────────────────────────────────
    gt_classes = sorted(scores_per_gt.keys())
    class_names = [ce['name'] for ce in class_entries]
    score_mat = np.zeros((len(gt_classes), N_C), dtype=np.float64)
    confusion = np.zeros((len(gt_classes), N_C), dtype=np.int64)
    counts = np.zeros(len(gt_classes), dtype=np.int64)
    winner_rate = {}
    margins = {}

    for gi, g in enumerate(gt_classes):
        arr = np.stack(scores_per_gt[g])                                # [N, N_C]
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

    # Within-class winner rates
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
        sub_arr = np.stack(bucket)                                     # [N, K]
        argmax = sub_arr.argmax(axis=1)
        rates = {p: float((argmax == k).mean())
                 for k, p in enumerate(ce['prompts'])}
        means = {p: float(sub_arr[:, k].mean())
                 for k, p in enumerate(ce['prompts'])}
        within_class[ce['name']] = dict(N=int(sub_arr.shape[0]),
                                        winner_rate=rates,
                                        mean_score=means)

    # ── Per-prompt leakage diagnostics ──────────────────────────────────
    # Trim to actually-seen GT classes
    n_gt_seen = len(gt_index)
    inv_gt_index = {v: k for k, v in gt_index.items()}
    seen_gt_names = [inv_gt_index[i] for i in range(n_gt_seen)]
    p_sum  = prompt_sum_on_gt[:, :n_gt_seen]
    p_cnt  = gt_count[:n_gt_seen]
    p_wins = global_win_count[:, :n_gt_seen]
    # mean score per (prompt, gt)
    safe_cnt = np.maximum(p_cnt, 1)
    mean_pxg = p_sum / safe_cnt[None, :]                                # [N_flat, n_gt]

    # On-target mask per prompt: GT classes whose test_class entry includes this prompt
    on_mask = np.zeros((N_flat, n_gt_seen), dtype=bool)
    for fi in range(N_flat):
        ci = prompt_owner[fi]
        if ci < 0:
            continue
        for alias in class_entries[ci]['gt_alias']:
            if alias in gt_index and gt_index[alias] < n_gt_seen:
                on_mask[fi, gt_index[alias]] = True

    # Specificity = mean_on / (mean_off + eps), and N counts
    eps = 1e-9
    on_n  = (on_mask  * p_cnt[None, :]).sum(axis=1)
    off_n = ((~on_mask) * p_cnt[None, :]).sum(axis=1)
    on_sum  = (p_sum * on_mask).sum(axis=1)
    off_sum = (p_sum * (~on_mask)).sum(axis=1)
    mean_on  = on_sum  / np.maximum(on_n,  1)
    mean_off = off_sum / np.maximum(off_n, 1)
    specificity = mean_on / (mean_off + eps)

    on_wins  = (p_wins * on_mask).sum(axis=1)
    off_wins = (p_wins * (~on_mask)).sum(axis=1)
    total_wins = on_wins + off_wins
    steal_rate = off_wins / np.maximum(total_wins, 1)
    net_useful = (on_wins.astype(np.int64) - off_wins.astype(np.int64))

    # Per-prompt within-class winner rate (extract from the within_class dict
    # which was computed above using sibling-only argmax per anchor)
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

    # Within-class CSV
    wc_rows = [['test_class', 'N', 'prompt', 'winner_rate', 'mean_score']]
    for cname, d in within_class.items():
        if d.get('N') is None:
            for p in next((ce['prompts'] for ce in class_entries if ce['name']==cname), []):
                wc_rows.append([cname, 0, p, '', ''])
            continue
        for p, wr in d['winner_rate'].items():
            wc_rows.append([cname, d['N'], p, f'{wr:.4f}',
                            f"{d['mean_score'][p]:.4f}"])
    with open(os.path.join(out_dir, 'within_class_winners.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        for r in wc_rows: w.writerow(r)

    # Per-prompt diagnostic table (the main "leakage / usefulness" view)
    pp_header = ['prompt', 'parent_class', 'N_on', 'N_off',
                 'mean_on', 'mean_off', 'specificity',
                 'within_class_winner', 'global_wins_total',
                 'on_target_wins', 'off_target_wins',
                 'steal_rate', 'net_useful']
    pp_rows = [pp_header]
    rank_idx = np.argsort(-net_useful)  # highest net_useful first
    for fi in rank_idx:
        parent = class_entries[prompt_owner[fi]]['name'] if prompt_owner[fi] >= 0 else ''
        wr = flat_within_wr[fi]
        pp_rows.append([
            flat_prompts[fi], parent,
            int(on_n[fi]), int(off_n[fi]),
            f'{mean_on[fi]:.4f}', f'{mean_off[fi]:.4f}',
            f'{specificity[fi]:.3f}',
            f'{wr:.4f}' if not np.isnan(wr) else '',
            int(total_wins[fi]),
            int(on_wins[fi]), int(off_wins[fi]),
            f'{steal_rate[fi]:.4f}',
            int(net_useful[fi]),
        ])
    with open(os.path.join(out_dir, 'prompt_diagnostics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        for r in pp_rows: w.writerow(r)

    # Stealing matrix: rows = victim GT, cols = prompts that aren't theirs
    sm_header = ['victim_gt'] + [f'{flat_prompts[fi]}__{class_entries[prompt_owner[fi]]["name"] if prompt_owner[fi]>=0 else ""}'
                                 for fi in range(N_flat)]
    sm_rows = [sm_header]
    for gi, gname in enumerate(seen_gt_names):
        row = [gname]
        for fi in range(N_flat):
            ci = prompt_owner[fi]
            is_own = (ci >= 0 and gname in class_entries[ci]['gt_alias'])
            cell = int(p_wins[fi, gi]) if not is_own else 0
            row.append(cell)
        sm_rows.append(row)
    with open(os.path.join(out_dir, 'stealing_matrix.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        for r in sm_rows: w.writerow(r)

    # Per-prompt mean activation on every GT class (including non-target)
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
        'test_classes': [
            {'name': ce['name'], 'gt_alias': ce['gt_alias'],
             'prompts': ce['prompts']} for ce in class_entries],
        'gt_classes_seen': gt_classes,
        'counts_per_gt': {g: int(counts[i]) for i, g in enumerate(gt_classes)},
        'cross_class_winner_rate': winner_rate,
        'cross_class_mean_margin': margins,
        'within_class': within_class,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Human-readable
    lines = [f'=== probe v={tag}  N_imgs={n_imgs}  N_classes={N_C}  '
             f'N_unique_prompts={N_flat} ===\n']
    lines.append('Cross-class winner rate (correct class = highest summed sigmoid):')
    for g in gt_classes:
        wr = winner_rate[g]; mg = margins[g]
        wr_s = f'{wr:.3f}' if wr is not None else '  n/a'
        mg_s = f'{mg:+.3f}' if mg is not None else '  n/a'
        lines.append(f'  {g:25s} N={counts[gt_classes.index(g)]:7d} '
                     f'winner={wr_s}  margin={mg_s}')

    if within_class:
        lines.append('\nWithin-class winner rate (which sub-prompt wins per anchor):')
        for cname, d in within_class.items():
            if d.get('N') is None:
                lines.append(f'  {cname:25s} (no GT samples)')
                continue
            lines.append(f'  {cname:25s} N={d["N"]:7d}')
            for p, wr in d['winner_rate'].items():
                lines.append(f'      {p:30s} winner={wr:.3f}  '
                             f'mean={d["mean_score"][p]:.4f}')

    # Per-prompt diagnostic ranked report
    lines.append('\nPer-prompt leakage diagnostic '
                 '(sorted by net_useful = on_target_wins - off_target_wins):')
    lines.append(f'  {"prompt":35s} {"parent":18s} '
                 f'{"spec":>6s} {"within":>7s} {"on_wins":>9s} '
                 f'{"off_wins":>9s} {"steal%":>7s} {"net":>9s}')
    for fi in rank_idx:
        parent = class_entries[prompt_owner[fi]]['name'] if prompt_owner[fi] >= 0 else ''
        wr = flat_within_wr[fi]
        wr_s = f'{wr:.3f}' if not np.isnan(wr) else '   - '
        lines.append(f'  {flat_prompts[fi][:35]:35s} {parent[:18]:18s} '
                     f'{specificity[fi]:6.2f} {wr_s:>7s} '
                     f'{int(on_wins[fi]):9d} {int(off_wins[fi]):9d} '
                     f'{steal_rate[fi]*100:6.1f}% {int(net_useful[fi]):9d}')

    # Top-leakers per victim GT class
    lines.append('\nTop-3 stealers per GT class (which prompts steal anchors from this class):')
    for gi, gname in enumerate(seen_gt_names):
        if p_cnt[gi] < 50:
            continue  # skip rare GT classes
        # Mask out prompts that are own-class; rank others by win count
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
            lines.append(f'  {gname:30s} N={int(p_cnt[gi]):7d}  '
                         + '  '.join(items))

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
