"""Discriminative attribute selection — single all-vs-all AUROC metric.

For each class c and candidate attribute q, score q by AUROC of
cosine(BN(F), e_q) on own-class anchors (P_c) vs all-other-class
anchors (∪_{r≠c} P_r). Pick top-K by AUROC.

Pipeline:
  1. Build reliable per-class support set from box_<K>shot_<gt_alias>_train.txt.
  2. Forward images, extract post-BN per-anchor features at ONE IoU.
  3. Encode 30 candidates per class via the model's CLIP text encoder.
  4. AUROC(score(c, q on P_c), score(c, q on P_¬c)).
  5. Top-K by AUROC → JSON.

Reuses build_model / encode_prompts_cached from probe_text_visual.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml


# ─── Reliable per-class support set ──────────────────────────────────────────
def build_reliable_support(fewshot_dir, fewshot_seed, fewshot_k, select_k,
                           classes_cfg, dataset_root, dataset_name):
    seed_dir = os.path.join(fewshot_dir, f'seed{fewshot_seed}')
    annot_dir = os.path.join(dataset_root, 'Annotations', dataset_name)
    rng = random.Random(fewshot_seed)
    pairs = defaultdict(list)
    img_paths = {}

    for ce in classes_cfg:
        cname = ce['name']
        fpath = None
        for alias in ce['gt_alias']:
            cand = os.path.join(seed_dir,
                                f'box_{fewshot_k}shot_{alias}_train.txt')
            if os.path.exists(cand):
                fpath = cand
                break
        if fpath is None:
            print(f'[reliable] no support file for "{cname}" — skipping')
            continue
        with open(fpath) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for line in lines:
            img_id = os.path.splitext(os.path.basename(line))[0]
            xml_path = os.path.join(annot_dir, f'{img_id}.xml')
            if not os.path.exists(xml_path):
                continue
            for o in ET.parse(xml_path).getroot().findall('object'):
                gt_name = o.find('name').text.strip()
                if gt_name not in ce['gt_alias']:
                    continue
                bb = o.find('bndbox')
                box = [float(bb.find('xmin').text) - 1.0,
                       float(bb.find('ymin').text) - 1.0,
                       float(bb.find('xmax').text) - 1.0,
                       float(bb.find('ymax').text) - 1.0]
                pairs[cname].append((img_id, box, gt_name))
                if img_id not in img_paths:
                    img_paths[img_id] = (line if os.path.isabs(line)
                                         else os.path.join(os.getcwd(), line))
        if len(pairs[cname]) > select_k:
            pairs[cname] = rng.sample(pairs[cname], select_k)

    reliable_gt = defaultdict(list)
    for cname, lst in pairs.items():
        for img_id, box, gt_name in lst:
            reliable_gt[img_id].append((gt_name, box))
    return (sorted(reliable_gt.keys()), dict(img_paths),
            dict(reliable_gt), {c: len(p) for c, p in pairs.items()})


class ReliableSupportDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_paths, pipeline):
        self.img_ids, self.img_paths, self.pipeline = img_ids, img_paths, pipeline
    def __len__(self): return len(self.img_ids)
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        data = self.pipeline(dict(img_path=self.img_paths[img_id],
                                  img_id=img_id, instances=[]))
        meta = data['data_samples'].metainfo
        return {'img': data['inputs'], 'img_id': img_id,
                'scale': np.asarray(meta['scale_factor'], dtype=np.float32),
                'pad': np.asarray(meta.get('pad_param',
                                           np.zeros(4, dtype=np.float32)),
                                  dtype=np.float32)}


def _import_probe():
    here = os.path.abspath(os.path.dirname(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from probe_text_visual import (build_model, collate_probe,
                                   encode_prompts_cached)
    return build_model, collate_probe, encode_prompts_cached


# ─── Stage 1: BN(F) extraction at one IoU (TAL geometric gate) ───────────────
@torch.no_grad()
def extract_visual_pools(model, head, head_module, loader, device,
                         classes_cfg, primary_iou, reliable_gt, use_amp):
    pools = defaultdict(list)
    gt_to_ci = {a: ci for ci, ce in enumerate(classes_cfg) for a in ce['gt_alias']}
    n_levels = head_module.num_levels
    o2o_cls = head_module.one2one_cls_preds
    o2o_reg = head_module.one2one_reg_preds
    o2o_contr = head_module.one2one_cls_contrasts
    proj = head_module.one2one_proj
    reg_max = head_module.reg_max
    radius = max(2.0 * (1.0 - float(primary_iou)), 1e-6)
    t0 = time.time(); n_seen = 0

    for batch in loader:
        imgs = batch['img'].to(device, non_blocking=True).float() / 255.0
        B = imgs.shape[0]
        with torch.cuda.amp.autocast(enabled=use_amp):
            img_feats = (model.neck(model.backbone.forward_image(imgs))
                         if model.with_neck
                         else model.backbone.forward_image(imgs))
            bn_lvl, bbox_lvl = [], []
            for li in range(n_levels):
                bn_lvl.append(o2o_contr[li].norm(o2o_cls[li](img_feats[li])).float())
                bbox_lvl.append(o2o_reg[li](img_feats[li]).float())

        featmap_sizes = [(t.shape[-2], t.shape[-1]) for t in bn_lvl]
        priors = torch.cat(head.prior_generator.grid_priors(
            featmap_sizes, dtype=torch.float32, device=device, with_stride=True),
            dim=0)
        bbox_per_lvl = []
        for bd in bbox_lvl:
            b, _, h, w = bd.shape
            if reg_max > 1:
                bd = bd.reshape(b, 4, reg_max, h*w).permute(0, 3, 1, 2)
                bp = bd.softmax(3).matmul(proj.view(-1, 1)).squeeze(-1)
                bp = bp.transpose(1, 2).reshape(b, 4, h, w)
            else:
                bp = bd
            bbox_per_lvl.append(bp.permute(0, 2, 3, 1).reshape(b, -1, 4))
        flat_bbox = torch.cat(bbox_per_lvl, dim=1)
        anchor_boxes = head.bbox_coder.decode(
            priors[..., :2], flat_bbox, priors[:, [2]][..., 0])
        bn_flat = torch.cat(
            [t.permute(0, 2, 3, 1).reshape(t.shape[0], -1, t.shape[1])
             for t in bn_lvl], dim=1)
        prior_xy = priors[:, :2]

        for bi in range(B):
            img_id = batch['img_id'][bi]
            gt_full = reliable_gt.get(img_id, [])
            if not gt_full:
                continue
            gt_orig = torch.tensor([g[1] for g in gt_full],
                                   dtype=torch.float32, device=device)
            sx, sy = float(batch['scale'][bi][0]), float(batch['scale'][bi][1])
            pad_top = float(batch['pad'][bi][0])
            pad_left = float(batch['pad'][bi][2])
            gt_boxes = gt_orig.clone()
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * sx + pad_left
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * sy + pad_top
            gt_names = [g[0] for g in gt_full]

            lt = gt_boxes[None, :, 0:2]
            rb = gt_boxes[None, :, 2:4]
            pxy = prior_xy[:, None, :]
            in_gt = torch.cat([pxy - lt, rb - pxy], dim=-1).min(dim=-1).values > 0
            gt_ctr = 0.5 * (gt_boxes[:, 0:2] + gt_boxes[:, 2:4])
            gt_diag = torch.sqrt(((gt_boxes[:, 2:4] - gt_boxes[:, 0:2]) ** 2).sum(-1))
            gt_half = (0.5 * gt_diag).clamp(min=1.0)
            d_norm = torch.cdist(prior_xy, gt_ctr) / gt_half[None, :]
            hit = in_gt & (d_norm <= radius)
            any_hit = hit.any(dim=1)
            if not any_hit.any():
                continue
            d_masked = torch.where(hit, d_norm,
                                   torch.full_like(d_norm, float('inf')))
            best_gt = d_masked.argmin(dim=1)
            gi_to_ci = torch.tensor(
                [gt_to_ci.get(g, -1) for g in gt_names],
                dtype=torch.long, device=device)
            anchor_ci = torch.where(any_hit, gi_to_ci[best_gt],
                                    torch.full_like(best_gt, -1))
            valid = anchor_ci >= 0
            if not valid.any():
                continue
            idxs = torch.where(valid)[0]
            cis = anchor_ci[valid].cpu().numpy()
            feats = bn_flat[bi][idxs].cpu().numpy()
            for ci_v, f in zip(cis, feats):
                pools[int(ci_v)].append(f.astype(np.float32))
        n_seen += B
        if n_seen % (loader.batch_size * 25) == 0:
            print(f'  [stage1] {n_seen} imgs ({n_seen/max(time.time()-t0,1e-6):.1f}/s)')
    return pools


# ─── Stage 2: text encoding ──────────────────────────────────────────────────
def encode_class_texts(model, classes_cfg, attr_dict, cache_dir, device,
                       encode_prompts_cached):
    """Returns txt_emb[ci] (30, D) for the 30 candidates of class ci."""
    all_strings, idx_map, per_class_idx, texts = [], {}, [], []
    for ce in classes_cfg:
        cands = list(attr_dict[ce['name']])
        if len(cands) != 30:
            raise ValueError(
                f'class {ce["name"]}: expected 30 candidates, got {len(cands)}')
        idxs = []
        for s in cands:
            if s not in idx_map:
                idx_map[s] = len(all_strings)
                all_strings.append(s)
            idxs.append(idx_map[s])
        per_class_idx.append(idxs)
        texts.append(cands)
    flat_np = encode_prompts_cached(
        model, all_strings, cache_dir, device).detach().cpu().numpy()
    return [flat_np[idxs] for idxs in per_class_idx], texts


# ─── Stage 3: AUROC (own vs all-others pooled) ──────────────────────────────
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U / (n_pos * n_neg). NaN if either side empty."""
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    all_v = np.concatenate([pos, neg])
    order = np.argsort(all_v, kind='stable')
    ranks = np.empty_like(order, dtype=np.float64)
    # Average ranks for ties.
    i = 0
    while i < len(all_v):
        j = i + 1
        while j < len(all_v) and all_v[order[j]] == all_v[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0  # 1-indexed
        ranks[order[i:j]] = avg
        i = j
    pos_rank_sum = ranks[:len(pos)].sum()
    return float((pos_rank_sum - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def compute_auroc_table(pools_arr, txt_emb, classes_cfg):
    """Returns auc[ci, q] for ci ∈ classes, q ∈ [0..30)."""
    N_C = len(classes_cfg)
    auc = np.full((N_C, 30), np.nan, dtype=np.float64)
    N_per_class = np.zeros(N_C, dtype=np.int64)
    for ci in range(N_C):
        P_c = pools_arr.get(ci)
        if P_c is None or len(P_c) == 0:
            continue
        N_per_class[ci] = len(P_c)
        # Pool all other classes' anchors.
        others = [pools_arr[r] for r in range(N_C)
                  if r != ci and r in pools_arr and len(pools_arr[r]) > 0]
        if not others:
            continue
        P_other = np.concatenate(others, axis=0)
        E = txt_emb[ci]                  # (30, D)
        S_c = P_c @ E.T                  # (N_c, 30)
        S_o = P_other @ E.T              # (N_o, 30)
        for q in range(30):
            auc[ci, q] = auroc(S_c[:, q], S_o[:, q])
    return auc, N_per_class


# ─── Output ──────────────────────────────────────────────────────────────────
def write_outputs(out_dir, classes_cfg, texts, auc, N_per_class, top_k):
    os.makedirs(out_dir, exist_ok=True)
    payload = {}
    for ci, ce in enumerate(classes_cfg):
        scores = auc[ci]                       # (30,)
        valid = ~np.isnan(scores)
        order = np.argsort(-np.where(valid, scores, -np.inf))
        sel = [int(q) for q in order[:top_k] if valid[q]]
        payload[ce['name']] = dict(
            support_size=int(N_per_class[ci]),
            selected=[texts[ci][q] for q in sel],
            selected_auroc=[float(scores[q]) for q in sel],
            all_scores={texts[ci][q]: (float(scores[q])
                                       if not np.isnan(scores[q]) else None)
                        for q in range(30)},
        )
    path = os.path.join(out_dir, f'selected_top{top_k}_auroc.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[write] {path}')
    return payload


def print_summary(classes_cfg, payload, top_k):
    print(f'\n=== top-{top_k} attributes per class (by all-vs-all AUROC) ===')
    print(f'  {"class":22s} {"N":>5s}  {"top-1 AUROC":>11s}  attribute')
    for ce in classes_cfg:
        info = payload[ce['name']]
        N = info['support_size']
        if not info['selected']:
            print(f'  {ce["name"][:22]:22s} {N:5d}  (no support)')
            continue
        for k, (s, a) in enumerate(zip(info['selected'], info['selected_auroc'])):
            head = (f'  {ce["name"][:22]:22s} {N:5d}  '
                    if k == 0 else ' ' * 32)
            print(f'{head}{a:11.3f}  {s}')


# ─── Driver ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--num-images', type=int, default=None)
    p.add_argument('--top-k', type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    build_model, collate_probe, encode_prompts_cached = _import_probe()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from mmengine.dataset import Compose
    from mmengine.registry import init_default_scope
    import mmyolo  # noqa
    import yolo_world  # noqa
    init_default_scope('mmyolo')

    with open(args.config) as f:
        pcfg = yaml.safe_load(f)
    if args.num_images is not None:
        pcfg['num_images'] = args.num_images
    out_dir = args.out_dir or pcfg['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = f'embeddings/attribute_select/{pcfg["experiment_tag"]}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(pcfg.get('amp', True)) and device.type == 'cuda'
    bs = int(pcfg.get('batch_size', 8))
    nw = int(pcfg.get('num_workers', 8))
    primary_iou = float(pcfg.get('primary_iou', 0.8))
    raw_k = pcfg.get('top_k_select', pcfg.get('top_k', args.top_k))
    top_k = int(raw_k[0] if isinstance(raw_k, list) else raw_k)

    with open(pcfg['attributes_json']) as f:
        attr_dict = json.load(f)
    classes_cfg = pcfg['classes']
    for ce in classes_cfg:
        if ce['name'] not in attr_dict:
            raise KeyError(f'class "{ce["name"]}" missing from attributes_json')

    print(f'[init] tag={pcfg["experiment_tag"]} '
          f'N_classes={len(classes_cfg)} IoU={primary_iou} K={top_k}')

    model, mmcfg = build_model(
        pcfg['model_config'], pcfg['checkpoint'],
        pcfg['text_encoder_pretrain_ckpt'],
        pcfg['text_encoder_model_name'],
        pcfg['text_encoder_use_lora'], device)
    model.eval()
    head, head_module = model.bbox_head, model.bbox_head.head_module

    fewshot_dir = os.environ.get('FEWSHOT_DIR') or pcfg.get('fewshot_dir')
    fewshot_seed = int(os.environ.get('FEWSHOT_SEED',
                                      pcfg.get('fewshot_seed', 1)))
    fewshot_k = int(os.environ.get('FEWSHOT_K', pcfg.get('fewshot_k', 10)))
    select_k = int(pcfg.get('select_k', 20))
    dataset_root = pcfg.get('dataset_root', 'data/OWOD')
    dataset_name = os.environ.get('DATASET') or pcfg.get('dataset_name')

    image_ids, img_paths, reliable_gt, per_class_count = build_reliable_support(
        fewshot_dir, fewshot_seed, fewshot_k, select_k,
        classes_cfg, dataset_root, dataset_name)
    print('[reliable] reliable boxes per class:')
    for ce in classes_cfg:
        print(f'  {ce["name"][:22]:22s}  '
              f'{per_class_count.get(ce["name"], 0):3d}/{select_k}')
    print(f'[reliable] {len(image_ids)} unique images')

    test_ds_cfg = mmcfg.test_dataloader.dataset
    raw_pipeline = (test_ds_cfg.pipeline if hasattr(test_ds_cfg, 'pipeline')
                    else test_ds_cfg.dataset.pipeline)
    img_pipeline_cfg = [t for t in raw_pipeline
                        if 'LoadAnnotations' not in t.get('type', '')
                        and 'PackDetInputs' not in t.get('type', '')]
    img_pipeline_cfg.append(dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param')))
    pipeline = Compose(img_pipeline_cfg)

    image_ids = [iid for iid in image_ids if os.path.exists(img_paths[iid])]
    if pcfg.get('num_images', 0):
        image_ids = image_ids[:int(pcfg['num_images'])]
    print(f'[init] N_support_images={len(image_ids)}')

    loader = torch.utils.data.DataLoader(
        ReliableSupportDataset(image_ids, img_paths, pipeline),
        batch_size=bs, num_workers=nw, shuffle=False,
        collate_fn=collate_probe, pin_memory=True,
        persistent_workers=(nw > 0))

    print(f'[stage1] extracting BN(F) at IoU={primary_iou}, '
          f'bs={bs} workers={nw} amp={use_amp}')
    t0 = time.time()
    pools_raw = extract_visual_pools(
        model, head, head_module, loader, device,
        classes_cfg, primary_iou, reliable_gt, use_amp)
    print(f'[stage1] done in {time.time()-t0:.1f}s')

    pools_arr = {}
    for ci, entries in pools_raw.items():
        if not entries:
            continue
        feats = np.stack(entries, axis=0)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        pools_arr[ci] = feats.astype(np.float32)
    print('[stage1] pool sizes:')
    for ci, ce in enumerate(classes_cfg):
        print(f'  {ce["name"][:22]:22s}  N={len(pools_arr.get(ci, [])):4d}')

    print('\n[stage2] encoding 30 candidates per class')
    txt_emb, texts = encode_class_texts(
        model, classes_cfg, attr_dict, cache_dir, device,
        encode_prompts_cached)

    print('[stage3] computing all-vs-all AUROC')
    auc, N_per_class = compute_auroc_table(pools_arr, txt_emb, classes_cfg)

    payload = write_outputs(out_dir, classes_cfg, texts, auc, N_per_class, top_k)
    print_summary(classes_cfg, payload, top_k)


if __name__ == '__main__':
    main()
