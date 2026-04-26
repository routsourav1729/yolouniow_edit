"""OWOD T2 diagnostic — anchor-score / energy / entropy / ratio per role.

Goal: characterise how the YOLO-UniOW head sees every object in the
few-shot training set, *without* the WAPR `anchor_score > max_known`
filter (which we suspect silently kills genuine objects).

Anchor-keep rule (matches YOLO-UniOW's anchor_label, minus the WAPR
gate we are studying):
    keep = (max_iou_to_GT < 0.5) & (anchor_score > 0.01)

We do NOT use `anchor_score > max_known` here — that's exactly what
we want to evaluate.

Buckets (per anchor):
  * matched_annot_<role>   IoU>=0.5 with a GT used in few-shot training
  * bg_unannot_base        passes keep, IoU>=0.5 with a base GT NOT in fewshot
  * bg_unannot_novel       passes keep, IoU>=0.5 with a novel GT NOT in fewshot
  * bg_unannot_unknown     passes keep, IoU>=0.5 with an unknown-class GT
  * bg_pure                passes keep, IoU<0.5 with every GT in the XML

Per anchor we record:
  anchor_score  = sigmoid(logit[-1])
  tunk_score    = sigmoid(logit[-2])
  max_known     = max(sigmoid(logit[:-2]))
  ratio         = max_known / anchor_score          (WAPR ratio)
  energy        = -logsumexp(known_logits)          (lower = more known)
  entropy       = -Σ p log p over softmax(known_logits)

Outputs: stdout summary + diagnose_summary.json + diagnose_thresholds.csv.
"""
import argparse
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description='OWOD anchor-stats diagnostic')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--num-images', type=int, default=0,
                   help='0 = all few-shot training images.')
    p.add_argument('--thresholds', type=str,
                   default='0.3,0.4,0.5,0.6,0.7,0.8,0.9',
                   help='Comma-separated WAPR ratio thresholds for the sweep.')
    return p.parse_args()


def percentiles(arr):
    if len(arr) == 0:
        return {}
    a = np.asarray(arr, dtype=np.float64)
    out = {'count': int(a.size),
           'min': float(a.min()), 'max': float(a.max()),
           'mean': float(a.mean()), 'std': float(a.std())}
    for q in (10, 25, 50, 75, 90, 95, 99):
        out[f'p{q}'] = float(np.percentile(a, q))
    return out


def box_iou_xyxy(a, b):
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    a_, b_ = a.unsqueeze(1), b.unsqueeze(0)
    lt = torch.maximum(a_[..., :2], b_[..., :2])
    rb = torch.minimum(a_[..., 2:], b_[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a_[..., 2] - a_[..., 0]) * (a_[..., 3] - a_[..., 1])
    area_b = (b_[..., 2] - b_[..., 0]) * (b_[..., 3] - b_[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)


def parse_voc_xml_full(xml_path):
    out = []
    for obj in ET.parse(xml_path).getroot().findall('object'):
        bb = obj.find('bndbox')
        out.append((obj.find('name').text.strip(), [
            float(bb.find('xmin').text) - 1.0,
            float(bb.find('ymin').text) - 1.0,
            float(bb.find('xmax').text) - 1.0,
            float(bb.find('ymax').text) - 1.0]))
    return out


def class_role(cls, base_set, novel_set):
    if cls in base_set: return 'base'
    if cls in novel_set: return 'novel'
    return 'unknown'


# Map img_id -> list of bboxes (original coords) used in few-shot training.
# A GT in the XML is "annotated_for_fewshot" iff a fewshot instance bbox
# matches it (within a small tolerance).
def build_fewshot_used_map(train_ds):
    """img_id -> list[(x1,y1,x2,y2)] of fewshot bboxes in original coords."""
    used = defaultdict(list)
    for i in range(len(train_ds)):
        info = train_ds.get_data_info(i)
        img_id = info['img_id']
        for inst in info['instances']:
            used[img_id].append(tuple(inst['bbox']))
    return used


def gt_is_annotated(gt_box_orig, fewshot_boxes, tol=1.0):
    """True if gt_box_orig matches any fewshot box within `tol` pixels."""
    for fb in fewshot_boxes:
        if (abs(gt_box_orig[0] - fb[0]) <= tol and
            abs(gt_box_orig[1] - fb[1]) <= tol and
            abs(gt_box_orig[2] - fb[2]) <= tol and
            abs(gt_box_orig[3] - fb[3]) <= tol):
            return True
    return False


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    thresholds = [float(t) for t in args.thresholds.split(',')]

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_state_dict
    from mmyolo.registry import MODELS, DATASETS
    import mmyolo  # noqa: F401
    import yolo_world  # noqa: F401
    init_default_scope('mmyolo')

    print(f'[init] config:     {args.config}')
    print(f'[init] checkpoint: {args.checkpoint}')
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.out_dir

    model = MODELS.build(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    state_dict = {k: v for k, v in state_dict.items()
                  if 'text_model' not in k}
    # Load checkpoint embeddings AS-IS — for a trained T2 ckpt we want the
    # actual trained novel + T_unk + T_anchor embeddings. The default
    # update_embeddings() path blends novel slots with config defaults
    # (used at the *start* of T2 training), which is wrong here.
    if 'embeddings' in state_dict:
        ckpt_emb = state_dict['embeddings']
        if ckpt_emb.shape == model.embeddings.shape:
            with torch.no_grad():
                model.embeddings.data.copy_(ckpt_emb.to(model.embeddings.device))
            print(f'[init] embeddings loaded from ckpt directly '
                  f'(shape={tuple(ckpt_emb.shape)})')
        else:
            print(f'[warn] embeddings shape mismatch: ckpt={tuple(ckpt_emb.shape)} '
                  f'model={tuple(model.embeddings.shape)} — falling back to update_embeddings')
            state_dict['embeddings'] = model.update_embeddings(ckpt_emb)
        print(f'[init] T_unk norm     = {ckpt_emb[-2].norm().item():.4f}')
        print(f'[init] T_anchor norm  = {ckpt_emb[-1].norm().item():.4f}')
        # Remove from state_dict so load_state_dict doesn't overwrite
        # our direct copy with anything stale.
        if ckpt_emb.shape == model.embeddings.shape:
            state_dict = {k: v for k, v in state_dict.items() if k != 'embeddings'}
    load_state_dict(model, state_dict, strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    head = model.bbox_head
    head_module = head.head_module

    num_classes = head_module.num_classes
    num_prev = model.num_prev_classes
    num_cur = num_classes - num_prev - 2
    print(f'[init] K(known)={num_prev + num_cur}  '
          f'(prev={num_prev}, cur={num_cur})  '
          f'unk_idx={num_classes - 2}  anchor_idx={num_classes - 1}')

    train_ds = DATASETS.build(cfg.train_dataloader.dataset)
    inner = getattr(train_ds, 'dataset', train_ds)
    class_names = list(inner.CLASS_NAMES)
    base_classes = class_names[:num_prev]
    novel_classes = class_names[num_prev:num_prev + num_cur]
    base_set, novel_set = set(base_classes), set(novel_classes)
    print(f'[init] base={base_classes}')
    print(f'[init] novel={novel_classes}')
    print(f'[init] {len(train_ds)} few-shot training images')

    imgid2ann = inner.imgid2annotations
    fewshot_used = build_fewshot_used_map(train_ds)

    # Build val/test pipeline (eval-mode forward on training images)
    val_ds_cfg = cfg.val_dataloader.dataset
    raw_pipeline = (val_ds_cfg.pipeline
                    if hasattr(val_ds_cfg, 'pipeline')
                    else val_ds_cfg.dataset.pipeline)
    img_pipeline_cfg = [t for t in raw_pipeline
                        if 'LoadAnnotations' not in t.get('type', '')
                        and 'PackDetInputs' not in t.get('type', '')]
    img_pipeline_cfg.append(dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param')))
    pipeline = Compose(img_pipeline_cfg)

    bucket_names = [
        'matched_annot_base', 'matched_annot_novel',
        'bg_unannot_base', 'bg_unannot_novel', 'bg_unannot_unknown',
        'bg_pure',
    ]
    metric_names = ['anchor', 'max_known', 'tunk', 'ratio',
                    'energy', 'entropy']
    dist = {b: {m: [] for m in metric_names} for b in bucket_names}
    per_cls = defaultdict(lambda: {m: [] for m in metric_names})  # cls_name -> metrics
    sweep = {thr: defaultdict(lambda: [0, 0]) for thr in thresholds}  # [kept, suppr]

    n_imgs = len(train_ds) if args.num_images == 0 else min(
        args.num_images, len(train_ds))
    print(f'\n[forward] running {n_imgs} images...')

    n_proc = 0
    with torch.no_grad():
        for i in range(n_imgs):
            info = train_ds.get_data_info(i)
            img_path, img_id = info['img_path'], info['img_id']
            data = pipeline(dict(img_path=img_path, img_id=img_id,
                                 instances=[]))
            img_tensor = data['inputs'].unsqueeze(0).float().to(device) / 255.0
            ds = data['data_samples']
            meta = ds.metainfo
            scale = np.asarray(meta['scale_factor'])
            pad = np.asarray(meta.get('pad_param', np.zeros(4, dtype=np.float32)))

            img_feats, txt_feats = model.extract_feat(img_tensor, [ds])
            cls_list, bbox_list = head_module.forward_one2one(
                img_feats, txt_feats)

            featmap_sizes = [(t.shape[-2], t.shape[-1]) for t in cls_list]
            mlvl_priors = head.prior_generator.grid_priors(
                featmap_sizes, dtype=img_tensor.dtype,
                device=img_tensor.device, with_stride=True)
            flat_priors = torch.cat(mlvl_priors, dim=0)

            flat_logits = torch.cat([
                t.permute(0, 2, 3, 1).reshape(1, -1, num_classes)
                for t in cls_list], dim=1)[0]
            flat_bbox = torch.cat([
                t.permute(0, 2, 3, 1).reshape(1, -1, 4)
                for t in bbox_list], dim=1)
            anchor_boxes = head.bbox_coder.decode(
                flat_priors[..., :2], flat_bbox, flat_priors[:, [2]][..., 0])[0]

            anchor_scores = flat_logits[:, -1].sigmoid()
            tunk_scores = flat_logits[:, -2].sigmoid()
            known_logits = flat_logits[:, :-2]
            known_scores = known_logits.sigmoid()
            max_known, _ = known_scores.max(dim=1)
            ratio = max_known / anchor_scores.clamp(min=1e-6)
            energy = -torch.logsumexp(known_logits, dim=1)
            log_softmax = torch.log_softmax(known_logits, dim=1)
            entropy = -(log_softmax.exp() * log_softmax).sum(dim=1)

            # Full XML annotations: keep ORIGINAL coords for fewshot match,
            # then transform to padded coords for IoU vs anchors.
            gt_full = parse_voc_xml_full(imgid2ann[img_id])
            fewshot_boxes_this_img = fewshot_used.get(img_id, [])
            if not gt_full:
                gt_boxes = torch.zeros((0, 4), device=device)
                gt_names, gt_annot = [], []
            else:
                orig_list = [g[1] for g in gt_full]
                gt_boxes_orig = torch.tensor(orig_list,
                                             dtype=torch.float32, device=device)
                sx, sy = float(scale[0]), float(scale[1])
                pad_top, pad_left = float(pad[0]), float(pad[2])
                gt_boxes = gt_boxes_orig.clone()
                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * sx + pad_left
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * sy + pad_top
                gt_names = [g[0] for g in gt_full]
                gt_annot = [gt_is_annotated(orig_list[i],
                                            fewshot_boxes_this_img)
                            for i in range(len(gt_full))]

            ious = box_iou_xyxy(anchor_boxes, gt_boxes)  # (N_anc, N_gt)
            ANCHOR_IOU = 0.5
            ANCHOR_SCORE_T = 0.01

            # Split GTs into fewshot vs non-fewshot for separate IoU stats
            if gt_boxes.shape[0] > 0:
                fewshot_mask = torch.tensor(gt_annot, dtype=torch.bool,
                                            device=device)
                if fewshot_mask.any():
                    ious_fs = ious[:, fewshot_mask]
                    max_iou_fs, best_gt_fs = ious_fs.max(dim=1)
                    fs_idx_map = fewshot_mask.nonzero(as_tuple=True)[0]
                else:
                    max_iou_fs = torch.zeros_like(anchor_scores)
                    best_gt_fs = torch.zeros_like(anchor_scores, dtype=torch.long)
                    fs_idx_map = torch.zeros(0, dtype=torch.long, device=device)

                if (~fewshot_mask).any():
                    ious_un = ious[:, ~fewshot_mask]
                    max_iou_un, best_gt_un = ious_un.max(dim=1)
                    un_idx_map = (~fewshot_mask).nonzero(as_tuple=True)[0]
                else:
                    max_iou_un = torch.zeros_like(anchor_scores)
                    best_gt_un = torch.zeros_like(anchor_scores, dtype=torch.long)
                    un_idx_map = torch.zeros(0, dtype=torch.long, device=device)
            else:
                max_iou_fs = torch.zeros_like(anchor_scores)
                max_iou_un = torch.zeros_like(anchor_scores)
                best_gt_fs = torch.zeros_like(anchor_scores, dtype=torch.long)
                best_gt_un = torch.zeros_like(anchor_scores, dtype=torch.long)
                fs_idx_map = torch.zeros(0, dtype=torch.long, device=device)
                un_idx_map = torch.zeros(0, dtype=torch.long, device=device)

            # YOLO-UniOW keep rule (without anchor_score>max_known):
            #   IoU<0.5 to ANY fewshot GT  AND  anchor_score>0.01.
            keep = (max_iou_fs < ANCHOR_IOU) & (anchor_scores > ANCHOR_SCORE_T)
            matched_fs = max_iou_fs >= ANCHOR_IOU

            # Pre-extract numpy for fast per-anchor work
            a_np = anchor_scores.cpu().numpy()
            mk_np = max_known.cpu().numpy()
            tu_np = tunk_scores.cpu().numpy()
            rt_np = ratio.cpu().numpy()
            en_np = energy.cpu().numpy()
            et_np = entropy.cpu().numpy()
            best_fs_np = best_gt_fs.cpu().numpy()
            best_un_np = best_gt_un.cpu().numpy()
            fs_idx_np = fs_idx_map.cpu().numpy()
            un_idx_np = un_idx_map.cpu().numpy()
            max_un_np = max_iou_un.cpu().numpy()

            def push(bucket, idx, cls_name=None):
                vals = (a_np[idx], mk_np[idx], tu_np[idx],
                        rt_np[idx], en_np[idx], et_np[idx])
                for k, v in zip(metric_names, vals):
                    dist[bucket][k].append(float(v))
                if cls_name is not None:
                    for k, v in zip(metric_names, vals):
                        per_cls[cls_name][k].append(float(v))

            # Anchors that positive-match a fewshot (annotated) GT
            for idx in matched_fs.nonzero(as_tuple=True)[0].tolist():
                g_local = best_fs_np[idx]
                g_global = int(fs_idx_np[g_local])
                role = class_role(gt_names[g_global], base_set, novel_set)
                if role == 'unknown':
                    continue
                push(f'matched_annot_{role}', idx, gt_names[g_global])

            # Anchors passing the keep rule: bucket by what unannotated
            # object they sit on (IoU>=0.5 to any non-fewshot GT) or pure bg.
            for idx in keep.nonzero(as_tuple=True)[0].tolist():
                bucket, cls_name = 'bg_pure', None
                if un_idx_np.size > 0 and max_un_np[idx] >= ANCHOR_IOU:
                    g_local = best_un_np[idx]
                    g_global = int(un_idx_np[g_local])
                    role = class_role(gt_names[g_global], base_set, novel_set)
                    bucket = f'bg_unannot_{role}'
                    cls_name = gt_names[g_global]
                push(bucket, idx, cls_name)

                rt = rt_np[idx]
                for thr in thresholds:
                    side = 0 if rt < thr else 1
                    sweep[thr][bucket][side] += 1

            n_proc += 1
            if n_proc % 10 == 0 or n_proc == n_imgs:
                print(f'  processed {n_proc}/{n_imgs}')

    # ── summarise ────────────────────────────────────────────────────
    print('\n[scores] per-bucket distributions '
          '(anchor / max_known / T_unk / ratio / energy / entropy):')
    summary_dist = {}
    for b in bucket_names:
        d = dist[b]
        summary_dist[b] = {m: percentiles(d[m]) for m in metric_names}
        n = len(d['anchor'])
        if n == 0:
            print(f'  {b:24s} (empty)')
            continue
        s = summary_dist[b]
        print(f'  {b:24s} N={n:6d}  '
              f'anchor[μ={s["anchor"]["mean"]:.3f} σ={s["anchor"]["std"]:.3f} '
              f'p50={s["anchor"]["p50"]:.3f} p95={s["anchor"]["p95"]:.3f}]  '
              f'maxK[μ={s["max_known"]["mean"]:.3f} σ={s["max_known"]["std"]:.3f}]  '
              f'Tunk[μ={s["tunk"]["mean"]:.3f} σ={s["tunk"]["std"]:.3f}]  '
              f'ratio[μ={s["ratio"]["mean"]:.3f} p50={s["ratio"]["p50"]:.3f}]  '
              f'energy[μ={s["energy"]["mean"]:.3f} σ={s["energy"]["std"]:.3f}]  '
              f'entropy[μ={s["entropy"]["mean"]:.3f} σ={s["entropy"]["std"]:.3f}]')

    print('\n[per-class] anchor_score / max_known / energy / entropy '
          '(all anchors with IoU>=0.5 to that class):')
    pc_summary = {}
    for cls_name in sorted(per_cls.keys()):
        d = per_cls[cls_name]
        n = len(d['anchor'])
        if n == 0:
            continue
        role = class_role(cls_name, base_set, novel_set)
        s = {m: percentiles(d[m]) for m in metric_names}
        pc_summary[cls_name] = {'role': role, 'count': n, 'stats': s}
        print(f'  {cls_name:25s} ({role:7s}) N={n:5d} '
              f'anchor[μ={s["anchor"]["mean"]:.3f} σ={s["anchor"]["std"]:.3f}] '
              f'maxK[μ={s["max_known"]["mean"]:.3f}] '
              f'Tunk[μ={s["tunk"]["mean"]:.3f}] '
              f'energy[μ={s["energy"]["mean"]:.3f} σ={s["energy"]["std"]:.3f}] '
              f'entropy[μ={s["entropy"]["mean"]:.3f}]')

    # ── ratio sweep ──────────────────────────────────────────────────
    print('\n[ablation] WAPR ratio sweep over keep-passing background anchors')
    print('  KEPT = ratio<thr (would survive WAPR as candidate unknown)')
    bg_buckets = ['bg_unannot_base', 'bg_unannot_novel',
                  'bg_unannot_unknown', 'bg_pure']
    print(f'  {"thr":>5s}  ' + '  '.join(f'{g:>22s}' for g in bg_buckets))
    print(f'  {"":5s}  ' + '  '.join(f'{"kept/suppr":>22s}' for _ in bg_buckets))
    sweep_rows = []
    for thr in thresholds:
        row = {'threshold': thr}
        for g in bg_buckets:
            kept, sup = sweep[thr][g]
            row[f'{g}_kept'] = kept
            row[f'{g}_suppressed'] = sup
        sweep_rows.append(row)
        print(f'  {thr:>5.2f}  ' + '  '.join(
            f'{row[g+"_kept"]:>10d}/{row[g+"_suppressed"]:<10d}'
            for g in bg_buckets))

    # ── write outputs ────────────────────────────────────────────────
    out_summary = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'num_images_processed': n_proc,
        'base_classes': base_classes,
        'novel_classes': novel_classes,
        'bucket_distributions': summary_dist,
        'per_class_stats': pc_summary,
        'thresholds_swept': thresholds,
        'sweep_rows': sweep_rows,
    }
    json_path = os.path.join(args.out_dir, 'diagnose_summary.json')
    with open(json_path, 'w') as f:
        json.dump(out_summary, f, indent=2)
    print(f'\n[write] {json_path}')

    csv_path = os.path.join(args.out_dir, 'diagnose_thresholds.csv')
    fieldnames = ['threshold'] + [
        f'{g}_{s}' for g in bg_buckets for s in ('kept', 'suppressed')]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sweep_rows:
            w.writerow(r)
    print(f'[write] {csv_path}')


if __name__ == '__main__':
    main()
