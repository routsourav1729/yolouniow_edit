#!/usr/bin/env python3
"""
Cosine similarity analysis: class embeddings vs T_anchor (wildcard), before and after T2 training.

For each dataset (IDD, FOOD_VOC, FOOD_VOCCOCO):
  - CLIP-init (before T2): load idd_t2.npy  (CLIP text embeddings)
  - T1-trained (after T1): extract from T1 checkpoint state_dict['embeddings']
  - T2-trained (after T2): extract from T2 checkpoint state_dict['embeddings']
  - T_anchor: taken from the checkpoint's embeddings[-1]
  - T_unk:    taken from the checkpoint's embeddings[-2]

Prints a ranked table + pairwise class similarity heatmap (saved as png).

Usage (offline, no GPU needed):
    python tools/owod_scripts/embedding_similarity_analysis.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(BASE_DIR)

# ─── Dataset metadata ─────────────────────────────────────────────────────────
DATASETS = {
    'IDD': {
        'embed_dir': 'embeddings/uniow-idd',
        'prefix':    'idd',
        'base_classes': ['car', 'motorcycle', 'rider', 'person',
                         'autorickshaw', 'bicycle', 'traffic sign',
                         'traffic light'],
        'novel_classes': ['bus', 'truck', 'tanker_vehicle', 'crane_truck',
                          'street_cart', 'excavator'],
        't1_ckpt':   'work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task1/best_owod_Both_epoch_20.pth',
        't2_ckpt':   'work_dirs/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_train_task2/best_owod_Both_epoch_80.pth',
    },
    'FOOD_VOC': {
        'embed_dir': 'embeddings/uniow-food-voc',
        'prefix':    'food_voc',
        'base_classes': [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
        ],
        'novel_classes': [
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
        ],
        't1_ckpt':   'work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voc_train_task1/best_owod_Both_epoch_15.pth',
        't2_ckpt':   'work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voc_train_task2_10shot_seed1_wapr/best_owod_Both_epoch_20.pth',
    },
    'FOOD_VOCCOCO': {
        'embed_dir': 'embeddings/uniow-food-voccoco',
        'prefix':    'food_voccoco',
        'base_classes': [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
        ],
        'novel_classes': [
            'truck', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator',
        ],
        't1_ckpt':   'work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voccoco_train_task1/best_owod_Both_epoch_20.pth',
        't2_ckpt':   'work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voccoco_train_task2_10shot_seed1_wapr/best_owod_Both_epoch_5.pth',
    },
}


def cosine(a, b):
    """a: (D,), b: (D,) -> scalar"""
    a = F.normalize(torch.as_tensor(a, dtype=torch.float32).flatten(), dim=0)
    b = F.normalize(torch.as_tensor(b, dtype=torch.float32).flatten(), dim=0)
    return float((a * b).sum())


def cosine_mat(A, b):
    """A: (N, D), b: (D,) -> (N,) cosine similarities"""
    A = F.normalize(torch.as_tensor(A, dtype=torch.float32), dim=1)
    b = F.normalize(torch.as_tensor(b, dtype=torch.float32).flatten(), dim=0)
    return (A @ b).numpy()


def find_best_ckpt(path_or_dir, pattern='best*.pth'):
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if os.path.isdir(path_or_dir):
        import glob
        matches = sorted(glob.glob(os.path.join(path_or_dir, pattern)), key=os.path.getmtime)
        if matches:
            return matches[-1]
        # fallback: epoch_*.pth
        matches = sorted(glob.glob(os.path.join(path_or_dir, 'epoch_*.pth')), key=os.path.getmtime)
        if matches:
            return matches[-1]
    return None


def load_ckpt_embeddings(ckpt_path):
    """Load embeddings tensor from checkpoint state_dict. Returns numpy (N, D)."""
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return None
    print(f"  Loading ckpt: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location='cpu')['state_dict']
    emb = sd['embeddings'].float().numpy()  # (n_known + 2, D): [..., T_unk, T_anchor]
    return emb


def print_table(rows, header, col_widths):
    """Print a simple ASCII table."""
    sep = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'
    fmt = '|' + '|'.join(f' {{:{w}}} ' for w in col_widths) + '|'
    print(sep)
    print(fmt.format(*header))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)


def analyze_dataset(name, info):
    print(f"\n{'='*72}")
    print(f"EMBEDDING COSINE SIMILARITY — {name}")
    print(f"{'='*72}")

    base_cls  = info['base_classes']
    novel_cls = info['novel_classes']
    all_cls   = base_cls + novel_cls
    n_base    = len(base_cls)
    n_novel   = len(novel_cls)
    n_known   = n_base + n_novel

    embed_dir = info['embed_dir']

    # ── 1. CLIP-initialized embeddings (before any training) ─────────────────
    clip_t2_path = os.path.join(embed_dir, f"{info['prefix']}_t2.npy")
    clip_t1_path = os.path.join(embed_dir, f"{info['prefix']}_t1.npy")
    clip_anc_path = os.path.join(embed_dir, 'object_tuned.npy')
    clip_unk_path = os.path.join(embed_dir, 'object.npy')

    clip_t2  = np.load(clip_t2_path) if os.path.isfile(clip_t2_path) else None
    clip_t1  = np.load(clip_t1_path) if os.path.isfile(clip_t1_path) else None
    clip_anc = np.load(clip_anc_path).flatten() if os.path.isfile(clip_anc_path) else None
    clip_unk = np.load(clip_unk_path).flatten() if os.path.isfile(clip_unk_path) else None

    # ── 2. Trained checkpoint embeddings ─────────────────────────────────────
    t1_ckpt_path = find_best_ckpt(info['t1_ckpt'])
    t2_ckpt_path = find_best_ckpt(info['t2_ckpt'])

    t1_emb = load_ckpt_embeddings(t1_ckpt_path)   # (8+2, D) or (14+2, D)?
    t2_emb = load_ckpt_embeddings(t2_ckpt_path)   # (n_known+2, D)

    # Extract T_anchor and T_unk from checkpoints
    t1_anc = t1_emb[-1] if t1_emb is not None else None
    t1_unk = t1_emb[-2] if t1_emb is not None else None
    t2_anc = t2_emb[-1] if t2_emb is not None else None
    t2_unk = t2_emb[-2] if t2_emb is not None else None

    # T1 ckpt class embeddings (base only, shape n_base x D)
    t1_class_emb = t1_emb[:n_base] if t1_emb is not None and len(t1_emb) >= n_base + 2 else None
    # T2 ckpt class embeddings (all known, shape n_known x D)
    t2_class_emb = t2_emb[:n_known] if t2_emb is not None and len(t2_emb) >= n_known + 2 else None

    print(f"\n  n_base={n_base}  n_novel={n_novel}  n_known={n_known}")
    if t1_emb is not None:
        print(f"  T1 ckpt embeddings shape: {t1_emb.shape}  "
              f"T_anchor norm={np.linalg.norm(t1_anc):.4f}  T_unk norm={np.linalg.norm(t1_unk):.4f}")
    if t2_emb is not None:
        print(f"  T2 ckpt embeddings shape: {t2_emb.shape}  "
              f"T_anchor norm={np.linalg.norm(t2_anc):.4f}  T_unk norm={np.linalg.norm(t2_unk):.4f}")

    # ── 3. Per-class cosine sim table ─────────────────────────────────────────
    print(f"\n  Cosine similarity with T_anchor (wildcard embedding)")
    print(f"  Legend: [B]=base  [N]=novel  | col A=CLIP-init  B=T1-trained  C=T2-trained")

    header = ['class', 'type', 'CLIP-init→anc', 'T1-ckpt→anc', 'T2-ckpt→anc', 'Δ(T2-CLIP)']
    col_w  = [22, 4, 14, 12, 12, 12]

    rows = []
    for i, cls in enumerate(all_cls):
        tag = '[B]' if i < n_base else '[N]'

        # CLIP init sim (vs clip_anc from object_tuned.npy)
        if clip_t2 is not None and clip_anc is not None and i < len(clip_t2):
            sim_clip = cosine(clip_t2[i], clip_anc)
        else:
            sim_clip = float('nan')

        # T1 ckpt sim (vs t1_anc) — only valid for base classes
        if t1_class_emb is not None and t1_anc is not None and i < len(t1_class_emb):
            sim_t1 = cosine(t1_class_emb[i], t1_anc)
        else:
            sim_t1 = float('nan')

        # T2 ckpt sim (vs t2_anc) — all known classes
        if t2_class_emb is not None and t2_anc is not None and i < len(t2_class_emb):
            sim_t2 = cosine(t2_class_emb[i], t2_anc)
        else:
            sim_t2 = float('nan')

        delta = sim_t2 - sim_clip if not (np.isnan(sim_t2) or np.isnan(sim_clip)) else float('nan')

        def fmt(v):
            return f'{v:+.4f}' if not np.isnan(v) else '    --  '

        rows.append((cls[:22], tag, fmt(sim_clip), fmt(sim_t1), fmt(sim_t2), fmt(delta)))

    print_table(rows, header, col_w)

    # ── 4. T_unk and T_anchor similarities vs each other ──────────────────────
    if t2_anc is not None and t2_unk is not None:
        print(f"\n  T_unk vs T_anchor cosine (T2 ckpt): {cosine(t2_unk, t2_anc):+.4f}")
    if t2_anc is not None and clip_anc is not None:
        print(f"  T_anchor shift (CLIP→T2): {cosine(clip_anc, t2_anc):+.4f}")

    # ── 5. Summary: mean sim by group ─────────────────────────────────────────
    if t2_class_emb is not None and t2_anc is not None:
        sims_t2 = cosine_mat(t2_class_emb, t2_anc)
        base_sims  = sims_t2[:n_base]
        novel_sims = sims_t2[n_base:n_base + n_novel]
        print(f"\n  T2-trained class→T_anchor similarity summary:")
        print(f"    Base  ({n_base:2d} classes): mean={base_sims.mean():+.4f}  std={base_sims.std():.4f}  "
              f"min={base_sims.min():+.4f}  max={base_sims.max():+.4f}")
        if n_novel:
            print(f"    Novel ({n_novel:2d} classes): mean={novel_sims.mean():+.4f}  std={novel_sims.std():.4f}  "
                  f"min={novel_sims.min():+.4f}  max={novel_sims.max():+.4f}")

    # ── 6. Pairwise class-class similarity heatmap ───────────────────────────
    if t2_class_emb is not None:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch

            E = F.normalize(torch.as_tensor(t2_class_emb, dtype=torch.float32), dim=1).numpy()
            sim_mat = E @ E.T  # (N, N)

            out_dir = f'vis_output/embedding_similarity'
            os.makedirs(out_dir, exist_ok=True)

            fig, ax = plt.subplots(figsize=(max(8, n_known * 0.55), max(7, n_known * 0.5)))
            im = ax.imshow(sim_mat, vmin=-0.3, vmax=1.0, cmap='RdYlGn', aspect='auto')
            ax.set_xticks(range(n_known))
            ax.set_yticks(range(n_known))
            tick_labels = [f'[B]{c[:12]}' if i < n_base else f'[N]{c[:12]}' for i, c in enumerate(all_cls)]
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
            ax.set_yticklabels(tick_labels, fontsize=7)
            # Overlay divider line between base and novel
            ax.axhline(n_base - 0.5, color='black', lw=1.5)
            ax.axvline(n_base - 0.5, color='black', lw=1.5)
            # Colorbar
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            ax.set_title(f'{name} T2 — pairwise class cosine similarity (post-T2 training)')
            legend_elements = [Patch(facecolor='#1a9641', label='Base classes'),
                               Patch(facecolor='#fdae61', label='Novel classes')]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=7)
            fig.tight_layout()
            out_path = os.path.join(out_dir, f'{name.lower()}_t2_class_sim.png')
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"\n  [VIS] Pairwise heatmap saved: {out_path}")

            # ── Bar chart: each class vs T_anchor ────────────────────────────
            if t2_anc is not None and clip_t2 is not None:
                fig2, ax2 = plt.subplots(figsize=(max(10, n_known * 0.55), 5))
                x = np.arange(n_known)
                w = 0.38
                sims_clip = cosine_mat(clip_t2, t2_anc)
                sims_t2_v = cosine_mat(t2_class_emb, t2_anc)
                colors = ['#2166ac' if i < n_base else '#d73027' for i in range(n_known)]
                ax2.bar(x - w/2, sims_clip, w, label='CLIP-init', color='#a6bddb', edgecolor='none')
                ax2.bar(x + w/2, sims_t2_v, w, label='T2-trained', color=colors, edgecolor='none', alpha=0.85)
                ax2.axvline(n_base - 0.5, color='black', lw=1.2, linestyle='--')
                ax2.axhline(0, color='gray', lw=0.8)
                ax2.set_xticks(x)
                ax2.set_xticklabels([c[:14] for c in all_cls], rotation=90, fontsize=7)
                ax2.set_ylabel('Cosine similarity with T_anchor')
                ax2.set_title(f'{name} — class↔T_anchor cosine: CLIP-init vs T2-trained')
                ax2.legend(fontsize=8)
                fig2.tight_layout()
                out_path2 = os.path.join(out_dir, f'{name.lower()}_t2_anchor_bar.png')
                fig2.savefig(out_path2, dpi=120)
                plt.close(fig2)
                print(f"  [VIS] Bar chart saved:     {out_path2}")

        except ImportError:
            print("  [SKIP] matplotlib not available — skipping plots")


def print_cross_dataset_summary():
    print(f"\n{'='*72}")
    print(f"CROSS-DATASET: T2-trained class → T_anchor cosine (mean per group)")
    print(f"{'='*72}")
    header = ['dataset', 'base_mean', 'novel_mean', 'Δ(novel-base)', 'anc_norm']
    col_w  = [16, 10, 11, 14, 10]
    rows = []
    for name, info in DATASETS.items():
        n_base  = len(info['base_classes'])
        n_novel = len(info['novel_classes'])
        t2_path = find_best_ckpt(info['t2_ckpt'])
        t2_emb  = load_ckpt_embeddings(t2_path)
        if t2_emb is None:
            rows.append((name, '--', '--', '--', '--'))
            continue
        t2_anc = t2_emb[-1]
        t2_cls = t2_emb[:n_base + n_novel]
        sims   = cosine_mat(t2_cls, t2_anc)
        bm     = sims[:n_base].mean()
        nm     = sims[n_base:].mean() if n_novel else float('nan')
        delta  = nm - bm if not np.isnan(nm) else float('nan')
        anorm  = float(np.linalg.norm(t2_anc))

        def f(v): return f'{v:+.4f}' if not np.isnan(v) else '    --'
        rows.append((name, f(bm), f(nm), f(delta), f'{anorm:.4f}'))
    print_table(rows, header, col_w)


def main():
    for name, info in DATASETS.items():
        analyze_dataset(name, info)
    print_cross_dataset_summary()
    print("\nDone.")


if __name__ == '__main__':
    main()
