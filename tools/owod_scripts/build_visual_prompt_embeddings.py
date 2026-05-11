"""Build T2 known-class prompts from visual-cache novel prototypes.

The visual cache stores post-BN one2one support features per FPN level.
For prompt initialization, avoid a raw mean over every feature from every
level.  Each level has its own BN geometry, so the safer visual center is:

    p_l,c = L2(mean(features_l,c))
    t_c   = L2(mean_l(p_l,c))

Only current-task novel rows are replaced. Base rows, T_unk, and T_anchor are
handled by the normal OWOD load/update path.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--base-embeddings', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--reduce',
                        default='mean_level_prototypes',
                        choices=['mean_level_prototypes', 'mean_all'])
    return parser.parse_args()


def novel_proto(cache, novel_idx, reduce):
    feats = []
    level_counts = []
    for level_dict in cache['cache_per_level'].values():
        class_feats = level_dict.get(novel_idx)
        if class_feats is None or class_feats.numel() == 0:
            continue
        class_feats = class_feats.float()
        level_counts.append(int(class_feats.shape[0]))
        if reduce == 'mean_level_prototypes':
            feats.append(F.normalize(class_feats.mean(dim=0), dim=0))
        else:
            feats.append(class_feats)
    if not feats:
        return None, level_counts
    if reduce == 'mean_level_prototypes':
        proto = torch.stack(feats, dim=0).mean(dim=0)
    else:
        proto = torch.cat(feats, dim=0).mean(dim=0)
    return F.normalize(proto, dim=0), level_counts


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    cache = torch.load(args.cache, map_location='cpu')
    embeddings = torch.from_numpy(np.load(args.base_embeddings)).float()

    n_base = int(cache['n_base'])
    n_novel = int(cache['n_novel'])
    names = cache['novel_class_names']
    if embeddings.shape[0] < n_base + n_novel:
        raise ValueError(
            f'Embedding rows {embeddings.shape[0]} < required '
            f'{n_base + n_novel}')

    print(f'[init] cache={args.cache}')
    print(f'[init] base_embeddings={args.base_embeddings} '
          f'shape={tuple(embeddings.shape)}')
    print(f'[init] n_base={n_base} n_novel={n_novel} reduce={args.reduce}')

    for ci in range(n_novel):
        proto, level_counts = novel_proto(cache, ci, args.reduce)
        if proto is None:
            print(f'[warn] no cache features for novel {ci}: {names[ci]}')
            continue
        row = n_base + ci
        old = F.normalize(embeddings[row], dim=0)
        cos = float((old * proto).sum())
        embeddings[row] = proto
        print(f'[replace] row={row:02d} {names[ci]:20s} '
              f'old_text_cos_to_visual={cos:.4f} '
              f'level_counts={level_counts}')

    np.save(args.out, embeddings.numpy().astype(np.float32))
    print(f'[write] {args.out} shape={tuple(embeddings.shape)}')


if __name__ == '__main__':
    main()
