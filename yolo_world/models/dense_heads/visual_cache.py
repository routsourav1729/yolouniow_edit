"""Visual K-shot prototype cache, PER-FPN-LEVEL.

For each (FPN level l, novel class c), build a unit prototype
    p_{l,c} = L2( mean_k(BN(g_k)) )
from K-shot supports. At inference, the per-anchor visual logit is
    visual_logit_c(g) = s · <BN(g), p_{l,c}> + b
which has the same scale/bias as the text branch (text uses
<BN(g), L2(t_c)>). That makes a convex combination
    fused_c = (1-α)·text_logit_c + α·visual_logit_c
geometrically meaningful — both terms live on the same logit scale.

Per-level matters because each FPN level has its own BN affine; features
across levels are not directly comparable.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualCache(nn.Module):

    def __init__(self,
                 n_base: int,
                 n_novel: int,
                 embed_dim: int,
                 num_levels: int = 3,
                 # `reduce` / `topk` retained for backwards-compat with the cfg
                 # but not used by the prototype path.
                 reduce: str = 'mean',
                 topk: int = 3):
        super().__init__()
        self.n_base = n_base
        self.n_novel = n_novel
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.reduce = reduce
        self.topk = topk
        # Per-level prototypes (n_novel, D), L2-normed direction per class.
        # Plain attrs (not buffers) so EMAHook(update_buffers=True) ignores them.
        self.prototypes: List[Optional[torch.Tensor]] = [None] * num_levels
        # Per-level mask (n_novel,) bool — True iff that class has a prototype.
        self.has_class: List[Optional[torch.Tensor]] = [None] * num_levels
        self.has_cache = False
        self.register_buffer('_device_anchor',
                             torch.zeros((), dtype=torch.float32),
                             persistent=False)

    @property
    def _device(self) -> torch.device:
        return self._device_anchor.device

    def _apply(self, fn):
        out = super()._apply(fn)
        for lvl in range(self.num_levels):
            if self.prototypes[lvl] is not None:
                self.prototypes[lvl] = fn(self.prototypes[lvl])
                self.has_class[lvl] = fn(self.has_class[lvl])
        return out

    def load_cache(self,
                   cache_per_level: Dict[int, Dict[int, torch.Tensor]]):
        """cache_per_level: {level: {novel_idx: Tensor(M, D) of raw BN(g)}}.

        Builds L2(mean(supports)) prototype per (level, class). Empty buckets
        get a zero prototype + mask=False so the head leaves text logit alone.
        """
        any_loaded = False
        for lvl in range(self.num_levels):
            level_dict = cache_per_level.get(lvl, {})
            proto = torch.zeros(self.n_novel, self.embed_dim)
            mask = torch.zeros(self.n_novel, dtype=torch.bool)
            for ci, feats in level_dict.items():
                assert 0 <= ci < self.n_novel, f'bad novel idx {ci}'
                assert feats.shape[1] == self.embed_dim, \
                    f'cache dim {feats.shape[1]} != {self.embed_dim}'
                if feats.shape[0] == 0:
                    continue
                mean = feats.mean(dim=0)
                norm = mean.norm() + 1e-9
                proto[ci] = mean / norm
                mask[ci] = True
                any_loaded = True
            self.prototypes[lvl] = proto.to(self._device)
            self.has_class[lvl] = mask.to(self._device)
        if not any_loaded:
            raise RuntimeError(
                'VisualCache.load_cache: no prototypes built (empty cache)')
        self.has_cache = True

    @torch.no_grad()
    def forward_level(self, bn_embed: torch.Tensor, level: int):
        """bn_embed: (B,D,H,W) at FPN level `level`.

        Returns:
          dot:  (B, n_novel, H, W) — raw <BN(g), prototype_c>.
                The head wraps this with cls_contrast.{logit_scale, bias}.
          mask: (n_novel,) bool — True iff that class has a prototype here.
        """
        proto = self.prototypes[level]
        mask = self.has_class[level]
        # Same einsum form as BNContrastiveHead: <BN(x), L2(w)>.
        dot = torch.einsum('bdhw,nd->bnhw', bn_embed, proto)
        return dot, mask
