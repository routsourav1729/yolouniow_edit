"""Visual K-shot cache for FS-OSOD logit-space ensembling at inference.

Holds (n_novel, K_max, D) post-BN visual features from K-shot GTs and
returns per-anchor mean cosine to each novel class. No training, no grad.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualCache(nn.Module):

    def __init__(self,
                 n_base: int,
                 n_novel: int,
                 embed_dim: int,
                 reduce: str = 'mean',
                 topk: int = 3):
        super().__init__()
        assert reduce in ('mean', 'topk_mean', 'max')
        self.n_base = n_base
        self.n_novel = n_novel
        self.embed_dim = embed_dim
        self.reduce = reduce
        self.topk = topk
        # Plain attributes — NOT register_buffer — so EMAHook(update_buffers=True)
        # doesn't track them and choke on shape change after load_cache().
        self.cache = None
        self.cache_mask = None
        self.has_cache = False
        # Tiny buffer that DOES move with .to()/.cuda() — tells us the live
        # device so load_cache puts the cache on the right one.
        self.register_buffer('_device_anchor',
                             torch.zeros((), dtype=torch.float32),
                             persistent=False)

    @property
    def _device(self):
        return self._device_anchor.device

    def _apply(self, fn):
        out = super()._apply(fn)
        if self.cache is not None:
            self.cache = fn(self.cache)
            self.cache_mask = fn(self.cache_mask)
        return out

    def load_cache(self, cache_dict: dict):
        """cache_dict: {novel_idx (0..n_novel-1): Tensor(K_c, D)}."""
        if not cache_dict:
            raise RuntimeError('VisualCache.load_cache got empty dict')
        K_max = max(v.shape[0] for v in cache_dict.values())
        cache = torch.zeros(self.n_novel, K_max, self.embed_dim)
        mask = torch.zeros(self.n_novel, K_max, dtype=torch.bool)
        for ci, feats in cache_dict.items():
            assert 0 <= ci < self.n_novel, f'bad novel idx {ci}'
            assert feats.shape[1] == self.embed_dim, \
                f'cache dim {feats.shape[1]} != {self.embed_dim}'
            cache[ci, :feats.shape[0]] = feats
            mask[ci, :feats.shape[0]] = True
        self.cache = cache.to(self._device)
        self.cache_mask = mask.to(self._device)
        self.has_cache = True

    @torch.no_grad()
    def forward(self, bn_embed: torch.Tensor) -> torch.Tensor:
        """bn_embed: (B, D, H, W) -> visual cosine (B, n_novel, H, W)."""
        v = F.normalize(bn_embed, dim=1)                          # (B,D,H,W)
        c = F.normalize(self.cache, dim=-1)                       # (n_novel,K,D)
        cos = torch.einsum('bdhw,nkd->bnkhw', v, c)               # (B,n_novel,K,H,W)
        mask = self.cache_mask[None, :, :, None, None]            # (1,n_novel,K,1,1)

        if self.reduce == 'mean':
            cos = cos * mask
            denom = mask.sum(dim=2).clamp(min=1).to(cos.dtype)    # (1,n_novel,1,1)
            return cos.sum(dim=2) / denom

        cos = cos.masked_fill(~mask, float('-inf'))
        if self.reduce == 'max':
            out = cos.max(dim=2).values
        else:  # topk_mean
            k = min(self.topk, cos.shape[2])
            top = cos.topk(k, dim=2).values
            out = top.mean(dim=2)
        return torch.where(out.isinf(), torch.zeros_like(out), out)
