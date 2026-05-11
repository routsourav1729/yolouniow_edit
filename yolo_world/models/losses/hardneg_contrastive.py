"""Hard-negative contrastive loss using T1-cached base-positive features.

Cache entries (built by HardNegativeCacheHook during T1) hold BN-normalized
o2o `cls_embed`s at base-class GT-positive anchors that scored high on T_unk.
At T2 fine-tuning we feed these features through the same contrastive math
the head uses and push DOWN the score for T_unk *and* every novel-class
prompt. Base-class prompts are intentionally untouched so we don't disturb
their already-learned positive signal.

Score per cached feature f at level l and target class c:
    logit = scale[l] * (f · L2(t_c)) + bias[l]
where scale[l] = cls_contrasts[l].logit_scale.exp() and bias[l] are the
per-scale contrastive head parameters and t_c = model.embeddings[c].

Loss: BCE-with-logits(target=0) summed across target classes and averaged
over cached features. Gradients flow only into:
    - model.embeddings[novel and T_unk]  (trainable at T2)
    - cls_contrasts[l].{logit_scale, bias}  (only if o2o head is unfrozen)
Base-class embeddings, the backbone, and the cached features themselves
remain untouched (cache tensors are pre-detached + CPU and copied to device
each step).
"""

from typing import List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class HardNegativeContrastiveLoss(nn.Module):
    """Hard-negative contrastive loss for T2 fine-tuning.

    Args:
        cache_path: torch.save'd dict produced by HardNegativeCacheHook with
            schema {'memory': {level: {base_cls: [entry, ...]}}, 'meta': ...}.
            Each entry has 'feature' (BN(cls_embed), 1D tensor), 'level',
            'base_cls', 'score'.
        num_base_classes: number of base / previously-introduced classes.
            Indices [0, num_base_classes) are base prompts (frozen, untouched).
        num_known_classes: total known classes (base + novel). Indices
            [num_base_classes, num_known_classes) are novel prompts and ARE
            targeted as hard negatives.
        unk_idx: index of the T_unk prompt in model.embeddings. Defaults to
            num_known_classes (the conventional layout: [base, novel, unk, anchor]).
        target_unk: include T_unk as a hard-negative target. Default True.
        target_novel: include novel prompts as hard-negative targets.
            Default True.
        weight: scalar multiplier on the final loss before being returned to
            the loss dict.
    """

    def __init__(self,
                 cache_path: str,
                 num_base_classes: int,
                 num_known_classes: int,
                 unk_idx: Optional[int] = None,
                 target_unk: bool = True,
                 target_novel: bool = True,
                 weight: float = 1.0) -> None:
        super().__init__()
        if not (target_unk or target_novel):
            raise ValueError(
                'HardNegativeContrastiveLoss requires at least one of '
                'target_unk / target_novel to be True.')

        self.cache_path = cache_path
        self.num_base_classes = int(num_base_classes)
        self.num_known_classes = int(num_known_classes)
        self.unk_idx = int(unk_idx if unk_idx is not None else num_known_classes)
        self.target_unk = bool(target_unk)
        self.target_novel = bool(target_novel)
        self.weight = float(weight)

        # Per-level stacked features [N_l, C]. CPU tensors registered as
        # buffers so they move with .to(device) but stay detached from grad.
        self._features_per_level: List[Tensor] = []
        # Per-level entry count, used only for logging.
        self._counts_per_level: List[int] = []
        self._load_cache()

    # ------------------------------------------------------------------ cache

    def _load_cache(self) -> None:
        blob = torch.load(self.cache_path, map_location='cpu')
        memory = blob['memory']
        levels = sorted(memory.keys())
        if not levels:
            raise RuntimeError(
                f'HardNegativeContrastiveLoss: empty cache at {self.cache_path}')

        for lvl_idx, level in enumerate(levels):
            bucket = memory[level]
            feats = []
            for cls_id, entries in bucket.items():
                if cls_id >= self.num_base_classes:
                    # Defensive: cache should only contain base-positive
                    # entries, but skip anything outside the base range.
                    continue
                for e in entries:
                    f = e['feature']
                    if not isinstance(f, Tensor):
                        f = torch.as_tensor(f)
                    feats.append(f.float().view(-1))
            stacked = (torch.stack(feats, dim=0) if feats
                       else torch.zeros(0, 1, dtype=torch.float32))
            self.register_buffer(f'_feat_lvl{lvl_idx}', stacked,
                                 persistent=False)
            self._features_per_level.append(stacked)
            self._counts_per_level.append(stacked.shape[0])

        if sum(self._counts_per_level) == 0:
            raise RuntimeError(
                f'HardNegativeContrastiveLoss: cache at {self.cache_path} '
                'has no usable base-positive entries.')

    # ------------------------------------------------------------------ math

    def _target_indices(self) -> List[int]:
        idxs: List[int] = []
        if self.target_novel and self.num_known_classes > self.num_base_classes:
            idxs.extend(range(self.num_base_classes, self.num_known_classes))
        if self.target_unk:
            idxs.append(self.unk_idx)
        return idxs

    def compute(self, embeddings: Tensor,
                cls_contrasts: nn.ModuleList) -> Tensor:
        """Return the scalar hard-negative contrastive loss.

        Args:
            embeddings: (num_prompts, C) text/prompt embeddings (trainable).
            cls_contrasts: per-level ModuleList of BNContrastiveHead modules
                (the o2o branch). Each has `logit_scale` and `bias`
                nn.Parameters. The BN inside is NOT applied here because the
                cached features were already BN-normalized at collection time.
        """
        target_idxs = self._target_indices()
        if not target_idxs:
            return embeddings.new_zeros(())

        target_embeds = embeddings[target_idxs]                # (T, C)
        target_norm = F.normalize(target_embeds.float(), dim=-1, p=2)

        device = embeddings.device
        total_loss = embeddings.new_zeros(())
        total_count = 0
        for lvl_idx, feats in enumerate(self._features_per_level):
            if feats.numel() == 0:
                continue
            feats = getattr(self, f'_feat_lvl{lvl_idx}')
            if feats.device != device:
                feats = feats.to(device)
            cc = cls_contrasts[lvl_idx]
            scale = cc.logit_scale.exp()
            bias = cc.bias
            # (N, T) = (N, C) @ (C, T)
            logits = feats.float() @ target_norm.t()
            logits = logits * scale + bias
            target = torch.zeros_like(logits)
            # sum so that "more cached features" → "stronger signal"; we
            # divide by total count below to give a per-feature average.
            total_loss = total_loss + F.binary_cross_entropy_with_logits(
                logits, target, reduction='sum')
            total_count += logits.numel()
        if total_count == 0:
            return embeddings.new_zeros(())
        return self.weight * total_loss / total_count

    # ----------------------------------------------------------------- utils

    def summary(self) -> str:
        return (f'HardNegativeContrastiveLoss: cache={self.cache_path}, '
                f'levels={len(self._counts_per_level)}, '
                f'features_per_level={self._counts_per_level}, '
                f'target_unk={self.target_unk}, '
                f'target_novel={self.target_novel}, '
                f'num_base={self.num_base_classes}, '
                f'num_known={self.num_known_classes}, '
                f'unk_idx={self.unk_idx}, '
                f'weight={self.weight}')
