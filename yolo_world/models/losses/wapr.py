"""Wildcard-Aware Pseudo-label Redistribution (WAPR) for Few-Shot Open-World Detection.

During T2 few-shot fine-tuning, unannotated known objects pass the gatekeeper Φ
and corrupt T_unk with known-class gradients. WAPR computes redistribution weights
using the model's own BNContrastiveHead logits for known classes:
  w_r = 1 - max_c sigmoid(logit_c)
to:
  - Scale down the unknown target for anchors similar to known classes
  - Mine novel class targets for anchors matching novel classes
  - Anchor T_unk with an L2 drift loss against its T1 value
"""

import torch
import torch.nn.functional as F
import numpy as np


class WAPRModule:
    """Computes redistribution weights and modifies assigned_scores in-place.

    Uses cached BNContrastiveHead logits (post-BN, scaled) to compute per-anchor
    similarity to known classes. This ensures the similarity is in the same
    calibrated space the model uses for classification.

    For each gatekeeper-passing anchor r:
      max_prob = max_c sigmoid(logit_c)     (how "known" this anchor looks)
      w_r = 1 - max_prob                    (low for known-like anchors)
      assigned_scores[b,r,-2] = w_r * anchor_scores[b,r]       (filtered unknown)
      assigned_scores[b,r,c]  += (1-w_r) * anchor_scores[b,r]  (novel mining, c >= num_prev)

    Args:
        frozen_embedding_path: Path to .npy (kept for config compat, not used for sim).
        num_prev_classes: Number of base classes (novel classes start at this index).
        num_known_classes: Total known classes (base + novel).
        warmup_epochs: Epochs where gatekeeper is skipped entirely (T_unk frozen).
        anchor_loss_weight: Lambda for L2 anchoring loss on T_unk drift.
    """

    def __init__(self,
                 frozen_embedding_path: str,
                 num_prev_classes: int,
                 num_known_classes: int,
                 warmup_epochs: int = 2,
                 anchor_loss_weight: float = 0.1):
        self.num_prev_classes = num_prev_classes
        self.num_known_classes = num_known_classes
        self.warmup_epochs = warmup_epochs
        self.anchor_loss_weight = anchor_loss_weight
        self._device = None

        # T_unk anchor — snapshot from T1 checkpoint, set lazily
        self.t_unk_anchor = None

    def _ensure_device(self, device):
        """Move frozen tensors to the correct device lazily."""
        if self._device != device:
            if self.t_unk_anchor is not None:
                self.t_unk_anchor = self.t_unk_anchor.to(device)
            self._device = device

    def set_t_unk_anchor(self, t_unk_embedding: torch.Tensor):
        """Snapshot T_unk from loaded T1 checkpoint. Called once."""
        self.t_unk_anchor = t_unk_embedding.detach().clone()

    def redistribute(self, unknown_mask, anchor_scores, assigned_scores,
                     cached_cls_logits, num_level_priors):
        """Modify assigned_scores in-place with WAPR redistribution.

        Args:
            unknown_mask: (B, N) bool — gatekeeper Φ mask.
            anchor_scores: (B, N) float — sigmoid objectness scores.
            assigned_scores: (B, N, K) float — mutable target tensor.
            cached_cls_logits: list of (B, K, H, W) logit tensors per FPN level.
                These are the BNContrastiveHead outputs (post-BN, scaled).
            num_level_priors: list of int — number of anchors per FPN level.

        Returns:
            dict with WAPR stats (or empty dict if no candidates).
        """
        if not unknown_mask.any():
            return {}

        device = assigned_scores.device
        self._ensure_device(device)

        # 1. Flatten cached logits: list of (B, K, H, W) → (B, N_total, K)
        flat_logits = []
        for lvl_logit in cached_cls_logits:
            b, k, h, w = lvl_logit.shape
            flat_logits.append(
                lvl_logit.permute(0, 2, 3, 1).reshape(b, h * w, k))
        flat_logits = torch.cat(flat_logits, dim=1)  # (B, N, K)

        # 2. Take known class channels only (0..num_known-1), apply sigmoid
        known_logits = flat_logits[:, :, :self.num_known_classes]  # (B, N, num_known)
        known_probs = known_logits.sigmoid()  # (B, N, num_known)

        # 3. Max known-class probability per anchor
        max_prob, best_class = known_probs.max(dim=-1)  # (B, N), (B, N)

        # 4. Redistribution weight: high known prob → low w_r (redirect away from T_unk)
        w_r = (1.0 - max_prob).clamp(0.0, 1.0)  # (B, N)

        # 5. Scale unknown target by w_r for gatekeeper-passing anchors
        assigned_scores[:, :, -2] = torch.where(
            unknown_mask,
            w_r * anchor_scores,
            assigned_scores[:, :, -2])

        # 6. Novel mining: add soft targets for novel class matches
        novel_mining_mask = unknown_mask & (best_class >= self.num_prev_classes)
        num_redirected = 0
        if novel_mining_mask.any():
            mining_scores = (1.0 - w_r) * anchor_scores  # (B, N)
            # Zero out non-mining positions
            mining_scores = torch.where(
                novel_mining_mask, mining_scores, torch.zeros_like(mining_scores))
            # Scatter add into the matched novel class channel
            idx = best_class.unsqueeze(-1)  # (B, N, 1)
            assigned_scores.scatter_add_(2, idx, mining_scores.unsqueeze(-1).to(assigned_scores.dtype))
            num_redirected = int(novel_mining_mask.sum().item())

        # 7. Collect stats for logging
        num_candidates = int(unknown_mask.sum().item())
        w_r_masked = w_r[unknown_mask]
        max_prob_masked = max_prob[unknown_mask]
        return {
            'wapr/mean_w_r': float(w_r_masked.mean().item()) if num_candidates > 0 else 0.0,
            'wapr/mean_max_prob': float(max_prob_masked.mean().item()) if num_candidates > 0 else 0.0,
            'wapr/num_redirected': num_redirected,
            'wapr/num_genuine_unk': num_candidates - num_redirected,
            'wapr/num_candidates': num_candidates,
        }

    def compute_anchor_loss(self, current_t_unk: torch.Tensor) -> torch.Tensor:
        """L2 anchoring loss: λ * ||T_unk_current - T_unk_T1||².

        Args:
            current_t_unk: Current T_unk embedding (D,), with gradient.
        Returns:
            Scalar loss tensor.
        """
        if self.t_unk_anchor is None:
            return current_t_unk.new_zeros(1).squeeze()
        self._ensure_device(current_t_unk.device)
        return self.anchor_loss_weight * F.mse_loss(
            current_t_unk, self.t_unk_anchor)
