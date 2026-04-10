"""Wildcard-Aware Pseudo-label Redistribution (WAPR) for Few-Shot Open-World Detection.

During T2 few-shot fine-tuning, unannotated known objects pass the gatekeeper Φ
and corrupt T_unk with known-class gradients. WAPR uses the model's own score
margin at the gatekeeper boundary to discriminate:

  ratio = max_known_score / anchor_score   (both already sigmoid-calibrated)
  w_r = (1 - ratio).clamp(0, 1)

- Genuine unknown: known scores << anchor → ratio ≈ 0 → w_r ≈ 1 (kept)
- Unannotated known: known score ≈ anchor (barely passed) → ratio ≈ 1 → w_r ≈ 0 (suppressed)

This is theoretically grounded: the gatekeeper already computes the decision
boundary (anchor > max_known), so the *margin* of that decision is the cleanest
discriminative signal. No embedding-space heuristics needed.
"""

import torch
import numpy as np


class WAPRModule:
    """Computes redistribution weights using the model's own score margin
    and modifies assigned_scores in-place.

    Args:
        frozen_embedding_path: Path to .npy (kept for config compat).
        num_prev_classes: Number of base classes (novel classes start at this index).
        num_known_classes: Total known classes (base + novel).
        warmup_epochs: Epochs before novel class filtering kicks in.
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
        self._current_epoch = 0

        # T_unk anchor — snapshot from T1 checkpoint, set lazily
        self.t_unk_anchor = None

    def _ensure_device(self, device):
        if self._device != device:
            if self.t_unk_anchor is not None:
                self.t_unk_anchor = self.t_unk_anchor.to(device)
            self._device = device

    def set_t_unk_anchor(self, t_unk_embedding: torch.Tensor):
        """Snapshot T_unk from loaded T1 checkpoint. Called once."""
        self.t_unk_anchor = t_unk_embedding.detach().clone()

    def redistribute(self, unknown_mask, anchor_scores, assigned_scores,
                     max_known_scores, best_known_class):
        """Modify assigned_scores in-place with WAPR redistribution.

        Uses the score ratio (max_known / anchor) as the discriminative signal.
        Both scores are already sigmoid-calibrated from the model's own logits.

        Args:
            unknown_mask: (B, N) bool — gatekeeper Φ mask.
            anchor_scores: (B, N) float — sigmoid anchor (objectness) scores.
            assigned_scores: (B, N, K) float — mutable target tensor.
            max_known_scores: (B, N) float — max sigmoid score over known classes.
            best_known_class: (B, N) long — argmax class index for novel mining.

        Returns:
            dict with WAPR stats (or empty dict if no candidates).
        """
        if not unknown_mask.any():
            return {}

        device = assigned_scores.device
        self._ensure_device(device)

        # 1. Score ratio: how close is max_known to anchor?
        #    For gatekeeper-passing anchors: max_known < anchor (guaranteed).
        #    ratio → 1 means "barely passed" (likely unannotated known).
        #    ratio → 0 means "large margin" (likely genuine unknown).
        ratio = (max_known_scores / anchor_scores.clamp(min=1e-6))  # (B, N)

        # 2. w_r = 1 - ratio: high margin (genuine unknown) → w_r ≈ 1
        w_r = (1.0 - ratio).clamp(0.0, 1.0)  # (B, N)

        # 3. Scale unknown target by w_r for gatekeeper-passing anchors
        assigned_scores[:, :, -2] = torch.where(
            unknown_mask,
            w_r * anchor_scores,
            assigned_scores[:, :, -2])

        # 4. Novel mining: redirect suppressed score to matched novel class
        novel_mining_mask = unknown_mask & (best_known_class >= self.num_prev_classes)
        num_redirected = 0
        if novel_mining_mask.any():
            mining_scores = (1.0 - w_r) * anchor_scores  # (B, N)
            mining_scores = torch.where(
                novel_mining_mask, mining_scores, torch.zeros_like(mining_scores))
            idx = best_known_class.unsqueeze(-1)  # (B, N, 1)
            assigned_scores.scatter_add_(2, idx, mining_scores.unsqueeze(-1).to(assigned_scores.dtype))
            num_redirected = int(novel_mining_mask.sum().item())

        # 5. Collect stats for logging
        num_candidates = int(unknown_mask.sum().item())
        w_r_masked = w_r[unknown_mask]
        ratio_masked = ratio[unknown_mask]
        return {
            'wapr/mean_w_r': float(w_r_masked.mean().item()) if num_candidates > 0 else 0.0,
            'wapr/mean_ratio': float(ratio_masked.mean().item()) if num_candidates > 0 else 0.0,
            'wapr/std_ratio': float(ratio_masked.std().item()) if num_candidates > 1 else 0.0,
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
        import torch.nn.functional as F
        self._ensure_device(current_t_unk.device)
        return self.anchor_loss_weight * F.mse_loss(
            current_t_unk, self.t_unk_anchor)
