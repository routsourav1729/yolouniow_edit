"""Wildcard-Aware Pseudo-label Redistribution (WAPR) for Few-Shot Open-World Detection.

During T2 few-shot fine-tuning, unannotated known objects pass the gatekeeper Φ
and corrupt T_unk with known-class gradients. WAPR uses the ratio of max known
score to anchor score as a discriminative signal:

  ratio = max_known_score / anchor_score   (both sigmoid-calibrated)
  w_r = 1 - ratio                          (soft suppression weight)

ALL gatekeeper-passing anchors get soft-suppressed by w_r:
  - High ratio (close to 1) → w_r ≈ 0 → strong suppression (likely known)
  - Low ratio (close to 0) → w_r ≈ 1 → minimal suppression (genuine unknown)

The ratio_threshold is used only for COUNTING/logging how many anchors
are above vs below threshold, not for changing the suppression behavior.
"""

import torch
import torch.nn.functional as F


class WAPRModule:
    """Soft ratio-based WAPR: all gatekeeper-passing anchors get w_r scaling.

    Args:
        frozen_embedding_path: Path to .npy (kept for config compat).
        num_prev_classes: Not used (kept for config compat).
        num_known_classes: Not used (kept for config compat).
        warmup_epochs: Epochs where gatekeeper is skipped entirely.
        anchor_loss_weight: Lambda for L2 anchoring loss on T_unk drift.
        ratio_threshold: Threshold for logging counts (not used in scoring).
    """

    def __init__(self,
                 frozen_embedding_path: str,
                 num_prev_classes: int = 0,
                 num_known_classes: int = 0,
                 warmup_epochs: int = 2,
                 anchor_loss_weight: float = 0.1,
                 ratio_threshold: float = 0.5):
        self.warmup_epochs = warmup_epochs
        self.anchor_loss_weight = anchor_loss_weight
        self.ratio_threshold = ratio_threshold
        self._device = None

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
        """Modify assigned_scores in-place: w_r * anchor_scores for ALL
        gatekeeper-passing anchors.

        Args:
            unknown_mask: (B, N) bool — gatekeeper Φ mask.
            anchor_scores: (B, N) float — sigmoid anchor scores.
            assigned_scores: (B, N, K) float — mutable target tensor.
            max_known_scores: (B, N) float — max sigmoid score over known classes.
            best_known_class: (B, N) long — (kept for interface compat).

        Returns:
            dict with WAPR stats (or empty dict if no candidates).
        """
        if not unknown_mask.any():
            return {}

        device = assigned_scores.device
        self._ensure_device(device)

        # ratio = max_known / anchor. High = barely passed gatekeeper.
        ratio = max_known_scores / anchor_scores.clamp(min=1e-6)  # (B, N)

        # w_r = 1 - ratio: soft suppression for ALL unknown_mask anchors
        w_r = (1.0 - ratio).clamp(0.0, 1.0)  # (B, N)

        # Save w_r for downstream consumers (e.g. KUME)
        self._last_w_r = w_r.detach()

        # Apply w_r * anchor_scores to ALL gatekeeper-passing anchors
        assigned_scores[:, :, -2] = torch.where(
            unknown_mask,
            w_r * anchor_scores,
            assigned_scores[:, :, -2])

        # Stats — threshold used only for counting
        num_candidates = int(unknown_mask.sum().item())
        suppress_mask = unknown_mask & (ratio >= self.ratio_threshold)
        genuine_mask = unknown_mask & (ratio < self.ratio_threshold)
        num_suppressed = int(suppress_mask.sum().item())
        num_genuine = int(genuine_mask.sum().item())
        ratio_masked = ratio[unknown_mask]
        w_r_masked = w_r[unknown_mask]
        return {
            'wapr/mean_w_r': float(w_r_masked.mean().item()) if num_candidates > 0 else 0.0,
            'wapr/mean_ratio': float(ratio_masked.mean().item()) if num_candidates > 0 else 0.0,
            'wapr/std_ratio': float(ratio_masked.std().item()) if num_candidates > 1 else 0.0,
            'wapr/num_suppressed': num_suppressed,
            'wapr/num_genuine_unk': num_genuine,
            'wapr/num_candidates': num_candidates,
        }

    def compute_anchor_loss(self, current_t_unk: torch.Tensor) -> torch.Tensor:
        """L2 anchoring loss: λ * ||T_unk_current - T_unk_T1||²."""
        if self.t_unk_anchor is None:
            return current_t_unk.new_zeros(1).squeeze()
        self._ensure_device(current_t_unk.device)
        return self.anchor_loss_weight * F.mse_loss(
            current_t_unk, self.t_unk_anchor)
