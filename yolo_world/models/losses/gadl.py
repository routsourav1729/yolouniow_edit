"""GT-Anchored Discriminative Loss (GADL) for YOLO-UniOW T2 fine-tuning.

Computes a softmax cross-entropy loss ONLY at annotated GT box locations
(anchors assigned to novel classes by TAL). This provides:
  1. Strong classification gradient to novel class embeddings (BCE near-zero gradient problem).
  2. Selective T_unk unlearning at novel GT locations via softmax denominator.

Only novel class channels + unknown channel enter the softmax denominator.
Base class channels (frozen, grad-masked) are excluded to avoid wasted gradient.
"""

import torch
import torch.nn.functional as F


class GADLModule:
    """GT-Anchored Discriminative Loss module.

    Args:
        num_prev (int): Number of base/previous classes (novel starts here).
        num_known (int): Total known classes (base + novel); unknown is at unk_idx.
        unk_idx (int): Index of the unknown class embedding in the logit tensor.
        weight (float): Loss weight scalar.
    """

    def __init__(self, num_prev: int, num_known: int, unk_idx: int, weight: float = 1.0):
        self.num_prev = num_prev
        self.num_known = num_known
        self.unk_idx = unk_idx
        self.weight = weight

        # Pre-build selected index list (novel indices + unknown index)
        self._novel_indices = list(range(num_prev, num_known))
        self._selected_indices = self._novel_indices + [unk_idx]

    def compute(self, cls_logits: torch.Tensor, assigned_labels: torch.Tensor) -> torch.Tensor:
        """Compute GADL loss.

        Args:
            cls_logits (Tensor): (B, N, K) raw logits from BNContrastiveHead
                (pre-sigmoid). Must retain grad for backprop into novel embeddings.
            assigned_labels (Tensor): (B, N) long tensor. Assigned GT class index
                per anchor (absolute, 0-indexed). -1 for negative/background anchors.

        Returns:
            Scalar loss tensor.
        """
        # Find anchors assigned to novel classes: [num_prev, num_known)
        novel_mask = (assigned_labels >= self.num_prev) & \
                     (assigned_labels < self.num_known)  # (B, N)

        num_novel = int(novel_mask.sum().item())
        if num_novel == 0:
            print(f"[GADL] novel_anchors=0 (no novel GT boxes assigned by TAL) loss=0.0000")
            return cls_logits.new_zeros(1).squeeze()

        # Extract logits and labels at novel GT anchor locations
        novel_logits = cls_logits[novel_mask]   # (M, K)
        novel_labels = assigned_labels[novel_mask]  # (M,)

        # Select only novel class channels + unknown channel for softmax
        # This excludes frozen base class channels (indices 0..num_prev-1)
        sel_idx = cls_logits.new_tensor(self._selected_indices, dtype=torch.long)
        selected_logits = novel_logits[:, sel_idx]  # (M, num_novel + 1)

        # Remap absolute class indices to relative positions within selected_logits
        # novel_labels in [num_prev, num_known) → targets in [0, num_novel)
        targets = (novel_labels - self.num_prev).long()  # (M,)

        # Softmax cross-entropy
        loss = F.cross_entropy(selected_logits, targets)

        # Diagnostic stats
        mean_logit = float(selected_logits.detach().mean().item())
        print(f"[GADL] novel_anchors={num_novel} "
              f"mean_selected_logit={mean_logit:.4f} "
              f"loss={loss.item():.6f} "
              f"weight={self.weight}")

        return self.weight * loss
