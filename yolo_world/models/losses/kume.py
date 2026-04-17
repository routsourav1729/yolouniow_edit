"""Known-Unknown Margin Enforcement (KUME) for YOLO-UniOW.

Adds explicit inter-channel separation between the correct-class logit
and the competing unknown/known logit at each anchor location.

BCE trains each channel independently — it never enforces that at a given
anchor, the correct channel beats the incorrect channel by a margin.
KUME fills this structural gap with a hinge-based margin loss operating
in the calibrated logit space (post-BN, post-scale, post-bias).

For GT-matched known-class anchors:
    L_known(a) = ReLU(l_unk(a) - l_k(a) + m)
    Push T_unk's logit at least m below the correct class logit.

For gatekeeper-assigned unknown anchors:
    L_unk(a) = w_r(a) * ReLU(max_k l_k(a) - l_unk(a) + m)
    Push T_unk's logit at least m above the best known-class logit.
    Weighted by WAPR's w_r so contaminated pseudo-labels contribute less.
"""

import torch
import torch.nn.functional as F


class KUMEModule:
    """Known-Unknown Margin Enforcement.

    Args:
        num_known_classes (int): Total known classes (base + novel).
        unk_idx (int): Absolute index of T_unk in the logit tensor.
            For a K+2 tensor (known + T_unk + T_anchor), unk_idx = K.
        margin (float): Required logit separation. Default 1.0.
        weight (float): Loss weight scalar. Default 0.5.
    """

    def __init__(self,
                 num_known_classes: int,
                 unk_idx: int,
                 margin: float = 1.0,
                 weight: float = 0.5):
        self.num_known_classes = num_known_classes
        self.unk_idx = unk_idx
        self.margin = margin
        self.weight = weight

    def compute(self,
                cls_logits: torch.Tensor,
                assigned_labels: torch.Tensor,
                fg_mask: torch.Tensor,
                unknown_mask: torch.Tensor = None,
                wapr_w_r: torch.Tensor = None) -> torch.Tensor:
        """Compute KUME margin loss.

        Args:
            cls_logits: (B, N, K) raw logits from BNContrastiveHead (pre-sigmoid,
                WITH gradient). K includes known classes + T_unk + T_anchor.
            assigned_labels: (B, N) long tensor. Argmax of assigned_scores from
                TAL. Values 0..num_known-1 for GT-matched, arbitrary for bg.
            fg_mask: (B, N) bool — fg_mask_pre_prior from TAL assignment.
            unknown_mask: (B, N) bool — gatekeeper mask. None during warmup.
            wapr_w_r: (B, N) float — WAPR redistribution weights. If provided,
                weight the unknown-side margin by w_r.

        Returns:
            Scalar loss tensor (weighted).
        """
        # --- Known-side margin ---
        # Only at GT-matched anchors with known class labels
        known_mask = fg_mask & (assigned_labels < self.num_known_classes)  # (B, N)

        if known_mask.any():
            known_labels = assigned_labels[known_mask]  # (M,)
            known_logits = cls_logits[known_mask]  # (M, K)
            l_correct = known_logits.gather(
                1, known_labels.unsqueeze(1)).squeeze(1)  # (M,)
            l_unk = known_logits[:, self.unk_idx]  # (M,)
            margin_known = F.relu(l_unk - l_correct + self.margin)  # (M,)
            num_known_violations = int((margin_known > 0).sum().item())
            num_known_anchors = int(known_mask.sum().item())
            loss_known = margin_known.mean()
        else:
            loss_known = cls_logits.new_zeros(1).squeeze()
            num_known_violations = 0
            num_known_anchors = 0

        # --- Unknown-side margin ---
        if unknown_mask is not None and unknown_mask.any():
            unk_logits = cls_logits[unknown_mask]  # (P, K)
            l_unk_at_unk = unk_logits[:, self.unk_idx]  # (P,)
            l_max_known = unk_logits[
                :, :self.num_known_classes].max(dim=1).values  # (P,)
            margin_unk = F.relu(
                l_max_known - l_unk_at_unk + self.margin)  # (P,)

            if wapr_w_r is not None:
                w_r_at_unk = wapr_w_r[unknown_mask]  # (P,)
                margin_unk = margin_unk * w_r_at_unk

            num_unk_violations = int((margin_unk > 0).sum().item())
            num_unk_anchors = int(unknown_mask.sum().item())
            loss_unk = margin_unk.mean()
        else:
            loss_unk = cls_logits.new_zeros(1).squeeze()
            num_unk_violations = 0
            num_unk_anchors = 0

        total_loss = self.weight * (loss_known + loss_unk) / 2.0

        # Diagnostic
        print(f"[KUME] known_anchors={num_known_anchors} "
              f"unk_anchors={num_unk_anchors} "
              f"loss_known={loss_known.item():.4f} "
              f"loss_unk={loss_unk.item():.4f} "
              f"margin_violations_known={num_known_violations}/{num_known_anchors} "
              f"margin_violations_unk={num_unk_violations}/{num_unk_anchors}")

        return total_loss
