"""Known-Unknown Margin Enforcement (KUME) — k-vs-k single-term variant.

This is the k-vs-k ablation: a single all-pairs margin hinge between the
correct-known logit and every other known logit at GT-matched anchors.

For a known-class anchor a with correct class c:
    L_kvk(a) = mean_{j != c, j in [0, K)} ReLU(l_j(a) - l_c(a) + m)

T_unk and T_anchor are excluded from the comparison set. The unknown-side
hinge and the original known-vs-T_unk hinge are removed in this variant.
WAPR weights and the unknown_mask are accepted in the signature but ignored,
so call-site compatibility is preserved.

Rationale: probe diagnostics show BCE provides no inter-known coupling, so
correct vs other-known logits are not separated. This term targets that
gap directly.
"""

import torch
import torch.nn.functional as F


class KUMEModule:
    """KUME k-vs-k margin loss.

    Args:
        num_known_classes (int): Total known classes (base + novel).
        unk_idx (int): Absolute index of T_unk in the logit tensor.
            Kept for signature compatibility; not used in this variant.
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
        """Compute k-vs-k margin loss.

        unknown_mask and wapr_w_r are accepted for call-site compatibility
        but ignored in the kvk variant.
        """
        K = self.num_known_classes
        known_mask = fg_mask & (assigned_labels < K)  # (B, N)

        if not known_mask.any() or K < 2:
            zero = cls_logits.new_zeros(()).requires_grad_(False)
            print(f"[KUME-KVK] num_known_anchors=0 loss_kvk=0.0000 "
                  f"num_violations=0 mean_violation_per_anchor=0.0000")
            return self.weight * zero

        labels = assigned_labels[known_mask]                       # (M,)
        logits = cls_logits[known_mask]                            # (M, K_total)
        known_logits = logits[:, :K]                               # (M, K)
        l_correct = known_logits.gather(
            1, labels.unsqueeze(1)).squeeze(1)                     # (M,)

        diff = known_logits - l_correct.unsqueeze(1) + self.margin  # (M, K)
        idx = torch.arange(K, device=diff.device).unsqueeze(0)      # (1, K)
        non_correct_mask = (idx != labels.unsqueeze(1))             # (M, K)
        hinge = F.relu(diff) * non_correct_mask.float()             # (M, K)

        loss_per_anchor = hinge.sum(dim=1) / (K - 1)                # (M,)
        loss_kvk = loss_per_anchor.mean()

        num_known_anchors = int(known_mask.sum().item())
        num_violations = int((hinge > 0).sum().item())
        mean_violation_per_anchor = (num_violations /
                                     max(num_known_anchors, 1))

        print(f"[KUME-KVK] num_known_anchors={num_known_anchors} "
              f"loss_kvk={loss_kvk.item():.4f} "
              f"num_violations={num_violations} "
              f"mean_violation_per_anchor={mean_violation_per_anchor:.4f}")

        return self.weight * loss_kvk
