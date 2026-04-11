"""GT-Anchored Softmax Discriminative Loss (GASDL) for YOLO-UniOW T2.

Auxiliary loss at raw GT box center positions (not TAL-assigned anchors).
For each novel GT box, maps the center to the correct FPN level and grid
cell, extracts the one2one cls logit vector, and applies temperature-scaled
softmax CE over novel + unknown channels. Replaces the original GADL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_FPN_STRIDES = (8, 16, 32)
_AREA_THRESH = (64 ** 2, 128 ** 2)  # small/medium/large boundaries


class GASDLModule(nn.Module):
    """GT-Anchored Softmax Discriminative Loss with learnable temperature.

    Args:
        num_prev (int): Base class count (novel starts here).
        num_known (int): Total known = base + novel (excludes unk/anchor).
        unk_idx (int): Unknown class index in the logit tensor.
        weight (float): Loss weight scalar.
        temperature (float): Initial softmax temperature (learned in log-space).
        include_unknown (bool): Include unknown channel in softmax denominator.
    """

    def __init__(self, num_prev: int, num_known: int, unk_idx: int,
                 weight: float = 1.0, temperature: float = 5.0,
                 include_unknown: bool = True):
        super().__init__()
        self.num_prev = num_prev
        self.num_known = num_known
        self.unk_idx = unk_idx
        self.weight = weight
        self.include_unknown = include_unknown

        # Learnable temperature in log-space
        self.log_temperature = nn.Parameter(
            torch.tensor(float(temperature)).log())

        # Selected channel indices: novel classes + optionally unknown
        sel = list(range(num_prev, num_known))
        if include_unknown:
            sel.append(unk_idx)
        self.register_buffer('_sel_idx',
                             torch.tensor(sel, dtype=torch.long))

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def compute(self, cached_logits, batch_data_samples):
        """Compute GASDL from one2one logits (with grad) and GT boxes.

        Args:
            cached_logits: list of 3 tensors [(B, K, H, W)] per FPN level.
            batch_data_samples: list of DataSample with gt_instances.

        Returns:
            Scalar loss tensor.
        """
        device = cached_logits[0].device
        sel = self._sel_idx.to(device)
        temp = self.temperature
        B = cached_logits[0].shape[0]

        all_logits = []
        all_targets = []

        for b in range(B):
            gt = batch_data_samples[b].gt_instances
            bboxes = gt.bboxes   # (M, 4) xyxy
            labels = gt.labels   # (M,)

            # Novel-only filter
            mask = (labels >= self.num_prev) & (labels < self.num_known)
            if not mask.any():
                continue

            bboxes = bboxes[mask]
            labels = labels[mask]

            cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
            cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

            # Assign FPN levels by area
            lvls = torch.zeros(areas.shape[0], dtype=torch.long, device=device)
            lvls[areas > _AREA_THRESH[0]] = 1
            lvls[areas > _AREA_THRESH[1]] = 2

            for lvl in range(3):
                lm = lvls == lvl
                if not lm.any():
                    continue

                feat = cached_logits[lvl]          # (B, K, H, W)
                s = _FPN_STRIDES[lvl]
                H, W = feat.shape[2], feat.shape[3]

                gx = (cx[lm] / s).long().clamp(0, W - 1)
                gy = (cy[lm] / s).long().clamp(0, H - 1)

                # (K, M_lvl) -> (M_lvl, K)
                logits_at_gt = feat[b, :, gy, gx].T
                all_logits.append(logits_at_gt[:, sel])
                all_targets.append((labels[lm] - self.num_prev).long())

        if not all_logits:
            return cached_logits[0].new_zeros(1).squeeze()

        cat_logits = torch.cat(all_logits, dim=0)    # (N, num_sel)
        cat_targets = torch.cat(all_targets, dim=0)   # (N,)

        loss = F.cross_entropy(cat_logits / temp, cat_targets)

        print(f"[GASDL] n_gt={cat_targets.shape[0]} "
              f"temp={temp.item():.3f} "
              f"loss={loss.item():.6f}")

        return self.weight * loss
