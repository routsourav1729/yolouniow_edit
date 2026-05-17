from typing import Any, Iterable, Optional

import torch
from mmengine.hooks import Hook

from mmyolo.registry import HOOKS


@HOOKS.register_module()
class OWODStageControlHook(Hook):
    """Small staged controls for OWOD T2 experiments.

    The detector already uses a row-wise prompt gradient mask. This hook updates
    that mask after a chosen epoch and also exposes the current 1-based epoch to
    heads that need scheduled training behavior.
    """

    def __init__(self,
                 switch_epoch: int,
                 freeze_embedding_rows: Optional[Iterable[int]] = None,
                 train_embedding_rows: Optional[Iterable[int]] = None,
                 disable_hardneg: bool = False,
                 fed_bce_updates: Optional[dict[str, Any]] = None,
                 set_head_epoch: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.switch_epoch = int(switch_epoch)
        self.freeze_embedding_rows = list(freeze_embedding_rows or [])
        self.train_embedding_rows = list(train_embedding_rows or [])
        self.disable_hardneg = bool(disable_hardneg)
        self.fed_bce_updates = dict(fed_bce_updates or {})
        self.set_head_epoch = set_head_epoch
        self._mask_updated = False
        self._stage_updated = False

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, 'module') else model

    def _set_head_epoch(self, model, epoch: int) -> int:
        count = 0
        for module in model.modules():
            if hasattr(module, 'anchor_label'):
                setattr(module, 'current_epoch', epoch)
                count += 1
        return count

    def _update_embedding_mask(self, model) -> tuple[list[int], list[int]]:
        if not hasattr(model, 'embeddings'):
            return [], []

        num_rows = int(model.embeddings.shape[0])
        mask = getattr(model, '_grad_mask', None)
        if mask is None or int(mask.shape[0]) != num_rows:
            mask = torch.ones(num_rows, 1, dtype=torch.bool)
            model.embeddings.register_hook(
                lambda grad: grad * model._grad_mask.to(grad.device))
        else:
            mask = mask.detach().cpu().bool().clone()

        frozen_rows = [
            row for row in self.freeze_embedding_rows
            if 0 <= int(row) < num_rows
        ]
        trained_rows = [
            row for row in self.train_embedding_rows
            if 0 <= int(row) < num_rows
        ]
        if frozen_rows:
            mask[frozen_rows] = False
        if trained_rows:
            mask[trained_rows] = True
        model._grad_mask = mask
        return frozen_rows, trained_rows

    def _disable_hardneg(self, model) -> bool:
        if not self.disable_hardneg or getattr(model, 'hardneg', None) is None:
            return False
        model.hardneg = None
        return True

    def _update_fed_bce(self, model) -> int:
        if not self.fed_bce_updates:
            return 0
        count = 0
        for module in model.modules():
            fed_bce = getattr(module, 'fed_bce', None)
            if fed_bce is None:
                continue
            fed_bce.update(self.fed_bce_updates)
            count += 1
        return count

    def before_train_epoch(self, runner) -> None:
        epoch = runner.epoch + 1
        model = self._unwrap_model(runner.model)

        if self.set_head_epoch:
            self._set_head_epoch(model, epoch)

        if epoch <= self.switch_epoch or self._stage_updated:
            return

        frozen_rows, trained_rows = self._update_embedding_mask(model)
        hardneg_disabled = self._disable_hardneg(model)
        fed_bce_update_count = self._update_fed_bce(model)
        runner.logger.info(
            '[OWODStageControlHook] epoch=%d switched prompt mask: '
            'frozen_rows=%s train_rows=%s hardneg_disabled=%s '
            'fed_bce_updates=%s fed_bce_modules=%d',
            epoch, frozen_rows, trained_rows, hardneg_disabled,
            self.fed_bce_updates, fed_bce_update_count)
        self._mask_updated = True
        self._stage_updated = True
