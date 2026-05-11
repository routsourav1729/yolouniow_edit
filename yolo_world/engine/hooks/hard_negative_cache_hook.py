"""Cache T1 O2O base positives that score highly as unknown."""

import os
import types
from typing import Dict, List, Optional

import torch
from mmdet.structures.bbox import bbox_overlaps
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmyolo.registry import HOOKS
from torch import Tensor, nn


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if is_model_wrapper(model) else model


@HOOKS.register_module()
class HardNegativeCacheHook(Hook):
    """Build a compact cache from T1/base GT-positive O2O assignments only."""

    def __init__(self,
                 cache_path: str,
                 topk_per_class_scale: int = 10,
                 num_base_classes: Optional[int] = None,
                 duplicate_iou_thr: float = 0.9,
                 min_unknown_score: float = 0.0,
                 no_step_collect: bool = True,
                 stop_when_full: bool = True,
                 freeze_bn: bool = True,
                 priority: int = 48):
        super().__init__()
        self.cache_path = cache_path
        self.topk_per_class_scale = int(topk_per_class_scale)
        self.num_base_classes = num_base_classes
        self.duplicate_iou_thr = float(duplicate_iou_thr)
        self.min_unknown_score = float(min_unknown_score)
        self.no_step_collect = bool(no_step_collect)
        self.stop_when_full = bool(stop_when_full)
        self.freeze_bn = bool(freeze_bn)
        self.priority = priority

        self.memory: Dict[int, Dict[int, List[dict]]] = {}
        self.original_train_step = None
        self.collect_iters = 0

    def before_train(self, runner: Runner) -> None:
        model = _unwrap(runner.model)
        if self.freeze_bn:
            self._freeze_bn_stats(model)
        self._init_memory(model)
        if self.no_step_collect:
            self._patch_collect_train_step(runner)
        runner.logger.info(
            '[HardNegCache] collect T1 O2O base-positive anchors: '
            f'cache={self.cache_path}, topk={self.topk_per_class_scale}, '
            f'dup_iou={self.duplicate_iou_thr}, '
            f'target={self._target_size(model)}')

    def before_train_epoch(self, runner: Runner) -> None:
        if self.freeze_bn:
            self._freeze_bn_stats(_unwrap(runner.model))

    def before_train_iter(self, runner: Runner, batch_idx: int,
                          data_batch=None) -> None:
        if self.freeze_bn:
            self._freeze_bn_stats(_unwrap(runner.model))

    def after_train_iter(self, runner: Runner, batch_idx: int,
                         data_batch=None, outputs=None) -> None:
        if data_batch is None:
            return
        self._collect_from_batch(runner, data_batch)
        self.collect_iters += 1
        if self.collect_iters % 50 == 0:
            self._save_cache(runner)
        if self.stop_when_full and self._is_full(_unwrap(runner.model)):
            runner.logger.info('[HardNegCache] target cache size reached')
            runner.train_loop.stop_training = True

    def after_train(self, runner: Runner) -> None:
        self._save_cache(runner)
        self._log_summary(runner)

    def _patch_collect_train_step(self, runner: Runner) -> None:
        def _train_step(patched_self, data, optim_wrapper):
            model = _unwrap(patched_self)
            with torch.no_grad():
                data = model.data_preprocessor(data, training=True)
                losses = patched_self._run_forward(data, mode='loss')
                _, log_vars = model.parse_losses(losses)
            log_vars['hardneg_collect_no_step'] = (
                next(model.parameters()).detach().new_tensor(0.0))
            return log_vars

        self.original_train_step = runner.model.train_step
        runner.model.train_step = types.MethodType(_train_step, runner.model)

    def _init_memory(self, model: nn.Module) -> None:
        num_levels = int(model.bbox_head.head_module.num_levels)
        num_base = self._num_base_classes(model)
        self.memory = {
            lvl: {cls_id: [] for cls_id in range(num_base)}
            for lvl in range(num_levels)
        }

    def _collect_from_batch(self, runner: Runner, data_batch) -> None:
        model = _unwrap(runner.model)
        head = model.bbox_head
        head_module = head.head_module
        feats = getattr(head_module, '_cached_cls_embeds_one2one', None)
        logits = getattr(head_module, '_cached_cls_logits_one2one', None)
        bbox_preds = getattr(head_module, '_cached_bbox_preds_one2one', None)
        if not feats or not logits or not bbox_preds:
            return

        with torch.no_grad():
            data = model.data_preprocessor(data_batch, training=True)
            labels = data['data_samples'].get('bboxes_labels', None)
            if labels is None or labels.numel() == 0:
                return

            flat_feats, flat_logits, flat_boxes, flat_priors, level_ids = (
                self._flatten_o2o_predictions(head, feats, logits,
                                              bbox_preds))
            gt_labels, gt_bboxes, pad_flag = self._gt_tensors(
                labels, flat_logits.shape[0], flat_logits.device,
                flat_logits.dtype)
            assigned = head.one2one_assigner(
                flat_boxes.detach().type(gt_bboxes.dtype),
                flat_logits.detach().sigmoid(),
                flat_priors,
                gt_labels,
                gt_bboxes,
                pad_flag)

            fg_mask = assigned['fg_mask_pre_prior']
            assigned_gt_idxs = self._assigned_gt_indices(assigned, fg_mask)
            assigned_cls = assigned['assigned_labels'].long()
            unk_idx = int(model.num_training_classes - 2)
            num_base = self._num_base_classes(model)
            unk_scores = flat_logits.sigmoid()[..., unk_idx]

            for img_idx in range(flat_logits.shape[0]):
                keep = fg_mask[img_idx] & (assigned_cls[img_idx] < num_base)
                keep = keep & (unk_scores[img_idx] >= self.min_unknown_score)
                if not bool(keep.any()):
                    continue

                best_by_gt = {}
                for prior_idx in torch.nonzero(keep, as_tuple=False).squeeze(1):
                    gt_idx = int(assigned_gt_idxs[img_idx, prior_idx].item())
                    if gt_idx < 0 or not bool(pad_flag[img_idx, gt_idx, 0]):
                        continue
                    score = float(unk_scores[img_idx, prior_idx].cpu())
                    old = best_by_gt.get(gt_idx)
                    if old is None or score > old['score']:
                        best_by_gt[gt_idx] = {
                            'prior_idx': int(prior_idx.item()),
                            'score': score,
                        }

                for gt_idx, chosen in best_by_gt.items():
                    prior_idx = chosen['prior_idx']
                    cls_id = int(gt_labels[img_idx, gt_idx, 0].item())
                    level = int(level_ids[prior_idx].item())
                    entry = {
                        'score': chosen['score'],
                        'feature': flat_feats[img_idx, prior_idx].detach().cpu().float(),
                        'box': flat_boxes[img_idx, prior_idx].detach().cpu().float(),
                        'gt_box': gt_bboxes[img_idx, gt_idx].detach().cpu().float(),
                        'base_cls': cls_id,
                        'level': level,
                    }
                    self._try_insert(level, cls_id, entry)

    def _flatten_o2o_predictions(self, head: nn.Module,
                                 feats: List[Tensor],
                                 logits: List[Tensor],
                                 bbox_preds: List[Tensor]):
        flat_feats = []
        flat_logits = []
        flat_boxes = []
        priors = []
        level_ids = []
        for level, (logit, bbox_pred) in enumerate(zip(logits, bbox_preds)):
            b, _, h, w = logit.shape
            n = h * w
            prior = head.prior_generator.single_level_grid_priors(
                (h, w),
                level_idx=level,
                dtype=logit.dtype,
                device=logit.device,
                with_stride=True)
            pred = bbox_pred.permute(0, 2, 3, 1).reshape(b, n, 4)
            box = head.bbox_coder.decode(prior[:, :2], pred, prior[:, 2])
            flat_feats.append(feats[level].permute(0, 2, 3, 1).reshape(
                b, n, feats[level].shape[1]))
            flat_logits.append(logit.permute(0, 2, 3, 1).reshape(
                b, n, logit.shape[1]))
            flat_boxes.append(box)
            priors.append(prior)
            level_ids.append(torch.full((n,), level, device=logit.device,
                                        dtype=torch.long))
        return (torch.cat(flat_feats, dim=1),
                torch.cat(flat_logits, dim=1),
                torch.cat(flat_boxes, dim=1),
                torch.cat(priors, dim=0),
                torch.cat(level_ids, dim=0))

    @staticmethod
    def _assigned_gt_indices(assigned: dict, fg_mask: Tensor) -> Tensor:
        if 'assigned_gt_idxs' in assigned:
            return assigned['assigned_gt_idxs'].long()
        if assigned.get('pos_mask', None) is not None:
            return assigned['pos_mask'].long().argmax(dim=1)
        return torch.full_like(fg_mask, -1, dtype=torch.long)

    @staticmethod
    def _gt_tensors(labels: Tensor, batch_size: int, device, dtype):
        counts = [
            int((labels[:, 0].long() == i).sum().item())
            for i in range(batch_size)
        ]
        max_count = max(max(counts), 1)
        gt_labels = torch.zeros(batch_size, max_count, 1,
                                device=device, dtype=dtype)
        gt_bboxes = torch.zeros(batch_size, max_count, 4,
                                device=device, dtype=dtype)
        pad_flag = torch.zeros(batch_size, max_count, 1,
                               device=device, dtype=dtype)
        for i in range(batch_size):
            rows = labels[labels[:, 0].long() == i]
            if rows.numel() == 0:
                continue
            n = rows.shape[0]
            gt_labels[i, :n, 0] = rows[:, 1].to(device=device, dtype=dtype)
            gt_bboxes[i, :n] = rows[:, 2:6].to(device=device, dtype=dtype)
            pad_flag[i, :n, 0] = 1.0
        return gt_labels, gt_bboxes, pad_flag

    def _try_insert(self, level: int, cls_id: int, entry: dict) -> None:
        bucket = self.memory[level][cls_id]
        dup = self._find_duplicate(bucket, entry['box'])
        if dup is not None:
            if entry['score'] > bucket[dup]['score']:
                bucket[dup] = entry
            return
        if len(bucket) < self.topk_per_class_scale:
            bucket.append(entry)
            return
        worst_idx, worst = min(enumerate(bucket), key=lambda x: x[1]['score'])
        if entry['score'] > worst['score']:
            bucket[worst_idx] = entry

    def _find_duplicate(self, bucket: List[dict], box: Tensor):
        if not bucket:
            return None
        boxes = torch.stack([item['box'] for item in bucket], dim=0)
        ious = bbox_overlaps(box[None].float(), boxes.float(), mode='iou')[0]
        max_iou, idx = ious.max(dim=0)
        return int(idx) if float(max_iou) >= self.duplicate_iou_thr else None

    def _save_cache(self, runner: Runner) -> None:
        os.makedirs(os.path.dirname(self.cache_path) or '.', exist_ok=True)
        torch.save({
            'memory': self.memory,
            'meta': {
                'topk_per_class_scale': self.topk_per_class_scale,
                'num_base_classes': self.num_base_classes,
                'duplicate_iou_thr': self.duplicate_iou_thr,
                'min_unknown_score': self.min_unknown_score,
            }
        }, self.cache_path)
        runner.logger.info(f'[HardNegCache] Saved cache: {self.cache_path}')

    def _num_base_classes(self, model: nn.Module) -> int:
        if self.num_base_classes is not None:
            return int(self.num_base_classes)
        num_prev = int(getattr(model, 'num_prev_classes', 0))
        if num_prev > 0:
            return num_prev
        return int(model.num_training_classes - 2)

    def _target_size(self, model: nn.Module) -> int:
        return (int(model.bbox_head.head_module.num_levels) *
                self._num_base_classes(model) *
                self.topk_per_class_scale)

    def _is_full(self, model: nn.Module) -> bool:
        return self._cache_size() >= self._target_size(model)

    def _cache_size(self) -> int:
        return sum(len(bucket) for level in self.memory.values()
                   for bucket in level.values())

    def _log_summary(self, runner: Runner) -> None:
        lines = ['[HardNegCache] Cache summary']
        for lvl in sorted(self.memory):
            counts = {cls_id: len(bucket)
                      for cls_id, bucket in self.memory[lvl].items()
                      if bucket}
            lines.append(
                f'  level {lvl}: {sum(counts.values())} features {counts}')
        lines.append(f'  total: {self._cache_size()} features')
        runner.logger.info('\n'.join(lines))

    @staticmethod
    def _freeze_bn_stats(model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
