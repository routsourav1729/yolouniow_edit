# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.utils import filter_scores_and_topk, multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)

from mmdet.utils import InstanceList
from mmdet.structures import SampleList
from mmengine.config import ConfigDict
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from ..utils import gt_instances_preprocess, make_divisible
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead


@MODELS.register_module()
class YOLOv10HeadModule(BaseModule):
    """YOLOv10HeadModule head module used in `YOLOv10`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
        reg_max (int): Max value of integral set :math: ``{0, ..., reg_max-1}``
            in QFL setting. Defaults to 16.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 reg_max: int = 16,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.reg_max = reg_max

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self.one2many_init_layers()
        self.one2one_init_layers()

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for one2many_reg_pred, one2many_cls_pred, one2one_reg_pred, one2one_cls_pred, stride in zip(self.one2many_reg_preds,
                                                                                                    self.one2many_cls_preds,
                                                                                                    self.one2one_reg_preds,
                                                                                                    self.one2one_cls_preds,
                                                                                                    self.featmap_strides):
            one2many_reg_pred[-1].bias.data[:] = 1.0  # box
            one2one_reg_pred[-1].bias.data[:] = 1.0  # box
            
            # cls (.01 objects, 80 classes, 640 img)
            one2many_cls_pred[-1].bias.data[:self.num_classes] = math.log(5 / self.num_classes / (640 / stride)**2)
            one2one_cls_pred[-1].bias.data[:self.num_classes] = math.log(5 / self.num_classes / (640 / stride)**2)


    def one2many_init_layers(self):
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.one2many_cls_preds = nn.ModuleList()
        self.one2many_reg_preds = nn.ModuleList()

        reg_out_channels = max((16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)
        c3 = max(self.in_channels[0], min(self.num_classes, 100))

        for i in range(self.num_levels):
            self.one2many_reg_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    ConvModule(
                        in_channels=reg_out_channels,
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.Conv2d(
                        in_channels=reg_out_channels,
                        out_channels=4 * self.reg_max,
                        kernel_size=1)))
            self.one2many_cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                            out_channels=self.in_channels[i],
                            kernel_size=3,
                            stride=1,
                            groups=self.in_channels[i],
                            padding=1,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg),
                    ConvModule(in_channels=self.in_channels[i],
                            out_channels=c3,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg), 
                    ConvModule(in_channels=c3,
                            out_channels=c3,
                            kernel_size=3,
                            groups=c3,
                            stride=1,
                            padding=1,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg),
                    ConvModule(in_channels=c3,
                            out_channels=c3,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=c3,
                            out_channels=self.num_classes,
                            kernel_size=1)
                )
            )

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('one2many_proj', proj, persistent=False)

    def one2one_init_layers(self):
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.one2one_cls_preds = nn.ModuleList()
        self.one2one_reg_preds = nn.ModuleList()

        reg_out_channels = max((16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)
        c3 = max(self.in_channels[0], min(self.num_classes, 100))
        for i in range(self.num_levels):
            self.one2one_reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=4 * self.reg_max,
                              kernel_size=1)
                    )
                    )
            self.one2one_cls_preds.append(
                nn.Sequential(
                        ConvModule(in_channels=self.in_channels[i],
                                out_channels=self.in_channels[i],
                                kernel_size=3,
                                stride=1,
                                groups=self.in_channels[i],
                                padding=1,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg),
                        ConvModule(in_channels=self.in_channels[i],
                                out_channels=c3,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg), 
                        ConvModule(in_channels=c3,
                                out_channels=c3,
                                kernel_size=3,
                                groups=c3,
                                stride=1,
                                padding=1,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg),
                        ConvModule(in_channels=c3,
                                out_channels=c3,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg),
                        nn.Conv2d(in_channels=c3,
                                out_channels=self.num_classes,
                                kernel_size=1)
                    )
                )

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('one2one_proj', proj, persistent=False)

    def forward_one2many(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions
        """
        assert len(x) == self.num_levels
        
        return multi_apply(self.one2many_forward_single, x, self.one2many_cls_preds, self.one2many_reg_preds)

    def forward_one2one(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions
        """
        assert len(x) == self.num_levels
        return multi_apply(self.one2one_forward_single, x, self.one2one_cls_preds, self.one2one_reg_preds)


    def one2many_forward_single(self, x: torch.Tensor, cls_pred: nn.ModuleList,
                                reg_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        cls_logit = cls_pred(x)
        bbox_dist_preds = reg_pred(x)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape([-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(self.one2many_proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds

    
    def one2one_forward_single(self, x: torch.Tensor, cls_pred: nn.ModuleList,
                               reg_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        cls_logit = cls_pred(x)
        bbox_dist_preds = reg_pred(x)

        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape([-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(self.one2one_proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


@MODELS.register_module()
class YOLOv10Head(BaseDenseHead):
    """YOLOv10Head head used in `YOLOv10`.

    Args:
        head_module(:obj:`ConfigDict` or dict): Base module used for YOLOv8Head
        prior_generator(dict): Points generator feature maps
            in 2D points-based detectors.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_dfl (:obj:`ConfigDict` or dict): Config of Distribution Focal
            Loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=0.5),
                 loss_bbox: ConfigType = dict(
                     type='IoULoss',
                     iou_mode='ciou',
                     bbox_format='xyxy',
                     reduction='sum',
                     loss_weight=7.5,
                     return_iou=False),
                 loss_dfl=dict(
                     type='mmdet.DistributionFocalLoss',
                     reduction='mean',
                     loss_weight=1.5 / 4),
                 train_cfg: OptConfigType = None,
                 infer_type: str = "one2one",
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        self.head_module = MODELS.build(head_module)
        self.num_classes = self.head_module.num_classes
        self.featmap_strides = self.head_module.featmap_strides
        self.num_levels = len(self.featmap_strides)

        self.loss_cls: nn.Module = MODELS.build(loss_cls)
        self.loss_bbox: nn.Module = MODELS.build(loss_bbox)

        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.num_base_priors = self.prior_generator.num_base_priors[0]

        self.featmap_sizes = [torch.empty(1)] * self.num_levels

        self.infer_type = infer_type
        self.one2many_train_cfg = train_cfg.get("one2many_assigner", None)
        self.one2one_train_cfg = train_cfg.get("one2one_assigner", None)
        # whether to use anchor class as peudo label
        self.anchor_label = train_cfg.get("anchor_label", None)
        self.fed_bce = train_cfg.get("fed_bce", None)
        self.test_cfg = test_cfg

        self.loss_dfl = MODELS.build(loss_dfl)
        # YOLOv8 doesn't need loss_obj
        self.loss_obj = None
        self.shape = None
        self.special_init()

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.one2many_train_cfg:
            self.one2many_assigner = TASK_UTILS.build(self.one2many_train_cfg)

            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

        if self.one2one_train_cfg:
            self.one2one_assigner = TASK_UTILS.build(self.one2one_train_cfg)

            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    def _pseudo_unknown_mask(self, flatten_cls_preds: Tensor,
                             flatten_pred_bboxes: Tensor,
                             gt_bboxes: Tensor,
                             fg_mask_pre_prior: Tensor) -> Tuple[Tensor, Tensor]:
        anchor_iou_thresh = self.anchor_label.get('iou_threshold', 0.5)
        anchor_score_thresh = self.anchor_label.get('score_threshold', 0.0)
        anchor_scores = flatten_cls_preds[:, :, -1].detach().sigmoid()
        overlaps = bbox_overlaps(flatten_pred_bboxes, gt_bboxes, mode='iou')
        unknown_mask = ((overlaps.amax(dim=2) < anchor_iou_thresh)
                        & (fg_mask_pre_prior == 0)
                        & (anchor_scores > anchor_score_thresh))

        gate_epoch = self.anchor_label.get(
            'tobj_gt_known_max_after_epoch',
            self.anchor_label.get('objectness_gt_known_max_after_epoch', None))
        current_epoch = getattr(self, 'current_epoch', 0)
        if gate_epoch is not None and current_epoch > int(gate_epoch):
            num_known_classes = int(
                self.anchor_label.get('num_known_classes',
                                      max(self.num_classes - 2, 1)))
            num_known_classes = max(1, min(num_known_classes, self.num_classes))
            max_known_scores = flatten_cls_preds[:, :, :num_known_classes].detach(
            ).sigmoid().amax(dim=2)
            unknown_mask = unknown_mask & (anchor_scores > max_known_scores)

        return unknown_mask, anchor_scores

    def _fed_bce_mask(self, assigned_labels: Tensor,
                      fg_mask_pre_prior: Tensor,
                      assigned_scores: Tensor) -> Optional[Tensor]:
        """Mask classification BCE for sparse few-shot T2 supervision.

        Current-class positives supervise only their own class channel. Base
        positives are optional safe negatives for current-class channels.
        Everything else receives zero classification gradient.
        """
        if self.fed_bce is None or not self.fed_bce.get('enabled', False):
            return None

        num_prev_classes = int(self.fed_bce.get('num_prev_classes', 0))
        num_known_classes = int(
            self.fed_bce.get('num_known_classes', self.num_classes - 2))
        include_base_negatives = self.fed_bce.get(
            'include_base_negatives', True)
        class_unmask_weights = self.fed_bce.get('class_unmask_weights', None)
        tunk_known_negative = self.fed_bce.get('tunk_known_negative', False)
        mask_tunk_pseudo_positive = self.fed_bce.get(
            'mask_tunk_pseudo_positive', False)
        mask_classes_on_tunk_pseudo_positive = self.fed_bce.get(
            'mask_classes_on_tunk_pseudo_positive', None)
        unk_idx = int(self.fed_bce.get('unk_idx', self.num_classes - 2))

        cls_mask = assigned_scores.new_zeros(assigned_scores.shape)
        labels = assigned_labels.long().clamp(min=0,
                                              max=self.num_classes - 1)
        fg_mask = fg_mask_pre_prior.bool()

        current_pos = ((assigned_labels >= num_prev_classes)
                       & (assigned_labels < num_known_classes)
                       & fg_mask)
        if current_pos.any():
            current_channel = torch.zeros_like(
                assigned_scores, dtype=torch.bool)
            current_channel.scatter_(2, labels.unsqueeze(-1), True)
            cls_mask = torch.where(
                current_pos.unsqueeze(-1) & current_channel,
                torch.ones_like(cls_mask), cls_mask)

        if include_base_negatives and num_prev_classes > 0:
            base_pos = ((assigned_labels >= 0)
                        & (assigned_labels < num_prev_classes)
                        & fg_mask)
            if base_pos.any() and num_known_classes > num_prev_classes:
                cls_mask[:, :, num_prev_classes:num_known_classes] = (
                    torch.where(
                        base_pos.unsqueeze(-1),
                        torch.ones_like(cls_mask[:, :, num_prev_classes:
                                                  num_known_classes]),
                        cls_mask[:, :, num_prev_classes:num_known_classes]))

        if tunk_known_negative:
            known_pos = ((assigned_labels >= 0)
                         & (assigned_labels < num_known_classes)
                         & fg_mask)
            if known_pos.any() and 0 <= unk_idx < self.num_classes:
                cls_mask[:, :, unk_idx] = torch.where(
                    known_pos,
                    torch.ones_like(cls_mask[:, :, unk_idx]),
                    cls_mask[:, :, unk_idx])

        if class_unmask_weights is not None:
            weights = assigned_scores.new_tensor(class_unmask_weights)
            if weights.numel() != self.num_classes:
                raise ValueError(
                    'fed_bce.class_unmask_weights must have '
                    f'{self.num_classes} entries, got {weights.numel()}.')
            weights = weights.clamp(0, 1).view(1, 1, -1)
            cls_mask = cls_mask + (1.0 - cls_mask) * weights

        if mask_tunk_pseudo_positive and 0 <= unk_idx < self.num_classes:
            tunk_pseudo_pos = (assigned_scores[:, :, unk_idx] > 0) & ~fg_mask
            if tunk_pseudo_pos.any():
                cls_mask[:, :, unk_idx] = torch.where(
                    tunk_pseudo_pos,
                    torch.zeros_like(cls_mask[:, :, unk_idx]),
                    cls_mask[:, :, unk_idx])

        if (mask_classes_on_tunk_pseudo_positive is not None
                and 0 <= unk_idx < self.num_classes):
            tunk_pseudo_pos = (assigned_scores[:, :, unk_idx] > 0) & ~fg_mask
            if tunk_pseudo_pos.any():
                class_ids = [
                    int(class_id)
                    for class_id in mask_classes_on_tunk_pseudo_positive
                ]
                valid_ids = [
                    class_id for class_id in class_ids
                    if 0 <= class_id < self.num_classes
                ]
                if valid_ids:
                    cls_mask[:, :, valid_ids] = torch.where(
                        tunk_pseudo_pos.unsqueeze(-1),
                        torch.zeros_like(cls_mask[:, :, valid_ids]),
                        cls_mask[:, :, valid_ids])

        return cls_mask

    def _loss_cls_with_fed_bce(self, flatten_cls_preds: Tensor,
                               assigned_scores: Tensor,
                               assigned_labels: Tensor,
                               fg_mask_pre_prior: Tensor,
                               normalizer: Tensor) -> Tensor:
        loss_cls_raw = self.loss_cls(flatten_cls_preds, assigned_scores)
        cls_mask = self._fed_bce_mask(assigned_labels, fg_mask_pre_prior,
                                      assigned_scores)
        if cls_mask is not None:
            loss_cls_raw = loss_cls_raw * cls_mask
            # Keep YOLO's original target-sum normalizer by default. Otherwise
            # masking changes the effective LR of every still-unmasked channel,
            # including T_unk when its BCE entries are intentionally left free.
            if self.fed_bce.get('renormalize_by_mask', False):
                normalizer = (assigned_scores * cls_mask).sum().clamp(min=1)
        return loss_cls_raw.sum() / normalizer

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """

        if self.training:
            one2many_result = self.head_module.forward_one2many(x)
            x_detach = [x_.detach() for x_ in x]
            one2one_result = self.head_module.forward_one2one(x_detach)
        else:
            one2many_result = self.head_module.forward_one2many(x) if self.infer_type == 'one2many' else None
            one2one_result = self.head_module.forward_one2one(x) if self.infer_type == 'one2one' else None

        return one2many_result, one2one_result

    def loss(self, x: Tuple[Tensor], batch_data_samples: Union[list,
                                                               dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`], dict): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """

        x_one2many, x_one2one =  self.forward(x)

        if isinstance(batch_data_samples, list):
            losses = super().loss(x_one2many, batch_data_samples)
        else:
            one2many_outs = x_one2many
            one2one_outs = x_one2one
            # Fast version
            one2many_loss_inputs = one2many_outs + (batch_data_samples['bboxes_labels'], batch_data_samples['img_metas'])
            one2one_loss_inputs = one2one_outs + (batch_data_samples['bboxes_labels'], batch_data_samples['img_metas'])
            losses = self.loss_by_feat([one2many_loss_inputs, one2one_loss_inputs])

        return losses

    def loss_by_feat(
            self,
            all2one
            ) -> dict:
        one2many_loss_inputs = all2one[0]
        one2one_loss_inputs = all2one[1]

        losses = self.one2many_loss_by_feat(*one2many_loss_inputs)
        losses.update(self.one2one_loss_by_feat(*one2one_loss_inputs)) 
        return losses
        
    
    def one2many_loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4) for bbox_pred_org in bbox_dist_preds]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        assigned_result = self.one2many_assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        _unknown_mask = None
        if self.anchor_label:
            unknown_mask, anchor_scores = self._pseudo_unknown_mask(
                flatten_cls_preds, flatten_pred_bboxes, gt_bboxes,
                fg_mask_pre_prior)
            _unknown_mask = unknown_mask
            assigned_scores[:, :, -2] = torch.where(
                unknown_mask, anchor_scores,
                assigned_scores[:, :, -2])  # unknown embedding score

        # [GADL] Cache raw logits and GT-assigned labels for GADL loss in OWODDetector.
        # - flatten_cls_preds: (B, N, K) raw logits WITH grad (pre-sigmoid).
        # - assigned_labels: (B, N) GT class index per anchor, -1 for negatives.
        if getattr(self, 'gadl', None) is not None:
            _gadl_raw_labels = assigned_scores.argmax(dim=-1)  # (B, N)
            self._gadl_assigned_labels = torch.where(
                fg_mask_pre_prior,
                _gadl_raw_labels,
                _gadl_raw_labels.new_full(_gadl_raw_labels.shape, -1))
            self._gadl_cls_logits = flatten_cls_preds  # (B, N, K) with grad

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)
        loss_cls = self._loss_cls_with_fed_bce(
            flatten_cls_preds, assigned_scores,
            assigned_result['assigned_labels'], fg_mask_pre_prior,
            assigned_scores_sum)

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        _, world_size = get_dist_info()

        losses = dict(
            one2many_loss_cls=loss_cls * num_imgs * world_size,
            one2many_loss_bbox=loss_bbox * num_imgs * world_size,
            one2many_loss_dfl=loss_dfl * num_imgs * world_size)

        # [KUME] Known-Unknown Margin Enforcement
        _kume = getattr(self, '_kume', None)
        if _kume is not None:
            _kume_labels = assigned_scores.argmax(dim=-1)  # (B, N)
            kume_loss = _kume.compute(
                flatten_cls_preds,       # (B, N, K) with grad
                _kume_labels,            # (B, N)
                fg_mask_pre_prior,       # (B, N) bool
                _unknown_mask,           # (B, N) bool or None
            )
            losses['kume_loss'] = kume_loss

        return losses

    def one2one_loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4) for bbox_pred_org in bbox_dist_preds]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        assigned_result = self.one2one_assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        if self.anchor_label:
            unknown_mask, anchor_scores = self._pseudo_unknown_mask(
                flatten_cls_preds, flatten_pred_bboxes, gt_bboxes,
                fg_mask_pre_prior)
            assigned_scores[:, :, -2] = torch.where(
                unknown_mask, anchor_scores,
                assigned_scores[:, :, -2])  # unknown embedding score

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)
        loss_cls = self._loss_cls_with_fed_bce(
            flatten_cls_preds, assigned_scores,
            assigned_result['assigned_labels'], fg_mask_pre_prior,
            assigned_scores_sum)

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        _, world_size = get_dist_info()

        return dict(
            one2one_loss_cls=loss_cls * num_imgs * world_size,
            one2one_loss_bbox=loss_bbox * num_imgs * world_size,
            one2one_loss_dfl=loss_dfl * num_imgs * world_size)

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """

        x_one2many, x_one2one = self.forward(x)
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
    
        if self.infer_type == "one2one":
            outs = x_one2one
            with_nms = self.test_cfg.get("one2one_withnms", False)
        elif self.infer_type == "one2many":
            outs = x_one2many
            with_nms = self.test_cfg.get("one2many_withnms", False)
        else:
            raise Exception("unsupported infer type")

        predictions = self.predict_by_feat(*outs,
                                            batch_img_metas=batch_img_metas,
                                            rescale=rescale,
                                            with_nms=with_nms)
        return predictions

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = False) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)

        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        score_diagnostics = flatten_cls_scores.clone()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]

        results_list = []
        
        if self.anchor_label:
            flatten_cls_scores = flatten_cls_scores.clone()
            flatten_cls_scores[:, :, -1] = 0 # IMPORTANT: ignore all anchor predictions

        for (bboxes, scores, diag_scores, objectness,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              score_diagnostics, flatten_objectness,
                              batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                diag_scores = diag_scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]
                diag_scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                empty_results.max_known_scores = scores[:, 0]
                empty_results.tunk_scores = scores[:, 0]
                empty_results.tobj_scores = scores[:, 0]
                results_list.append(empty_results)
                continue

            if self.anchor_label:
                known_end = max(self.num_classes - 2, 1)
                tunk_idx = self.num_classes - 2
                tobj_scores = diag_scores[:, -1]
            else:
                known_end = max(self.num_classes - 1, 1)
                tunk_idx = self.num_classes - 1
                tobj_scores = diag_scores.new_zeros(diag_scores.shape[0])
            max_known_scores = diag_scores[:, :known_end].max(dim=1).values
            tunk_scores = diag_scores[:, tunk_idx]

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes[keep_idxs],
                max_known_scores=max_known_scores[keep_idxs],
                tunk_scores=tunk_scores[keep_idxs],
                tobj_scores=tobj_scores[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)
            
            if cfg.get('unknown_nms', None):
                ori_max_per_img = cfg.max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta)

            if not with_nms:
                results = results[:cfg.max_per_img]
                
            if cfg.get('unknown_nms', None):
                cfg.max_per_img = ori_max_per_img
                results = self._unknown_post_process(results, cfg)

            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list
    
    def _unknown_post_process(self, 
                              results: InstanceData,
                              cfg: ConfigDict) -> InstanceData:
        all_boxes = results.bboxes
        all_scores = results.scores
        all_labels = results.labels
        
        ious = bbox_overlaps(all_boxes, all_boxes)

        iou_mask = ious >= cfg.unknown_nms.get('iou_threshold', 0.99)
        iou_mask = torch.triu(iou_mask, diagonal=1)
        iou_mask[all_scores < cfg.unknown_nms.get('score_threshold', 0.2)] = False
        
        unknown_mask = all_labels == (self.num_classes - 2 if self.anchor_label else self.num_classes - 1)
        unknown_iou_mask = iou_mask & unknown_mask[None, :]
        retained_indices = torch.arange(len(all_boxes), dtype=torch.long).to(all_boxes.device)
        retained_indices = retained_indices[~unknown_iou_mask.any(dim=0)]

        results = results[retained_indices]
        results = results[:cfg.max_per_img]
        return results

    def _convert_gt_to_norm_format(self,
                                   batch_gt_instances: Sequence[InstanceData],
                                   batch_img_metas: Sequence[dict]) -> Tensor:
        if isinstance(batch_gt_instances, torch.Tensor):
            # fast version
            img_shape = batch_img_metas[0]['batch_input_shape']
            gt_bboxes_xyxy = batch_gt_instances[:, 2:]
            xy1, xy2 = gt_bboxes_xyxy.split((2, 2), dim=-1)
            gt_bboxes_xywh = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
            gt_bboxes_xywh[:, 1::2] /= img_shape[0]
            gt_bboxes_xywh[:, 0::2] /= img_shape[1]
            batch_gt_instances[:, 2:] = gt_bboxes_xywh

            # (num_base_priors, num_bboxes, 6)
            batch_targets_normed = batch_gt_instances.repeat(
                self.num_base_priors, 1, 1)
        else:
            batch_target_list = []
            # Convert xyxy bbox to yolo format.
            for i, gt_instances in enumerate(batch_gt_instances):
                img_shape = batch_img_metas[i]['batch_input_shape']
                bboxes = gt_instances.bboxes
                labels = gt_instances.labels

                xy1, xy2 = bboxes.split((2, 2), dim=-1)
                bboxes = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
                # normalized to 0-1
                bboxes[:, 1::2] /= img_shape[0]
                bboxes[:, 0::2] /= img_shape[1]

                index = bboxes.new_full((len(bboxes), 1), i)
                # (batch_idx, label, normed_bbox)
                target = torch.cat((index, labels[:, None].float(), bboxes),
                                   dim=1)
                batch_target_list.append(target)

            # (num_base_priors, num_bboxes, 6)
            batch_targets_normed = torch.cat(
                batch_target_list, dim=0).repeat(self.num_base_priors, 1, 1)

        # (num_base_priors, num_bboxes, 1)
        batch_targets_prior_inds = self.prior_inds.repeat(
            1, batch_targets_normed.shape[1])[..., None]
        # (num_base_priors, num_bboxes, 7)
        # (img_ind, labels, bbox_cx, bbox_cy, bbox_w, bbox_h, prior_ind)
        batch_targets_normed = torch.cat(
            (batch_targets_normed, batch_targets_prior_inds), 2)
        return batch_targets_normed

    def _decode_bbox_to_xywh(self, bbox_pred, priors_base_sizes) -> Tensor:
        bbox_pred = bbox_pred.sigmoid()
        pred_xy = bbox_pred[:, :2] * 2 - 0.5
        pred_wh = (bbox_pred[:, 2:] * 2)**2 * priors_base_sizes
        decoded_bbox_pred = torch.cat((pred_xy, pred_wh), dim=-1)
        return decoded_bbox_pred
