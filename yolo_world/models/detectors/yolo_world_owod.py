# Copyright (c) Tencent Inc. All rights reserved.
import gc
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS


@MODELS.register_module()
class OWODDetector(YOLODetector):
    """Implementation of Open-World YOLO"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes: int = 80,
                 num_test_classes: int = 80,
                 num_prev_classes: int = 0,
                 num_prompts: int = 80,
                 prompt_dim: int = 512,
                 embedding_path: str = '',
                 unknown_embedding_path: str = '',
                 anchor_embedding_path: str = '',
                 embedding_mask: Union[List, int] = None,
                 freeze_prompt: bool = False,
                 use_mlp_adapter: bool = False,
                 wapr: dict = None,
                 scpi: dict = None,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.num_prev_classes = num_prev_classes
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if len(embedding_path) > 0:
            self.embeddings = torch.nn.Parameter(
                torch.from_numpy(np.load(embedding_path)).float())
        else:
            # random init
            embeddings = nn.functional.normalize(torch.randn(
                (num_train_classes, prompt_dim)),
                                                    dim=-1)
            self.embeddings = nn.Parameter(embeddings)

        if len(unknown_embedding_path) > 0:
            unknown_embeddings = nn.Parameter(torch.from_numpy(
                np.load(unknown_embedding_path)).float())
            self.embeddings = nn.Parameter(torch.cat([self.embeddings, unknown_embeddings], dim=0))
        
        if len(anchor_embedding_path) > 0:
            anchor_embeddings = nn.Parameter(torch.from_numpy(
                np.load(anchor_embedding_path)).float())
            self.embeddings = nn.Parameter(torch.cat([self.embeddings, anchor_embeddings], dim=0))

        if self.freeze_prompt:
            self.embeddings.requires_grad = False
        else:
            self.embeddings.requires_grad = True

        if embedding_mask and not self.freeze_prompt:
            if isinstance(embedding_mask, int):
                self._grad_mask = torch.ones(num_train_classes, dtype=torch.bool)[:, None]
                self._grad_mask[:embedding_mask] = False
            else:
                self._grad_mask = torch.Tensor(embedding_mask).bool()[:, None]
            assert len(self._grad_mask) == num_train_classes
            self.embeddings.register_hook(lambda grad: grad * self._grad_mask.to(grad.device))

        if use_mlp_adapter:
            self.adapter = nn.Sequential(
                nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                nn.Linear(prompt_dim * 2, prompt_dim))
        else:
            self.adapter = None

        # WAPR: Wildcard-Aware Pseudo-label Redistribution (T2 only)
        self.wapr = None
        if wapr is not None:
            from yolo_world.models.losses.wapr import WAPRModule
            self.wapr = WAPRModule(
                frozen_embedding_path=wapr['frozen_embedding_path'],
                num_prev_classes=self.num_prev_classes,
                num_known_classes=wapr.get(
                    'num_known_classes', num_train_classes - 2),
                warmup_epochs=wapr.get('warmup_epochs', 2),
                anchor_loss_weight=wapr.get('anchor_loss_weight', 0.1),
            )
            self.bbox_head.wapr = self.wapr
            print(f"[WAPR] Initialized: num_known={self.wapr.num_known_classes}, "
                  f"num_prev={self.wapr.num_prev_classes}, "
                  f"warmup={self.wapr.warmup_epochs} epochs, "
                  f"anchor_weight={self.wapr.anchor_loss_weight}")

        # SCPI: Support-Calibrated Prompt Interpolation (training-free T2)
        self.scpi_cfg = scpi

    def update_embeddings(self, embeddings):
        # update embeddings when loading from checkpoint
        prev_embeddings = embeddings[:self.num_prev_classes]
        cur_embeddings = self.embeddings[self.num_prev_classes:].detach().cpu()
        embeddings = torch.cat([prev_embeddings, cur_embeddings], dim=0)
        return embeddings

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # WAPR: snapshot T_unk anchor on first call (after T1 checkpoint load)
        if self.wapr is not None and self.wapr.t_unk_anchor is None:
            self.wapr.set_t_unk_anchor(self.embeddings[-2])
            print(f"[WAPR] T_unk anchor snapshot saved (norm={self.embeddings[-2].norm().item():.4f})")

        # WAPR: set warmup flag on head (skip gatekeeper during warmup)
        if self.wapr is not None:
            from mmengine.logging import MessageHub
            try:
                epoch = MessageHub.get_current_instance().get_info('epoch')
            except (KeyError, RuntimeError):
                epoch = 0
            self.bbox_head._wapr_in_warmup = (epoch < self.wapr.warmup_epochs)

        losses = self.bbox_head.loss(img_feats, txt_feats,
                                        batch_data_samples)

        # WAPR: add T_unk anchor drift loss and log stats
        if self.wapr is not None:
            anchor_loss = self.wapr.compute_anchor_loss(
                self.embeddings[-2])
            losses['wapr_anchor_loss'] = anchor_loss
            # Log WAPR stats
            wapr_stats = getattr(self.bbox_head, '_wapr_stats', {})
            warmup = getattr(self.bbox_head, '_wapr_in_warmup', False)
            # Print every call so user can see it's running
            if wapr_stats:
                print(f"[WAPR] epoch={epoch} "
                      f"warmup={'Y' if warmup else 'N'} "
                      f"anchor_loss={anchor_loss.item():.8f} "
                      f"candidates={wapr_stats.get('wapr/num_candidates', 0)} "
                      f"redirected={wapr_stats.get('wapr/num_redirected', 0)} "
                      f"genuine_unk={wapr_stats.get('wapr/num_genuine_unk', 0)} "
                      f"mean_max_prob={wapr_stats.get('wapr/mean_max_prob', 0):.4f} "
                      f"mean_w_r={wapr_stats.get('wapr/mean_w_r', 0):.4f}")
            else:
                print(f"[WAPR] epoch={epoch} "
                      f"warmup={'Y' if warmup else 'N'} "
                      f"anchor_loss={anchor_loss.item():.8f} "
                      f"no redistribute stats (0 candidates or cached logits missing)")
            # Also push to MessageHub for TensorBoard
            if wapr_stats:
                from mmengine.logging import MessageHub
                hub = MessageHub.get_current_instance()
                for k, v in wapr_stats.items():
                    hub.update_scalar(k, v)
                hub.update_scalar('wapr/warmup_active',
                                  1.0 if warmup else 0.0)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes

        results_list = self.bbox_head.predict(img_feats,
                                                txt_feats,
                                                batch_data_samples,
                                                rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        # use embeddings
        txt_feats = self.embeddings[None]
        if self.adapter is not None:
            txt_feats = self.adapter(txt_feats) + txt_feats
            txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
        txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)

        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats

    # ── SCPI: Support-Calibrated Prompt Interpolation ────────────────────

    # FPN strides and area thresholds (same as embedding_diagnostic.py)
    _FPN_STRIDES = [8, 16, 32]
    _AREA_THRESHOLDS = [96**2, 192**2]

    @torch.no_grad()
    def calibrate_scpi(self, support_cfg: dict):
        """Support-Calibrated Prompt Interpolation.

        Replaces novel class embeddings with an adaptive blend of zero-shot
        CLIP prompts and visual centroids from K-shot support images.

        Called once after checkpoint load, before eval.  No training needed.

        Args:
            support_cfg: dict with keys
                fewshot_dir, fewshot_k, fewshot_seed, data_root, dataset,
                novel_classes (list[str]), beta (float), tau (float)
        """
        from mmengine.dataset import Compose

        fewshot_dir = support_cfg['fewshot_dir']
        fewshot_k = support_cfg['fewshot_k']
        fewshot_seed = support_cfg['fewshot_seed']
        data_root = support_cfg['data_root']
        dataset_name = support_cfg['dataset']
        novel_classes = support_cfg['novel_classes']
        beta = support_cfg.get('beta', 10.0)
        tau = support_cfg.get('tau', 0.15)

        device = self.embeddings.device
        num_novel = len(novel_classes)
        novel_start = self.num_prev_classes  # base classes end here

        # CRITICAL: Switch to eval mode so BN uses running stats (not batch
        # stats) and does NOT corrupt running_mean/running_var.
        # mmengine calls model.eval() AFTER before_test_epoch, so we must
        # do it ourselves here.
        was_training = self.training
        self.eval()

        print(f'[SCPI] Starting calibration: {num_novel} novel classes, '
              f'beta={beta}, tau={tau}')
        print(f'[SCPI] Support: k={fewshot_k}, seed={fewshot_seed}, '
              f'dir={fewshot_dir}')

        # ── Build image preprocessing pipeline ───────────────────────────
        # Hardcoded YOLOv10 test pipeline (640x640).  Avoids Config.fromfile()
        # which triggers heavy registry imports and can leave side effects.
        img_scale = (640, 640)
        img_pipeline_cfg = [
            dict(type='LoadImageFromFile'),
            dict(type='YOLOv5KeepRatioResize', scale=img_scale),
            dict(type='LetterResize', scale=img_scale,
                 allow_scale_up=False, pad_val=dict(img=114)),
            dict(type='mmdet.PackDetInputs',
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'pad_param')),
        ]
        pipeline = Compose(img_pipeline_cfg)

        # ── Register hooks on BN layers inside BNContrastiveHead ─────────
        head_module = self.bbox_head.head_module
        hooked = {}

        def make_hook(name):
            def hook_fn(module, inp, out):
                hooked[name] = out.detach()
            return hook_fn

        handles = []
        for i, layer in enumerate(head_module.one2one_cls_contrasts):
            if hasattr(layer, 'norm'):
                handles.append(
                    layer.norm.register_forward_hook(make_hook(f'bn_{i}')))

        # ── Collect per-class visual features from support set ───────────
        ann_dir = os.path.join(data_root, 'Annotations', dataset_name)
        seed_dir = os.path.join(fewshot_dir, f'seed{fewshot_seed}')

        class_features = defaultdict(list)  # cls_name -> list of 512-d

        with torch.no_grad():
            for cls_idx, cls_name in enumerate(novel_classes):
                shot_file = os.path.join(
                    seed_dir, f'box_{fewshot_k}shot_{cls_name}_train.txt')
                with open(shot_file) as f:
                    img_paths = [line.strip() for line in f if line.strip()]

                for img_path in img_paths:
                    # Derive image ID and annotation path
                    img_id = os.path.splitext(os.path.basename(img_path))[0]
                    xml_path = os.path.join(ann_dir, img_id + '.xml')
                    full_img_path = os.path.join(
                        data_root, '..', img_path) if not os.path.isabs(
                            img_path) else img_path
                    # The few-shot paths are relative to repo root, not data_root
                    if not os.path.exists(full_img_path):
                        full_img_path = img_path  # try as-is from repo root

                    if not os.path.exists(xml_path) or not os.path.exists(
                            full_img_path):
                        print(f'[SCPI] WARNING: missing {xml_path} or '
                              f'{full_img_path}')
                        continue

                    # Parse GT boxes for this class only
                    gt_boxes = self._parse_xml_for_class(xml_path, cls_name)
                    if not gt_boxes:
                        continue

                    # Preprocess image
                    data = dict(img_path=full_img_path, img_id=img_id,
                                instances=[])
                    data = pipeline(data)
                    img_tensor = data['inputs'].unsqueeze(0).float().to(device)
                    img_tensor = img_tensor / 255.0
                    data_sample = data['data_samples']

                    # Forward to populate hooks
                    img_feats, txt_feats = self.extract_feat(
                        img_tensor, [data_sample])
                    head_module.forward_one2one(img_feats, txt_feats)

                    # Extract features at each GT box center
                    meta = data_sample.metainfo
                    scale_factor = np.array(
                        meta.get('scale_factor', [1.0, 1.0]))
                    pad_param = meta.get('pad_param', (0, 0, 0, 0))

                    for box in gt_boxes:
                        feat = self._extract_feat_at_box(
                            box, scale_factor, pad_param, hooked)
                        if feat is not None:
                            class_features[cls_name].append(feat)

        # ── Remove hooks and free GPU memory ──────────────────────────────
        for h in handles:
            h.remove()
        hooked.clear()
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        # ── Compute per-class interpolated embeddings ────────────────────
        # Measure base embedding norms for target_norm
        base_norms = self.embeddings[:self.num_prev_classes].norm(
            dim=-1).mean().item()
        print(f'[SCPI] Base embedding mean norm: {base_norms:.4f}')

        print(f'[SCPI] {"Class":20s} {"n_feat":>6s} {"s_c":>6s} '
              f'{"alpha":>6s} {"action":>10s}')
        print('-' * 60)

        for cls_idx, cls_name in enumerate(novel_classes):
            emb_idx = novel_start + cls_idx
            e_zs = self.embeddings[emb_idx].clone()  # zero-shot prompt

            feats = class_features.get(cls_name, [])
            if not feats:
                print(f'[SCPI] {cls_name:20s} {"0":>6s} {"--":>6s} '
                      f'{"--":>6s} {"skip":>10s}')
                continue

            feat_stack = torch.stack(feats)  # (N, 512)

            # Visual centroid: mean of raw post-BN features, then normalize
            centroid = feat_stack.mean(dim=0)
            e_vis = F.normalize(centroid, dim=0)

            # Compatibility: cosine between L2-normed prompt and each
            # post-BN feature (also L2-normed for scale-invariance)
            e_zs_normed = F.normalize(e_zs, dim=0)
            feat_normed = F.normalize(feat_stack, dim=-1)
            cos_scores = (feat_normed @ e_zs_normed).cpu()
            s_c = cos_scores.mean().item()

            # Interpolation weight
            alpha_c = 1.0 - torch.sigmoid(
                torch.tensor(beta * (s_c - tau))).item()

            # Blend and rescale to match base norm range
            e_final = F.normalize(
                alpha_c * e_vis + (1.0 - alpha_c) * e_zs, dim=0
            ) * base_norms

            # Overwrite embedding in-place
            self.embeddings.data[emb_idx] = e_final

            action = ('visual' if alpha_c > 0.7
                      else 'blend' if alpha_c > 0.3
                      else 'zero-shot')
            print(f'[SCPI] {cls_name:20s} {len(feats):6d} {s_c:6.3f} '
                  f'{alpha_c:6.3f} {action:>10s}')

        # Free support-set feature tensors and flush GPU cache
        del class_features
        gc.collect()
        torch.cuda.empty_cache()

        # Restore original training mode (mmengine calls model.eval()
        # right after before_test_epoch, but be safe).
        if was_training:
            self.train()

        print(f'[SCPI] Calibration complete. Novel embeddings updated.')

    @staticmethod
    def _parse_xml_for_class(xml_path: str, cls_name: str) -> list:
        """Parse VOC XML, return list of [xmin, ymin, xmax, ymax] for cls."""
        tree = ET.parse(xml_path)
        boxes = []
        for obj in tree.findall('object'):
            if obj.find('name').text != cls_name:
                continue
            bbox = obj.find('bndbox')
            box = [float(bbox.find(x).text)
                   for x in ('xmin', 'ymin', 'xmax', 'ymax')]
            boxes.append(box)
        return boxes

    def _extract_feat_at_box(self, box, scale_factor, pad_param,
                             hooked) -> Tensor:
        """Extract post-BN feature vector at the center of a GT box."""
        x1 = box[0] * scale_factor[1] + pad_param[2]
        y1 = box[1] * scale_factor[0] + pad_param[0]
        x2 = box[2] * scale_factor[1] + pad_param[2]
        y2 = box[3] * scale_factor[0] + pad_param[0]

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = (x2 - x1) * (y2 - y1)

        # FPN level assignment
        if area < self._AREA_THRESHOLDS[0]:
            level = 0
        elif area < self._AREA_THRESHOLDS[1]:
            level = 1
        else:
            level = 2

        bn_key = f'bn_{level}'
        if bn_key not in hooked:
            return None

        feat_map = hooked[bn_key]  # (1, 512, H, W)
        stride = self._FPN_STRIDES[level]
        _, C, H, W = feat_map.shape
        gx = max(0, min(int(cx / stride), W - 1))
        gy = max(0, min(int(cy / stride), H - 1))

        return feat_map[0, :, gy, gx].clone()  # (512,)
