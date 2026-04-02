"""SCPIHook — patches ONLY novel-class embeddings from a pre-calibrated npy.

The T2 checkpoint is already loaded by the Runner (has correct base/unk/anchor).
This hook ONLY overwrites novel embedding slots (indices num_prev..num_prev+num_novel-1)
with the SCPI-calibrated values from the npy file.

The npy is produced offline by tools/owod_scripts/scpi_calibrate.py.
"""
import os

import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmyolo.registry import HOOKS


@HOOKS.register_module()
class SCPIHook(Hook):
    """Patch novel-class embeddings from SCPI npy before eval.

    Runs at before_test_epoch with priority > 49 (after EMAHook swaps
    EMA weights into the model).

    The npy must have shape [num_known, 512] (base + novel, no unk/anchor).
    Only the novel portion (indices num_prev:) is written into the model.

    Config usage::

        custom_hooks = [
            ...,
            dict(type='SCPIHook',
                 scpi_emb_path='embeddings/uniow-idd/idd_t2_scpi.npy',
                 priority=51),
        ]
    """

    def __init__(self, scpi_emb_path: str = '', **kwargs):
        super().__init__(**kwargs)
        self.scpi_emb_path = scpi_emb_path

    def before_test_epoch(self, runner: Runner):
        if not self.scpi_emb_path or not os.path.exists(self.scpi_emb_path):
            print(f'[SCPIHook] No embedding file at {self.scpi_emb_path}, skipping')
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        device = model.embeddings.device
        num_prev = model.num_prev_classes

        scpi_emb = torch.from_numpy(np.load(self.scpi_emb_path)).float()
        num_known = scpi_emb.shape[0]
        print(f'[SCPIHook] Loaded {self.scpi_emb_path}: shape={scpi_emb.shape}')

        # Extract only the novel portion from the npy
        novel_emb = scpi_emb[num_prev:num_known]
        num_novel = novel_emb.shape[0]

        with torch.no_grad():
            model.embeddings.data[num_prev:num_prev + num_novel] = novel_emb.to(device)

        print(f'[SCPIHook] Patched novel slots {num_prev}..{num_prev + num_novel - 1}')
        print(f'[SCPIHook] Embedding norms: {model.embeddings.data.norm(dim=-1)}')
