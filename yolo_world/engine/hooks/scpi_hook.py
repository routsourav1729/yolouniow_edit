"""SCPIHook — patches model embeddings from a pre-calibrated npy file.

The heavy SCPI calibration is done offline by
tools/owod_scripts/scpi_calibrate.py, which saves a [num_known, 512] npy.

This hook simply:
  1. Loads the T1 checkpoint to get base/unk/anchor embeddings
  2. Overwrites novel embeddings from the pre-calibrated npy
  3. Zero runtime overhead — no forward passes, no model building
"""
import os

import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmyolo.registry import HOOKS


@HOOKS.register_module()
class SCPIHook(Hook):
    """Patch embeddings from pre-calibrated SCPI npy before eval.

    Config usage::

        custom_hooks = [
            ...,
            dict(type='SCPIHook',
                 scpi_emb_path='embeddings/uniow-idd/idd_t2_scpi_b10_t0.15.npy',
                 priority=50),
        ]
    """

    def __init__(self, scpi_emb_path: str = '', **kwargs):
        super().__init__(**kwargs)
        self.scpi_emb_path = scpi_emb_path

    def before_test_epoch(self, runner: Runner):
        if not self.scpi_emb_path or not os.path.exists(self.scpi_emb_path):
            print(f'[SCPI] No embedding file at {self.scpi_emb_path}, skipping')
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        cfg = runner.cfg
        device = model.embeddings.device
        num_prev = model.num_prev_classes

        # ── Load pre-calibrated embeddings (novel only or all known) ────
        scpi_emb = torch.from_numpy(np.load(self.scpi_emb_path)).float()
        # scpi_emb shape: [num_known, 512] — base + novel (no unk/anchor)
        num_known = scpi_emb.shape[0]
        print(f'[SCPI] Loaded {self.scpi_emb_path}: shape={scpi_emb.shape}')

        # ── Patch base/unk/anchor from T1 checkpoint ───────────────────
        # Use state_dict (not EMA) — matches what Runner loads into model.
        ckpt_path = cfg.get('load_from', '')
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            sd = ckpt.get('state_dict', ckpt)
            t1_emb = sd.get('embeddings', None)

            if t1_emb is not None:
                with torch.no_grad():
                    model.embeddings.data[:num_prev] = t1_emb[:num_prev].to(device)
                    model.embeddings.data[-2] = t1_emb[-2].to(device)
                    model.embeddings.data[-1] = t1_emb[-1].to(device)
                print(f'[SCPI] Patched base/unk/anchor from {ckpt_path}')

        # ── Overwrite novel embeddings from SCPI npy ───────────────────
        with torch.no_grad():
            novel_emb = scpi_emb[num_prev:num_known]
            num_novel = novel_emb.shape[0]
            model.embeddings.data[num_prev:num_prev + num_novel] = novel_emb.to(device)

        print(f'[SCPI] Final embedding norms: {model.embeddings.data.norm(dim=-1)}')
        print(f'[SCPI] Patched {num_novel} novel embeddings. Done.')
