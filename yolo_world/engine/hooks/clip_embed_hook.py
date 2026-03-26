"""CLIPEmbedHook — overwrite known-class embeddings with fresh CLIP npy.

Fires at both after_load_checkpoint AND before_test_epoch (after EMAHook
swaps EMA weights in), so the CLIP values always win.
"""
import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmyolo.registry import HOOKS


@HOOKS.register_module()
class CLIPEmbedHook(Hook):
    """Overwrite known-class embedding slots with a CLIP npy file.

    Config usage::

        custom_hooks = [
            dict(type='CLIPEmbedHook',
                 clip_emb_path='embeddings/uniow-l/idd_all.npy',
                 priority=51),   # must be > 49 so it runs AFTER EMAHook
        ]
    """

    def __init__(self, clip_emb_path: str, **kwargs):
        super().__init__(**kwargs)
        self.clip_emb_path = clip_emb_path
        self._clip_emb = torch.from_numpy(np.load(clip_emb_path)).float()
        print(f'[CLIPEmbedHook] Loaded {clip_emb_path}: shape={self._clip_emb.shape}')

    def _patch(self, runner: Runner, tag: str):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        n = self._clip_emb.shape[0]
        with torch.no_grad():
            model.embeddings.data[:n] = self._clip_emb.to(model.embeddings.device)
        print(f'[CLIPEmbedHook] ({tag}) Replaced slots 0..{n-1}')
        print(f'[CLIPEmbedHook] Norms: {model.embeddings.data.norm(dim=-1)}')

    def after_load_checkpoint(self, runner: Runner, checkpoint: dict):
        self._patch(runner, 'after_load_checkpoint')

    def before_test_epoch(self, runner: Runner):
        self._patch(runner, 'before_test_epoch')
