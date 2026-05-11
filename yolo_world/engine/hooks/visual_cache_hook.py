"""VisualCacheLoadHook — load a pre-built (n_novel, K, D) cache into the
head's VisualCache buffer before evaluation.
"""
import torch
from mmengine.hooks import Hook
from mmyolo.registry import HOOKS


@HOOKS.register_module()
class VisualCacheLoadHook(Hook):

    def __init__(self, cache_path: str, **kwargs):
        super().__init__(**kwargs)
        self.cache_path = cache_path
        self._loaded = False

    def _load(self, runner):
        if self._loaded:
            return
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        hm = model.bbox_head.head_module
        if getattr(hm, 'visual_cache', None) is None:
            raise RuntimeError(
                'VisualCacheLoadHook: head_module.visual_cache is None — '
                'set visual_cache_cfg in head_module config.')
        ckpt = torch.load(self.cache_path, map_location='cpu')
        device = next(model.parameters()).device
        hm.visual_cache.to(device)
        hm.visual_cache.load_cache(ckpt['cache_per_level'])
        runner.logger.info(
            f'[VisualCache] {self.cache_path}  '
            f'matched={ckpt["matched_per_class_per_level"]}  '
            f'alpha={hm.visual_alpha}  reduce={hm.visual_cache.reduce}')
        self._loaded = True

    def before_test(self, runner) -> None:
        self._load(runner)

    def before_val(self, runner) -> None:
        self._load(runner)
