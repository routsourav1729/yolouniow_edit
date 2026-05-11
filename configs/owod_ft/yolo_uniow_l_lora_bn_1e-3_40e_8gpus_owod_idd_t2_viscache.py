_base_ = ['yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2.py']

# Visual K-shot cache (FS-OSOD) — inference-time logit fusion only.
# IDD: n_base=8, n_novel=6.
visual_cache_cfg = dict(n_base=8, n_novel=6, reduce='mean', topk=3)
visual_alpha = 1.0
visual_cache_path = 'cache/idd/viscache_10shot_seed1.pt'

model = dict(
    bbox_head=dict(
        head_module=dict(
            visual_cache_cfg=visual_cache_cfg,
            visual_alpha=visual_alpha)))

# Increase test batch_size so CuDNN benchmark picks fast algorithms
# (batch_size=1 cold-start selects ~15x slower convolution kernels).
test_dataloader = dict(batch_size=8)

custom_hooks = [
    *_base_.custom_hooks,
    dict(type='VisualCacheLoadHook',
         cache_path=visual_cache_path,
         priority=99),
]
