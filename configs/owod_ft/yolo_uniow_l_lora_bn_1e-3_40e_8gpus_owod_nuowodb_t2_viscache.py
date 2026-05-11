_base_ = ['yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_nuowodb_t2.py']

# nuOWODB: n_base=10, n_novel=7.
visual_cache_cfg = dict(n_base=10, n_novel=7, reduce='mean', topk=3)
visual_alpha = 1.0
visual_cache_path = 'cache/nuowodb/viscache_10shot_seed1.pt'

model = dict(
    bbox_head=dict(
        head_module=dict(
            visual_cache_cfg=visual_cache_cfg,
            visual_alpha=visual_alpha)))

test_dataloader = dict(batch_size=8)

custom_hooks = [
    *_base_.custom_hooks,
    dict(type='VisualCacheLoadHook',
         cache_path=visual_cache_path,
         priority=99),
]
