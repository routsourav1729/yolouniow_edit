_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py']

# IDD T2 with KUME k-vs-k ONLY (no WAPR, no k-vs-unk).
# Tests whether all-pairs correct-known vs other-knowns margin reduces
# inter-known stealing on the probe.
import os

max_epochs = 40
close_mosaic_epochs = max_epochs
val_interval = 10
val_interval_stage2 = 10

default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(interval=val_interval,
                    save_best=['owod/Both'],
                    rule='greater'))

train_cfg = dict(max_epochs=max_epochs,
                 val_interval=val_interval,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     val_interval_stage2)])

custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline={{_base_.train_pipeline_stage2}})
]

# KUME k-vs-k only — no WAPR.
# Margin sweep via env var KUME_KVK_MARGIN (default 0.5).
_margin = float(os.environ.get('KUME_KVK_MARGIN', '0.5'))

# IDD: 8 base + 6 novel = 14 total known classes.
# T_unk is at index 14 in the (K+2) logit tensor.
model = dict(
    kume=dict(
        num_known_classes=14,
        unk_idx=14,
        margin=_margin,
        weight=0.5,
    ),
    # NOTE: wapr is intentionally absent. KUME runs alone.
)
