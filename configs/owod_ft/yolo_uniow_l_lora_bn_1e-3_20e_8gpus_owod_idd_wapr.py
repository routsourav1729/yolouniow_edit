_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py']

# Extended training: 20 → 40 epochs.
# With only ~60 training images (10-shot × 6 novel classes), the 20-epoch
# run shows metrics still rising at the end — clear underfitting.
max_epochs = 40
close_mosaic_epochs = max_epochs  # mosaic already disabled from start in base config
save_epoch_intervals = 20
val_interval = 20
val_interval_stage2 = 20

default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(interval=save_epoch_intervals,
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

# WAPR: Wildcard-Aware Pseudo-label Redistribution
# Only active during T2 (few-shot fine-tuning).
# Uses model's own BNContrastiveHead logits to detect known-class pseudo-unknowns
# and redirect their pseudo-label signal to novel class channels.
model = dict(
    wapr=dict(
        frozen_embedding_path='embeddings/uniow-idd/idd_t2.npy',
        num_known_classes=14,      # 8 base + 6 novel (IDD T2)
        warmup_epochs=1,           # reduced from 2→1: only 2.5% of training wasted
        anchor_loss_weight=0.1,    # lambda for T_unk drift L2 loss
    ),
)
