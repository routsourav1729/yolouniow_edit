_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py']

# IDD T2 WAPR — 40-epoch schedule.
# IDD has only ~60 training images (10-shot × 6 novel classes), so it
# needs more epochs than FOOD_VOCCOCO. Metrics still rising at ep20;
# keep 40 epochs and evaluate every 10.
max_epochs = 40
close_mosaic_epochs = max_epochs
val_interval = 10
val_interval_stage2 = 10

default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(interval=10, save_best=['owod/Both'], rule='greater'))

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
# Uses score-ratio (max_known_score / anchor_score) as the discriminative signal.
# IDD: 8 base + 6 novel = 14 total known classes.
model = dict(
    wapr=dict(
        frozen_embedding_path='embeddings/uniow-idd/idd_t2.npy',
        num_known_classes=14,
        warmup_epochs=1,
        anchor_loss_weight=0.05,
    ),
    concept_negation=dict(
        alpha=1.0,
    ),
    gasdl=dict(
        weight=1.0,
        temperature=5.0,
        include_unknown=True,
    ),
)
