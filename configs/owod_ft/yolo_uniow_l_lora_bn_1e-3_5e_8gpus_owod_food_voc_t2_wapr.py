_base_ = ['yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voc_t2.py']

# FOOD VOC10-5-5 — T2 WAPR config.
# 20 epochs with val every 5 to track WAPR effect over time.
max_epochs = 20
close_mosaic_epochs = max_epochs
save_epoch_intervals = 5
val_interval = 5
val_interval_stage2 = 5

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
# FOOD_VOC T2 has 10 base + 5 novel = 15 known classes.
model = dict(
    wapr=dict(
        frozen_embedding_path='embeddings/uniow-food-voc/food_voc_t2.npy',
        num_known_classes=15,
        warmup_epochs=1,
        anchor_loss_weight=0.05,
        ratio_threshold=0.5,
    ),
)
