_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voc.py']

# FOOD VOC10-5-5 — T2 few-shot fine-tuning config.
# 20 epochs max — U-Recall collapses after epoch 20 (73.6% @ e20 vs 31% @ e40).
# Validate at 5, 10, 15, 20 to find the best U-Recall checkpoint.
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
