_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voccoco.py']

# FOOD VOC-COCO — T2 few-shot fine-tuning config (no WAPR).
# Scaling rationale vs FOOD_VOC T2:
#   FOOD_VOC:     5 novel classes, ~50 fewshot instances → 20 epochs, val/5
#   FOOD_VOCCOCO: 20 novel classes, ~200 fewshot instances → 80 epochs, val/10
#   4× more data → 4× more epochs (matches MOWODB T2 protocol: 20 novel, 80ep)
#   val every 10 → 8 checkpoints (same density as FOOD_VOC's 20/5=4 per 20ep)
max_epochs = 80
close_mosaic_epochs = max_epochs
save_epoch_intervals = 10
val_interval = 10
val_interval_stage2 = 10

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
