_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py']

# T2 fine-tuning: 80 epochs, 10-shot, no WAPR, no SCPI.
# Load from T1 best checkpoint; base-class embeddings are frozen via
# embedding_mask (PREV=8 frozen, CUR=6 trainable, unk+anchor trainable).
# LR scheduler is linear decay — max_epochs=80 scales it correctly.
# val/save every 20 epochs → checkpoints at 20, 40, 60, 80.
max_epochs = 80
close_mosaic_epochs = max_epochs  # mosaic disabled from start (same as base)
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
