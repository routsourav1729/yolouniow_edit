_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py']

# T2 fine-tuning: 20 epochs, 10-shot, no WAPR, no SCPI.
# Load from T1 best checkpoint; base-class embeddings are frozen via
# embedding_mask (PREV=8 frozen, CUR=6 trainable, unk+anchor trainable).
max_epochs = 20
close_mosaic_epochs = max_epochs
val_interval = 5
val_interval_stage2 = 5

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
