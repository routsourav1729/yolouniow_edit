_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py']

# IDD T2 with KUME (Known-Unknown Margin Enforcement) — 40 epochs.
# IDD has only ~60 training images (10-shot × 6 novel classes), so it
# needs more epochs. Metrics typically rise through epoch 40. Eval every 10.
#
# KUME adds explicit inter-channel separation between correct-class logit
# and competing unknown logit at each anchor. This enforces the separation
# that BCE cannot provide independently per channel.
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

# KUME: Known-Unknown Margin Enforcement + WAPR together.
# IDD: 8 base + 6 novel = 14 total known classes.
# T_unk is at index 14 in the (K+2) logit tensor.
# Margin in logit space: logit_unk must be at least m below correct class
# for known anchors, and at least m above best known class for unknown anchors.
model = dict(
    kume=dict(
        num_known_classes=14,  # base (8) + novel (6)
        unk_idx=14,             # absolute index of T_unk in K+2 logit tensor
        margin=1.0,             # logit-space separation margin
        weight=0.5,             # loss weight
    ),
    wapr=dict(
        frozen_embedding_path='embeddings/uniow-idd/idd_t2.npy',
        num_known_classes=14,
        warmup_epochs=1,
        anchor_loss_weight=0.05,
        ratio_threshold=0.5,
    ),
)
