_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py']

# ── Training duration ──────────────────────────────────────────────────────
# 60 images (10-shot × 6 classes), batch_size=8 → ~8 steps/epoch.
# 200 epochs → ~1600 iters (closer to TFA's 12K iter few-shot regime).
# Previous 40 epochs gave only ~320 steps — severe underfitting.
max_epochs = 200
close_mosaic_epochs = max_epochs  # mosaic already disabled from start in base config
save_epoch_intervals = 50
val_interval = 50
val_interval_stage2 = 50

# ── Learning rate ──────────────────────────────────────────────────────────
# Only 7 embedding vectors × 512 dims = 3584 trainable params.
# 10x higher lr (0.01 vs 1e-3) is safe and necessary for convergence.
base_lr = 0.01

# ── LR schedule: constant after warmup ────────────────────────────────────
# The inherited YOLOv5ParamSchedulerHook uses linear decay (lr_factor=0.01),
# which kills the lr before embeddings converge in few-shot.
# Setting lr_factor=1.0 makes it constant: lr(x) = 1.0 for all epochs.
# warmup_epochs=3 (default in hook) provides a short ramp-up.
default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs,
                         lr_factor=1.0),         # constant lr (no decay)
    checkpoint=dict(interval=save_epoch_intervals,
                    save_best=['owod/Both'],
                    rule='greater'))

train_cfg = dict(max_epochs=max_epochs,
                 val_interval=val_interval,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     val_interval_stage2)])

# ── EMA: faster update for few-shot ───────────────────────────────────────
# momentum=0.0001 with 320 steps → 0.9999^320 ≈ 0.969 retention of T1 init.
# The EMA model barely moves, so evaluation (which uses EMA by default)
# sees essentially the T1 model. This alone explains cos_ctrd ≈ 0 for novel.
#
# With momentum=0.005 and 1600 steps → 0.995^1600 ≈ 0.0003 retention.
# EMA fully tracks the learned novel embeddings.
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.005,           # was 0.0001 (50x increase)
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline={{_base_.train_pipeline_stage2}})
]

# ── Optimizer override ─────────────────────────────────────────────────────
# Raise base_lr to 0.01 and zero weight_decay on embeddings.
optim_wrapper = dict(
    optimizer=dict(lr=base_lr, weight_decay=0.025),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone.text_model': dict(lr_mult=1),
            'logit_scale': dict(weight_decay=0.0),
            'embeddings': dict(lr_mult=1.0, weight_decay=0.0),
        }),
)

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
