_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod.py']

# ============================================================================
# nuOWODB  Task-2  Few-Shot Fine-tuning  (TFA / CED-FOOD style)
# ============================================================================
# Inherits the default OWOD config (same backbone/neck/head architecture).
# DATASET=nuOWODB TASK=2 → env vars set PREV_INTRODUCED_CLS=10, CUR_INTRODUCED_CLS=7
# embedding_mask is already correct from base:
#   [0]*10 (T1 frozen) + [1]*7 (T2 trainable) + [1] (unknown) + [0] (anchor frozen)
#
# Differences vs default T1 config:
#   - anchor_embedding_path: nuOWODB-specific anchor extracted from T1 ckpt
#   - max_epochs: 40  (IDD-style: small fewshot set needs more epochs)
#   - val/save intervals: every 10 epochs
#   - load_from: injected by sbatch via --cfg-options load_from=<t1_best.pth>
#
# DO NOT use this config for default OWOD incremental T2 (full t2_train).
# That uses yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod.py with TASK=2, no FEWSHOT_K.
# ============================================================================

# Embeddings: known classes from CLIP (pretrained, read-only .npy loaded at init)
# nuowodb_t2.npy was generated with human-readable prompts (Table 11):
#   [bicycle, motorcycle, car, ..., adult, child, ..., wheelchair, ...]
# anchor: nuOWODB-specific T_anchor extracted from T1-trained checkpoint
model = dict(
    embedding_path='embeddings/uniow-l/nuowodb_t2.npy',
    unknown_embedding_path='embeddings/uniow-l/object.npy',
    anchor_embedding_path='embeddings/uniow-l/nuowodb_object_tuned.npy',
)

# 40 epochs — nuOWODB 10-shot seed1 has ~50-130 fewshot images (similar to IDD)
max_epochs = 40
close_mosaic_epochs = max_epochs
val_interval = 10
val_interval_stage2 = 10

default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(
        interval=val_interval,
        save_best=['owod/Both'],
        rule='greater',
    ),
)

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=val_interval,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), val_interval_stage2)],
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49,
    ),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline={{_base_.train_pipeline_stage2}},
    ),
]
