_base_ = ['yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voccoco_t2.py']

# FOOD VOC-COCO — T2 WAPR config.
# 20-epoch schedule (model peaks around ep10, overfits beyond ep20).
# Evaluate every 5 epochs to track the curve.
max_epochs = 20
close_mosaic_epochs = max_epochs
val_interval = 5
val_interval_stage2 = 5

train_cfg = dict(max_epochs=max_epochs,
                 val_interval=val_interval,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     val_interval_stage2)])

default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(interval=5, save_best=['owod/Both'], rule='greater'))

model = dict(
    wapr=dict(
        frozen_embedding_path='embeddings/uniow-food-voccoco/food_voccoco_t2.npy',
        num_known_classes=40,
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