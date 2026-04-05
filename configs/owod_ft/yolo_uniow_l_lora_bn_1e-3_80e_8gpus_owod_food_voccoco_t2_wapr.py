_base_ = ['yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voccoco_t2.py']

# FOOD VOC-COCO — T2 WAPR config.
# Keep the same 80-epoch schedule as the plain T2 run and only enable
# WAPR-specific pseudo-label redistribution plus T_unk anchoring.
model = dict(
    wapr=dict(
        frozen_embedding_path='embeddings/uniow-food-voccoco/food_voccoco_t2.npy',
        num_known_classes=40,
        warmup_epochs=1,
        anchor_loss_weight=0.05,
    ),
)