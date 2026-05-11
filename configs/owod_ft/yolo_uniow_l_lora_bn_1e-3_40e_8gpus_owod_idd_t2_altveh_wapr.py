_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_wapr.py']

# IDD T2 prompt-name experiment.
# Dataset labels/eval names stay unchanged:
#   bus, truck, tanker_vehicle, crane_truck, street_cart, excavator
# Only the CLIP initialization names for the known prompts change. This tests
# whether using more pretraining-natural vehicle names gives better T2 prompt
# tuning than dataset-specific underscored names.

model = dict(
    embedding_path='embeddings/uniow-idd/idd_t2_altveh.npy',
    wapr=dict(
        frozen_embedding_path='embeddings/uniow-idd/idd_t2_altveh.npy',
        num_known_classes=14,
        warmup_epochs=1,
        anchor_loss_weight=0.05,
        ratio_threshold=0.5,
    ),
)
