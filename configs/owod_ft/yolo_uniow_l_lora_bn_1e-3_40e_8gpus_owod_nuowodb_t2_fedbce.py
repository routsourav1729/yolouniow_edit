_base_ = ['yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_nuowodb_t2.py']

# T2 ablation 1: anchor-aware federated BCE only.
# Supervise current-class positives on their own class channel, use k-shot
# base-class positives as safe negatives for current-class channels, and mask
# all other classification gradients.
model = dict(
    train_cfg=dict(
        fed_bce=dict(
            enabled=True,
            num_prev_classes=10,
            num_known_classes=17,
            include_base_negatives=True,
        ),
    ),
)
