_base_ = ['yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_fedbce.py']

# T2 ablation 5:
# - Give bus/truck full BCE background negatives to suppress over-selection.
# - Do not use pseudo-unknown anchors as "not bus/not truck" supervision.
# - Keep T_unk fully unmasked and keep T_unk-only hard negatives active.
hardneg_cache_path = 'cache/idd/hardneg_t1_o2o_10x8x3.pt'

model = dict(
    train_cfg=dict(
        fed_bce=dict(
            enabled=True,
            num_prev_classes=8,
            num_known_classes=14,
            include_base_negatives=True,
            class_unmask_weights=[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 0.0,
            ],
            mask_classes_on_tunk_pseudo_positive=[8, 9],
            unk_idx=14,
        ),
    ),
    hardneg=dict(
        cache_path=hardneg_cache_path,
        target_unk=True,
        target_novel=False,
        weight=1.0,
    ),
)
