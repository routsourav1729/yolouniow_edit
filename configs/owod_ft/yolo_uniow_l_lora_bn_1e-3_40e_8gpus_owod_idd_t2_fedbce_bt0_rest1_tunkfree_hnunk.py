_base_ = ['yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_fedbce.py']

# T2 ablation 4:
# - Keep novel-class masking from bt0_rest1:
#     bus/truck protected, confused novel rows get full BCE negatives.
# - For T_unk:
#     no FedBCE masking at all, so original pseudo-unknown positives,
#     background negatives, and known-positive negatives are all active.
# - Add T1 cached base-positive hard negatives targeting T_unk only.
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
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 0.0,
            ],
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
