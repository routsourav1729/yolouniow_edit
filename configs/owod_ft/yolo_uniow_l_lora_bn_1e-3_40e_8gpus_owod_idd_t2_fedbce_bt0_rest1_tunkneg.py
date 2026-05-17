_base_ = ['yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_fedbce.py']

# T2 ablation 2: soft-unmask current-class BCE.
# Weight meaning:
#   0.0 -> keep aggressive FedBCE masking for this class
#   1.0 -> restore full BCE for this class channel
#
# IDD order:
#   0 car, 1 motorcycle, 2 rider, 3 person, 4 autorickshaw, 5 bicycle,
#   6 traffic sign, 7 traffic light,
#   8 bus, 9 truck, 10 tanker_vehicle, 11 crane_truck,
#   12 street_cart, 13 excavator, 14 T_unk, 15 T_anchor
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
                0.0, 0.0,
            ],
            tunk_known_negative=True,
            unk_idx=14,
        ),
    ),
)
