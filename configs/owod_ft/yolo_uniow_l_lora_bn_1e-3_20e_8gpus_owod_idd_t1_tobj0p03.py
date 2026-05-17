_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py']

# T1 pseudo-unknown mining: IoU < 0.5 and Tobj > 0.03.
# Everything else stays inherited from the original IDD T1 config.
model = dict(
    train_cfg=dict(
        anchor_label=dict(iou_threshold=0.5, score_threshold=0.03)))
