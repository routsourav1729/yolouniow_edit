_base_ = [
    'yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_fedbce_btbg_maskpuk_tunkfree_hnunk.py'
]

# Stage experiment:
# - no HardNeg anywhere; the last run made AOSE worse, so remove that pressure.
# - epochs 1..20: train T_unk with the vanilla pseudo-unknown rule.
# - epochs 21..40: freeze novel prompt rows 8..13 and keep training T_unk.
# - on pseudo-unknown anchors, mask only novel class gradients 8..13.
#   Background anchors still provide normal novel-class negative gradients.
# - keep the all-known 10-shot cap, i.e. 14 classes * 10 boxes = 140 boxes.
max_epochs = 40
val_interval = 10

model = dict(
    train_cfg=dict(
        anchor_label=dict(
            iou_threshold=0.5,
            score_threshold=0.01,
            num_known_classes=14,
        ),
        fed_bce=dict(
            mask_classes_on_tunk_pseudo_positive=list(range(8, 14)),
        )),
    hardneg=None)

val_dataloader = dict(batch_size=4, num_workers=8)
test_dataloader = val_dataloader
train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            fewshot_cap_all_classes=True,
        )))

default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(interval=val_interval,
                    save_best=['owod/Both'],
                    rule='greater'))

custom_hooks = [
    *_base_.custom_hooks,
    dict(type='OWODStageControlHook',
         switch_epoch=20,
         freeze_embedding_rows=list(range(8, 14)),
         train_embedding_rows=[14],
         priority=48),
]

train_cfg = dict(max_epochs=max_epochs,
                 val_interval=val_interval,
                 dynamic_intervals=[(0, val_interval)])
