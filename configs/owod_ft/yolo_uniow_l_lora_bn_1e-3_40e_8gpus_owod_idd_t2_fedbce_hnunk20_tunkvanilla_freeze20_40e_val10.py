_base_ = [
    'yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_fedbce_btbg_maskpuk_tunkfree_hnunk.py'
]

# Stage experiment:
# - epochs 1..20: match the strong HN-to-T_unk recipe from
#   idd_t2_fedbce_hn_freeze20_objgtknown_40e_val10 up to the switch point.
# - epochs 21..40: freeze novel prompt rows 8..13, disable HardNeg, and keep
#   vanilla T_unk tuning by masking all current-class pseudo-unknown negatives.
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
            mask_classes_on_tunk_pseudo_positive=[8, 9],
        )),
    hardneg=dict(
        cache_path='cache/idd/hardneg_t1_o2o_10x8x3.pt',
        target_unk=True,
        target_novel=False,
        weight=1.0,
    ))

val_dataloader = dict(batch_size=4, num_workers=8)
test_dataloader = val_dataloader

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
         disable_hardneg=True,
         fed_bce_updates=dict(
             mask_classes_on_tunk_pseudo_positive=list(range(8, 14))),
         priority=48),
]

train_cfg = dict(max_epochs=max_epochs,
                 val_interval=val_interval,
                 dynamic_intervals=[(0, val_interval)])
