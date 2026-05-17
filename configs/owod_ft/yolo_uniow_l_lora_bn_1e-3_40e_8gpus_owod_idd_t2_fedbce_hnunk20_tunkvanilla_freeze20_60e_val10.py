_base_ = [
    'yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_fedbce_hnunk20_tunkvanilla_freeze20_40e_val10.py'
]

# Same staged recipe as the 40e run:
# - epochs 1..20: HardNeg targets T_unk, matching the strong old run.
# - epochs 21..60: disable HardNeg, freeze novel prompt rows, tune T_unk
#   with the vanilla pseudo-unknown masking rule.
max_epochs = 60
val_interval = 10

default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(interval=val_interval,
                    save_best=['owod/Both'],
                    rule='greater'))

train_cfg = dict(max_epochs=max_epochs,
                 val_interval=val_interval,
                 dynamic_intervals=[(0, val_interval)])
