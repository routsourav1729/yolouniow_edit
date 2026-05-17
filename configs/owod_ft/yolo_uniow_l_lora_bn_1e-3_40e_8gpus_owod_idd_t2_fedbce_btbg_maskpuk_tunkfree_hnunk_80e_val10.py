_base_ = ['yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_fedbce_btbg_maskpuk_tunkfree_hnunk.py']

# Combined FedBCE + HardNeg IDD T2 sweep: train 80 epochs and
# validate/checkpoint every 10, matching the vanilla 80e comparison run.
max_epochs = 80
val_interval = 10

default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(interval=val_interval,
                    save_best=['owod/Both'],
                    rule='greater'))

train_cfg = dict(max_epochs=max_epochs,
                 val_interval=val_interval,
                 dynamic_intervals=[(0, val_interval)])
