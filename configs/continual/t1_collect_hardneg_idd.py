_base_ = ['../owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py']

# Frozen/no-step pass over T1/base train annotations to build a compact O2O
# base-positive cache ranked by T_unk score.

max_epochs = 1
hardneg_cache_path = '{{$HARDNEG_CACHE_PATH:cache/idd/hardneg_t1_o2o_10x8x3.pt}}'
cache_image_set = '{{$CACHE_IMAGE_SET:train}}'

owod_train_dataset = dict(**_base_.owod_train_dataset)
owod_train_dataset['dataset'] = dict(**_base_.owod_train_dataset['dataset'])
owod_train_dataset['dataset']['image_set'] = cache_image_set

train_dataloader = dict(dataset=owod_train_dataset)

default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(interval=999))

train_cfg = dict(max_epochs=max_epochs, val_interval=999)

val_dataloader = dict(dataset=owod_train_dataset)
test_dataloader = dict(dataset=owod_train_dataset)

custom_hooks = [
    dict(type='HardNegativeCacheHook',
         cache_path=hardneg_cache_path,
         topk_per_class_scale=10,
         num_base_classes=8,
         duplicate_iou_thr=0.9,
         min_unknown_score=0.0,
         no_step_collect=True,
         stop_when_full=False,
         freeze_bn=True,
         priority=48),
]
