CUR_INTRODUCED_CLS = 8
PREV_INTRODUCED_CLS = 0
_backend_args = None
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
affine_scale = 0.5
albu_train_transforms = [
    dict(p=0.01, type='Blur'),
    dict(p=0.01, type='MedianBlur'),
    dict(p=0.01, type='ToGray'),
    dict(p=0.01, type='CLAHE'),
]
analyze = 0
backend_args = None
base_lr = 0.001
batch_shapes_cfg = None
class_text_path = 'data/OWOD/ImageSets/IDD/t1_known.txt'
close_mosaic_epochs = 20
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=0,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114.0),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.5,
                    1.5,
                ),
                type='YOLOv5RandomAffine'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ],
                type='mmdet.Albu'),
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'yolo_world',
    ])
data_root = 'data/coco/'
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 1
default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=2,
        rule='greater',
        save_best=[
            'owod/Both',
        ],
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(
        lr_factor=0.01,
        max_epochs=20,
        scheduler_type='linear',
        type='YOLOv5ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
embedding_mask = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
]
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
fewshot_dir = 'data/OWOD/none'
fewshot_k = 0
fewshot_scope = 'novel'
fewshot_seed = 1
img_scale = (
    640,
    640,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
infer_type = 'one2one'
last_stage_out_channels = 512
last_transform = [
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
launcher = 'pytorch'
load_from = 'pretrained/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth'
log_interval = 5
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 7.5
loss_cls_weight = 0.5
loss_dfl_weight = 0.375
lr_factor = 0.01
max_aspect_ratio = 100
max_epochs = 20
max_keep_ckpts = 2
model = dict(
    anchor_embedding_path='embeddings/uniow-idd/object_tuned.npy',
    backbone=dict(
        frozen_stages=4,
        image_model=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            arch='P5',
            deepen_factor=1,
            last_stage_out_channels=512,
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            type='YOLOv10Backbone',
            use_c2fcib=True,
            widen_factor=1),
        text_model=None,
        type='MultiModalYOLOBackbone',
        with_text_model=False),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            embed_dims=512,
            featmap_strides=[
                8,
                16,
                32,
            ],
            freeze_one2many=True,
            freeze_one2one=True,
            in_channels=[
                256,
                512,
                512,
            ],
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=10,
            reg_max=16,
            type='YOLOv10WorldHeadModule',
            use_bn_head=True,
            widen_factor=1),
        infer_type='one2one',
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='YOLOv10WorldHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOv5DetDataPreprocessor'),
    embedding_mask=[
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
    ],
    embedding_path='embeddings/uniow-idd/idd_t1.npy',
    freeze_prompt=False,
    mm_neck=False,
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=1,
        freeze_all=True,
        in_channels=[
            256,
            512,
            512,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        out_channels=[
            256,
            512,
            512,
        ],
        type='YOLOv10PAFPN',
        widen_factor=1),
    num_prev_classes=0,
    num_prompts=10,
    num_test_classes=10,
    num_train_classes=10,
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=30000,
        one2many_withnms=True,
        one2one_withnms=False,
        score_thr=0.001,
        unknown_nms=dict(iou_threshold=0.99, score_threshold=0.2)),
    train_cfg=dict(
        anchor_label=dict(iou_threshold=0.5, score_threshold=0.03),
        one2many_assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=10,
            topk=10,
            type='BatchTaskAlignedAssigner',
            use_ciou=True),
        one2one_assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=10,
            topk=1,
            type='BatchTaskAlignedAssigner',
            use_ciou=True)),
    type='OWODDetector',
    unknown_embedding_path='embeddings/uniow-idd/object.npy')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.7, type='nms'),
    nms_pre=30000,
    one2many_withnms=True,
    one2one_withnms=False,
    score_thr=0.001)
neck_embed_channels = [
    128,
    256,
    256,
]
neck_num_heads = [
    4,
    8,
    8,
]
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')
num_classes = 10
num_det_layers = 3
num_training_classes = 10
one2many_tal_alpha = 0.5
one2many_tal_beta = 6.0
one2many_tal_topk = 10
one2one_tal_alpha = 0.5
one2one_tal_beta = 6.0
one2one_tal_topk = 1
optim_wrapper = dict(
    clip_grad=dict(max_norm=10.0),
    constructor='YOLOWv5OptimizerConstructor',
    loss_scale='dynamic',
    optimizer=dict(
        batch_size_per_gpu=8, lr=0.001, type='AdamW', weight_decay=0.025),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys=dict({
            'backbone.text_model': dict(lr_mult=1),
            'embeddings': dict(weight_decay=0.0),
            'logit_scale': dict(weight_decay=0.0)
        }),
        norm_decay_mult=0.0),
    type='AmpOptimWrapper')
owod_cfg = dict(
    CUR_INTRODUCED_CLS=8,
    PREV_INTRODUCED_CLS=0,
    num_classes=9,
    split='test',
    task_num=1)
owod_dataset = 'IDD'
owod_root = 'data/OWOD'
owod_settings = dict(
    FOOD_VOC=dict(task_list=[
        0,
        10,
        15,
    ], test_image_set='test'),
    FOOD_VOCCOCO=dict(task_list=[
        0,
        20,
        40,
    ], test_image_set='test'),
    IDD=dict(task_list=[
        0,
        8,
        14,
    ], test_image_set='test'),
    MOWODB=dict(
        task_list=[
            0,
            20,
            40,
            60,
            80,
        ], test_image_set='all_task_test'),
    SOWODB=dict(task_list=[
        0,
        19,
        40,
        60,
        80,
    ], test_image_set='test'),
    nuOWODB=dict(task_list=[
        0,
        10,
        17,
        23,
    ], test_image_set='test'))
owod_task = 1
owod_train_dataset = dict(
    class_text_path='data/OWOD/ImageSets/IDD/t1_known.txt',
    dataset=dict(
        data_root='data/OWOD',
        dataset='IDD',
        fewshot_dir='data/OWOD/none',
        fewshot_k=0,
        fewshot_scope='novel',
        fewshot_seed=1,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        image_set='train',
        owod_cfg=dict(
            CUR_INTRODUCED_CLS=8,
            PREV_INTRODUCED_CLS=0,
            num_classes=9,
            split='test',
            task_num=1),
        training_strategy=0,
        type='OWODDataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            img_scale=(
                640,
                640,
            ),
            pad_val=114.0,
            pre_transform=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            type='Mosaic'),
        dict(
            border=(
                -320,
                -320,
            ),
            border_val=(
                114,
                114,
                114,
            ),
            max_aspect_ratio=100,
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            scaling_ratio_range=(
                0.5,
                1.5,
            ),
            type='YOLOv5RandomAffine'),
        dict(
            bbox_params=dict(
                format='pascal_voc',
                label_fields=[
                    'gt_bboxes_labels',
                    'gt_ignore_flags',
                ],
                type='BboxParams'),
            keymap=dict(gt_bboxes='bboxes', img='image'),
            transforms=[
                dict(p=0.01, type='Blur'),
                dict(p=0.01, type='MedianBlur'),
                dict(p=0.01, type='ToGray'),
                dict(p=0.01, type='CLAHE'),
            ],
            type='mmdet.Albu'),
        dict(type='YOLOv5HSVRandomAug'),
        dict(prob=0.5, type='mmdet.RandomFlip'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'flip',
                'flip_direction',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='MultiModalOWDataset')
owod_val_dataset = dict(
    class_text_path='data/OWOD/ImageSets/IDD/t1_known.txt',
    dataset=dict(
        data_root='data/OWOD',
        dataset='IDD',
        image_set='test',
        owod_cfg=dict(
            CUR_INTRODUCED_CLS=8,
            PREV_INTRODUCED_CLS=0,
            num_classes=9,
            split='test',
            task_num=1),
        test_mode=True,
        type='OWODDataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(scale=(
            640,
            640,
        ), type='YOLOv5KeepRatioResize'),
        dict(
            allow_scale_up=False,
            pad_val=dict(img=114),
            scale=(
                640,
                640,
            ),
            type='LetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='MultiModalOWDataset')
owod_val_evaluator = dict(
    data_root='data/OWOD',
    dataset_name='IDD',
    owod_cfg=dict(
        CUR_INTRODUCED_CLS=8,
        PREV_INTRODUCED_CLS=0,
        num_classes=9,
        split='test',
        task_num=1),
    save_rets=False,
    threshold=0.05,
    type='OpenWorldMetric',
    unknown_threshold=0.0)
param_scheduler = None
persistent_workers = True
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
resume = False
save_epoch_intervals = 5
save_rets = False
strides = [
    8,
    16,
    32,
]
task_list = [
    0,
    8,
    14,
]
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path='data/OWOD/ImageSets/IDD/t1_known.txt',
        dataset=dict(
            data_root='data/OWOD',
            dataset='IDD',
            image_set='test',
            owod_cfg=dict(
                CUR_INTRODUCED_CLS=8,
                PREV_INTRODUCED_CLS=0,
                num_classes=9,
                split='test',
                task_num=1),
            test_mode=True,
            type='OWODDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalOWDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    data_root='data/OWOD',
    dataset_name='IDD',
    owod_cfg=dict(
        CUR_INTRODUCED_CLS=8,
        PREV_INTRODUCED_CLS=0,
        num_classes=9,
        split='test',
        task_num=1),
    save_rets=False,
    threshold=0.05,
    type='OpenWorldMetric',
    unknown_threshold=0.0)
test_image_set = 'test'
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]
text_channels = 512
threshold = 0.05
train_ann_file = 'annotations/instances_train2017.json'
train_batch_size_per_gpu = 8
train_cfg = dict(
    dynamic_intervals=[
        (
            0,
            5,
        ),
    ],
    max_epochs=20,
    type='EpochBasedTrainLoop',
    val_interval=5)
train_data_prefix = 'train2017/'
train_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        class_text_path='data/OWOD/ImageSets/IDD/t1_known.txt',
        dataset=dict(
            data_root='data/OWOD',
            dataset='IDD',
            fewshot_dir='data/OWOD/none',
            fewshot_k=0,
            fewshot_scope='novel',
            fewshot_seed=1,
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            image_set='train',
            owod_cfg=dict(
                CUR_INTRODUCED_CLS=8,
                PREV_INTRODUCED_CLS=0,
                num_classes=9,
                split='test',
                task_num=1),
            training_strategy=0,
            type='OWODDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                ],
                type='Mosaic'),
            dict(
                border=(
                    -320,
                    -320,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.5,
                    1.5,
                ),
                type='YOLOv5RandomAffine'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ],
                type='mmdet.Albu'),
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalOWDataset'),
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_image_set = 'train'
train_num_workers = 12
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='Mosaic'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
training_strategy = 0
tta_model = dict(
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.65, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
unknown_threshold = 0.0
val_ann_file = 'annotations/instances_val2017.json'
val_batch_size_per_gpu = 1
val_cfg = dict(type='ValLoop')
val_data_prefix = 'val2017/'
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path='data/OWOD/ImageSets/IDD/t1_known.txt',
        dataset=dict(
            data_root='data/OWOD',
            dataset='IDD',
            image_set='test',
            owod_cfg=dict(
                CUR_INTRODUCED_CLS=8,
                PREV_INTRODUCED_CLS=0,
                num_classes=9,
                split='test',
                task_num=1),
            test_mode=True,
            type='OWODDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalOWDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    data_root='data/OWOD',
    dataset_name='IDD',
    owod_cfg=dict(
        CUR_INTRODUCED_CLS=8,
        PREV_INTRODUCED_CLS=0,
        num_classes=9,
        split='test',
        task_num=1),
    save_rets=False,
    threshold=0.05,
    type='OpenWorldMetric',
    unknown_threshold=0.0)
val_interval = 5
val_interval_stage2 = 5
val_num_workers = 8
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.025
widen_factor = 1
work_dir = 'model_weights/t1/idd_t1_tobj0p03'
