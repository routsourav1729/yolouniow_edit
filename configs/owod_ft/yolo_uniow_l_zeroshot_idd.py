"""Zero-shot eval: pretrained YOLO-UniOW-L on IDD, all 14 classes as known.

Uses fresh CLIP embeddings (idd_all.npy) — NOT fine-tuned OWOD embeddings.
PREV=0, CUR=14 so every class is 'current known', nothing is unknown.

Run:
    DATASET=IDD TASK=2 THRESHOLD=0.05 FEWSHOT_K=0 FEWSHOT_DIR=none \
    FEWSHOT_SEED=1 IMAGESET=train TRAINING_STRATEGY=0 SAVE=True ANALYZE=0 \
    ./tools/dist_test.sh configs/owod_ft/yolo_uniow_l_zeroshot_idd.py \
        pretrained/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth 1 \
        --work-dir work_dirs/zeroshot_idd_l
"""

_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod.py']

# ── Override: all 14 IDD classes are "current known" ──────────────────────
# The base config computes num_classes from PREV + CUR + 2, but those come
# from env-var TASK.  We override everything that depends on the class split.

num_classes = 14 + 2              # 14 known + 1 unk + 1 anchor
num_training_classes = num_classes

# All embeddings frozen — zero-shot, no gradient
embedding_mask = [0] * num_classes

model = dict(
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    num_prev_classes=0,
    num_prompts=num_classes,
    freeze_prompt=False,           # DDP requires at least one param with requires_grad
    embedding_path='embeddings/uniow-l/idd_all.npy',       # fresh CLIP embeds for all 14
    unknown_embedding_path='embeddings/uniow-l/object.npy', # wildcard "object"
    anchor_embedding_path='embeddings/uniow-l/object.npy',  # reuse wildcard as anchor
    embedding_mask=embedding_mask, # all zeros → hook zeroes all grads, nothing updates
    bbox_head=dict(head_module=dict(num_classes=num_training_classes)),
    train_cfg=dict(one2many_assigner=dict(num_classes=num_training_classes),
                   one2one_assigner=dict(num_classes=num_training_classes)),
)

# ── Override dataset/evaluator owod_cfg: PREV=0, CUR=14 ──────────────────
_zs_owod_cfg = dict(
    split='test',
    task_num=2,
    PREV_INTRODUCED_CLS=0,
    CUR_INTRODUCED_CLS=14,
    num_classes=15,  # 14 known + 1 unk (for metric labels)
)

_zs_class_text_path = 'data/OWOD/ImageSets/IDD/t_all_known.txt'

owod_val_dataset = dict(
    _delete_=True,
    type='MultiModalOWDataset',
    dataset=dict(type='OWODDataset',
                 data_root='data/OWOD',
                 image_set='test',
                 dataset='IDD',
                 owod_cfg=_zs_owod_cfg,
                 test_mode=True),
    class_text_path=_zs_class_text_path,
    pipeline=_base_.test_pipeline)

val_dataloader = dict(dataset=owod_val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='OpenWorldMetric',
    data_root='data/OWOD',
    dataset_name='IDD',
    threshold=0.05,
    save_rets=False,
    owod_cfg=_zs_owod_cfg,
)
test_evaluator = val_evaluator
