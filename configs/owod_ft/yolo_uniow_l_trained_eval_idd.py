"""Eval the fine-tuned YOLO-UniOW-L (task2 10-shot WAPR) on all 14 IDD classes.

Inherits the training config (16-slot layout: 8 base + 6 novel + unk + anchor).
Prompts come from the checkpoint — NOT re-extracted from CLIP.
All 14 known classes treated as current (PREV=0, CUR=14) for a flat eval.

Run:
    DATASET=IDD TASK=2 THRESHOLD=0.05 FEWSHOT_K=0 FEWSHOT_DIR=none \
    FEWSHOT_SEED=1 IMAGESET=train TRAINING_STRATEGY=0 SAVE=True ANALYZE=0 \
    ./tools/dist_test.sh configs/owod_ft/yolo_uniow_l_trained_eval_idd.py \
        work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task2_10shot_wapr/best_owod_Both_epoch_40.pth 1 \
        --work-dir work_dirs/trained_eval_idd_l
"""

_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_wapr.py']

# ── Eval-only overrides ────────────────────────────────────────────────────
# Keep the 16-slot layout so it matches the checkpoint exactly.
# PREV=0, CUR=14: treat all 14 known as "current" for a clean all-class eval.

_eval_owod_cfg = dict(
    split='test',
    task_num=2,
    PREV_INTRODUCED_CLS=0,
    CUR_INTRODUCED_CLS=14,
    num_classes=15,          # 14 known + 1 unk (for metric labels)
)

_eval_class_text_path = 'data/OWOD/ImageSets/IDD/t_all_known.txt'

# Prompts come from checkpoint; embedding_path only used if prompts are absent.
# We leave embedding_mask all-zero so no gradients flow (eval is frozen anyway).
embedding_mask = [0] * 16

model = dict(
    embedding_mask=embedding_mask,
    freeze_prompt=False,          # DDP needs >=1 param with requires_grad
    # Do NOT override embedding_path — use what's in the checkpoint
)

val_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='MultiModalOWDataset',
        dataset=dict(
            type='OWODDataset',
            data_root='data/OWOD',
            image_set='test',
            dataset='IDD',
            owod_cfg=_eval_owod_cfg,
            test_mode=True,
        ),
        class_text_path=_eval_class_text_path,
        pipeline={{_base_.test_pipeline}},
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='OpenWorldMetric',
    data_root='data/OWOD',
    dataset_name='IDD',
    threshold=0.05,
    save_rets=False,
    owod_cfg=_eval_owod_cfg,
)
test_evaluator = val_evaluator
