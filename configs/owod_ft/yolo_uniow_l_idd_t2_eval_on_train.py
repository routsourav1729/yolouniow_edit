_base_ = ['yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2.py']

# Quick experiment: evaluate T2 finetuned model on the few-shot training data
# itself (the same 10-shot/seed1 split used for T2 finetuning) to inspect
# class confusion between base and novel classes after few-shot tuning.
#
# Predictions: model runs on the ~54 few-shot images; per-image GT seen by the
# dataset is restricted to the per-class fewshot file (image listed under
# class C contributes only C's annotations -> ~140 boxes total for IDD).
# Evaluator GT: loaded directly from XMLs of the same images via
# t2_train_10shot.txt -- not capped to 140, see /home/.../scripts/experiment.

# --- val/test dataloader: point at fewshot training images, test_mode=True ---
_fewshot_val_dataset = dict(
    type='MultiModalOWDataset',
    dataset=dict(
        type='OWODDataset',
        data_root=_base_.owod_root,
        image_set='train',
        dataset=_base_.owod_dataset,
        owod_cfg=_base_.owod_cfg,
        training_strategy=1,  # oracle: keep base+novel labels, mark others unk
        fewshot_dir=_base_.fewshot_dir,
        fewshot_k=_base_.fewshot_k,
        fewshot_seed=_base_.fewshot_seed,
        test_mode=True),
    class_text_path=_base_.class_text_path,
    pipeline=_base_.test_pipeline,
)

val_dataloader = dict(dataset={'_delete_': True, **_fewshot_val_dataset})
test_dataloader = val_dataloader

# --- evaluator: load GT from the 10-shot training image list ---
_eval_owod_cfg = dict(
    split='t2_train_10shot',
    task_num=_base_.owod_task,
    PREV_INTRODUCED_CLS=_base_.PREV_INTRODUCED_CLS,
    CUR_INTRODUCED_CLS=_base_.CUR_INTRODUCED_CLS,
    num_classes=_base_.PREV_INTRODUCED_CLS + _base_.CUR_INTRODUCED_CLS + 1,
)

val_evaluator = dict(
    _delete_=True,
    type='OpenWorldMetric',
    data_root=_base_.owod_root,
    dataset_name=_base_.owod_dataset,
    threshold=_base_.threshold,
    save_rets=_base_.save_rets,
    owod_cfg=_eval_owod_cfg,
)
test_evaluator = val_evaluator
