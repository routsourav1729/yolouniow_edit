_base_ = ['yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2.py']

# T2 IDD fine-tuning with hard-negative contrastive loss.
# Uses T1-cached base-positive features (high T_unk score on base GT anchors)
# as hard negatives for both T_unk and novel-class prompts.

hardneg_cache_path = '{{$HARDNEG_CACHE_PATH:cache/idd/hardneg_t1_o2o_10x8x3.pt}}'
hardneg_weight = {{'$HARDNEG_WEIGHT:1.0'}}
hardneg_target_unk = {{'$HARDNEG_TARGET_UNK:True'}}
hardneg_target_novel = {{'$HARDNEG_TARGET_NOVEL:True'}}

model = dict(
    hardneg=dict(
        cache_path=hardneg_cache_path,
        # num_base_classes / num_known_classes / unk_idx default from
        # OWODDetector init (num_prev_classes and num_train_classes - 2).
        target_unk=hardneg_target_unk,
        target_novel=hardneg_target_novel,
        weight=hardneg_weight,
    ),
)
