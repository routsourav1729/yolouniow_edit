_base_ = ['../owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2.py']

# Vanilla T2 prompt tuning, but initialize current/novel prompt rows from
# visual-cache prototypes instead of CLIP text embeddings.

visual_prompt_embedding_path = '{{$VIS_PROMPT_EMBED_PATH:embeddings/uniow-idd/idd_t2_visprompt_mean.npy}}'

model = dict(embedding_path=visual_prompt_embedding_path)
