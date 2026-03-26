"""Eval fine-tuned YOLO-UniOW-L (t2 10-shot WAPR) with fresh CLIP prompts.

Same checkpoint and 16-slot layout as the trained model.
CLIPEmbedHook fires after_load_checkpoint and overwrites slots 0..13
with idd_all.npy (fresh CLIP embeddings), so trained prompts are replaced.
Slots -2 (unk) and -1 (anchor) stay from the checkpoint.
"""

_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_wapr.py']

custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='CLIPEmbedHook',
         clip_emb_path='embeddings/uniow-l/idd_all.npy',
         priority=51),  # must run AFTER EMAHook (49) which swaps EMA weights
]
