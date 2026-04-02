_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py']

# SCPI: post-training calibration for T2.
#
# Runner loads T2 checkpoint (backbone, neck, heads, base/unk/anchor embeddings).
# SCPIHook patches ONLY novel embeddings (slots 8-13) from the SCPI npy.
# Base (0-7), unknown (14), anchor (15) stay from the T2 checkpoint.
#
# Usage:
#   DATASET=IDD TASK=2 ... ./tools/dist_test.sh <this_config> <T2_ckpt> 1 \
#     --cfg-options custom_hooks.2.scpi_emb_path=<scpi_npy_path>

custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=0,
         switch_pipeline={{_base_.train_pipeline_stage2}}),
    dict(type='SCPIHook',
         scpi_emb_path='embeddings/uniow-idd/idd_t2_scpi.npy',
         priority=51),
]

max_epochs = 0
