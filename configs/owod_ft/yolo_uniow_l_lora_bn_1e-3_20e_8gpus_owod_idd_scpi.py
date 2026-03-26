_base_ = ['yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py']

# SCPI: Support-Calibrated Prompt Interpolation (training-free T2).
#
# Pre-calibrated embeddings are produced offline by:
#   python tools/owod_scripts/scpi_calibrate.py --output-npy <path>
#
# This config just loads the T1 checkpoint and patches novel embeddings
# from the pre-calibrated npy. No model forward passes during eval setup.
#
# Usage:
#   DATASET=IDD TASK=2 ... ./tools/dist_test.sh <this_config> <T1_ckpt> 1 \
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
         scpi_emb_path='embeddings/uniow-idd/idd_t2_scpi_b10_t0.15.npy',
         priority=50),
]

max_epochs = 0
