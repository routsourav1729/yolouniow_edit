#!/bin/bash
#SBATCH --job-name=zs_idd
#SBATCH --output=slurm_logs/zeroshot_idd_%j.out
#SBATCH --error=slurm_logs/zeroshot_idd_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00

# ============================================================================
# Zero-shot eval: pretrained YOLO-UniOW-L on IDD (all 14 classes)
#
# Step 1: Extract fresh CLIP embeddings for all 14 IDD classes → idd_all.npy
# Step 2: Run inference on IDD test set using pretrained weights
#
# Usage:
#   sbatch scripts/zeroshot_eval_idd.sbatch
# ============================================================================

PRETRAIN_CONFIG=configs/pretrain/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py
PRETRAIN_CKPT=pretrained/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth
ZS_CONFIG=configs/owod_ft/yolo_uniow_l_zeroshot_idd.py
EMBED_DIR=embeddings/uniow-l
EMBED_FILE=${EMBED_DIR}/idd_all.npy
WORK_DIR=work_dirs/zeroshot_idd_l

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))

echo "=========================================="
echo "YOLO-UniOW-L  IDD  Zero-Shot Evaluation"
echo "=========================================="
echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"
echo "  Pretrain ckpt: $PRETRAIN_CKPT"
echo "  ZS config:     $ZS_CONFIG"
echo "  Embed file:    $EMBED_FILE"
echo "=========================================="

if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "ERROR: Pretrained checkpoint not found: $PRETRAIN_CKPT"
    exit 1
fi

# ─── Environment ─────────────────────────────────────────────────────────
source /home/agipml/sourav.rout/miniconda3/bin/activate yolouniow

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW:$PYTHONPATH

export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export MASTER_PORT

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.device_count()} GPUs')"
echo ""

mkdir -p slurm_logs "$WORK_DIR" "$EMBED_DIR"

# ============================================================================
# Step 1: Extract fresh CLIP embeddings for all 14 IDD classes
# ============================================================================
if [ -f "$EMBED_FILE" ]; then
    echo "Embeddings already exist: $EMBED_FILE — skipping extraction."
else
    echo "=========================================="
    echo "Step 1: Extracting CLIP embeddings for all 14 IDD classes"
    echo "  Started at: $(date)"
    echo "=========================================="

    python -c "
import torch, numpy as np
from pathlib import Path
from mmengine.config import Config
from mmengine.runner import Runner

# Read class names
with open('data/OWOD/ImageSets/IDD/t_all_known.txt') as f:
    classes = [l.strip() for l in f if l.strip()]
print(f'Classes ({len(classes)}): {classes}')

# Load pretrained model with CLIP text encoder
cfg = Config.fromfile('${PRETRAIN_CONFIG}')
cfg.work_dir = 'work_dirs/_extract_tmp'
runner = Runner.from_cfg(cfg)
runner.call_hook('before_run')
runner.load_checkpoint('${PRETRAIN_CKPT}', map_location='cpu')
model = runner.model.to('cuda')
model.eval()

with torch.inference_mode():
    feats = model.backbone.forward_text([classes]).squeeze(0).detach().cpu().numpy()
print(f'Extracted: {feats.shape}')
np.save('${EMBED_FILE}', feats)
print(f'Saved: ${EMBED_FILE}')
"

    if [ $? -ne 0 ]; then
        echo "ERROR: Embedding extraction failed"
        exit 1
    fi
    echo "Done."
    echo ""
fi

# Verify
python -c "
import numpy as np
e = np.load('${EMBED_FILE}')
print(f'idd_all.npy: {e.shape}')
assert e.shape == (14, 512), f'Bad shape: {e.shape}'
print('OK')
"

# ============================================================================
# Step 2: Zero-shot inference + evaluation
# ============================================================================
echo "=========================================="
echo "Step 2: Running zero-shot eval on IDD test set"
echo "  Started at: $(date)"
echo "=========================================="

DATASET=IDD TASK=2 THRESHOLD=0.05 SAVE=True \
    FEWSHOT_K=0 FEWSHOT_SEED=1 FEWSHOT_DIR=none \
    IMAGESET=train TRAINING_STRATEGY=0 ANALYZE=0 \
    PORT=${MASTER_PORT} \
    ./tools/dist_test.sh ${ZS_CONFIG} ${PRETRAIN_CKPT} ${NUM_GPUS} \
    --work-dir ${WORK_DIR}

EVAL_RC=$?
echo ""
echo "=========================================="
if [ $EVAL_RC -eq 0 ]; then
    echo "Zero-shot eval complete at: $(date)"
    echo "Results in: $WORK_DIR"
else
    echo "ERROR: Eval failed (exit code $EVAL_RC)"
fi
echo "=========================================="
exit $EVAL_RC
