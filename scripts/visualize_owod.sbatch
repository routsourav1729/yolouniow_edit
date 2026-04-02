#!/bin/bash
#SBATCH --job-name=vis_owod
#SBATCH --output=slurm_logs/vis_owod_%j.out
#SBATCH --error=slurm_logs/vis_owod_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# ============================================================================
# YOLO-UniOW-L  IDD  — Side-by-side GT vs Prediction visualisation
# ============================================================================
# Usage:
#   sbatch scripts/visualize_owod.sbatch
#   sbatch scripts/visualize_owod.sbatch /path/to/custom_checkpoint.pth
# ============================================================================

CUSTOM_CKPT=${1:-""}

WORK_DIR=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task2_10shot_wapr
CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_wapr.py
OUTPUT_DIR=visualizations/inference

# Resolve checkpoint
if [ -n "$CUSTOM_CKPT" ]; then
    CKPT="$CUSTOM_CKPT"
else
    CKPT=$(ls -t "${WORK_DIR}"/best*.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        CKPT=$(ls -t "${WORK_DIR}"/epoch_*.pth 2>/dev/null | head -1)
    fi
fi

echo "=========================================="
echo "YOLO-UniOW-L  IDD  — Visualisation"
echo "=========================================="
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Config:     $CONFIG"
echo "Checkpoint: $CKPT"
echo "Output:     $OUTPUT_DIR"
echo "=========================================="

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

# ─── Environment ─────────────────────────────────────────────────────────────
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

# ─── Env vars for mmengine config interpolation ─────────────────────────────
export DATASET=IDD
export TASK=2
export THRESHOLD=0.05
export SAVE=False
export FEWSHOT_DIR=data/OWOD/iddsplit
export FEWSHOT_K=10
export FEWSHOT_SEED=1
export TRAINING_STRATEGY=0
export IMAGESET=train
export ANALYZE=0

mkdir -p slurm_logs "$OUTPUT_DIR"

echo ""
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.device_count()} GPUs')"
echo ""

# ─── Run visualisation ──────────────────────────────────────────────────────
python utils/visualize_owod.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --data-root data/OWOD \
    --dataset IDD \
    --task 2 \
    --threshold 0.05 \
    --unk-threshold 0.15 \
    --output-dir "$OUTPUT_DIR" \
    --amp

RC=$?
echo ""
if [ $RC -eq 0 ]; then
    echo "Done! Images saved to: $OUTPUT_DIR/"
    ls -lh "$OUTPUT_DIR"/*.jpg 2>/dev/null | wc -l
    echo "images generated."
else
    echo "ERROR: Visualisation failed (exit code $RC)"
fi
