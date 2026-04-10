#!/bin/bash
#SBATCH --job-name=diag_idd
#SBATCH --output=slurm_logs/idd/diag_t1_on_t2_%j.out
#SBATCH --error=slurm_logs/idd/diag_t1_on_t2_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00

# ============================================================================
# IDD DIAGNOSTIC: Evaluate T1 checkpoint on T2 test set (14 known + 6 unknown)
#
# Purpose: Isolate whether the U-Recall drop comes from:
#   (a) Simply adding novel class embeddings (expected: similar classes steal detections)
#   (b) T2 training / WAPR actively making it worse
#
# If T1-on-T2 already shows the drop → it's (a), not a code bug.
# If T1-on-T2 is fine but T2-trained drops → T2 training is the problem.
# ============================================================================

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
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))
export MASTER_PORT

cd /home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW
mkdir -p slurm_logs/idd

# T2 config (vanilla, no WAPR — we just want T2 eval setup)
CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2.py

# T1 best checkpoint (trained with corrected embeddings)
CKPT=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task1/best_owod_Both_epoch_20.pth

WORK_DIR=work_dirs/diag_idd_t1_on_t2_eval

echo "=========================================="
echo "DIAGNOSTIC: T1 checkpoint → T2 eval (14 known + 6 unknown)"
echo "=========================================="
echo "  Config:     $CONFIG"
echo "  Checkpoint: $CKPT"
echo "  Work dir:   $WORK_DIR"
echo "  Purpose:    Isolate U-Recall drop cause"
echo "=========================================="

if [ ! -f "$CKPT" ]; then
    echo "ERROR: T1 checkpoint not found: $CKPT"
    exit 1
fi

echo "Checkpoint: $(du -sh "$CKPT" | cut -f1)"
echo ""

# Run eval with T2 settings
DATASET=IDD TASK=2 THRESHOLD=0.05 SAVE=False \
    FEWSHOT_K=10 FEWSHOT_SEED=1 FEWSHOT_DIR=data/OWOD/iddsplit \
    PORT=${MASTER_PORT} \
    ./tools/dist_test.sh ${CONFIG} ${CKPT} 1 \
    --work-dir ${WORK_DIR}

RC=$?
echo ""
if [ $RC -eq 0 ]; then
    echo "DIAGNOSTIC COMPLETE at $(date)"
else
    echo "DIAGNOSTIC FAILED (exit code $RC)"
fi
exit $RC
