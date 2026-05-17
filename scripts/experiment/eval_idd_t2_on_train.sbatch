#!/bin/bash
#SBATCH --job-name=uniow_idd_t2_evaltrain
#SBATCH --output=slurm_logs/idd/eval_t2_on_train_%j.out
#SBATCH --error=slurm_logs/idd/eval_t2_on_train_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# ============================================================================
# Quick experiment: run T2-finetuned IDD weights on the 10-shot training data
# (same images/boxes used for T2 finetuning) to surface base-vs-novel class
# confusion. Reports mAP/recall via OpenWorldMetric.
#
# Usage:
#   sbatch scripts/experiment/eval_idd_t2_on_train.sbatch                 # uses baseline T2 ckpt by default
#   sbatch scripts/experiment/eval_idd_t2_on_train.sbatch --variant kume
#   sbatch scripts/experiment/eval_idd_t2_on_train.sbatch --ckpt /path/to/best.pth
# ============================================================================

VARIANT="plain"
CKPT_OVERRIDE=""
SHOT=10
SEED=1
THRESHOLD=0.05

while [[ $# -gt 0 ]]; do
    case $1 in
        --variant)   VARIANT="$2"; shift 2 ;;
        --ckpt)      CKPT_OVERRIDE="$2"; shift 2 ;;
        --shot)      SHOT="$2"; shift 2 ;;
        --seed)      SEED="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p slurm_logs/idd

case "$VARIANT" in
    plain) SUFFIX="" ;;
    kume)  SUFFIX="_kume" ;;
    *) SUFFIX="_${VARIANT}" ;;
esac

T2_WORK_DIR=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task2_${SHOT}shot_seed${SEED}${SUFFIX}
if [ -n "$CKPT_OVERRIDE" ]; then
    CKPT="$CKPT_OVERRIDE"
else
    CKPT=$(ls -t "${T2_WORK_DIR}"/best*.pth 2>/dev/null | head -1)
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found in $T2_WORK_DIR"
    exit 1
fi

CONFIG=configs/owod_ft/yolo_uniow_l_idd_t2_eval_on_train.py
EVAL_DIR=work_dirs/eval_IDD_L_t2_on_train_${SHOT}shot_seed${SEED}${SUFFIX}
mkdir -p "$EVAL_DIR"

echo "=========================================="
echo "IDD T2 -> eval on fewshot TRAINING data"
echo "  variant:    $VARIANT"
echo "  checkpoint: $CKPT"
echo "  config:     $CONFIG"
echo "  eval dir:   $EVAL_DIR"
echo "  shot/seed:  ${SHOT}/${SEED}"
echo "  threshold:  $THRESHOLD"
echo "=========================================="

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
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))

DATASET=IDD TASK=2 THRESHOLD=${THRESHOLD} SAVE=True \
    FEWSHOT_K=${SHOT} FEWSHOT_SEED=${SEED} FEWSHOT_DIR=data/OWOD/iddsplit \
    PORT=${MASTER_PORT} \
    ./tools/dist_test.sh ${CONFIG} ${CKPT} 1 \
    --work-dir ${EVAL_DIR}

echo "Done at $(date)"
