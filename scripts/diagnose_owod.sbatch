#!/bin/bash
#SBATCH --job-name=diag_owod
#SBATCH --output=slurm_logs/analysis/diag_%j.out
#SBATCH --error=slurm_logs/analysis/diag_%j.err
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:30:00

# ============================================================================
# Universal OWOD Diagnostic
#
# Usage:
#   sbatch scripts/diagnose_owod.sbatch --dataset IDD --task 2 [--checkpoint /path/to/ckpt]
#   sbatch scripts/diagnose_owod.sbatch --dataset FOOD_VOCCOCO --task 2
#   sbatch scripts/diagnose_owod.sbatch --dataset FOOD_VOC --task 2
#
# Options:
#   --dataset         IDD | FOOD_VOCCOCO | FOOD_VOC
#   --task            1 | 2
#   --checkpoint      Override checkpoint (default: auto-detect T1 best for T2 diag)
#   --config          Override config (default: auto-detect)
#   --num-images      Number of val images (default: 50)
#   --t1-on-t2        Evaluate T1 ckpt on T2 config (diag mode)
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

cd /home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW
mkdir -p slurm_logs/{idd,voccoco,voc,analysis}

# ─── Parse arguments ──────────────────────────────────────────────────────
DATASET="IDD"
TASK=2
CKPT=""
CONFIG=""
NUM_IMAGES=0
T1_ON_T2=0
SECTIONS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)   DATASET="$2"; shift 2;;
        --task)      TASK="$2"; shift 2;;
        --checkpoint) CKPT="$2"; shift 2;;
        --config)    CONFIG="$2"; shift 2;;
        --num-images) NUM_IMAGES="$2"; shift 2;;
        --t1-on-t2)  T1_ON_T2=1; shift;;
        --sections)  shift; SECTIONS=""; while [[ $# -gt 0 && "$1" != --* ]]; do SECTIONS="$SECTIONS $1"; shift; done;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

export DATASET TASK

# ─── Auto-detect config and checkpoint ────────────────────────────────────
case $DATASET in
    IDD)
        LOG_DIR=slurm_logs/idd
        FEWSHOT_DIR=data/OWOD/iddsplit
        if [ -z "$CONFIG" ]; then
            if [ "$TASK" -eq 1 ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py
            else
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2.py
            fi
        fi
        T1_CKPT_DIR=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task1
        T2_CKPT_DIR=work_dirs/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_train_task2
        T2_CKPT_DIR_VANILLA=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task2_10shot_seed1
        ;;
    FOOD_VOCCOCO)
        LOG_DIR=slurm_logs/voccoco
        FEWSHOT_DIR=data/OWOD/voccocosplit
        if [ -z "$CONFIG" ]; then
            if [ "$TASK" -eq 1 ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voccoco.py
            else
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voccoco_t2_wapr.py
            fi
        fi
        T1_CKPT_DIR=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voccoco_train_task1
        T2_CKPT_DIR=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voccoco_train_task2_10shot_seed1_wapr
        T2_CKPT_DIR_VANILLA=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voccoco_train_task2_10shot_seed1
        ;;
    FOOD_VOC)
        LOG_DIR=slurm_logs/voc
        FEWSHOT_DIR=data/OWOD/vocsplit
        if [ -z "$CONFIG" ]; then
            if [ "$TASK" -eq 1 ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voc.py
            else
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voc_t2.py
            fi
        fi
        T1_CKPT_DIR=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voc_train_task1
        T2_CKPT_DIR=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voc_train_task2_10shot_seed1
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'. Use IDD | FOOD_VOCCOCO | FOOD_VOC"
        exit 1
        ;;
esac

# Auto-detect checkpoint
if [ -z "$CKPT" ]; then
    if [ "$T1_ON_T2" -eq 1 ] || [ "$TASK" -eq 1 ]; then
        CKPT=$(ls -t ${T1_CKPT_DIR}/best*.pth 2>/dev/null | head -1)
    else
        # Try WAPR dir first, then vanilla dir, then latest epoch in either
        CKPT=$(ls -t ${T2_CKPT_DIR}/best*.pth 2>/dev/null | head -1)
        if [ -z "$CKPT" ] && [ -n "${T2_CKPT_DIR_VANILLA:-}" ]; then
            CKPT=$(ls -t ${T2_CKPT_DIR_VANILLA}/best*.pth 2>/dev/null | head -1)
        fi
        if [ -z "$CKPT" ]; then
            CKPT=$(ls -t ${T2_CKPT_DIR}/epoch_*.pth 2>/dev/null | head -1)
        fi
    fi
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found. Searched:"
    echo "  T1: ${T1_CKPT_DIR}/best*.pth"
    echo "  T2: ${T2_CKPT_DIR}/best*.pth"
    if [ -n "${T2_CKPT_DIR_VANILLA:-}" ]; then
        echo "  T2 vanilla: ${T2_CKPT_DIR_VANILLA}/best*.pth"
    fi
    echo "Use --checkpoint to specify manually."
    exit 1
fi

# Symlink logs to dataset-specific dir
ln -sf "$(pwd)/slurm_logs/analysis/diag_${SLURM_JOB_ID}.out" "${LOG_DIR}/diag_${SLURM_JOB_ID}.out" 2>/dev/null
ln -sf "$(pwd)/slurm_logs/analysis/diag_${SLURM_JOB_ID}.err" "${LOG_DIR}/diag_${SLURM_JOB_ID}.err" 2>/dev/null

# Set env vars for the config
export THRESHOLD=0.05
export SAVE=False
export FEWSHOT_K=10
export FEWSHOT_SEED=1
export FEWSHOT_DIR

echo "=========================================="
echo "OWOD DIAGNOSTIC"
echo "=========================================="
echo "  Dataset:    $DATASET"
echo "  Task:       T${TASK}"
echo "  Config:     $CONFIG"
echo "  Checkpoint: $CKPT ($(du -sh "$CKPT" 2>/dev/null | cut -f1))"
echo "  Num images: $NUM_IMAGES"
echo "  T1-on-T2:   $T1_ON_T2"
echo "  Sections:   ${SECTIONS:-all}"
echo "  Log dir:    $LOG_DIR"
echo "=========================================="

# Auto-detect T1 checkpoint
T1_CKPT=$(ls -t ${T1_CKPT_DIR}/best*.pth 2>/dev/null | head -1)
T1_FLAG=""
if [ -n "$T1_CKPT" ] && [ -f "$T1_CKPT" ]; then
    T1_FLAG="--t1-checkpoint $T1_CKPT"
    echo "  T1 Ckpt:    $T1_CKPT"
else
    echo "  T1 Ckpt:    (not found, skipping visual centroid)"
fi

SECTIONS_FLAG=""
if [ -n "$SECTIONS" ]; then
    SECTIONS_FLAG="--sections $SECTIONS"
fi

python tools/diagnose_owod.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    $T1_FLAG \
    --num-images "$NUM_IMAGES" \
    $SECTIONS_FLAG

echo ""
echo "DIAGNOSTIC FINISHED at $(date)"
