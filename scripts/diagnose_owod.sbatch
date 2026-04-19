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
#SBATCH --time=02:00:00

# ============================================================================
# Universal OWOD Diagnostic
#
# Usage:
#   sbatch scripts/diagnose_owod.sbatch --dataset IDD --task 2
#   sbatch scripts/diagnose_owod.sbatch --dataset IDD --task 2 --method wapr-kume
#   sbatch scripts/diagnose_owod.sbatch --dataset IDD --task 2 --checkpoint /path/to/ckpt
#   sbatch scripts/diagnose_owod.sbatch --dataset FOOD_VOCCOCO --task 2 --method wapr
#
# Options:
#   --dataset         IDD | FOOD_VOCCOCO | FOOD_VOC
#   --task            1 | 2
#   --method          baseline | wapr | kume | wapr-kume  (default: wapr-kume)
#                     Selects config and work_dir suffix, matches train_owod_l_t2.sbatch
#   --checkpoint      Override checkpoint path (default: auto-detect best in method work_dir)
#   --config          Override config file path
#   --shot            Few-shot K (default: 10)
#   --seed            Few-shot seed (default: 1)
#   --out-dir         Output directory for results (default: <work_dir>/diagnose)
#   --num-images      Number of images to process (default: all)
#   --thresholds      Comma-separated ratio thresholds (default: 0.3,0.4,0.5,0.6,0.7,0.8,0.9)
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
METHOD="wapr-kume"
CKPT=""
CONFIG=""
FEWSHOT_K=10
FEWSHOT_SEED=1
OUT_DIR_OVERRIDE=""
NUM_IMAGES=0
THRESHOLDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)    DATASET="$2"; shift 2;;
        --task)       TASK="$2"; shift 2;;
        --method)     METHOD="$2"; shift 2;;
        --checkpoint) CKPT="$2"; shift 2;;
        --config)     CONFIG="$2"; shift 2;;
        --shot)        FEWSHOT_K="$2"; shift 2;;
        --seed)        FEWSHOT_SEED="$2"; shift 2;;
        --out-dir)     OUT_DIR_OVERRIDE="$2"; shift 2;;
        --num-images)  NUM_IMAGES="$2"; shift 2;;
        --thresholds)  THRESHOLDS="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

export DATASET TASK

# ─── Resolve method suffix (mirrors train_owod_l_t2.sbatch logic) ────────────
case "$METHOD" in
    wapr-kume)   METHOD_SUFFIX="_wapr_kume" ;;
    kume)        METHOD_SUFFIX="_kume" ;;
    wapr)        METHOD_SUFFIX="_wapr" ;;
    baseline|"")  METHOD_SUFFIX="" ;;
    *)           echo "ERROR: Unknown --method '$METHOD'. Use: baseline wapr kume wapr-kume"; exit 1 ;;
esac

# ─── Auto-detect config and checkpoint ────────────────────────────────────
DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
STEM=yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod

case $DATASET in
    IDD)
        LOG_DIR=slurm_logs/idd
        FEWSHOT_DIR=data/OWOD/iddsplit
        T1_CKPT_DIR=work_dirs/${STEM}_idd_train_task1
        if [ -z "$CONFIG" ]; then
            if [ "$TASK" -eq 1 ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py
            elif [ "$METHOD" = "wapr-kume" ] || [ "$METHOD" = "kume" ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_kume.py
            elif [ "$METHOD" = "wapr" ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_wapr.py
            else
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2.py
            fi
        fi
        ;;
    FOOD_VOCCOCO)
        LOG_DIR=slurm_logs/voccoco
        FEWSHOT_DIR=data/OWOD/voccocosplit
        T1_CKPT_DIR=work_dirs/${STEM}_food_voccoco_train_task1
        if [ -z "$CONFIG" ]; then
            if [ "$TASK" -eq 1 ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voccoco.py
            elif [ "$METHOD" = "wapr" ] || [ "$METHOD" = "wapr-kume" ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voccoco_t2_wapr.py
            else
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voccoco_t2.py
            fi
        fi
        ;;
    FOOD_VOC)
        LOG_DIR=slurm_logs/voc
        FEWSHOT_DIR=data/OWOD/vocsplit
        T1_CKPT_DIR=work_dirs/${STEM}_food_voc_train_task1
        if [ -z "$CONFIG" ]; then
            if [ "$TASK" -eq 1 ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voc.py
            elif [ "$METHOD" = "wapr" ] || [ "$METHOD" = "wapr-kume" ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_5e_8gpus_owod_food_voc_t2_wapr.py
            else
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voc_t2.py
            fi
        fi
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'. Use IDD | FOOD_VOCCOCO | FOOD_VOC"
        exit 1
        ;;
esac

# T2 work dir (matches what train_owod_l_t2.sbatch produces)
if [ "$TASK" -eq 1 ]; then
    T2_WORK_DIR="$T1_CKPT_DIR"
else
    T2_WORK_DIR=work_dirs/${STEM}_${DATASET_LOWER}_train_task2_${FEWSHOT_K}shot_seed${FEWSHOT_SEED}${METHOD_SUFFIX}
fi

# Auto-detect checkpoint
if [ -z "$CKPT" ]; then
    CKPT=$(ls -t "${T2_WORK_DIR}"/best*.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        CKPT=$(ls -t "${T2_WORK_DIR}"/epoch_*.pth 2>/dev/null | head -1)
    fi
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found in ${T2_WORK_DIR}"
    echo "  Use --checkpoint /path/to/ckpt.pth to specify manually."
    exit 1
fi

# Output dir
if [ -n "$OUT_DIR_OVERRIDE" ]; then
    OUT_DIR="$OUT_DIR_OVERRIDE"
else
    OUT_DIR="${T2_WORK_DIR}/diagnose"
fi
mkdir -p "$OUT_DIR"

# Symlink logs to dataset-specific dir
ln -sf "$(pwd)/slurm_logs/analysis/diag_${SLURM_JOB_ID}.out" "${LOG_DIR}/diag_${SLURM_JOB_ID}.out" 2>/dev/null
ln -sf "$(pwd)/slurm_logs/analysis/diag_${SLURM_JOB_ID}.err" "${LOG_DIR}/diag_${SLURM_JOB_ID}.err" 2>/dev/null

# Set env vars for the config
export THRESHOLD=0.05
export SAVE=False
export FEWSHOT_K=${FEWSHOT_K}
export FEWSHOT_SEED=${FEWSHOT_SEED}
export FEWSHOT_DIR
export VISUALIZE=${VISUALIZE:-0}

echo "=========================================="
echo "OWOD DIAGNOSTIC"
echo "=========================================="
echo "  Dataset:    $DATASET"
echo "  Task:       T${TASK}"
echo "  Method:     $METHOD"
echo "  Config:     $CONFIG"
echo "  Checkpoint: $CKPT ($(du -sh "$CKPT" 2>/dev/null | cut -f1))"
echo "  Out dir:    $OUT_DIR"
echo "  Num images: ${NUM_IMAGES:-all}"
echo "  Thresholds: ${THRESHOLDS:-default (0.3-0.9)}"
echo "  Log dir:    $LOG_DIR"
echo "=========================================="

# Auto-detect T1 checkpoint (informational only)
T1_CKPT=$(ls -t ${T1_CKPT_DIR}/best*.pth 2>/dev/null | head -1)
if [ -n "$T1_CKPT" ] && [ -f "$T1_CKPT" ]; then
    echo "  T1 Ckpt:    $T1_CKPT"
else
    echo "  T1 Ckpt:    (not found)"
fi

NUM_IMAGES_FLAG=""
if [ "$NUM_IMAGES" -gt 0 ] 2>/dev/null; then
    NUM_IMAGES_FLAG="--num-images $NUM_IMAGES"
fi

THRESHOLDS_FLAG=""
if [ -n "$THRESHOLDS" ]; then
    THRESHOLDS_FLAG="--thresholds $THRESHOLDS"
fi

python tools/diagnose_owod.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --out-dir "$OUT_DIR" \
    $NUM_IMAGES_FLAG \
    $THRESHOLDS_FLAG

echo ""
echo "Diagnosis complete. Results in: $OUT_DIR"

echo ""
echo "DIAGNOSTIC FINISHED at $(date)"
