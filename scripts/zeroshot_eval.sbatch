#!/bin/bash
#SBATCH --job-name=uniow_zs
#SBATCH --output=slurm_logs/zeroshot_%x_%j.out
#SBATCH --error=slurm_logs/zeroshot_%x_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# ============================================================================
# YOLO-UniOW-L  Zero-Shot / T1-Checkpoint Evaluation  (all datasets)
#
# Runs inference on a dataset's T1 test set using a given checkpoint
# WITHOUT any few-shot finetuning. Purpose: verify whether T2 finetuning
# actually helps, or whether it destroys zero-shot generalisation.
#
# Usage:
#   sbatch scripts/zeroshot_eval.sbatch --dataset FOOD_VOCCOCO
#   sbatch scripts/zeroshot_eval.sbatch --dataset FOOD_VOC
#   sbatch scripts/zeroshot_eval.sbatch --dataset IDD
#
#   # Use a specific checkpoint (e.g. T2 best, to compare)
#   sbatch scripts/zeroshot_eval.sbatch --dataset FOOD_VOCCOCO --ckpt work_dirs/.../best.pth
#
#   # Use pretrained backbone (downstream-naive zero-shot)
#   sbatch scripts/zeroshot_eval.sbatch --dataset FOOD_VOCCOCO --pretrained
#
# Outputs to: work_dirs/zeroshot_<DATASET>_<ckpt-stem>/
# ============================================================================

# ─── Parse arguments ──────────────────────────────────────────────────────────
DATASET=""
CKPT_OVERRIDE=""
USE_PRETRAINED=0
THRESHOLD=0.05
NUM_GPUS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)    DATASET="$2"; shift 2 ;;
        --ckpt)       CKPT_OVERRIDE="$2"; shift 2 ;;
        --pretrained) USE_PRETRAINED=1; shift 1 ;;
        --threshold)  THRESHOLD="$2"; shift 2 ;;
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: sbatch scripts/zeroshot_eval.sbatch --dataset DATASET [--ckpt PATH] [--pretrained] [--threshold T] [--gpus N]"
            exit 1
            ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset is required."
    echo "Available datasets: IDD, FOOD_VOC, FOOD_VOCCOCO"
    exit 1
fi

NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))
STEM=yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod
DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

# ─── Route logs to dataset subfolder ─────────────────────────────────────────
case "$DATASET" in
    IDD)           _LOG_SUBDIR=slurm_logs/idd ;;
    FOOD_VOCCOCO)  _LOG_SUBDIR=slurm_logs/voccoco ;;
    FOOD_VOC)      _LOG_SUBDIR=slurm_logs/voc ;;
    *)             _LOG_SUBDIR=slurm_logs/zeroshot ;;
esac
mkdir -p "$_LOG_SUBDIR"
_ZS_LOG="zeroshot_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
ln -sf "../${_ZS_LOG}.out" "${_LOG_SUBDIR}/${_ZS_LOG}.out" 2>/dev/null || true
ln -sf "../${_ZS_LOG}.err" "${_LOG_SUBDIR}/${_ZS_LOG}.err" 2>/dev/null || true
echo "[Log] ${_LOG_SUBDIR}/${_ZS_LOG}.{out,err}"

# ─── Dataset routing ──────────────────────────────────────────────────────────
case $DATASET in
    IDD)
        T1_CONFIG=configs/owod_ft/${STEM}_idd.py
        T1_WORK_DIR=work_dirs/${STEM}_idd_train_task1
        EMBED_FILE=embeddings/uniow-idd/idd_t1.npy
        NUM_BASE_CLASSES=8
        FEWSHOT_DIR=data/OWOD/iddsplit
        ;;
    FOOD_VOC)
        T1_CONFIG=configs/owod_ft/${STEM}_food_voc.py
        T1_WORK_DIR=work_dirs/${STEM}_food_voc_train_task1
        EMBED_FILE=embeddings/uniow-food-voc/food_voc_t1.npy
        NUM_BASE_CLASSES=10
        FEWSHOT_DIR=data/OWOD/vocsplit
        ;;
    FOOD_VOCCOCO)
        T1_CONFIG=configs/owod_ft/${STEM}_food_voccoco.py
        T1_WORK_DIR=work_dirs/${STEM}_food_voccoco_train_task1
        EMBED_FILE=embeddings/uniow-food-voccoco/food_voccoco_t1.npy
        NUM_BASE_CLASSES=20
        FEWSHOT_DIR=data/OWOD/voccocosplit
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'. Available: IDD, FOOD_VOC, FOOD_VOCCOCO"
        exit 1
        ;;
esac

# ─── Checkpoint selection ─────────────────────────────────────────────────────
PRETRAIN_CKPT=pretrained/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth

if [ -n "$CKPT_OVERRIDE" ]; then
    CKPT="$CKPT_OVERRIDE"
elif [ "$USE_PRETRAINED" -eq 1 ]; then
    CKPT="$PRETRAIN_CKPT"
else
    # Default: T1 best checkpoint
    CKPT=$(ls -t "${T1_WORK_DIR}"/best*.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: No T1 checkpoint found in ${T1_WORK_DIR}/"
        echo "  Train T1 first, or pass --ckpt PATH or --pretrained."
        exit 1
    fi
fi

CKPT_STEM=$(basename "$CKPT" .pth)
EVAL_DIR=work_dirs/zeroshot_${DATASET}_${CKPT_STEM}

echo "=========================================="
echo "YOLO-UniOW-L  Zero-Shot Eval — $DATASET"
echo "=========================================="
echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"
echo ""
echo "  Dataset:       $DATASET ($NUM_BASE_CLASSES base classes)"
echo "  Config:        $T1_CONFIG"
echo "  Checkpoint:    $CKPT"
echo "  Embed file:    $EMBED_FILE"
echo "  Eval dir:      $EVAL_DIR"
echo "  Threshold:     $THRESHOLD"
echo "  GPUs:          $NUM_GPUS"
echo "=========================================="

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
export NCCL_DEBUG=WARN
export MASTER_PORT

cd /home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.device_count()} GPUs')"
echo ""

mkdir -p slurm_logs "$EVAL_DIR"

# ─── Verify files exist ───────────────────────────────────────────────────────
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi
if [ ! -f "$EMBED_FILE" ]; then
    echo "ERROR: Embedding file not found: $EMBED_FILE"
    exit 1
fi

echo "Checkpoint: $(du -sh "$CKPT" | cut -f1) — $CKPT"
echo "Embeddings: $(python -c "import numpy as np; e=np.load('$EMBED_FILE'); print(e.shape)")"
echo ""

# ─── Run evaluation ───────────────────────────────────────────────────────────
echo "=========================================="
echo "Running zero-shot evaluation on T1 test set"
echo "  Started at: $(date)"
echo "=========================================="

DATASET=$DATASET TASK=1 THRESHOLD=$THRESHOLD SAVE=False \
    FEWSHOT_K=0 FEWSHOT_SEED=1 FEWSHOT_DIR=$FEWSHOT_DIR \
    PORT=${MASTER_PORT} \
    ./tools/dist_test.sh ${T1_CONFIG} ${CKPT} ${NUM_GPUS} \
    --work-dir ${EVAL_DIR}

EVAL_RC=$?
echo ""
echo "=========================================="
if [ $EVAL_RC -eq 0 ]; then
    echo "Zero-shot eval DONE at: $(date)"
    echo "Results in: $EVAL_DIR"
    # Print summary if results exist
    python -c "
import json, glob, os
files = sorted(glob.glob('${EVAL_DIR}/*.json'))
if files:
    with open(files[-1]) as f:
        r = json.load(f)
    print('--- Metric summary ---')
    for k, v in r.items():
        if 'owod' in k.lower() or 'map' in k.lower() or 'recall' in k.lower():
            print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
" 2>/dev/null || true
else
    echo "ERROR: Eval failed (exit code $EVAL_RC)"
fi
echo "=========================================="
exit $EVAL_RC
