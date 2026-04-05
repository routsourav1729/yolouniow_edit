#!/bin/bash
#SBATCH --job-name=uniow_l_t1
#SBATCH --output=slurm_logs/owod_l_t1_%j.out
#SBATCH --error=slurm_logs/owod_l_t1_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=30:00:00

# ============================================================================
# YOLO-UniOW-L  Parameterized OWOD Training — T1 + Evaluation
# ============================================================================
# Generic T1 training script that works with any registered OWOD dataset.
#
# Usage:
#   sbatch scripts/train_owod_l.sbatch --dataset FOOD_VOC
#   sbatch scripts/train_owod_l.sbatch --dataset FOOD_VOCCOCO
#   sbatch scripts/train_owod_l.sbatch --dataset IDD
#   sbatch scripts/train_owod_l.sbatch --dataset FOOD_VOC --gpus 4 --threshold 0.1
#
# Required arguments:
#   --dataset DATASET     Dataset name (IDD, FOOD_VOC, FOOD_VOCCOCO, etc.)
#
# Optional arguments:
#   --gpus N              Number of GPUs (default: SLURM_GPUS_ON_NODE or 2)
#   --threshold T         Detection threshold (default: 0.05)
#   --config PATH         Override config file path
#   --pretrained PATH     Override pretrained checkpoint path
# ============================================================================

# ─── Parse arguments ─────────────────────────────────────────────────────────
DATASET=""
NUM_GPUS=""
THRESHOLD=0.05
CONFIG_OVERRIDE=""
PRETRAINED_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)    DATASET="$2"; shift 2 ;;
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        --threshold)  THRESHOLD="$2"; shift 2 ;;
        --config)     CONFIG_OVERRIDE="$2"; shift 2 ;;
        --pretrained) PRETRAINED_OVERRIDE="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: sbatch scripts/train_owod_l.sbatch --dataset DATASET [--gpus N] [--threshold T] [--config PATH] [--pretrained PATH]"
            exit 1
            ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset is required."
    echo "Usage: sbatch scripts/train_owod_l.sbatch --dataset DATASET"
    echo "Available datasets: IDD, FOOD_VOC, FOOD_VOCCOCO"
    exit 1
fi

NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-2}}

# ─── Dataset-specific configuration ──────────────────────────────────────────
STEM=yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod
PRETRAINED_CKPT=${PRETRAINED_OVERRIDE:-pretrained/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth}

DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

case $DATASET in
    IDD)
        CONFIG=${CONFIG_OVERRIDE:-configs/owod_ft/${STEM}_idd.py}
        EMBED_DIR=embeddings/uniow-idd
        EMBED_PREFIX=idd
        ;;
    FOOD_VOC)
        CONFIG=${CONFIG_OVERRIDE:-configs/owod_ft/${STEM}_food_voc.py}
        EMBED_DIR=embeddings/uniow-food-voc
        EMBED_PREFIX=food_voc
        ;;
    FOOD_VOCCOCO)
        CONFIG=${CONFIG_OVERRIDE:-configs/owod_ft/${STEM}_food_voccoco.py}
        EMBED_DIR=embeddings/uniow-food-voccoco
        EMBED_PREFIX=food_voccoco
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'. Available: IDD, FOOD_VOC, FOOD_VOCCOCO"
        exit 1
        ;;
esac

T1_WORK_DIR=work_dirs/${STEM}_${DATASET_LOWER}_train_task1
T1_EVAL_DIR=work_dirs/eval_${DATASET}_L_test_task1

echo "=========================================="
echo "YOLO-UniOW-L  OWOD TRAINING (T1)"
echo "=========================================="
echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"
echo ""
echo "Configuration:"
echo "  Dataset:       $DATASET"
echo "  Config:        $CONFIG"
echo "  Pretrained:    $PRETRAINED_CKPT"
echo "  GPUs:          $NUM_GPUS"
echo "  Threshold:     $THRESHOLD"
echo "  Embed dir:     $EMBED_DIR"
echo "  T1 train dir:  $T1_WORK_DIR"
echo "  T1 eval dir:   $T1_EVAL_DIR"
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
export NCCL_IB_DISABLE=0
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))
export MASTER_PORT
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / NUM_GPUS ))

echo ""
echo "Environment:"
echo "  Python:        $(which python)"
echo "  Conda env:     $CONDA_DEFAULT_ENV"
echo "  MASTER_PORT:   $MASTER_PORT"
python -c "import torch; print(f'  PyTorch:       {torch.__version__}')"
python -c "import torch; print(f'  CUDA devices:  {torch.cuda.device_count()}')"
echo ""

# ─── Ensure dirs exist ───────────────────────────────────────────────────────
mkdir -p slurm_logs "$EMBED_DIR"

# ─── Verify embeddings exist ─────────────────────────────────────────────────
if [ ! -f "${EMBED_DIR}/${EMBED_PREFIX}_t1.npy" ]; then
    echo "Embeddings not found — generating from cached CLIP model..."
    python tools/owod_scripts/extract_embeddings.py --dataset "$DATASET" --task 1 --device auto
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to extract embeddings. Aborting."
        exit 1
    fi
else
    echo "Embeddings already exist: ${EMBED_DIR}/${EMBED_PREFIX}_t1.npy"
fi

echo ""
echo "Embeddings:"
ls -la "$EMBED_DIR"/
echo ""

# ─── Verify pretrained weights exist ─────────────────────────────────────────
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: Pretrained weights not found at $PRETRAINED_CKPT"
    exit 1
fi
echo "Pretrained weights: $(du -h $PRETRAINED_CKPT | cut -f1) — $PRETRAINED_CKPT"
echo ""

# ─── Helper ──────────────────────────────────────────────────────────────────
run_train() {
    local task=$1
    local work_dir=$2
    local load_from=$3

    echo "=========================================="
    echo "Training Task ${task}"
    echo "  work_dir:   $work_dir"
    echo "  load_from:  ${load_from:-<none>}"
    echo "  Started at: $(date)"
    echo "=========================================="

    local cmd="DATASET=${DATASET} TASK=${task} THRESHOLD=${THRESHOLD} SAVE=False \
        PORT=${MASTER_PORT} \
        ./tools/dist_train_owod.sh ${CONFIG} ${NUM_GPUS} \
        --amp \
        --work-dir ${work_dir}"

    if [ -n "$load_from" ]; then
        cmd+=" --cfg-options load_from=${load_from}"
    fi

    echo "CMD: $cmd"
    eval "$cmd"
    local rc=$?

    if [ $rc -ne 0 ]; then
        echo "ERROR: Task ${task} training failed (exit code $rc). Aborting."
        exit $rc
    fi

    echo "Task ${task} finished at: $(date)"
    echo ""
}

# ─── Task 1: Train ───────────────────────────────────────────────────────────
mkdir -p "$T1_WORK_DIR"
run_train 1 "$T1_WORK_DIR" "$PRETRAINED_CKPT"

# ─── Find T1 best checkpoint ─────────────────────────────────────────────────
T1_BEST=$(ls -t "${T1_WORK_DIR}"/best*.pth 2>/dev/null | head -1)
if [ -z "$T1_BEST" ]; then
    echo "ERROR: No best_*.pth checkpoint found in ${T1_WORK_DIR}."
    exit 1
fi
echo "T1 best checkpoint: $T1_BEST"

# ─── Task 1: Evaluation ──────────────────────────────────────────────────────
echo "=========================================="
echo "Running Inference / Evaluation on T1"
echo "  checkpoint: $T1_BEST"
echo "  eval dir:   $T1_EVAL_DIR"
echo "  Started at: $(date)"
echo "=========================================="

mkdir -p "$T1_EVAL_DIR"
DATASET=${DATASET} TASK=1 THRESHOLD=${THRESHOLD} SAVE=True \
    PORT=${MASTER_PORT} \
    ./tools/dist_test.sh ${CONFIG} ${T1_BEST} ${NUM_GPUS} \
    --work-dir ${T1_EVAL_DIR}

echo ""
echo "=========================================="
echo "YOLO-UniOW-L  T1 TRAINING + EVAL COMPLETE"
echo "  Dataset: $DATASET"
echo "Job finished at: $(date)"
echo "=========================================="
