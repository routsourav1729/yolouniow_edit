#!/bin/bash
#SBATCH --job-name=uniow_l_idd
#SBATCH --output=slurm_logs/idd_owod_l_%j.out
#SBATCH --error=slurm_logs/idd_owod_l_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=30:00:00

# ============================================================================
# YOLO-UniOW-L  IDD OWOD Training — T1 + Evaluation
# ============================================================================
# Largest official YOLO-UniOW variant (29.4M params).
# Comparable to YOLO-World XL used in hypyolov2 for fair comparison.
#
# T1: 8 known IDD classes.
# T2 classes (bus, truck, tanker_vehicle, …) treated as UNKNOWN.
#
# Usage:
#   sbatch scripts/train_idd_owod_l.sbatch
# ============================================================================

NUM_GPUS=${1:-${SLURM_GPUS_ON_NODE:-2}}

DATASET=IDD
CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py
PRETRAINED_CKPT=pretrained/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth
THRESHOLD=0.05

STEM=yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod
T1_WORK_DIR=work_dirs/${STEM}_idd_train_task1
T1_EVAL_DIR=work_dirs/eval_IDD_L_test_task1

echo "=========================================="
echo "YOLO-UniOW-L  IDD OWOD TRAINING (T1)"
echo "=========================================="
echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"
echo ""
echo "Configuration:"
echo "  Dataset:       $DATASET (T1: 8 known, T2 classes = unknown)"
echo "  Config:        $CONFIG"
echo "  Pretrained:    $PRETRAINED_CKPT"
echo "  GPUs:          $NUM_GPUS"
echo "  Threshold:     $THRESHOLD"
echo "  T1 train dir:  $T1_WORK_DIR"
echo "  T1 eval dir:   $T1_EVAL_DIR"
echo "=========================================="

# ─── Environment ─────────────────────────────────────────────────────────────

source /home/agipml/sourav.rout/miniconda3/bin/activate yolouniow

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# YOLO-UniOW as the importable package root
export PYTHONPATH=/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW:$PYTHONPATH

# HuggingFace / CLIP cache — no internet on compute nodes
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# DDP / NCCL
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
mkdir -p slurm_logs embeddings/uniow-idd

# ─── Verify embeddings exist ─────────────────────────────────────────────────
if [ ! -f "embeddings/uniow-idd/idd_t1.npy" ]; then
    echo "IDD embeddings not found — generating from cached CLIP model..."
    python tools/owod_scripts/extract_idd_embeddings.py --task 1 --device auto
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to extract IDD embeddings. Aborting."
        exit 1
    fi
else
    echo "IDD embeddings already exist: embeddings/uniow-idd/idd_t1.npy"
fi

echo ""
echo "Embeddings:"
ls -la embeddings/uniow-idd/
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
echo "Job finished at: $(date)"
echo "=========================================="
