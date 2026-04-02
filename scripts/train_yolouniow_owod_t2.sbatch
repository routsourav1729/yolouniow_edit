#!/bin/bash
#SBATCH --job-name=yolouniow_idd_t2
#SBATCH --output=slurm_logs/idd_owod_t2_%j.out
#SBATCH --error=slurm_logs/idd_owod_t2_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=30:00:00

# ============================================================================
# YOLO-UniOW IDD T2 Fine-tuning — 10-shot, 40 epochs, no WAPR, no SCPI
# ============================================================================
# Loads from the T1 best checkpoint and fine-tunes on 10-shot T2 data.
# T2 novel classes: bus, truck, tanker_vehicle, crane_truck, street_cart,
#                   excavator (6 classes, PREV=8, CUR=6, total known=14).
# No WAPR, no SCPI — plain YOLO-UniOW anchor_label pseudo-labeling.
#
# Usage (submit from YOLO-UniOW root):
#   sbatch scripts/train_idd_owod_t2.sbatch
# ============================================================================

NUM_GPUS=${1:-${SLURM_GPUS_ON_NODE:-2}}

DATASET=IDD
CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2.py
T1_CKPT=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task1/best_owod_Both_epoch_20.pth
THRESHOLD=0.05
FEWSHOT_K=10
FEWSHOT_SEED=1
FEWSHOT_DIR=data/OWOD/iddsplit

STEM=yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2
T2_WORK_DIR=work_dirs/${STEM}_train_task2
T2_EVAL_DIR=work_dirs/eval_IDD_L_t2_40e_nowapr_task2

echo "=========================================="
echo "YOLO-UniOW IDD T2 FINE-TUNING (no WAPR/SCPI)"
echo "=========================================="
echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"
echo ""
echo "Configuration:"
echo "  Dataset:       $DATASET T2 (PREV=8, CUR=6, total=14)"
echo "  Config:        $CONFIG"
echo "  T1 checkpoint: $T1_CKPT"
echo "  Few-shot:      ${FEWSHOT_K}-shot, seed${FEWSHOT_SEED}"
echo "  GPUs:          $NUM_GPUS"
echo "  T2 train dir:  $T2_WORK_DIR"
echo "  T2 eval dir:   $T2_EVAL_DIR"
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

mkdir -p slurm_logs "$T2_WORK_DIR"

# ─── Verify T1 checkpoint exists ─────────────────────────────────────────────
if [ ! -f "$T1_CKPT" ]; then
    echo "ERROR: T1 checkpoint not found: $T1_CKPT"
    exit 1
fi
echo "T1 checkpoint: $T1_CKPT"

# ─── Verify T2 embeddings exist ──────────────────────────────────────────────
if [ ! -f "embeddings/uniow-idd/idd_t2.npy" ]; then
    echo "ERROR: idd_t2.npy not found. Run extract_idd_embeddings.py --task 2 first."
    exit 1
fi
echo "T2 embeddings: embeddings/uniow-idd/idd_t2.npy"

# ─── T2 Train ─────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Training T2 (10-shot, 40 epochs)"
echo "  Started at: $(date)"
echo "=========================================="

DATASET=${DATASET} TASK=2 THRESHOLD=${THRESHOLD} SAVE=False \
    FEWSHOT_K=${FEWSHOT_K} FEWSHOT_SEED=${FEWSHOT_SEED} FEWSHOT_DIR=${FEWSHOT_DIR} \
    PORT=${MASTER_PORT} \
    ./tools/dist_train_owod.sh ${CONFIG} ${NUM_GPUS} \
    --amp \
    --work-dir ${T2_WORK_DIR} \
    --cfg-options load_from=${T1_CKPT}

if [ $? -ne 0 ]; then
    echo "ERROR: T2 training failed. Aborting."
    exit 1
fi
echo "T2 training finished at: $(date)"

# ─── Find T2 best checkpoint ─────────────────────────────────────────────────
T2_BEST=$(ls -t "${T2_WORK_DIR}"/best*.pth 2>/dev/null | head -1)
if [ -z "$T2_BEST" ]; then
    echo "ERROR: No best_*.pth found in ${T2_WORK_DIR}. Cannot run inference."
    exit 1
fi
echo "T2 best checkpoint: $T2_BEST"

# ─── T2 Inference / Evaluation ───────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Running Inference / Evaluation on T2"
echo "  checkpoint: $T2_BEST"
echo "  eval dir:   $T2_EVAL_DIR"
echo "  Started at: $(date)"
echo "=========================================="

mkdir -p "$T2_EVAL_DIR"
DATASET=${DATASET} TASK=2 THRESHOLD=${THRESHOLD} SAVE=True \
    PORT=${MASTER_PORT} \
    ./tools/dist_test.sh ${CONFIG} ${T2_BEST} ${NUM_GPUS} \
    --work-dir ${T2_EVAL_DIR}

echo ""
echo "=========================================="
echo "T2 TRAINING + INFERENCE COMPLETE"
echo "Job finished at: $(date)"
echo "=========================================="
