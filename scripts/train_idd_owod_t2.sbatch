#!/bin/bash
#SBATCH --job-name=uniow_l_t2_nowapr
#SBATCH --output=slurm_logs/idd_owod_l_t2_%j.out
#SBATCH --error=slurm_logs/idd_owod_l_t2_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# ============================================================================
# YOLO-UniOW-L  IDD OWOD  — Task 2 (Few-Shot, 80 epochs, no WAPR / no SCPI)
# ============================================================================
#
# Prerequisites:
#   - T1 best checkpoint must exist in $T1_WORK_DIR
#   - 10-shot split files: data/OWOD/iddsplit/seed1/box_10shot_<cls>_train.txt
#   - T2 embeddings:       embeddings/uniow-idd/idd_t2.npy  (14 classes)
#
# What gets trained in T2:
#   - 6 novel class embeddings (bus, truck, tanker_vehicle, crane_truck,
#     street_cart, excavator)
#   - 1 unknown embedding
#   Frozen: backbone, neck, heads, 8 base embeddings, anchor embedding.
#
# Evaluation: runs inference at every saved checkpoint (ep 20/40/60/80).
#
# Usage:
#   sbatch scripts/train_idd_owod_t2.sbatch
#   sbatch scripts/train_idd_owod_t2.sbatch [num_gpus]
# ============================================================================

NUM_GPUS=${1:-${SLURM_GPUS_ON_NODE:-1}}

DATASET=IDD
CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2.py
THRESHOLD=0.05

# Few-shot settings
FEWSHOT_K=10
FEWSHOT_SEED=1
FEWSHOT_DIR=data/OWOD/iddsplit

STEM=yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod

# T1 checkpoint
T1_WORK_DIR=work_dirs/${STEM}_idd_train_task1
T1_BEST=$(ls -t "${T1_WORK_DIR}"/best*.pth 2>/dev/null | head -1)

# T2 directories
T2_WORK_DIR=work_dirs/${STEM}_idd_train_task2_${FEWSHOT_K}shot_nowapr_80e

echo "=========================================="
echo "YOLO-UniOW-L  IDD T2 (80 ep, no WAPR/SCPI)"
echo "=========================================="
echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"
echo ""
echo "Configuration:"
echo "  Dataset:       $DATASET T2 (PREV=8, CUR=6, total=14)"
echo "  Config:        $CONFIG"
echo "  Few-shot:      k=${FEWSHOT_K}, seed=${FEWSHOT_SEED}"
echo "  T1 checkpoint: $T1_BEST"
echo "  GPUs:          $NUM_GPUS"
echo "  T2 train dir:  $T2_WORK_DIR"
echo "=========================================="

# ─── Validate T1 checkpoint ──────────────────────────────────────────────────
if [ -z "$T1_BEST" ] || [ ! -f "$T1_BEST" ]; then
    echo "ERROR: T1 best checkpoint not found in ${T1_WORK_DIR}."
    exit 1
fi
echo "T1 best checkpoint: $(du -h "$T1_BEST" | cut -f1) — $T1_BEST"

# ─── Validate few-shot split files ────────────────────────────────────────────
SEED_DIR="${FEWSHOT_DIR}/seed${FEWSHOT_SEED}"
if [ ! -d "$SEED_DIR" ]; then
    echo "ERROR: Few-shot seed directory not found: $SEED_DIR"
    exit 1
fi
echo "Few-shot split dir: $SEED_DIR"
for cls in bus truck tanker_vehicle crane_truck street_cart excavator; do
    shot_file="${SEED_DIR}/box_${FEWSHOT_K}shot_${cls}_train.txt"
    if [ ! -f "$shot_file" ]; then
        echo "ERROR: Missing few-shot file: $shot_file"
        exit 1
    fi
    echo "  ${cls}: $(wc -l < "$shot_file") images"
done
echo ""

# ─── Validate T2 embeddings ──────────────────────────────────────────────────
if [ ! -f "embeddings/uniow-idd/idd_t2.npy" ]; then
    echo "ERROR: T2 embeddings not found: embeddings/uniow-idd/idd_t2.npy"
    exit 1
fi
echo "T2 embeddings: $(du -h embeddings/uniow-idd/idd_t2.npy | cut -f1) — embeddings/uniow-idd/idd_t2.npy"
echo ""

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

echo "Environment:"
echo "  Python:        $(which python)"
echo "  Conda env:     $CONDA_DEFAULT_ENV"
echo "  MASTER_PORT:   $MASTER_PORT"
python -c "import torch; print(f'  PyTorch:       {torch.__version__}')"
python -c "import torch; print(f'  CUDA devices:  {torch.cuda.device_count()}')"
echo ""

mkdir -p slurm_logs "$T2_WORK_DIR"

# ─── Task 2: Train (80 epochs) ───────────────────────────────────────────────
echo "=========================================="
echo "Training T2 (${FEWSHOT_K}-shot, 80 epochs, no WAPR)"
echo "  work_dir:  $T2_WORK_DIR"
echo "  load_from: $T1_BEST"
echo "  Started at: $(date)"
echo "=========================================="

DATASET=${DATASET} TASK=2 THRESHOLD=${THRESHOLD} SAVE=False \
    FEWSHOT_K=${FEWSHOT_K} FEWSHOT_SEED=${FEWSHOT_SEED} FEWSHOT_DIR=${FEWSHOT_DIR} \
    PORT=${MASTER_PORT} \
    ./tools/dist_train_owod.sh ${CONFIG} ${NUM_GPUS} \
    --amp \
    --work-dir ${T2_WORK_DIR} \
    --cfg-options load_from=${T1_BEST}

T2_TRAIN_RC=$?
if [ $T2_TRAIN_RC -ne 0 ]; then
    echo "ERROR: T2 training failed (exit code $T2_TRAIN_RC). Aborting."
    exit $T2_TRAIN_RC
fi
echo "T2 training finished at: $(date)"
echo ""

# ─── Helper: run inference on one checkpoint ─────────────────────────────────
run_eval() {
    local ckpt=$1
    local eval_dir=$2
    echo "=========================================="
    echo "Inference: $(basename $ckpt)"
    echo "  eval_dir:   $eval_dir"
    echo "  Started at: $(date)"
    echo "=========================================="
    mkdir -p "$eval_dir"
    DATASET=${DATASET} TASK=2 THRESHOLD=${THRESHOLD} SAVE=True \
        FEWSHOT_DIR=${FEWSHOT_DIR} \
        PORT=${MASTER_PORT} \
        ./tools/dist_test.sh ${CONFIG} ${ckpt} ${NUM_GPUS} \
        --work-dir ${eval_dir}
    echo "  Finished at: $(date)"
    echo ""
}

# ─── Inference at every 20-epoch checkpoint ──────────────────────────────────
echo "=========================================="
echo "Running inference at each 20-epoch checkpoint"
echo "=========================================="

for ep in 20 40 60 80; do
    # Try best checkpoint first, fall back to epoch checkpoint
    ckpt=$(ls -t "${T2_WORK_DIR}"/best_owod_Both_epoch_${ep}.pth 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        ckpt="${T2_WORK_DIR}/epoch_${ep}.pth"
    fi
    if [ ! -f "$ckpt" ]; then
        echo "WARNING: No checkpoint found for epoch ${ep}, skipping."
        continue
    fi
    run_eval "$ckpt" "work_dirs/eval_IDD_L_t2_nowapr_80e_ep${ep}"
done

echo "=========================================="
echo "T2 TRAINING + INFERENCE COMPLETE"
echo "Job finished at: $(date)"
echo "=========================================="
