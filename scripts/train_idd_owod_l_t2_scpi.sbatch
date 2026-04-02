#!/bin/bash
#SBATCH --job-name=uniow_l_scpi
#SBATCH --output=slurm_logs/idd_owod_l_t2_scpi_%j.out
#SBATCH --error=slurm_logs/idd_owod_l_t2_scpi_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00

# ============================================================================
# YOLO-UniOW-L  IDD OWOD  — Task 2  SCPI (Post-Training Calibration)
# ============================================================================
#
# SCPI runs AFTER T2 fine-tuning (WAPR, 40 epochs).
#
#   Step 1: SCPI calibration — load T2 checkpoint, extract post-BN visual
#           features from K-shot support, decide per novel class: keep
#           fine-tuned prompt or revert to zero-shot CLIP prompt → .npy
#   Step 2: Eval — load T2 checkpoint, SCPIHook patches only novel embeddings
#           (slots 8-13) from the .npy. Base/unk/anchor stay from T2 ckpt.
#
# Requires: T2 fine-tuned checkpoint already exists.
#
# Usage:
#   sbatch scripts/train_idd_owod_l_t2_scpi.sbatch
# ============================================================================

NUM_GPUS=${1:-${SLURM_GPUS_ON_NODE:-1}}

DATASET=IDD
THRESHOLD=0.05

# Few-shot settings
FEWSHOT_K=10
FEWSHOT_SEED=1
FEWSHOT_DIR=data/OWOD/iddsplit

STEM=yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod

# T2 fine-tuned checkpoint (produced by WAPR training)
T2_WORK_DIR=work_dirs/${STEM}_idd_train_task2_${FEWSHOT_K}shot_wapr
T2_BEST=$(ls -t "${T2_WORK_DIR}"/best*.pth 2>/dev/null | head -1)

# Zero-shot CLIP embeddings (original, pre-training)
ZEROSHOT_EMB=embeddings/uniow-idd/idd_t2.npy

# Configs
CALIB_CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_wapr.py
EVAL_CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_scpi.py

# Output paths
SCPI_NPY=embeddings/uniow-idd/idd_t2_scpi.npy
EVAL_DIR=work_dirs/eval_IDD_L_test_task2_${FEWSHOT_K}shot_scpi

echo "=========================================="
echo "YOLO-UniOW-L  IDD T2 SCPI (Post-Training)"
echo "=========================================="
echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"
echo ""
echo "Configuration:"
echo "  Dataset:       $DATASET"
echo "  Task:          2 (6 novel classes)"
echo "  Few-shot:      k=${FEWSHOT_K}, seed=${FEWSHOT_SEED}"
echo "  T2 checkpoint: $T2_BEST"
echo "  Zero-shot emb: $ZEROSHOT_EMB"
echo "  SCPI npy:      $SCPI_NPY"
echo "  Eval dir:      $EVAL_DIR"
echo "=========================================="

# ─── Validate ─────────────────────────────────────────────────────────────────
if [ -z "$T2_BEST" ] || [ ! -f "$T2_BEST" ]; then
    echo "ERROR: T2 best checkpoint not found in ${T2_WORK_DIR}."
    echo "       Run T2 fine-tuning (WAPR) first."
    exit 1
fi
echo "T2 best checkpoint: $(du -h "$T2_BEST" | cut -f1) — $T2_BEST"

if [ ! -f "$ZEROSHOT_EMB" ]; then
    echo "ERROR: Zero-shot embeddings not found: $ZEROSHOT_EMB"
    exit 1
fi

SEED_DIR="${FEWSHOT_DIR}/seed${FEWSHOT_SEED}"
for cls in bus truck tanker_vehicle crane_truck street_cart excavator; do
    shot_file="${SEED_DIR}/box_${FEWSHOT_K}shot_${cls}_train.txt"
    if [ ! -f "$shot_file" ]; then
        echo "ERROR: Missing few-shot file: $shot_file"
        exit 1
    fi
    echo "  ${cls}: $(wc -l < "$shot_file") images"
done
echo ""

# ─── Environment ──────────────────────────────────────────────────────────────
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
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / NUM_GPUS ))

mkdir -p slurm_logs "$EVAL_DIR" embeddings/uniow-idd

# ─── Step 1: SCPI Calibration → save .npy ─────────────────────────────────────
echo "=========================================="
echo "Step 1: SCPI Calibration (post-training)"
echo "  Started at: $(date)"
echo "=========================================="

DATASET=${DATASET} TASK=2 THRESHOLD=${THRESHOLD} \
    FEWSHOT_K=${FEWSHOT_K} FEWSHOT_SEED=${FEWSHOT_SEED} FEWSHOT_DIR=${FEWSHOT_DIR} \
    python tools/owod_scripts/scpi_calibrate.py \
    --config ${CALIB_CONFIG} \
    --checkpoint ${T2_BEST} \
    --zeroshot-emb ${ZEROSHOT_EMB} \
    --output ${SCPI_NPY} \
    --fewshot-k ${FEWSHOT_K} --fewshot-seed ${FEWSHOT_SEED} \
    --fewshot-dir ${FEWSHOT_DIR} \
    --data-root data/OWOD --dataset ${DATASET}

CALIB_RC=$?
if [ $CALIB_RC -ne 0 ]; then
    echo "ERROR: SCPI calibration failed (exit code $CALIB_RC)"
    exit $CALIB_RC
fi
echo ""

# ─── Step 2: Eval with T2 checkpoint + SCPI novel embeddings ──────────────────
echo "=========================================="
echo "Step 2: Evaluation (T2 ckpt + SCPI novel prompts)"
echo "  Started at: $(date)"
echo "=========================================="

DATASET=${DATASET} TASK=2 THRESHOLD=${THRESHOLD} SAVE=False \
    FEWSHOT_DIR=${FEWSHOT_DIR} FEWSHOT_K=${FEWSHOT_K} FEWSHOT_SEED=${FEWSHOT_SEED} \
    PORT=${MASTER_PORT} \
    ./tools/dist_test.sh ${EVAL_CONFIG} ${T2_BEST} ${NUM_GPUS} \
    --work-dir ${EVAL_DIR} \
    --cfg-options \
    custom_hooks.2.scpi_emb_path=${SCPI_NPY} \
    val_evaluator.save_rets=False \
    test_evaluator.save_rets=False

echo ""
echo "=========================================="
echo "YOLO-UniOW-L  T2 SCPI COMPLETE"
echo "Job finished at: $(date)"
echo "=========================================="
