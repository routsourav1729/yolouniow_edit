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
# YOLO-UniOW-L  IDD OWOD  — Task 2 (SCPI: Training-Free Novel Calibration)
# ============================================================================
#
# Two-step approach:
#   Step 1: Offline calibration — builds model, extracts visual centroids from
#           K-shot support, blends with CLIP prompts → saves .npy (~30s)
#   Step 2: Normal eval — loads T1 checkpoint, SCPIHook patches novel
#           embeddings from npy before test starts → normal speed (~5min)
#
# Usage:
#   sbatch scripts/train_idd_owod_l_t2_scpi.sbatch
#   BETA=15 TAU=0.07 sbatch scripts/train_idd_owod_l_t2_scpi.sbatch
# ============================================================================

NUM_GPUS=${1:-${SLURM_GPUS_ON_NODE:-1}}

DATASET=IDD
THRESHOLD=0.05

# Few-shot settings
FEWSHOT_K=10
FEWSHOT_SEED=1
FEWSHOT_DIR=data/OWOD/iddsplit

# SCPI hyperparameters (overridable via env)
BETA=${BETA:-15.0}
TAU=${TAU:-0.07}

STEM=yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod

# T1 checkpoint
T1_WORK_DIR=work_dirs/${STEM}_idd_train_task1
T1_BEST=$(ls -t "${T1_WORK_DIR}"/best*.pth 2>/dev/null | head -1)

# Configs
CALIB_CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_wapr.py
EVAL_CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_scpi.py

# Output paths
SCPI_NPY=embeddings/uniow-idd/idd_t2_scpi_b${BETA}_t${TAU}.npy
EVAL_DIR=work_dirs/eval_IDD_L_test_task2_${FEWSHOT_K}shot_scpi_b${BETA}_t${TAU}

echo "=========================================="
echo "YOLO-UniOW-L  IDD T2 SCPI (Training-Free)"
echo "=========================================="
echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"
echo ""
echo "Configuration:"
echo "  Dataset:       $DATASET"
echo "  Task:          2 (6 novel classes, 6 unknown)"
echo "  Few-shot:      k=${FEWSHOT_K}, seed=${FEWSHOT_SEED}"
echo "  T1 checkpoint: $T1_BEST"
echo "  SCPI beta:     $BETA"
echo "  SCPI tau:      $TAU"
echo "  SCPI npy:      $SCPI_NPY"
echo "  Eval dir:      $EVAL_DIR"
echo "=========================================="

# ─── Validate ─────────────────────────────────────────────────────────────────
if [ -z "$T1_BEST" ] || [ ! -f "$T1_BEST" ]; then
    echo "ERROR: T1 best checkpoint not found in ${T1_WORK_DIR}."
    exit 1
fi
echo "T1 best checkpoint: $(du -h "$T1_BEST" | cut -f1) — $T1_BEST"

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

# ─── Step 1: SCPI Calibration → save .npy (~30s) ─────────────────────────────
echo "=========================================="
echo "Step 1: SCPI Calibration"
echo "  Started at: $(date)"
echo "=========================================="

DATASET=${DATASET} TASK=2 THRESHOLD=${THRESHOLD} \
    FEWSHOT_K=${FEWSHOT_K} FEWSHOT_SEED=${FEWSHOT_SEED} FEWSHOT_DIR=${FEWSHOT_DIR} \
    python tools/owod_scripts/scpi_calibrate.py \
    --config ${CALIB_CONFIG} \
    --checkpoint ${T1_BEST} \
    --output ${SCPI_NPY} \
    --beta ${BETA} --tau ${TAU} \
    --fewshot-k ${FEWSHOT_K} --fewshot-seed ${FEWSHOT_SEED} \
    --fewshot-dir ${FEWSHOT_DIR} \
    --data-root data/OWOD --dataset ${DATASET}

CALIB_RC=$?
if [ $CALIB_RC -ne 0 ]; then
    echo "ERROR: SCPI calibration failed (exit code $CALIB_RC)"
    exit $CALIB_RC
fi
echo ""

# ─── Step 2: Normal eval with T1 checkpoint + SCPI embedding override (~5min) ─
echo "=========================================="
echo "Step 2: Evaluation"
echo "  Started at: $(date)"
echo "=========================================="

DATASET=${DATASET} TASK=2 THRESHOLD=${THRESHOLD} SAVE=True \
    FEWSHOT_DIR=${FEWSHOT_DIR} FEWSHOT_K=${FEWSHOT_K} FEWSHOT_SEED=${FEWSHOT_SEED} \
    PORT=${MASTER_PORT} \
    ./tools/dist_test.sh ${EVAL_CONFIG} ${T1_BEST} ${NUM_GPUS} \
    --work-dir ${EVAL_DIR} \
    --cfg-options custom_hooks.2.scpi_emb_path=${SCPI_NPY}

echo ""
echo "=========================================="
echo "YOLO-UniOW-L  T2 SCPI COMPLETE"
echo "Job finished at: $(date)"
echo "=========================================="
