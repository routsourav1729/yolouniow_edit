#!/bin/bash
#SBATCH --job-name=uniow_l_t2
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00

# ============================================================================
# YOLO-UniOW-L  Parameterized OWOD — Task 2 (Few-Shot Novel Class Finetuning)
# ============================================================================
# Generic T2 training script that works with any registered OWOD dataset.
#
# Usage:
#   sbatch scripts/train_owod_l_t2.sbatch --dataset FOOD_VOC
#   sbatch scripts/train_owod_l_t2.sbatch --dataset FOOD_VOC --shot 5 --seed 2
#   sbatch scripts/train_owod_l_t2.sbatch --dataset FOOD_VOCCOCO --shot 10 --seed 1
#   sbatch scripts/train_owod_l_t2.sbatch --dataset IDD --shot 10 --seed 1
#
# Required arguments:
#   --dataset DATASET     Dataset name (IDD, FOOD_VOC, FOOD_VOCCOCO, etc.)
#
# Optional arguments:
#   --shot K              Number of shots per novel class (default: 10)
#   --seed S              Fewshot seed index (default: 1)
#   --fedbce              Enable first FedBCE ablation (k-shot base negatives + novel positives)
#   --hardneg             Enable contrastive hardneg from T1 cache (forces FEWSHOT_SCOPE=all_known)
#   --hardneg-cache PATH  Override hardneg cache .pt path
#   --hardneg-weight W    Hardneg loss weight (default: 1.0)
#   --hardneg-no-novel    Target T_unk only (skip novel-prompt hard-negative term)
#   --hardneg-no-unk      Target novel only (skip T_unk hard-negative term)
#   --gpus N              Number of GPUs (default: SLURM_GPUS_ON_NODE or 1)
#   --threshold T         Detection threshold (default: 0.05)
#   --config PATH         Override T2 config file path
#   --t1-ckpt PATH        Override T1 best checkpoint path (auto-detected by default)
#   --alpha A             TAL assigner cls-score exponent (default: 0.5, try 0.25)
#   --beta B              TAL assigner IoU exponent (default: 6.0, try 8.0)
#   --suffix S            Extra suffix appended to T2 work/eval dirs and log names
#   --final-eval          Run an extra post-training test pass (default: off)
# ============================================================================

# ─── Parse arguments ─────────────────────────────────────────────────────────
DATASET=""
FEWSHOT_K=10
FEWSHOT_SEED=1
USE_FEDBCE=0
USE_HARDNEG=0
HARDNEG_CACHE_ARG=""
HARDNEG_WEIGHT_ARG=""
HARDNEG_TARGET_NOVEL_ARG=""
HARDNEG_TARGET_UNK_ARG=""
NUM_GPUS=""
THRESHOLD=0.05
CONFIG_OVERRIDE=""
T1_CKPT_OVERRIDE=""
TAL_ALPHA=""
TAL_BETA=""
EXTRA_SUFFIX=""
RUN_FINAL_EVAL=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)    DATASET="$2"; shift 2 ;;
        --shot)       FEWSHOT_K="$2"; shift 2 ;;
        --seed)       FEWSHOT_SEED="$2"; shift 2 ;;
        --fedbce)               USE_FEDBCE=1; shift 1 ;;
        --hardneg)              USE_HARDNEG=1; shift 1 ;;
        --hardneg-cache)        HARDNEG_CACHE_ARG="$2"; shift 2 ;;
        --hardneg-weight)       HARDNEG_WEIGHT_ARG="$2"; shift 2 ;;
        --hardneg-no-novel)     HARDNEG_TARGET_NOVEL_ARG="False"; shift 1 ;;
        --hardneg-no-unk)       HARDNEG_TARGET_UNK_ARG="False"; shift 1 ;;
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        --threshold)  THRESHOLD="$2"; shift 2 ;;
        --config)     CONFIG_OVERRIDE="$2"; shift 2 ;;
        --t1-ckpt)    T1_CKPT_OVERRIDE="$2"; shift 2 ;;
        --alpha)      TAL_ALPHA="$2"; shift 2 ;;
        --beta)       TAL_BETA="$2"; shift 2 ;;
        --suffix)     EXTRA_SUFFIX="$2"; shift 2 ;;
        --final-eval) RUN_FINAL_EVAL=1; shift 1 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: sbatch scripts/train_owod_l_t2.sbatch --dataset DATASET [--shot K] [--seed S] [--fedbce|--hardneg] [--hardneg-weight W] [--hardneg-no-novel] [--hardneg-no-unk] [--gpus N] [--threshold T] [--config PATH] [--t1-ckpt PATH] [--alpha A] [--beta B] [--suffix S] [--final-eval]"
            exit 1
            ;;
    esac
done

# ─── Route logs directly to dataset subfolder ────────────────────────────────
DATASET_FOR_LOG=${DATASET:-unknown}
DATASET_LOG_LOWER=$(echo "$DATASET_FOR_LOG" | tr '[:upper:]' '[:lower:]')
case "$DATASET_FOR_LOG" in
    nuOWODB)       _LOG_SUBDIR=slurm_logs/nuscenes ;;
    IDD)           _LOG_SUBDIR=slurm_logs/idd ;;
    FOOD_VOCCOCO)  _LOG_SUBDIR=slurm_logs/voccoco ;;
    FOOD_VOC)      _LOG_SUBDIR=slurm_logs/voc ;;
    *)             _LOG_SUBDIR=slurm_logs/${DATASET_LOG_LOWER} ;;
esac
METHOD_NAME=vanilla
if [ "$USE_FEDBCE" -eq 1 ]; then
    METHOD_NAME=fedbce
fi
if [ "$USE_HARDNEG" -eq 1 ]; then
    if [ "$METHOD_NAME" = "vanilla" ]; then
        METHOD_NAME=hardneg
    else
        METHOD_NAME="${METHOD_NAME}_hardneg"
    fi
fi
_LOG_SUBDIR="${_LOG_SUBDIR}/${METHOD_NAME}"
LOG_SUFFIX=$(printf '%s' "${EXTRA_SUFFIX}" | sed 's/^_//' | tr -c '[:alnum:]_.-' '_')
if [ -n "$LOG_SUFFIX" ]; then
    RUN_TAG="${DATASET_LOG_LOWER}_t2_${METHOD_NAME}_${LOG_SUFFIX}"
else
    RUN_TAG="${DATASET_LOG_LOWER}_t2_${METHOD_NAME}"
fi
mkdir -p "$_LOG_SUBDIR"
LOG_OUT="${_LOG_SUBDIR}/${RUN_TAG}_${SLURM_JOB_ID}.out"
LOG_ERR="${_LOG_SUBDIR}/${RUN_TAG}_${SLURM_JOB_ID}.err"
exec > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)
echo "[Log] ${LOG_OUT}"
echo "[Err] ${LOG_ERR}"

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset is required."
    echo "Usage: sbatch scripts/train_owod_l_t2.sbatch --dataset DATASET"
    echo "Available datasets: nuOWODB, IDD, FOOD_VOC, FOOD_VOCCOCO"
    exit 1
fi

if [ "$USE_FEDBCE" -eq 1 ] && [ "$USE_HARDNEG" -eq 1 ]; then
    echo "ERROR: --fedbce is the first isolated ablation. Do not combine with HardNeg."
    exit 1
fi

NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}

# ─── Build TAL alpha/beta cfg-options string ──────────────────────────────────
TAL_CFG_OPTIONS=""
if [ -n "$TAL_ALPHA" ] || [ -n "$TAL_BETA" ]; then
    _A=${TAL_ALPHA:-0.5}
    _B=${TAL_BETA:-6.0}
    TAL_CFG_OPTIONS="model.train_cfg.one2many_assigner.alpha=${_A} model.train_cfg.one2many_assigner.beta=${_B} model.train_cfg.one2one_assigner.alpha=${_A} model.train_cfg.one2one_assigner.beta=${_B}"
fi

# ─── Dataset-specific configuration ──────────────────────────────────────────
STEM=yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod
DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

case $DATASET in
    nuOWODB)
        if [ -n "$CONFIG_OVERRIDE" ]; then
            CONFIG="$CONFIG_OVERRIDE"
        elif [ "$USE_FEDBCE" -eq 1 ]; then
            CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_nuowodb_t2_fedbce.py
        else
            CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_nuowodb_t2.py
        fi
        EMBED_DIR=embeddings/uniow-l
        EMBED_PREFIX=nuowodb
        FEWSHOT_DIR=data/OWOD/nuowodb_fewshot_tfa
        ;;
    IDD)
        if [ -n "$CONFIG_OVERRIDE" ]; then
            CONFIG="$CONFIG_OVERRIDE"
        elif [ "$USE_FEDBCE" -eq 1 ]; then
            CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_fedbce.py
        elif [ "$USE_HARDNEG" -eq 1 ]; then
            CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_hardneg.py
        else
            CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2.py
        fi
        EMBED_DIR=embeddings/uniow-idd
        EMBED_PREFIX=idd
        FEWSHOT_DIR=data/OWOD/iddsplit
        ;;
    FOOD_VOC)
        if [ -n "$CONFIG_OVERRIDE" ]; then
            CONFIG="$CONFIG_OVERRIDE"
        else
            CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voc_t2.py
        fi
        EMBED_DIR=embeddings/uniow-food-voc
        EMBED_PREFIX=food_voc
        FEWSHOT_DIR=data/OWOD/vocsplit
        ;;
    FOOD_VOCCOCO)
        if [ -n "$CONFIG_OVERRIDE" ]; then
            CONFIG="$CONFIG_OVERRIDE"
        else
            CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voccoco_t2.py
        fi
        EMBED_DIR=embeddings/uniow-food-voccoco
        EMBED_PREFIX=food_voccoco
        FEWSHOT_DIR=data/OWOD/voccocosplit
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'. Available: nuOWODB, IDD, FOOD_VOC, FOOD_VOCCOCO"
        exit 1
        ;;
esac

# T1 checkpoint — auto-detect from T1 work dir or use override
T1_WORK_DIR=work_dirs/${STEM}_${DATASET_LOWER}_train_task1
if [ -n "$T1_CKPT_OVERRIDE" ]; then
    T1_BEST="$T1_CKPT_OVERRIDE"
else
    T1_BEST=$(ls -t "${T1_WORK_DIR}"/best*.pth 2>/dev/null | head -1)
fi

# T2 directories
METHOD_SUFFIX=""
if [ "$USE_FEDBCE" -eq 1 ]; then
    METHOD_SUFFIX="_fedbce"
fi
if [ "$USE_HARDNEG" -eq 1 ]; then
    HW_TAG=$(echo "${HARDNEG_WEIGHT_ARG:-1.0}" | sed 's/\./p/g')
    HN_TARGET_TAG=""
    if [ "$HARDNEG_TARGET_NOVEL_ARG" = "False" ] && [ "$HARDNEG_TARGET_UNK_ARG" != "False" ]; then
        HN_TARGET_TAG="_unkonly"
    elif [ "$HARDNEG_TARGET_UNK_ARG" = "False" ] && [ "$HARDNEG_TARGET_NOVEL_ARG" != "False" ]; then
        HN_TARGET_TAG="_novelonly"
    fi
    METHOD_SUFFIX="${METHOD_SUFFIX}_hardneg_w${HW_TAG}${HN_TARGET_TAG}"
fi
METHOD_SUFFIX="${METHOD_SUFFIX}${EXTRA_SUFFIX}"
T2_WORK_DIR=work_dirs/${STEM}_${DATASET_LOWER}_train_task2_${FEWSHOT_K}shot_seed${FEWSHOT_SEED}${METHOD_SUFFIX}
T2_EVAL_DIR=work_dirs/eval_${DATASET}_L_test_task2_${FEWSHOT_K}shot_seed${FEWSHOT_SEED}${METHOD_SUFFIX}
FEWSHOT_SCOPE_ENV=novel
if [ "$USE_HARDNEG" -eq 1 ] || [ "$USE_FEDBCE" -eq 1 ]; then
    FEWSHOT_SCOPE_ENV=all_known
fi

echo "=========================================="
echo "YOLO-UniOW-L  T2 FEW-SHOT FINETUNING"
echo "=========================================="
echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"
echo ""
echo "Configuration:"
echo "  Dataset:       $DATASET"
echo "  Task:          2 (Few-shot finetuning)"
echo "  FedBCE:        $USE_FEDBCE"
echo "  HardNeg:       $USE_HARDNEG"
echo "  Config:        $CONFIG"
echo "  TAL alpha:     ${TAL_ALPHA:-0.5 (default)}"
echo "  TAL beta:      ${TAL_BETA:-6.0 (default)}"
echo "  Few-shot:      k=${FEWSHOT_K}, seed=${FEWSHOT_SEED}, scope=${FEWSHOT_SCOPE_ENV}"
echo "  Fewshot dir:   $FEWSHOT_DIR"
echo "  T1 checkpoint: ${T1_BEST:-<not found>}"
echo "  GPUs:          $NUM_GPUS"
echo "  Threshold:     $THRESHOLD"
echo "  Embed dir:     $EMBED_DIR"
echo "  T2 train dir:  $T2_WORK_DIR"
echo "  T2 eval dir:   $T2_EVAL_DIR"
echo "  Final eval:    $RUN_FINAL_EVAL"
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

echo "Environment:"
echo "  Python:        $(which python)"
echo "  Conda env:     $CONDA_DEFAULT_ENV"
echo "  MASTER_PORT:   $MASTER_PORT"
python -c "import torch; print(f'  PyTorch:       {torch.__version__}')"
python -c "import torch; print(f'  CUDA devices:  {torch.cuda.device_count()}')"
echo ""

# ─── Validate T1 checkpoint ──────────────────────────────────────────────────
if [ -z "$T1_BEST" ] || [ ! -f "$T1_BEST" ]; then
    echo "ERROR: T1 best checkpoint not found in ${T1_WORK_DIR}."
    echo "  Run T1 training first: sbatch scripts/train_owod_l.sbatch --dataset $DATASET"
    echo "  Or provide manually:   --t1-ckpt /path/to/best.pth"
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
echo "Sample few-shot files:"
ls "${SEED_DIR}"/box_${FEWSHOT_K}shot_*_train.txt 2>/dev/null | head -5
NSHOT_FILES=$(ls "${SEED_DIR}"/box_${FEWSHOT_K}shot_*_train.txt 2>/dev/null | wc -l)
echo "  Total ${FEWSHOT_K}-shot files: $NSHOT_FILES"
if [ "$NSHOT_FILES" -eq 0 ]; then
    echo "ERROR: No ${FEWSHOT_K}-shot files found in $SEED_DIR"
    exit 1
fi
echo ""

# ─── Validate T2 embeddings ──────────────────────────────────────────────────
if [ ! -f "${EMBED_DIR}/${EMBED_PREFIX}_t2.npy" ]; then
    echo "T2 embeddings not found — generating from cached CLIP model..."
    python tools/owod_scripts/extract_embeddings.py --dataset "$DATASET" --task 2 --device auto
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to extract T2 embeddings. Aborting."
        exit 1
    fi
else
    echo "T2 embeddings: $(du -h ${EMBED_DIR}/${EMBED_PREFIX}_t2.npy | cut -f1) — ${EMBED_DIR}/${EMBED_PREFIX}_t2.npy"
fi
echo ""

echo "Environment:"
echo "  Python:        $(which python)"
echo "  Conda env:     $CONDA_DEFAULT_ENV"
echo "  MASTER_PORT:   $MASTER_PORT"
python -c "import torch; print(f'  PyTorch:       {torch.__version__}')"
python -c "import torch; print(f'  CUDA devices:  {torch.cuda.device_count()}')"
echo ""

# ─── Ensure dirs exist ───────────────────────────────────────────────────────
mkdir -p slurm_logs "$T2_WORK_DIR" "$T2_EVAL_DIR"

# ─── Task 2: Train ───────────────────────────────────────────────────────────
echo "=========================================="
echo "Training Task 2 (Few-Shot, ${FEWSHOT_K}-shot, seed ${FEWSHOT_SEED})"
echo "  work_dir:   $T2_WORK_DIR"
echo "  load_from:  $T1_BEST"
echo "  fewshot:    k=${FEWSHOT_K}, seed=${FEWSHOT_SEED}, dir=${FEWSHOT_DIR}"
echo "  Started at: $(date)"
echo "=========================================="

if [ "$USE_HARDNEG" -eq 1 ]; then
    HARDNEG_CACHE_PATH_EFFECTIVE="${HARDNEG_CACHE_ARG:-cache/${DATASET_LOWER}/hardneg_t1_o2o_10x8x3.pt}"
    HARDNEG_WEIGHT_EFFECTIVE="${HARDNEG_WEIGHT_ARG:-1.0}"
    if [ ! -f "$HARDNEG_CACHE_PATH_EFFECTIVE" ]; then
        echo "ERROR: hardneg cache not found at $HARDNEG_CACHE_PATH_EFFECTIVE"
        echo "  Build it first via scripts/experiment/collect_hardneg_idd.sbatch (IDD) or equivalent."
        exit 1
    fi
    export HARDNEG_CACHE_PATH="$HARDNEG_CACHE_PATH_EFFECTIVE"
    export HARDNEG_WEIGHT="$HARDNEG_WEIGHT_EFFECTIVE"
    if [ -n "$HARDNEG_TARGET_NOVEL_ARG" ]; then
        export HARDNEG_TARGET_NOVEL="$HARDNEG_TARGET_NOVEL_ARG"
    fi
    if [ -n "$HARDNEG_TARGET_UNK_ARG" ]; then
        export HARDNEG_TARGET_UNK="$HARDNEG_TARGET_UNK_ARG"
    fi
    echo "HardNeg cache:        $HARDNEG_CACHE_PATH"
    echo "HardNeg weight:       $HARDNEG_WEIGHT"
    echo "HardNeg target_unk:   ${HARDNEG_TARGET_UNK:-True (default)}"
    echo "HardNeg target_novel: ${HARDNEG_TARGET_NOVEL:-True (default)}"
fi

DATASET=${DATASET} TASK=2 THRESHOLD=${THRESHOLD} SAVE=False \
    FEWSHOT_K=${FEWSHOT_K} FEWSHOT_SEED=${FEWSHOT_SEED} FEWSHOT_DIR=${FEWSHOT_DIR} FEWSHOT_SCOPE=${FEWSHOT_SCOPE_ENV} \
    PORT=${MASTER_PORT} \
    ./tools/dist_train_owod.sh ${CONFIG} ${NUM_GPUS} \
    --amp \
    --work-dir ${T2_WORK_DIR} \
    --cfg-options load_from=${T1_BEST} ${TAL_CFG_OPTIONS}

T2_TRAIN_RC=$?
if [ $T2_TRAIN_RC -ne 0 ]; then
    echo "ERROR: T2 training failed (exit code $T2_TRAIN_RC). Aborting."
    exit $T2_TRAIN_RC
fi
echo "T2 training finished at: $(date)"
echo ""

# ─── Find T2 best checkpoint ─────────────────────────────────────────────────
T2_BEST=$(ls -t "${T2_WORK_DIR}"/best*.pth 2>/dev/null | head -1)
if [ -z "$T2_BEST" ]; then
    echo "WARNING: No best_*.pth found in ${T2_WORK_DIR}. Using last checkpoint."
    T2_BEST=$(ls -t "${T2_WORK_DIR}"/epoch_*.pth 2>/dev/null | head -1)
fi
if [ -z "$T2_BEST" ]; then
    echo "ERROR: No checkpoint found in ${T2_WORK_DIR}."
    exit 1
fi
echo "T2 best checkpoint: $T2_BEST"

if [ "$RUN_FINAL_EVAL" -eq 1 ]; then
    # ─── Task 2: Optional post-training evaluation ──────────────────────────
    echo "=========================================="
    echo "Running Inference / Evaluation on T2"
    echo "  checkpoint: $T2_BEST"
    echo "  eval dir:   $T2_EVAL_DIR"
    echo "  Started at: $(date)"
    echo "=========================================="

    DATASET=${DATASET} TASK=2 THRESHOLD=${THRESHOLD} SAVE=True \
        FEWSHOT_DIR=${FEWSHOT_DIR} FEWSHOT_SCOPE=${FEWSHOT_SCOPE_ENV} \
        PORT=${MASTER_PORT} \
        ./tools/dist_test.sh ${CONFIG} ${T2_BEST} ${NUM_GPUS} \
        --work-dir ${T2_EVAL_DIR}
else
    echo "Skipping extra post-training eval; using the final validation metrics."
fi

echo ""
echo "=========================================="
echo "YOLO-UniOW-L  T2 FEW-SHOT TRAINING + EVAL COMPLETE"
echo "  Dataset:  $DATASET"
echo "  Shot:     $FEWSHOT_K"
echo "  Seed:     $FEWSHOT_SEED"
echo "Job finished at: $(date)"
echo "=========================================="
