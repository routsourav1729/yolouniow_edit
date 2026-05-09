#!/bin/bash
#SBATCH --job-name=uniow_l_t2
#SBATCH --output=slurm_logs/owod_l_t2_%j.out
#SBATCH --error=slurm_logs/owod_l_t2_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:30:00

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
#   --wapr                Enable WAPR only
#   --kume                Enable KUME only
#   --wapr-kume           Enable both WAPR and KUME together (uses kume config which includes wapr dict)
#   --gpus N              Number of GPUs (default: SLURM_GPUS_ON_NODE or 1)
#   --threshold T         Detection threshold (default: 0.05)
#   --config PATH         Override T2 config file path
#   --t1-ckpt PATH        Override T1 best checkpoint path (auto-detected by default)
#   --alpha A             TAL assigner cls-score exponent (default: 0.5, try 0.25)
#   --beta B              TAL assigner IoU exponent (default: 6.0, try 8.0)
# ============================================================================

# ─── Parse arguments ─────────────────────────────────────────────────────────
DATASET=""
FEWSHOT_K=10
FEWSHOT_SEED=1
USE_WAPR=0
USE_KUME=0
USE_KUME_KVK=0
KUME_KVK_MARGIN_ARG=""
NUM_GPUS=""
THRESHOLD=0.05
CONFIG_OVERRIDE=""
T1_CKPT_OVERRIDE=""
TAL_ALPHA=""
TAL_BETA=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)    DATASET="$2"; shift 2 ;;
        --shot)       FEWSHOT_K="$2"; shift 2 ;;
        --seed)       FEWSHOT_SEED="$2"; shift 2 ;;
        --wapr)       USE_WAPR=1; shift 1 ;;
        --kume)       USE_KUME=1; shift 1 ;;
        --wapr-kume)  USE_WAPR=1; USE_KUME=1; shift 1 ;;
        --kume-kvk)   USE_KUME_KVK=1; shift 1 ;;
        --kvk-margin) KUME_KVK_MARGIN_ARG="$2"; shift 2 ;;
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        --threshold)  THRESHOLD="$2"; shift 2 ;;
        --config)     CONFIG_OVERRIDE="$2"; shift 2 ;;
        --t1-ckpt)    T1_CKPT_OVERRIDE="$2"; shift 2 ;;
        --alpha)      TAL_ALPHA="$2"; shift 2 ;;
        --beta)       TAL_BETA="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: sbatch scripts/train_owod_l_t2.sbatch --dataset DATASET [--shot K] [--seed S] [--wapr|--kume|--wapr-kume] [--gpus N] [--threshold T] [--config PATH] [--t1-ckpt PATH] [--alpha A] [--beta B]"
            exit 1
            ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset is required."
    echo "Usage: sbatch scripts/train_owod_l_t2.sbatch --dataset DATASET"
    echo "Available datasets: nuOWODB, IDD, FOOD_VOC, FOOD_VOCCOCO"
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

# ─── Route logs to dataset subfolder ─────────────────────────────────────────
case "$DATASET" in
    nuOWODB)       _LOG_SUBDIR=slurm_logs/nuscenes ;;
    IDD)           _LOG_SUBDIR=slurm_logs/idd ;;
    FOOD_VOCCOCO)  _LOG_SUBDIR=slurm_logs/voccoco ;;
    FOOD_VOC)      _LOG_SUBDIR=slurm_logs/voc ;;
    *)             _LOG_SUBDIR=slurm_logs/$(echo "$DATASET" | tr '[:upper:]' '[:lower:]') ;;
esac
mkdir -p "$_LOG_SUBDIR"
ln -sf "../owod_l_t2_${SLURM_JOB_ID}.out" "${_LOG_SUBDIR}/owod_l_t2_${SLURM_JOB_ID}.out" 2>/dev/null || true
ln -sf "../owod_l_t2_${SLURM_JOB_ID}.err" "${_LOG_SUBDIR}/owod_l_t2_${SLURM_JOB_ID}.err" 2>/dev/null || true
echo "[Log] ${_LOG_SUBDIR}/owod_l_t2_${SLURM_JOB_ID}.{out,err}"

# ─── Dataset-specific configuration ──────────────────────────────────────────
STEM=yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod
DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

case $DATASET in
    nuOWODB)
        # nuOWODB T2 fewshot — uses dedicated T2 config (NOT the default OWOD T2 config).
        # Fewshot split: data/OWOD/nuowodb_fewshot_tfa/seed{S}/box_{K}shot_{cls}_train.txt
        # Embeddings: nuowodb_t2.npy (17 classes, human-readable CLIP prompts from Table 11)
        # Anchor: nuowodb_object_tuned.npy (extracted from nuOWODB T1 checkpoint)
        if [ -n "$CONFIG_OVERRIDE" ]; then
            CONFIG="$CONFIG_OVERRIDE"
        elif [ "$USE_KUME_KVK" -eq 1 ]; then
            CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_nuowodb_t2_kume_kvk.py
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
        elif [ "$USE_KUME_KVK" -eq 1 ]; then
            CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_kume_kvk.py
        elif [ "$USE_KUME" -eq 1 ]; then
            # kume config now includes wapr dict — covers both --kume and --wapr --kume
            CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_kume.py
        elif [ "$USE_WAPR" -eq 1 ]; then
            CONFIG=configs/owod_ft/${STEM}_idd_wapr.py
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
        elif [ "$USE_WAPR" -eq 1 ]; then
            CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_5e_8gpus_owod_food_voc_t2_wapr.py
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
        elif [ "$USE_WAPR" -eq 1 ]; then
            CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voccoco_t2_wapr.py
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
if [ "$USE_KUME_KVK" -eq 1 ]; then
    # Resolve effective margin: --kvk-margin arg > env var > default 0.5
    if [ -n "$KUME_KVK_MARGIN_ARG" ]; then
        export KUME_KVK_MARGIN="$KUME_KVK_MARGIN_ARG"
    fi
    EFFECTIVE_MARGIN="${KUME_KVK_MARGIN:-0.5}"
    METHOD_SUFFIX="_kume_kvk_m${EFFECTIVE_MARGIN}"
elif [ "$USE_KUME" -eq 1 ] && [ "$USE_WAPR" -eq 1 ]; then
    METHOD_SUFFIX="_wapr_kume"
elif [ "$USE_KUME" -eq 1 ]; then
    METHOD_SUFFIX="_kume"
elif [ "$USE_WAPR" -eq 1 ]; then
    METHOD_SUFFIX="_wapr"
fi
T2_WORK_DIR=work_dirs/${STEM}_${DATASET_LOWER}_train_task2_${FEWSHOT_K}shot_seed${FEWSHOT_SEED}${METHOD_SUFFIX}
T2_EVAL_DIR=work_dirs/eval_${DATASET}_L_test_task2_${FEWSHOT_K}shot_seed${FEWSHOT_SEED}${METHOD_SUFFIX}

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
echo "  WAPR:          $USE_WAPR"
echo "  KUME:          $USE_KUME"
echo "  KUME-KVK:      $USE_KUME_KVK${USE_KUME_KVK:+ (margin=${KUME_KVK_MARGIN:-0.5})}"
echo "  Config:        $CONFIG"
echo "  TAL alpha:     ${TAL_ALPHA:-0.5 (default)}"
echo "  TAL beta:      ${TAL_BETA:-6.0 (default)}"
echo "  Few-shot:      k=${FEWSHOT_K}, seed=${FEWSHOT_SEED}"
echo "  Fewshot dir:   $FEWSHOT_DIR"
echo "  T1 checkpoint: ${T1_BEST:-<not found>}"
echo "  GPUs:          $NUM_GPUS"
echo "  Threshold:     $THRESHOLD"
echo "  Embed dir:     $EMBED_DIR"
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

DATASET=${DATASET} TASK=2 THRESHOLD=${THRESHOLD} SAVE=False \
    FEWSHOT_K=${FEWSHOT_K} FEWSHOT_SEED=${FEWSHOT_SEED} FEWSHOT_DIR=${FEWSHOT_DIR} \
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

# ─── Task 2: Evaluation ──────────────────────────────────────────────────────
echo "=========================================="
echo "Running Inference / Evaluation on T2"
echo "  checkpoint: $T2_BEST"
echo "  eval dir:   $T2_EVAL_DIR"
echo "  Started at: $(date)"
echo "=========================================="

DATASET=${DATASET} TASK=2 THRESHOLD=${THRESHOLD} SAVE=True \
    FEWSHOT_DIR=${FEWSHOT_DIR} \
    PORT=${MASTER_PORT} \
    ./tools/dist_test.sh ${CONFIG} ${T2_BEST} ${NUM_GPUS} \
    --work-dir ${T2_EVAL_DIR}

echo ""
echo "=========================================="
echo "YOLO-UniOW-L  T2 FEW-SHOT TRAINING + EVAL COMPLETE"
echo "  Dataset:  $DATASET"
echo "  Shot:     $FEWSHOT_K"
echo "  Seed:     $FEWSHOT_SEED"
echo "Job finished at: $(date)"
echo "=========================================="
