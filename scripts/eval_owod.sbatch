#!/bin/bash
#SBATCH --job-name=uniow_eval
#SBATCH --output=slurm_logs/eval_idd_%j.out
#SBATCH --error=slurm_logs/eval_idd_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# ============================================================================
# YOLO-UniOW-L  IDD OWOD — Eval-only script
# ============================================================================
#
# Usage:
#   sbatch scripts/eval_owod.sbatch             → T2 best checkpoint (default)
#   sbatch scripts/eval_owod.sbatch 2           → T2 best checkpoint
#   sbatch scripts/eval_owod.sbatch 1           → T1 best checkpoint
#   sbatch scripts/eval_owod.sbatch 2 "" 1      → T2 + embedding diagnostic
#   sbatch scripts/eval_owod.sbatch 2 /path.pth → custom checkpoint
#   sbatch scripts/eval_owod.sbatch 2 /path.pth 1 → custom ckpt + diagnostic
#
# Output: slurm_logs/eval_idd_<jobid>.out
#   Prints per-class AP, U-Recall, A-OSE, and A-OSE per-class breakdown.
#   If ANALYZE=1: also generates visualizations/embed_diag_t{task}/
# ============================================================================

TASK=${1:-2}
CUSTOM_CKPT=${2:-""}
ANALYZE=${3:-0}        # 1 = run embedding diagnostic after eval

DATASET=IDD
THRESHOLD=0.05
FEWSHOT_K=10
FEWSHOT_SEED=1
FEWSHOT_DIR=data/OWOD/iddsplit   # needed by config parser even at eval time

STEM=yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))

# ─── Select config and work_dir by task ──────────────────────────────────────
if [ "$TASK" = "1" ]; then
    CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py
    WORK_DIR=work_dirs/${STEM}_idd_train_task1
    FEWSHOT_K=0   # T1 has no few-shot; K=0 disables filtering
else
    CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_wapr.py
    WORK_DIR=work_dirs/${STEM}_idd_train_task2_${FEWSHOT_K}shot_wapr
fi

# ─── Resolve checkpoint ───────────────────────────────────────────────────────
if [ -n "$CUSTOM_CKPT" ]; then
    CKPT="$CUSTOM_CKPT"
else
    CKPT=$(ls -t "${WORK_DIR}"/best*.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        CKPT=$(ls -t "${WORK_DIR}"/epoch_*.pth 2>/dev/null | head -1)
    fi
fi

CKPT_STEM=$(basename "$CKPT" .pth)
EVAL_DIR=work_dirs/eval_IDD_L_task${TASK}_${CKPT_STEM}

echo "=========================================="
echo "YOLO-UniOW-L  IDD OWOD  — Eval only"
echo "=========================================="
echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"
echo ""
echo "  Task:       $TASK"
echo "  Config:     $CONFIG"
echo "  Checkpoint: $CKPT"
echo "  Eval dir:   $EVAL_DIR"
echo "  GPUs:       $NUM_GPUS"
echo "  Threshold:  $THRESHOLD"
echo "  Analyze:    $ANALYZE"
echo "=========================================="

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    echo "  Check that ${WORK_DIR} contains best*.pth or epoch_*.pth"
    exit 1
fi
echo "Checkpoint: $(du -h "$CKPT" | cut -f1) — $CKPT"
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
export MASTER_PORT

echo "Environment:"
echo "  Python:      $(which python)"
echo "  Conda env:   $CONDA_DEFAULT_ENV"
echo "  MASTER_PORT: $MASTER_PORT"
python -c "import torch; print(f'  PyTorch:     {torch.__version__}')"
python -c "import torch; print(f'  CUDA devs:   {torch.cuda.device_count()}')"
echo ""

mkdir -p slurm_logs "$EVAL_DIR"

# ─── Run evaluation ──────────────────────────────────────────────────────────
echo "=========================================="
echo "Running inference + evaluation"
echo "  Started at: $(date)"
echo "=========================================="

DATASET=${DATASET} TASK=${TASK} THRESHOLD=${THRESHOLD} SAVE=True \
    FEWSHOT_DIR=${FEWSHOT_DIR} FEWSHOT_K=${FEWSHOT_K} FEWSHOT_SEED=${FEWSHOT_SEED} \
    ANALYZE=${ANALYZE} \
    PORT=${MASTER_PORT} \
    ./tools/dist_test.sh ${CONFIG} ${CKPT} ${NUM_GPUS} \
    --work-dir ${EVAL_DIR}

EVAL_RC=$?
echo ""
echo "=========================================="
if [ $EVAL_RC -eq 0 ]; then
    echo "Eval complete at: $(date)"
    echo "Results in: $EVAL_DIR"
else
    echo "ERROR: Eval failed (exit code $EVAL_RC)"
fi
echo "=========================================="

# ─── Embedding diagnostic (optional) ────────────────────────────────────────
if [ "$ANALYZE" = "1" ] && [ $EVAL_RC -eq 0 ]; then
    DIAG_DIR=visualizations/embed_diag_t${TASK}
    mkdir -p "${DIAG_DIR}"

    echo ""
    echo "=========================================="
    echo "Running embedding diagnostic"
    echo "  Started at: $(date)"
    echo "  Output dir: ${DIAG_DIR}"
    echo "=========================================="

    # Config parser needs these env vars
    export DATASET TASK THRESHOLD SAVE=False
    export FEWSHOT_DIR FEWSHOT_K FEWSHOT_SEED

    python tools/embedding_diagnostic.py \
        --config ${CONFIG} \
        --checkpoint ${CKPT} \
        --task ${TASK} \
        --max-images 0 \
        --output-dir ${DIAG_DIR}

    DIAG_RC=$?
    echo ""
    if [ $DIAG_RC -eq 0 ]; then
        echo "Diagnostic complete at: $(date)"
        echo "Outputs in: ${DIAG_DIR}/"
        ls -lh ${DIAG_DIR}/*.png ${DIAG_DIR}/*.txt ${DIAG_DIR}/*.pt 2>/dev/null
        echo ""
        echo "=========================================="
        echo "ANALYSIS SUMMARY"
        echo "=========================================="
        cat ${DIAG_DIR}/analysis.txt
    else
        echo "WARNING: Diagnostic failed (exit code $DIAG_RC) — eval results are still valid"
    fi
    echo "=========================================="
fi

exit $EVAL_RC
