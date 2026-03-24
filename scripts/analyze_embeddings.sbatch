#!/bin/bash
#SBATCH --job-name=embed_diag
#SBATCH --output=slurm_logs/embed_diag_%j.out
#SBATCH --error=slurm_logs/embed_diag_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# ============================================================================
# Embedding Space Diagnostic for YOLO-UniOW
# ============================================================================
#
# Extracts vision features at GT locations, computes cosine alignments,
# and generates diagnostic plots for A-OSE / unknown recall analysis.
#
# Usage:
#   sbatch scripts/analyze_embeddings.sbatch                  → T2 best ckpt, all images
#   sbatch scripts/analyze_embeddings.sbatch 2                → T2 best ckpt
#   sbatch scripts/analyze_embeddings.sbatch 2 /path/to.pth   → T2 custom ckpt
#   sbatch scripts/analyze_embeddings.sbatch 2 "" 500         → T2 best ckpt, 500 images
#   sbatch scripts/analyze_embeddings.sbatch 2 "" 0           → T2 best ckpt, ALL images
#
# Outputs:  visualizations/embed_diag_t{task}/
#   embedding_overview.png      — UMAP of all vision features + prompt embeddings
#   per_class/{name}.png        — per-class feature cloud + cosine stats
#   aose_confusion_matrix.png   — unknown→known confusion heatmap
#   score_distributions.png     — score histograms for top A-OSE offenders
#   cosine_heatmaps.png         — prompt×prompt and centroid×centroid similarity
#   decision_boundary_top3.png  — known vs unknown score scatter
#   analysis.txt                — full quantitative summary
#   cached_features.pt          — cached features for fast re-visualization
#
# Re-visualize without GPU (after first run caches features):
#   python tools/embedding_diagnostic.py \
#       --cache visualizations/embed_diag_t2/cached_features.pt \
#       --config <config> --checkpoint <ckpt> --task 2
# ============================================================================

TASK=${1:-2}
CUSTOM_CKPT=${2:-""}
MAX_IMAGES=${3:-0}        # 0 = all images

DATASET=IDD
FEWSHOT_K=10
FEWSHOT_SEED=1
FEWSHOT_DIR=data/OWOD/iddsplit

STEM=yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod

# ─── Select config and work_dir by task ──────────────────────────────────────
if [ "$TASK" = "1" ]; then
    CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py
    WORK_DIR=work_dirs/${STEM}_idd_train_task1
    FEWSHOT_K=0
else
    CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_wapr.py
    WORK_DIR=work_dirs/${STEM}_idd_train_task2_${FEWSHOT_K}shot_wapr
fi

# ─── Resolve checkpoint ─────────────────────────────────────────────────────
if [ -n "$CUSTOM_CKPT" ]; then
    CKPT="$CUSTOM_CKPT"
else
    CKPT=$(ls -t "${WORK_DIR}"/best*.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        CKPT=$(ls -t "${WORK_DIR}"/epoch_*.pth 2>/dev/null | head -1)
    fi
fi

OUTPUT_DIR=visualizations/embed_diag_t${TASK}

echo "=========================================="
echo "Embedding Space Diagnostic"
echo "=========================================="
echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"
echo ""
echo "  Task:         $TASK"
echo "  Config:       $CONFIG"
echo "  Checkpoint:   $CKPT"
echo "  Max images:   $MAX_IMAGES"
echo "  Output dir:   $OUTPUT_DIR"
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

# Set env vars needed by config parser
export DATASET=${DATASET}
export TASK=${TASK}
export THRESHOLD=0.05
export SAVE=False
export FEWSHOT_DIR=${FEWSHOT_DIR}
export FEWSHOT_K=${FEWSHOT_K}
export FEWSHOT_SEED=${FEWSHOT_SEED}
export ANALYZE=0

mkdir -p slurm_logs "${OUTPUT_DIR}"

# ─── Run diagnostic ──────────────────────────────────────────────────────────
echo "=========================================="
echo "Running embedding diagnostic"
echo "  Started at: $(date)"
echo "=========================================="

/home/agipml/sourav.rout/miniconda3/envs/yolouniow/bin/python tools/embedding_diagnostic.py \
    --config ${CONFIG} \
    --checkpoint ${CKPT} \
    --task ${TASK} \
    --max-images ${MAX_IMAGES} \
    --output-dir ${OUTPUT_DIR}

RC=$?
echo ""
echo "=========================================="
if [ $RC -eq 0 ]; then
    echo "Diagnostic complete at: $(date)"
    echo "Outputs in: ${OUTPUT_DIR}/"
    echo ""
    echo "Files:"
    ls -lh ${OUTPUT_DIR}/*.png ${OUTPUT_DIR}/*.txt ${OUTPUT_DIR}/*.pt 2>/dev/null
    echo ""
    echo "Per-class plots:"
    ls ${OUTPUT_DIR}/per_class/ 2>/dev/null
    echo ""
    # Print analysis.txt to log
    echo "=========================================="
    echo "ANALYSIS SUMMARY"
    echo "=========================================="
    cat ${OUTPUT_DIR}/analysis.txt
else
    echo "ERROR: Diagnostic failed (exit code $RC)"
fi
echo "=========================================="
exit $RC
