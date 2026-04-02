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
#SBATCH --time=03:00:00

# ============================================================================
# Embedding Space Diagnostic for YOLO-UniOW  (full test set + targeted exps)
# ============================================================================
#
# Extracts vision features at ALL GT locations on the full test set,
# computes cosine alignments, generates diagnostic plots for A-OSE / unknown
# recall analysis, and runs 5 targeted prompt-drift experiments.
#
# Usage:
#   sbatch scripts/analyze_embeddings.sbatch                  → T2 best ckpt
#   sbatch scripts/analyze_embeddings.sbatch 2                → T2 best ckpt
#   sbatch scripts/analyze_embeddings.sbatch 1                → T1 best ckpt
#   sbatch scripts/analyze_embeddings.sbatch 2 /path/to.pth   → T2 custom ckpt
#
# Outputs:  visualizations/embed_diag_t{task}/
#   analysis.txt                — full quantitative summary (printed to log)
#   targeted_analysis.txt       — 5 targeted experiment results (printed to log)
#   cached_features.pt          — cached features (re-run analysis without GPU)
#   embedding_overview.png      — UMAP of all vision features + prompt embeddings
#   per_class/{name}.png        — per-class feature cloud + cosine stats
#   aose_confusion_matrix.png   — unknown→known confusion heatmap
#   score_distributions.png     — score histograms for top A-OSE offenders
#   cosine_heatmaps.png         — prompt×prompt and centroid×centroid similarity
#   decision_boundary_top3.png  — known vs unknown score scatter
#   exp1_cross_prompt/          — per-instance FT vs ZS score scatter per novel class
#   exp2_score_leakage/         — per-channel score delta heatmap at novel GT locs
#   exp3_prompt_drift/          — prompt drift table + CSV
#   exp4_aose_mechanism/        — A-OSE margin histograms (confident vs marginal)
#   exp5_cross_alignment/       — prompt×centroid cross-alignment (FT, ZS, delta)
#
# Re-run analysis only (no GPU needed, uses cached features):
#   python tools/embedding_diagnostic.py \
#       --cache visualizations/embed_diag_t2/cached_features.pt \
#       --config <config> --checkpoint <ckpt> --task 2 \
#       --zeroshot-emb embeddings/uniow-idd/idd_t2.npy
# ============================================================================

TASK=${1:-2}
CUSTOM_CKPT=${2:-""}
MAX_IMAGES=0              # always use full test set

DATASET=IDD
FEWSHOT_K=10
FEWSHOT_SEED=1
FEWSHOT_DIR=data/OWOD/iddsplit

STEM=yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod
ZS_EMB_BASE=embeddings/uniow-idd

# ─── Select config, work_dir, and zero-shot embeddings by task ───────────────
if [ "$TASK" = "1" ]; then
    CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py
    WORK_DIR=work_dirs/${STEM}_idd_train_task1
    FEWSHOT_K=0
    ZS_EMB=${ZS_EMB_BASE}/idd_t1.npy
else
    CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_wapr.py
    WORK_DIR=work_dirs/${STEM}_idd_train_task2_${FEWSHOT_K}shot_wapr
    ZS_EMB=${ZS_EMB_BASE}/idd_t2.npy
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
echo "  Max images:   all (full test set)"
echo "  ZS embeddings:$ZS_EMB"
echo "  Output dir:   $OUTPUT_DIR"
echo "=========================================="

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    echo "  Check that ${WORK_DIR} contains best*.pth or epoch_*.pth"
    exit 1
fi
echo "Checkpoint: $(du -h "$CKPT" | cut -f1) — $CKPT"

if [ ! -f "$ZS_EMB" ]; then
    echo "ERROR: Zero-shot embedding not found: $ZS_EMB"
    echo "  Run tools/owod_scripts/extract_idd_embeddings.py to generate it."
    exit 1
fi
echo "ZS embeddings: $(du -h "$ZS_EMB" | cut -f1) — $ZS_EMB"
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
    --config      ${CONFIG}     \
    --checkpoint  ${CKPT}       \
    --task        ${TASK}       \
    --max-images  0             \
    --zeroshot-emb ${ZS_EMB}   \
    --output-dir  ${OUTPUT_DIR}

RC=$?
echo ""
echo "=========================================="
if [ $RC -eq 0 ]; then
    echo "Diagnostic complete at: $(date)"
    echo "Outputs in: ${OUTPUT_DIR}/"
    echo ""
    echo "Top-level files:"
    ls -lh ${OUTPUT_DIR}/*.png ${OUTPUT_DIR}/*.txt ${OUTPUT_DIR}/*.pt 2>/dev/null
    echo ""
    echo "Targeted experiment dirs:"
    ls -d ${OUTPUT_DIR}/exp*/ 2>/dev/null
    echo ""
    echo "Per-class plots:"
    ls ${OUTPUT_DIR}/per_class/ 2>/dev/null
    echo ""
    echo "=========================================="
    echo "ANALYSIS SUMMARY"
    echo "=========================================="
    cat ${OUTPUT_DIR}/analysis.txt
    echo ""
    echo "=========================================="
    echo "TARGETED EXPERIMENTS SUMMARY"
    echo "=========================================="
    cat ${OUTPUT_DIR}/targeted_analysis.txt 2>/dev/null || echo "(targeted_analysis.txt not found)"
else
    echo "ERROR: Diagnostic failed (exit code $RC)"
fi
echo "=========================================="
exit $RC
