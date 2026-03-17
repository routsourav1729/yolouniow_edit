#!/bin/bash
#SBATCH --job-name=yolouniow_vis
#SBATCH --output=slurm_logs/idd_vis_%j.out
#SBATCH --error=slurm_logs/idd_vis_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# ============================================================================
# YOLO-UniOW IDD Visualization Pipeline  (one-time analysis script)
# ============================================================================
#
# Runs 3 steps in sequence:
#
#   Step 1  Scan all IDD test-set GT annotations.
#           Finds images with novel T2 objects >= 100×100 px (tanker_vehicle,
#           crane_truck, excavator, truck, bus, street_cart).
#           Output: vis_output/candidates.txt
#
#   Step 2  Run per-image inference on candidates.
#           Scores each image by how well the model detects novel objects as
#           "unknown" (open-world quality).  Selects top TOP_N images that
#           are NOT too dense (≤ MAX_TOTAL_PREDS boxes) so they visualise well.
#           Output: vis_output/selected_images.txt
#
#   Step 3  Render colour-coded bounding-box visualizations for selected images.
#           GREEN  = known (T1) class predictions
#           ORANGE = unknown predictions that overlap a novel T2 GT box
#           RED    = unknown predictions with no GT overlap
#           Dashed outlines = GT reference boxes (green=known, orange=novel)
#           Output: visualizations/idd_t1_vis/  + image_list.txt
#
# Usage
# -----
#   sbatch scripts/visualize_idd.sbatch                    # auto-detect checkpoint
#   sbatch scripts/visualize_idd.sbatch path/to/best.pth   # explicit checkpoint
#
# ============================================================================

CHECKPOINT=${1:-""}   # first arg = checkpoint; auto-detected if empty

# ─── Config ──────────────────────────────────────────────────────────────────
DATASET=IDD
TASK=1
THRESHOLD=0.05
UNK_THRESHOLD=0.15
CONFIG=configs/owod_ft/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod_idd.py
DATA_ROOT=data/OWOD

CANDIDATES_N=120           # Step 1: shortlist size before inference ranking
TOP_N=10                   # Step 2: final number of images to select
MAX_TOTAL_PREDS=25         # Step 2: density cap (boxes per image)
MIN_BOX_SIZE=90            # Step 1: minimum GT box side in pixels

# Animal-focused filtering (unknown + novel co-occurrence)
TARGET_CLASS=animal
MIN_TARGET_COUNT=1
RUN_STEP1=0                # 0 = skip GT pre-scan, run Step-2 on full test set

CANDIDATES=vis_output/candidates_animal_novel.txt
SELECTED=vis_output/selected_images_animal_novel.txt
VIS_OUT=visualizations/idd_t1_vis_animal_novel

# ─── Environment ─────────────────────────────────────────────────────────────
source /home/agipml/sourav.rout/miniconda3/bin/activate yolouniow

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW:$PYTHONPATH

# HuggingFace / CLIP offline cache (no internet on compute nodes)
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Config template variables
export DATASET=$DATASET
export TASK=$TASK
export THRESHOLD=$THRESHOLD
export SAVE=False
export FEWSHOT_DIR="none"
export FEWSHOT_K=0
export FEWSHOT_SEED=1

echo "=========================================="
echo "YOLO-UniOW IDD Visualization Pipeline"
echo "=========================================="
echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          ${SLURM_JOB_ID:-local}"
echo ""
echo "Configuration:"
echo "  Dataset     : $DATASET (Task $TASK)"
echo "  Config      : $CONFIG"
echo "  Target cls  : $TARGET_CLASS (min count: $MIN_TARGET_COUNT)"
echo "  Candidates  : $CANDIDATES_N images (GT scan)"
echo "  Top-N       : $TOP_N images (final selection)"
echo "  Min box     : ${MIN_BOX_SIZE}px"
echo "  Thresholds  : known=$THRESHOLD  unknown=$UNK_THRESHOLD"
echo "  Output dir  : $VIS_OUT"
echo "=========================================="

python -c "import torch; \
    print(f'  PyTorch {torch.__version__}, CUDA devices: {torch.cuda.device_count()}')"

mkdir -p slurm_logs vis_output "$VIS_OUT"

# ─── Auto-detect checkpoint if not supplied ───────────────────────────────────
if [ -z "$CHECKPOINT" ]; then
    STEM=yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod
    WORK_DIR=work_dirs/${STEM}_idd_train_task1
    CHECKPOINT=$(ls -t "${WORK_DIR}"/best*.pth 2>/dev/null | head -1)
    if [ -z "$CHECKPOINT" ]; then
        echo ""
        echo "ERROR: No best_*.pth checkpoint found in ${WORK_DIR}."
        echo "       Please supply the checkpoint explicitly:"
        echo "         sbatch scripts/visualize_idd.sbatch <path/to/best.pth>"
        exit 1
    fi
    echo "Auto-detected checkpoint: $CHECKPOINT"
else
    echo "Using checkpoint: $CHECKPOINT"
fi
echo ""

# ─── Step 1: Optional GT pre-scan ────────────────────────────────────────────
if [ "$RUN_STEP1" -eq 1 ]; then
    echo "=========================================="
    echo "Step 1: Scanning GT annotations"
    echo "  Started at: $(date)"
    echo "=========================================="

    python tools/vis/step1_find_gt_candidates.py \
        --data-root  "$DATA_ROOT"     \
        --dataset    "$DATASET"       \
        --task       "$TASK"          \
        --min-size   "$MIN_BOX_SIZE"  \
        --target-class "$TARGET_CLASS" \
        --min-target-count "$MIN_TARGET_COUNT" \
        --require-novel \
        --target-weight 8.0 \
        --novel-weight 2.5 \
        --other-weight 1.0 \
        --known-weight 0.0 \
        --top-n      "$CANDIDATES_N"  \
        --output     "$CANDIDATES"

    if [ $? -ne 0 ]; then
        echo "ERROR: Step 1 failed. Aborting."
        exit 1
    fi
    echo "Step 1 done at: $(date)"
    echo ""
else
    echo "=========================================="
    echo "Step 1: Skipped (RUN_STEP1=0)"
    echo "Using full-testset model-driven selection in Step 2"
    echo "=========================================="
    echo ""
fi

# ─── Step 2: Inference-based image selection ──────────────────────────────────
echo "=========================================="
echo "Step 2: Inference & image selection"
echo "  Checkpoint: $CHECKPOINT"
echo "  Started at: $(date)"
echo "=========================================="

python tools/vis/step2_select_vis_images.py \
    --config        "$CONFIG"         \
    --checkpoint    "$CHECKPOINT"     \
    --candidates    "$CANDIDATES"     \
    --use-full-testset \
    --image-set     test \
    --data-root     "$DATA_ROOT"      \
    --dataset       "$DATASET"        \
    --task          "$TASK"           \
    --target-class  "$TARGET_CLASS"   \
    --min-box-size  "$MIN_BOX_SIZE"   \
    --min-target-count "$MIN_TARGET_COUNT" \
    --require-novel-gt \
    --threshold     "$THRESHOLD"      \
    --unk-threshold "$UNK_THRESHOLD"  \
    --top-n         "$TOP_N"          \
    --max-total-preds "$MAX_TOTAL_PREDS" \
    --output        "$SELECTED"

if [ $? -ne 0 ]; then
    echo "ERROR: Step 2 failed. Aborting."
    exit 1
fi
echo "Step 2 done at: $(date)"
echo ""

# ─── Step 3: Colour-coded visualization ───────────────────────────────────────
echo "=========================================="
echo "Step 3: Generating visualizations"
echo "  Output: $VIS_OUT"
echo "  Started at: $(date)"
echo "=========================================="

python tools/vis/step3_visualize_owod.py \
    --config          "$CONFIG"         \
    --checkpoint      "$CHECKPOINT"     \
    --selected-images "$SELECTED"       \
    --data-root       "$DATA_ROOT"      \
    --dataset         "$DATASET"        \
    --task            "$TASK"           \
    --threshold       "$THRESHOLD"      \
    --unk-threshold   "$UNK_THRESHOLD"  \
    --output-dir      "$VIS_OUT"

if [ $? -ne 0 ]; then
    echo "ERROR: Step 3 failed."
    exit 1
fi
echo "Step 3 done at: $(date)"
echo ""

# ─── Summary ──────────────────────────────────────────────────────────────────
echo "=========================================="
echo "VISUALIZATION PIPELINE COMPLETE"
echo "  Annotated images : $VIS_OUT/"
echo "  Image list       : $VIS_OUT/image_list.txt"
echo "  GT candidates    : $CANDIDATES"
echo "  Selected images  : $SELECTED"
echo ""
echo "  Colour key:"
echo "    GREEN  → known (T1) class predictions"
echo "    ORANGE → unknown predictions matching novel T2 GT"
echo "    RED    → unknown predictions with no GT overlap"
echo "    Dashed → GT reference boxes"
echo ""
echo "Job finished at: $(date)"
echo "=========================================="
