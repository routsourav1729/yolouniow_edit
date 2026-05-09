#!/bin/bash
#SBATCH --job-name=probe_norms
#SBATCH --output=slurm_logs/experiment/probe_norms_%j.out
#SBATCH --error=slurm_logs/experiment/probe_norms_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# ============================================================================
# NORM PROBE EXPERIMENT
#
# Measures ||BN(F_v)||_2 per matched GT anchor, grouped by class role
# (base / novel / unknown). Tests whether novel-class instances have
# systematically lower post-BN visual feature norms than base classes,
# which would suppress their classification scores regardless of angular alignment.
#
# Runs 4 cells in sequence (single job):
#   IDD      T1 ckpt  ×  {test, t1_train}
#   nuScenes T1 ckpt  ×  {test, t1_train}
#
# Usage:
#   sbatch scripts/experiment/probe_norms.sbatch
#   sbatch scripts/experiment/probe_norms.sbatch --match-iou 0.5   # looser matching
#   sbatch scripts/experiment/probe_norms.sbatch --num-images 2000  # quick test
#
# Outputs (in probe_out/):
#   idd_t1_test.csv / .json
#   idd_t1_train.csv / .json
#   nu_t1_test.csv / .json
#   nu_t1_train.csv / .json
# ============================================================================

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

cd /home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW

mkdir -p slurm_logs/experiment probe_out

# ── Parse args ────────────────────────────────────────────────────────────
MATCH_IOU=0.9
NUM_IMAGES=0
BATCH_SIZE=64
NUM_WORKERS=8
VIS=0
VIS_NEIGHBORS=30
VIS_MIN_DIST=0.1
VIS_MAX_PER_CLASS=2000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --match-iou)         MATCH_IOU="$2";         shift 2 ;;
        --num-images)        NUM_IMAGES="$2";        shift 2 ;;
        --batch-size)        BATCH_SIZE="$2";        shift 2 ;;
        --num-workers)       NUM_WORKERS="$2";       shift 2 ;;
        --vis)               VIS=1;                  shift 1 ;;
        --vis-n-neighbors)   VIS_NEIGHBORS="$2";     shift 2 ;;
        --vis-min-dist)      VIS_MIN_DIST="$2";      shift 2 ;;
        --vis-max-per-class) VIS_MAX_PER_CLASS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

NUM_IMAGES_FLAG=""
[ "$NUM_IMAGES" -gt 0 ] 2>/dev/null && NUM_IMAGES_FLAG="--num-images $NUM_IMAGES"

VIS_FLAGS=""
if [ "$VIS" = "1" ]; then
    VIS_FLAGS="--vis --vis-n-neighbors $VIS_NEIGHBORS --vis-min-dist $VIS_MIN_DIST --vis-max-per-class $VIS_MAX_PER_CLASS"
    mkdir -p probe_out/umap
fi

# ── Checkpoint + config table ─────────────────────────────────────────────
# T1 ckpts (norm-mode default)
IDD_T1_CKPT=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task1/best_owod_Both_epoch_20.pth
IDD_T1_CFG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py
NU_T1_CKPT=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_nuowodb_train_task1/best_owod_Both_epoch_20.pth
NU_T1_CFG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod.py

# T2 ckpts (vis-mode default — known prompts include both base + novel)
IDD_T2_CKPT=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task2_10shot_seed1/best_owod_Both_epoch_40.pth
IDD_T2_CFG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2.py
NU_T2_CKPT=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_nuowodb_train_task2_10shot_seed1/best_owod_Both_epoch_10.pth
NU_T2_CFG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod.py

if [ "$VIS" = "1" ]; then
    MODE_DESC="VIS (UMAP only, train splits, T2 weights)"
else
    MODE_DESC="NORM (CSV+JSON, all splits, T1 weights)"
fi

echo "============================================================"
echo "  PROBE: BN(F_v)  —  $MODE_DESC"
echo "  match_iou=$MATCH_IOU  batch=$BATCH_SIZE  num_images=${NUM_IMAGES:-all}"
echo "  Job: $SLURM_JOB_ID"
echo "============================================================"
echo ""

if [ "$VIS" = "1" ]; then
    # ─────────────────────────────────────────────────────────────────────
    # VIS MODE: only run the two TRAIN splits using T2 weights so that
    # Tunk + base + novel prompt vectors are all available on the UMAP.
    # Matching is best-IoU only — choice of weights does NOT affect anchor
    # selection (label comes from VOC XML).
    # ─────────────────────────────────────────────────────────────────────

    echo ">>> [1/2] IDD  T2-ckpt  t1_train  (vis)"
    DATASET=IDD TASK=2 THRESHOLD=0.05 SAVE=False \
        FEWSHOT_K=10 FEWSHOT_SEED=1 FEWSHOT_DIR=data/OWOD/iddsplit \
        python tools/probe_norms.py \
            --config      "$IDD_T2_CFG" \
            --checkpoint  "$IDD_T2_CKPT" \
            --split       t1_train \
            --out-csv     probe_out/umap/idd_t2_train.csv \
            --match-iou   "$MATCH_IOU" \
            --batch-size  "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
            $NUM_IMAGES_FLAG $VIS_FLAGS
    echo ""

    echo ">>> [2/2] nuScenes  T2-ckpt  t1_train  (vis)"
    DATASET=nuOWODB TASK=2 THRESHOLD=0.05 SAVE=False \
        FEWSHOT_K=10 FEWSHOT_SEED=1 FEWSHOT_DIR=data/OWOD/nuowodb_fewshot_tfa \
        python tools/probe_norms.py \
            --config      "$NU_T2_CFG" \
            --checkpoint  "$NU_T2_CKPT" \
            --split       t1_train \
            --out-csv     probe_out/umap/nu_t2_train.csv \
            --match-iou   "$MATCH_IOU" \
            --batch-size  "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
            $NUM_IMAGES_FLAG $VIS_FLAGS
    echo ""

    echo "============================================================"
    echo "VIS DONE  —  $(date)"
    ls -lh probe_out/umap/*.png probe_out/umap/*.npz 2>/dev/null
    echo "============================================================"
    exit 0
fi

# ── NORM MODE: 4 cells ───────────────────────────────────────────────────
echo ">>> [1/4] IDD  T1-ckpt  test"
DATASET=IDD TASK=1 THRESHOLD=0.05 SAVE=False \
    FEWSHOT_K=10 FEWSHOT_SEED=1 FEWSHOT_DIR=data/OWOD/iddsplit \
    python tools/probe_norms.py \
        --config      "$IDD_T1_CFG" \
        --checkpoint  "$IDD_T1_CKPT" \
        --split       test \
        --out-csv     probe_out/idd_t1_test.csv \
        --out-json    probe_out/idd_t1_test.json \
        --match-iou   "$MATCH_IOU" \
        --batch-size  "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        $NUM_IMAGES_FLAG
echo ""

echo ">>> [2/4] IDD  T1-ckpt  t1_train"
DATASET=IDD TASK=1 THRESHOLD=0.05 SAVE=False \
    FEWSHOT_K=10 FEWSHOT_SEED=1 FEWSHOT_DIR=data/OWOD/iddsplit \
    python tools/probe_norms.py \
        --config      "$IDD_T1_CFG" \
        --checkpoint  "$IDD_T1_CKPT" \
        --split       t1_train \
        --out-csv     probe_out/idd_t1_train.csv \
        --out-json    probe_out/idd_t1_train.json \
        --match-iou   "$MATCH_IOU" \
        --batch-size  "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        $NUM_IMAGES_FLAG
echo ""

echo ">>> [3/4] nuScenes  T1-ckpt  test"
DATASET=nuOWODB TASK=1 THRESHOLD=0.05 SAVE=False \
    FEWSHOT_K=10 FEWSHOT_SEED=1 FEWSHOT_DIR=data/OWOD/nuowodb_fewshot_tfa \
    python tools/probe_norms.py \
        --config      "$NU_T1_CFG" \
        --checkpoint  "$NU_T1_CKPT" \
        --split       test \
        --out-csv     probe_out/nu_t1_test.csv \
        --out-json    probe_out/nu_t1_test.json \
        --match-iou   "$MATCH_IOU" \
        --batch-size  "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        $NUM_IMAGES_FLAG
echo ""

echo ">>> [4/4] nuScenes  T1-ckpt  t1_train"
DATASET=nuOWODB TASK=1 THRESHOLD=0.05 SAVE=False \
    FEWSHOT_K=10 FEWSHOT_SEED=1 FEWSHOT_DIR=data/OWOD/nuowodb_fewshot_tfa \
    python tools/probe_norms.py \
        --config      "$NU_T1_CFG" \
        --checkpoint  "$NU_T1_CKPT" \
        --split       t1_train \
        --out-csv     probe_out/nu_t1_train.csv \
        --out-json    probe_out/nu_t1_train.json \
        --match-iou   "$MATCH_IOU" \
        --batch-size  "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        $NUM_IMAGES_FLAG
echo ""

echo "============================================================"
echo "NORM DONE  —  $(date)"
ls -lh probe_out/*.csv 2>/dev/null
echo "============================================================"
