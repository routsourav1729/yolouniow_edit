#!/bin/bash
#SBATCH --job-name=unknown_t1
#SBATCH --output=slurm_logs/analysis/unknown_analysis_t1_%j.out
#SBATCH --error=slurm_logs/analysis/unknown_analysis_t1_%j.err
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=384G
#SBATCH --time=12:00:00

set -euo pipefail

# ============================================================================
# T1 PSEUDO-UNKNOWN QUALITY ANALYSIS
#
# Scans a T1 training split with a T1-trained model and mines
# pseudo-unknown candidates using:
#   max IoU to T1-known GT < --known-iou-thr  and  Tobj >= --tobj-thr
# plus the default TAL foreground exclusion used by the training head.
#
# The raw XML annotations are then used to classify each mined candidate as:
#   - one of the T2 novel classes,
#   - a raw XML unknown class outside T1/T2,
#   - background / non-target.
#
# Usage:
#   sbatch scripts/experiment/unknown_analysis_t1.sbatch --dataset IDD
#   sbatch scripts/experiment/unknown_analysis_t1.sbatch --dataset nuOWODB
#   sbatch scripts/experiment/unknown_analysis_t1.sbatch --dataset IDD --num-images 1000
#   sbatch scripts/experiment/unknown_analysis_t1.sbatch --tobj-thr 0.05
#   sbatch scripts/experiment/unknown_analysis_t1.sbatch --max-candidates-per-image 300
#   sbatch scripts/experiment/unknown_analysis_t1.sbatch --ckpt work_dirs/.../best.pth
#   sbatch scripts/experiment/unknown_analysis_t1.sbatch --unknown-classes pole,tractor,road_roller,pull_cart,concrete_mixer,animal
# ============================================================================

DATASET="IDD"
CONFIG=""
T1_CKPT=""
IMAGE_SET="train"
SPLIT_FILE=""
OUT_DIR=""
NUM_IMAGES=0
BATCH_SIZE=64
NUM_WORKERS=24
TOBJ_THR=0.01
KNOWN_IOU_THR=0.5
MATCH_IOU=0.5
MAX_CANDIDATES_PER_IMAGE=0
CANDIDATE_CSV_MAX=200000
UNKNOWN_CLASSES=""
EXCLUDE_TAL_FG=1
USE_AMP=0
PRELOAD_DATA=1
CACHE_DEVICE="cpu"
PIN_CACHE_MEMORY=0
PROGRESS_INTERVAL=10000
SWEEP_THRESHOLDS="0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)                  DATASET="$2"; shift 2 ;;
        --config)                   CONFIG="$2"; shift 2 ;;
        --ckpt|--checkpoint|--t1-ckpt) T1_CKPT="$2"; shift 2 ;;
        --image-set)                IMAGE_SET="$2"; shift 2 ;;
        --split-file)               SPLIT_FILE="$2"; shift 2 ;;
        --out-dir)                  OUT_DIR="$2"; shift 2 ;;
        --num-images)               NUM_IMAGES="$2"; shift 2 ;;
        --batch-size)               BATCH_SIZE="$2"; shift 2 ;;
        --num-workers)              NUM_WORKERS="$2"; shift 2 ;;
        --tobj-thr)                 TOBJ_THR="$2"; shift 2 ;;
        --known-iou-thr)            KNOWN_IOU_THR="$2"; shift 2 ;;
        --match-iou)                MATCH_IOU="$2"; shift 2 ;;
        --max-candidates-per-image) MAX_CANDIDATES_PER_IMAGE="$2"; shift 2 ;;
        --candidate-csv-max)        CANDIDATE_CSV_MAX="$2"; shift 2 ;;
        --unknown-classes)          UNKNOWN_CLASSES="$2"; shift 2 ;;
        --no-exclude-tal-fg)        EXCLUDE_TAL_FG=0; shift 1 ;;
        --amp)                      USE_AMP=1; shift 1 ;;
        --no-preload-data)          PRELOAD_DATA=0; shift 1 ;;
        --cache-device)             CACHE_DEVICE="$2"; shift 2 ;;
        --no-pin-cache-memory)      PIN_CACHE_MEMORY=0; shift 1 ;;
        --progress-interval)        PROGRESS_INTERVAL="$2"; shift 2 ;;
        --sweep-thresholds)         SWEEP_THRESHOLDS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
case "$DATASET" in
    IDD)
        DEFAULT_CONFIG="configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py"
        LOG_SUBDIR="idd"
        ;;
    nuOWODB)
        DEFAULT_CONFIG="configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod.py"
        LOG_SUBDIR="nuscenes"
        ;;
    *)
        echo "ERROR: Unsupported dataset '$DATASET'. Use IDD or nuOWODB."
        exit 1
        ;;
esac
[ -z "$CONFIG" ] && CONFIG="$DEFAULT_CONFIG"

source /home/agipml/sourav.rout/miniconda3/bin/activate yolouniow

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW:${PYTHONPATH:-}
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW
mkdir -p "slurm_logs/analysis/${LOG_SUBDIR}" probe_out/unknown_analysis_t1

if [ -z "$T1_CKPT" ]; then
    T1_CKPT=$(ls -t "work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_${DATASET_LOWER}_train_task1"/best*.pth 2>/dev/null | head -1)
fi
if [ -z "$T1_CKPT" ] || [ ! -f "$T1_CKPT" ]; then
    echo "ERROR: T1 $DATASET checkpoint not found. Pass --ckpt /path/to/best.pth"
    exit 1
fi

tag_tobj=${TOBJ_THR//./p}
tag_iou=${KNOWN_IOU_THR//./p}
if [ -z "$OUT_DIR" ]; then
    OUT_DIR="probe_out/unknown_analysis_t1/${DATASET_LOWER}_t1_${IMAGE_SET}_tobj${tag_tobj}_knowniou${tag_iou}"
    if [ "$MAX_CANDIDATES_PER_IMAGE" -gt 0 ] 2>/dev/null; then
        OUT_DIR="${OUT_DIR}_top${MAX_CANDIDATES_PER_IMAGE}"
    fi
fi
mkdir -p "$OUT_DIR"

PY_ARGS=(
    --dataset "$DATASET"
    --config "$CONFIG"
    --checkpoint "$T1_CKPT"
    --image-set "$IMAGE_SET"
    --out-dir "$OUT_DIR"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --tobj-thr "$TOBJ_THR"
    --known-iou-thr "$KNOWN_IOU_THR"
    --match-iou "$MATCH_IOU"
    --max-candidates-per-image "$MAX_CANDIDATES_PER_IMAGE"
    --candidate-csv-max "$CANDIDATE_CSV_MAX"
    --cache-device "$CACHE_DEVICE"
    --progress-interval "$PROGRESS_INTERVAL"
    --sweep-thresholds "$SWEEP_THRESHOLDS"
)

[ "$NUM_IMAGES" -gt 0 ] 2>/dev/null && PY_ARGS+=(--num-images "$NUM_IMAGES")
[ -n "$SPLIT_FILE" ] && PY_ARGS+=(--split-file "$SPLIT_FILE")
[ -n "$UNKNOWN_CLASSES" ] && PY_ARGS+=(--unknown-classes "$UNKNOWN_CLASSES")
[ "$EXCLUDE_TAL_FG" -eq 0 ] && PY_ARGS+=(--no-exclude-tal-fg)
[ "$USE_AMP" -eq 1 ] && PY_ARGS+=(--amp)
[ "$PRELOAD_DATA" -eq 1 ] && PY_ARGS+=(--preload-data)
[ "$PIN_CACHE_MEMORY" -eq 1 ] && PY_ARGS+=(--pin-cache-memory)

echo "============================================================"
echo "  $DATASET T1 PSEUDO-UNKNOWN ANALYSIS"
echo "  dataset:      $DATASET"
echo "  config:       $CONFIG"
echo "  checkpoint:   $T1_CKPT"
echo "  image_set:    $IMAGE_SET"
echo "  split_file:   ${SPLIT_FILE:-auto}"
echo "  Tobj thr:     $TOBJ_THR"
echo "  known IoU:    $KNOWN_IOU_THR"
echo "  match IoU:    $MATCH_IOU"
echo "  TAL exclude:  $EXCLUDE_TAL_FG"
echo "  top per img:  $MAX_CANDIDATES_PER_IMAGE (0 = no cap)"
echo "  images:       ${NUM_IMAGES:-all}"
echo "  batch size:   $BATCH_SIZE"
echo "  workers:      $NUM_WORKERS"
echo "  preload:      $PRELOAD_DATA ($CACHE_DEVICE, pin=$PIN_CACHE_MEMORY)"
echo "  progress int: $PROGRESS_INTERVAL"
echo "  sweep Tobj:   $SWEEP_THRESHOLDS"
echo "  out_dir:      $OUT_DIR"
echo "  job:          ${SLURM_JOB_ID:-local}"
echo "============================================================"

DATASET="$DATASET" TASK=1 THRESHOLD=0.05 SAVE=False \
    python scripts/experiment/unknown_analysis_t1.py "${PY_ARGS[@]}"

echo "============================================================"
echo "DONE -- $(date)"
ls -lh "$OUT_DIR"/summary.json "$OUT_DIR"/summary_by_class.csv "$OUT_DIR"/summary_by_role.csv "$OUT_DIR"/threshold_sweep_summary.csv 2>/dev/null
echo "============================================================"
