#!/bin/bash
#SBATCH --job-name=idd_unk_novel_probe
#SBATCH --output=slurm_logs/experiment/unk_novel_probe_%j.out
#SBATCH --error=slurm_logs/experiment/unk_novel_probe_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# ============================================================================
# IDD UNKNOWN-vs-NOVEL-CACHE EVIDENCE PROBE
#
# Replaces the old BN-norm probe. Uses the existing K-shot novel visual cache
# (post-BN features, usually built with IoU>=0.9) and scans the IDD test set.
#
# For each GT object in IDD test (base, novel, and true unknown):
#   - select a matched one2one anchor by correct-class/T_unk TAL or best IoU,
#   - compare BN(anchor) with per-level cached novel prototypes,
#   - write a class x novel-cache cosine matrix.
#
# This answers: if we suppress T_unk when an unknown-selected anchor looks
# novel-like, is that similarity specific to novel GTs, or shared by true
# unknown/base GTs too?
#
# Usage:
#   sbatch scripts/experiment/probe_norms.sbatch
#   sbatch scripts/experiment/probe_norms.sbatch --variant plain
#   sbatch scripts/experiment/probe_norms.sbatch --match-iou 0.8 --selection best_iou
#   sbatch scripts/experiment/probe_norms.sbatch --num-images 500
#   sbatch scripts/experiment/probe_norms.sbatch --ckpt /path/to/t2.pth --cache /path/to/cache.pt
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

mkdir -p slurm_logs/experiment probe_out/unknown_novel

VARIANT="wapr"
SHOT=10
SEED=1
MATCH_IOU=0.9
SELECTION="tunk_tal"
NUM_IMAGES=0
BATCH_SIZE=32
NUM_WORKERS=8
CKPT_OVERRIDE=""
T1_CKPT_OVERRIDE=""
CACHE_OVERRIDE=""
OUT_TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)       VARIANT="$2"; shift 2 ;;
        --shot)          SHOT="$2"; shift 2 ;;
        --seed)          SEED="$2"; shift 2 ;;
        --match-iou)     MATCH_IOU="$2"; shift 2 ;;
        --selection)     SELECTION="$2"; shift 2 ;;
        --num-images)    NUM_IMAGES="$2"; shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        --num-workers)   NUM_WORKERS="$2"; shift 2 ;;
        --ckpt)          CKPT_OVERRIDE="$2"; shift 2 ;;
        --t1-ckpt)       T1_CKPT_OVERRIDE="$2"; shift 2 ;;
        --cache)         CACHE_OVERRIDE="$2"; shift 2 ;;
        --out-tag)       OUT_TAG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

case "$VARIANT" in
    plain) SUFFIX="" ;;
    wapr)  SUFFIX="_wapr" ;;
    kume)  SUFFIX="_kume" ;;
    wapr_kume) SUFFIX="_wapr_kume" ;;
    *) SUFFIX="_${VARIANT}" ;;
esac

CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2.py
T2_WORK_DIR=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task2_${SHOT}shot_seed${SEED}${SUFFIX}
if [ -n "$CKPT_OVERRIDE" ]; then
    CKPT="$CKPT_OVERRIDE"
else
    CKPT=$(ls -t "${T2_WORK_DIR}"/best*.pth 2>/dev/null | head -1)
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no T2 checkpoint found."
    echo "  work dir: $T2_WORK_DIR"
    echo "  pass --ckpt explicitly"
    exit 1
fi

if [ -n "$CACHE_OVERRIDE" ]; then
    CACHE="$CACHE_OVERRIDE"
else
    CACHE=cache/idd/viscache_${SHOT}shot_seed${SEED}.pt
fi
if [ ! -f "$CACHE" ]; then
    echo "ERROR: visual cache not found: $CACHE"
    echo "Build it first, e.g. scripts/eval_viscache.sbatch or tools/owod_scripts/build_visual_cache.py"
    exit 1
fi

if [ -n "$T1_CKPT_OVERRIDE" ]; then
    T1_CKPT="$T1_CKPT_OVERRIDE"
else
    T1_CKPT=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task1/best_owod_Both_epoch_20.pth
fi
T1_FLAG=""
if [ -f "$T1_CKPT" ]; then
    T1_FLAG="--t1-tunk-checkpoint $T1_CKPT --t1-unk-index 8"
else
    echo "WARN: T1 checkpoint not found; using current T2 T_unk as fallback."
    echo "  missing: $T1_CKPT"
fi

NUM_IMAGES_FLAG=""
[ "$NUM_IMAGES" -gt 0 ] 2>/dev/null && NUM_IMAGES_FLAG="--num-images $NUM_IMAGES"

TAG=${OUT_TAG:-idd_t2_${VARIANT}_${SHOT}shot_seed${SEED}_${SELECTION}_iou${MATCH_IOU}}
OUT_CSV=probe_out/unknown_novel/${TAG}.csv
OUT_JSON=probe_out/unknown_novel/${TAG}.json
OUT_MATRIX=probe_out/unknown_novel/${TAG}_matrix.csv

echo "============================================================"
echo "  IDD UNKNOWN-vs-NOVEL-CACHE PROBE"
echo "  variant:     $VARIANT"
echo "  config:      $CONFIG"
echo "  T2 ckpt:     $CKPT"
echo "  cache:       $CACHE"
echo "  T1 Tunk:     ${T1_CKPT:-none}"
echo "  match_iou:   $MATCH_IOU"
echo "  selection:   $SELECTION"
echo "  images:      ${NUM_IMAGES:-all}"
echo "  output:      $OUT_CSV"
echo "  matrix:      $OUT_MATRIX"
echo "  job:         $SLURM_JOB_ID"
echo "============================================================"

DATASET=IDD TASK=2 THRESHOLD=0.05 SAVE=False \
    FEWSHOT_K=${SHOT} FEWSHOT_SEED=${SEED} FEWSHOT_DIR=data/OWOD/iddsplit \
    python tools/probe_norms.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --cache "$CACHE" \
        --split test \
        --out-csv "$OUT_CSV" \
        --out-matrix "$OUT_MATRIX" \
        --out-json "$OUT_JSON" \
        --match-iou "$MATCH_IOU" \
        --selection "$SELECTION" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        $NUM_IMAGES_FLAG \
        $T1_FLAG

echo "============================================================"
echo "DONE — $(date)"
ls -lh "$OUT_CSV" "$OUT_MATRIX" "$OUT_JSON" 2>/dev/null
echo "============================================================"
