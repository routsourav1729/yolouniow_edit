#!/bin/bash
#SBATCH --job-name=idd_tunk_vc_cal
#SBATCH --output=slurm_logs/experiment/tunk_vc_cal_%j.out
#SBATCH --error=slurm_logs/experiment/tunk_vc_cal_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Inference-only T_unk calibration sweep.
# Only anchors whose current winner is T_unk are modified. Novel visual-cache
# evidence softly subtracts only from the excess over the best novel logit.
#
# Usage:
#   sbatch scripts/experiment/calibrate_tunk_viscache.sbatch
#   sbatch scripts/experiment/calibrate_tunk_viscache.sbatch --num-images 1000
#   sbatch scripts/experiment/calibrate_tunk_viscache.sbatch --taus 0.50,0.55,0.60 --alphas 0.1,0.2,0.3
#   sbatch scripts/experiment/calibrate_tunk_viscache.sbatch --score-deltas 0.003,0.005,0.008 --floor-score 0.0015 --pred-score-thr 0.001
#   sbatch scripts/experiment/calibrate_tunk_viscache.sbatch --score-deltas 0.001,0.002,0.003 --margin-eps 0.0005

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
mkdir -p slurm_logs/experiment probe_out/tunk_calibration

VARIANT="plain"
SHOT=10
SEED=1
NUM_IMAGES=0
BATCH_SIZE=8
CONF_THR=0.05
PRED_SCORE_THR=-1
TAUS="0.52,0.56,0.60"
ALPHAS="0.10,0.20,0.30"
SCORE_DELTAS=""
MARGIN_EPS=0.0005
TEMP=0.04
FLOOR_SCORE=0.05
CKPT_OVERRIDE=""
CACHE_OVERRIDE=""
OUT_TAG=""
PROTECT_BASE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)      VARIANT="$2"; shift 2 ;;
        --shot)         SHOT="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --num-images)   NUM_IMAGES="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --conf-thr)     CONF_THR="$2"; shift 2 ;;
        --pred-score-thr) PRED_SCORE_THR="$2"; shift 2 ;;
        --taus)         TAUS="$2"; shift 2 ;;
        --alphas)       ALPHAS="$2"; shift 2 ;;
        --score-deltas) SCORE_DELTAS="$2"; shift 2 ;;
        --margin-eps)   MARGIN_EPS="$2"; shift 2 ;;
        --temp)         TEMP="$2"; shift 2 ;;
        --floor-score)  FLOOR_SCORE="$2"; shift 2 ;;
        --ckpt)         CKPT_OVERRIDE="$2"; shift 2 ;;
        --cache)        CACHE_OVERRIDE="$2"; shift 2 ;;
        --out-tag)      OUT_TAG="$2"; shift 2 ;;
        --protect-base) PROTECT_BASE=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

case "$VARIANT" in
    plain) SUFFIX="" ;;
    kume)  SUFFIX="_kume" ;;
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
    echo "ERROR: no checkpoint found. work dir: $T2_WORK_DIR"
    exit 1
fi

if [ -n "$CACHE_OVERRIDE" ]; then
    CACHE="$CACHE_OVERRIDE"
else
    CACHE=cache/idd/viscache_${SHOT}shot_seed${SEED}.pt
fi
if [ ! -f "$CACHE" ]; then
    echo "ERROR: cache not found: $CACHE"
    exit 1
fi

TAG=${OUT_TAG:-idd_${VARIANT}_${SHOT}shot_seed${SEED}}
OUT_DIR=probe_out/tunk_calibration/${TAG}
mkdir -p "$OUT_DIR"

NUM_IMAGES_FLAG=""
[ "$NUM_IMAGES" -gt 0 ] 2>/dev/null && NUM_IMAGES_FLAG="--num-images $NUM_IMAGES"

echo "============================================================"
echo "  T_unk visual-cache calibration sweep"
echo "  checkpoint:   $CKPT"
echo "  cache:        $CACHE"
echo "  output:       $OUT_DIR"
echo "  taus:         $TAUS"
echo "  alphas:       $ALPHAS"
echo "  score_deltas: ${SCORE_DELTAS:-none}"
echo "  margin_eps:   $MARGIN_EPS"
echo "  floor_score:  $FLOOR_SCORE"
echo "  images:       ${NUM_IMAGES:-all}"
echo "  pred_score_thr: $PRED_SCORE_THR"
echo "  protect_base: $PROTECT_BASE"
echo "  job:          $SLURM_JOB_ID"
echo "============================================================"

PROTECT_BASE_FLAG=""
[ "$PROTECT_BASE" -eq 1 ] && PROTECT_BASE_FLAG="--protect-base"

DATASET=IDD TASK=2 THRESHOLD=${CONF_THR} SAVE=False \
    FEWSHOT_K=${SHOT} FEWSHOT_SEED=${SEED} FEWSHOT_DIR=data/OWOD/iddsplit \
    python tools/experiment_tunk_viscache_calibration.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --cache "$CACHE" \
        --out-dir "$OUT_DIR" \
        --split test \
        --batch-size "$BATCH_SIZE" \
        --conf-thr "$CONF_THR" \
        --pred-score-thr "$PRED_SCORE_THR" \
        --taus "$TAUS" \
        --alphas "$ALPHAS" \
        --score-deltas "$SCORE_DELTAS" \
        --margin-eps "$MARGIN_EPS" \
        --temp "$TEMP" \
        --floor-score "$FLOOR_SCORE" \
        $PROTECT_BASE_FLAG \
        $NUM_IMAGES_FLAG

echo "============================================================"
echo "DONE — $(date)"
ls -lh "$OUT_DIR"
echo "============================================================"
