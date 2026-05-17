#!/bin/bash
#SBATCH --job-name=idd_tunk_diag
#SBATCH --output=slurm_logs/experiment/tunk_diag_%j.out
#SBATCH --error=slurm_logs/experiment/tunk_diag_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

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
mkdir -p slurm_logs/experiment probe_out/tunk_diagnostic

VARIANT="plain"
SHOT=10
SEED=1
NUM_IMAGES=5000
BATCH_SIZE=8
TAUS="0.50,0.55,0.60,0.65"
MARGINS="0.0005,0.001,0.002,0.003,0.005"
OUT_TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)    VARIANT="$2"; shift 2 ;;
        --shot)       SHOT="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --num-images) NUM_IMAGES="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --taus)       TAUS="$2"; shift 2 ;;
        --margins)    MARGINS="$2"; shift 2 ;;
        --out-tag)    OUT_TAG="$2"; shift 2 ;;
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
CKPT=$(ls -t "${T2_WORK_DIR}"/best*.pth 2>/dev/null | head -1)
CACHE=cache/idd/viscache_${SHOT}shot_seed${SEED}.pt
TAG=${OUT_TAG:-idd_${VARIANT}_${SHOT}shot_seed${SEED}_5k}
OUT_DIR=probe_out/tunk_diagnostic/${TAG}
mkdir -p "$OUT_DIR"

NUM_IMAGES_FLAG=""
[ "$NUM_IMAGES" -gt 0 ] 2>/dev/null && NUM_IMAGES_FLAG="--num-images $NUM_IMAGES"

echo "============================================================"
echo "  T_unk margin/cache diagnostic"
echo "  checkpoint: $CKPT"
echo "  cache:      $CACHE"
echo "  output:     $OUT_DIR"
echo "  taus:       $TAUS"
echo "  margins:    $MARGINS"
echo "  images:     ${NUM_IMAGES:-all}"
echo "  job:        $SLURM_JOB_ID"
echo "============================================================"

DATASET=IDD TASK=2 THRESHOLD=0.05 SAVE=False \
    FEWSHOT_K=${SHOT} FEWSHOT_SEED=${SEED} FEWSHOT_DIR=data/OWOD/iddsplit \
    python tools/diagnose_tunk_margin_cache.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --cache "$CACHE" \
        --out-dir "$OUT_DIR" \
        --split test \
        --batch-size "$BATCH_SIZE" \
        --taus "$TAUS" \
        --margins "$MARGINS" \
        $NUM_IMAGES_FLAG

echo "============================================================"
echo "DONE — $(date)"
ls -lh "$OUT_DIR"
echo "============================================================"
