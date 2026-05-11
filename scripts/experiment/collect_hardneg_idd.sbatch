#!/bin/bash
#SBATCH --job-name=idd_hardneg
#SBATCH --output=slurm_logs/hardneg_collect_%j.out
#SBATCH --error=slurm_logs/hardneg_collect_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00

# Collect T1 O2O base-positive hard negatives for IDD using the FULL T1
# training set (data/OWOD/ImageSets/IDD/t1_train.txt). No subsampling.
#
# Usage:
#   sbatch scripts/experiment/collect_hardneg_idd.sbatch
#   sbatch scripts/experiment/collect_hardneg_idd.sbatch --cache cache/idd/hardneg.pt
#   sbatch scripts/experiment/collect_hardneg_idd.sbatch --t1-ckpt work_dirs/.../best.pth

CACHE_PATH="cache/idd/hardneg_t1_o2o_10x8x3.pt"
T1_CKPT=""
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
THRESHOLD=0.05
CACHE_IMAGE_SET=train
CONFIG=configs/continual/t1_collect_hardneg_idd.py
WORK_DIR=work_dirs/hardneg_collect_idd_t1

while [[ $# -gt 0 ]]; do
    case $1 in
        --cache) CACHE_PATH="$2"; shift 2 ;;
        --t1-ckpt) T1_CKPT="$2"; shift 2 ;;
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --image-set) CACHE_IMAGE_SET="$2"; shift 2 ;;
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

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

MASTER_PORT=$(( 29500 + RANDOM % 1000 ))
export MASTER_PORT
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / NUM_GPUS ))

if [ -z "$T1_CKPT" ]; then
    T1_CKPT=$(ls -t work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task1/best*.pth 2>/dev/null | head -1)
fi
if [ -z "$T1_CKPT" ] || [ ! -f "$T1_CKPT" ]; then
    echo "ERROR: T1 IDD checkpoint not found. Pass --t1-ckpt /path/to/best.pth"
    exit 1
fi

mkdir -p "$(dirname "$CACHE_PATH")" "$WORK_DIR" slurm_logs
SPLIT_FILE="data/OWOD/ImageSets/IDD/t1_${CACHE_IMAGE_SET}.txt"
if [ ! -f "$SPLIT_FILE" ]; then
    echo "ERROR: Missing split: $SPLIT_FILE"
    exit 1
fi

echo "=========================================="
echo "Collecting IDD T1 O2O hard-negative cache"
echo "  config:    $CONFIG"
echo "  T1 ckpt:   $T1_CKPT"
echo "  cache:     $CACHE_PATH"
echo "  split:     $SPLIT_FILE ($(wc -l < "$SPLIT_FILE") images, FULL T1 train)"
echo "  work_dir:  $WORK_DIR"
echo "  GPUs:      $NUM_GPUS"
echo "=========================================="

DATASET=IDD TASK=1 THRESHOLD=${THRESHOLD} SAVE=False \
    CACHE_IMAGE_SET=${CACHE_IMAGE_SET} HARDNEG_CACHE_PATH=${CACHE_PATH} PORT=${MASTER_PORT} \
    ./tools/dist_train_owod.sh ${CONFIG} ${NUM_GPUS} \
    --amp \
    --work-dir ${WORK_DIR} \
    --cfg-options load_from=${T1_CKPT}
