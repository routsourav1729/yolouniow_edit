#!/bin/bash
#SBATCH --job-name=attr_select
#SBATCH --output=slurm_logs/analysis/attr_select/attr_select_%j.out
#SBATCH --error=slurm_logs/analysis/attr_select/attr_select_%j.err
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00

# ============================================================================
# Discriminative Attribute Selection (Stage 1-9 + sanity checks)
#
# Usage:
#   sbatch scripts/select_attributes.sbatch --dataset idd
#   sbatch scripts/select_attributes.sbatch --dataset nuowodb
#   sbatch scripts/select_attributes.sbatch --dataset idd --cache-pools
#   sbatch scripts/select_attributes.sbatch --dataset idd --num-images 10  # smoke test
#   sbatch scripts/select_attributes.sbatch --dataset nuscenes \
#       --cached-pools results/attributes/nuowodb/visual_pools_iou08.pt \
#       --cache-only --ratio-only --out-dir slurm_logs/analysis/attr_select \
#       --class-scope current --output-prefix nuscenes_attrselect
#
# Outputs → results/attributes/{idd,nuowodb}/
#   selected_top{3,5}_lambda{0.5,0.7,0.9}.json
#   scores_iou08.csv
#   visual_pools_iou{07,08,09}.pt
#   default_name_confusion.npy
# ============================================================================

SELECTED_DATASET=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)    SELECTED_DATASET="$2"; shift 2;;
        --num-images) EXTRA_ARGS="$EXTRA_ARGS --num-images $2"; shift 2;;
        --out-dir)    EXTRA_ARGS="$EXTRA_ARGS --out-dir $2"; shift 2;;
        --top-k)      EXTRA_ARGS="$EXTRA_ARGS --top-k $2"; shift 2;;
        --cached-pools) EXTRA_ARGS="$EXTRA_ARGS --cached-pools $2"; shift 2;;
        --cache-only) EXTRA_ARGS="$EXTRA_ARGS --cache-only"; shift;;
        --ratio-only) EXTRA_ARGS="$EXTRA_ARGS --ratio-only"; shift;;
        --class-scope) EXTRA_ARGS="$EXTRA_ARGS --class-scope $2"; shift 2;;
        --output-prefix) EXTRA_ARGS="$EXTRA_ARGS --output-prefix $2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ -z "$SELECTED_DATASET" ]; then
    echo "ERROR: --dataset required (idd or nuowodb/nuscenes)"; exit 1
fi

source /home/agipml/sourav.rout/miniconda3/bin/activate yolouniow

set -eo pipefail
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
mkdir -p slurm_logs/analysis/attr_select

case "${SELECTED_DATASET}" in
    idd)
        export DATASET="${DATASET:-IDD}"
        export TASK="${TASK:-2}"
        export FEWSHOT_K="${FEWSHOT_K:-10}"
        export FEWSHOT_SEED="${FEWSHOT_SEED:-1}"
        export FEWSHOT_DIR="${FEWSHOT_DIR:-data/OWOD/iddsplit}"
        CONFIG="configs/test/idd_attribute_select.yaml"
        ;;
    nuowodb|nuimages|nuscenes)
        export DATASET="${DATASET:-nuOWODB}"
        export TASK="${TASK:-2}"
        export FEWSHOT_K="${FEWSHOT_K:-10}"
        export FEWSHOT_SEED="${FEWSHOT_SEED:-1}"
        export FEWSHOT_DIR="${FEWSHOT_DIR:-data/OWOD/nuowosplit}"
        CONFIG="configs/test/nuowodb_attribute_select.yaml"
        ;;
    *)
        echo "ERROR: unknown dataset '${SELECTED_DATASET}'. Use idd or nuowodb/nuscenes."; exit 1;;
esac

echo "=========================================="
echo "DISCRIMINATIVE ATTRIBUTE SELECTION"
echo "  Dataset:  ${SELECTED_DATASET}"
echo "  Config:   ${CONFIG}"
echo "  Extra:    ${EXTRA_ARGS}"
echo "=========================================="

python tools/select_attributes.py --config "${CONFIG}" ${EXTRA_ARGS}

echo ""
echo "ATTR SELECT FINISHED at $(date)"
