#!/bin/bash
#SBATCH --job-name=probe_tv
#SBATCH --output=slurm_logs/analysis/probe/probe_%j.out
#SBATCH --error=slurm_logs/analysis/probe/probe_%j.err
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00

# ============================================================================
# Text-Visual Alignment Probe
#
# Usage (config path):
#   sbatch scripts/diagnose_owod.sbatch --config configs/test/idd_prompt_eval.yaml
#   sbatch scripts/diagnose_owod.sbatch --config configs/test/idd_prompt_eval.yaml --out-dir results/probe/v3
#
# Usage (dataset shorthand — selects default config automatically):
#   sbatch scripts/diagnose_owod.sbatch --dataset idd
#   sbatch scripts/diagnose_owod.sbatch --dataset nuimages
#   sbatch scripts/diagnose_owod.sbatch --dataset nuimages --out-dir results/probe/nu_v1
#   sbatch scripts/diagnose_owod.sbatch --config configs/test/idd_train_top_logits.yaml --top-logits
#
# All experiment knobs (prompts, checkpoint, split, IoU threshold) live in the
# YAML config. Bump experiment_tag in the YAML to version a new run.
# ============================================================================

PYTHON_ARGS=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)   PYTHON_ARGS="$PYTHON_ARGS --dataset $2"; SELECTED_DATASET="$2"; shift 2;;
        --config)    PYTHON_ARGS="$PYTHON_ARGS --config $2";
                     # Infer dataset from config path to set correct env vars
                     if echo "$2" | grep -qi "nuimage\|nuowodb"; then
                         SELECTED_DATASET="nuimages"
                     elif echo "$2" | grep -qi "idd"; then
                         SELECTED_DATASET="idd"
                     else
                         SELECTED_DATASET="custom"
                     fi
                     shift 2;;
        --out-dir)   OUT_DIR="$2"; shift 2;;
        --checkpoint) PYTHON_ARGS="$PYTHON_ARGS --checkpoint $2"; shift 2;;
        --compare)   PYTHON_ARGS="$PYTHON_ARGS --compare"; shift 1;;
        --top-logits) PYTHON_ARGS="$PYTHON_ARGS --top-logits"; shift 1;;
        --top-k)     PYTHON_ARGS="$PYTHON_ARGS --top-k $2"; shift 2;;
        --max-boxes-per-class) PYTHON_ARGS="$PYTHON_ARGS --max-boxes-per-class $2"; shift 2;;
        --only-task) PYTHON_ARGS="$PYTHON_ARGS --only-task $2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ -z "$PYTHON_ARGS" ]; then
    # Default to IDD if no arg given
    PYTHON_ARGS="--dataset idd"
    SELECTED_DATASET="idd"
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
mkdir -p slurm_logs/analysis/probe

# Dataset-specific env vars (OWOD config uses these for $-substitution)
case "${SELECTED_DATASET}" in
    idd)
        export DATASET="${DATASET:-IDD}"
        export TASK="${TASK:-2}"
        export FEWSHOT_K="${FEWSHOT_K:-10}"
        export FEWSHOT_SEED="${FEWSHOT_SEED:-1}"
        export FEWSHOT_DIR="${FEWSHOT_DIR:-data/OWOD/iddsplit}"
        ;;
    nuimages)
        export DATASET="${DATASET:-nuOWODB}"
        export TASK="${TASK:-2}"
        export FEWSHOT_K="${FEWSHOT_K:-10}"
        export FEWSHOT_SEED="${FEWSHOT_SEED:-1}"
        export FEWSHOT_DIR="${FEWSHOT_DIR:-data/OWOD/nuowosplit}"
        ;;
    *)
        # custom config: caller must set DATASET/TASK etc. via env if needed
        export DATASET="${DATASET:-IDD}"
        export TASK="${TASK:-2}"
        export FEWSHOT_K="${FEWSHOT_K:-10}"
        export FEWSHOT_SEED="${FEWSHOT_SEED:-1}"
        export FEWSHOT_DIR="${FEWSHOT_DIR:-data/OWOD/iddsplit}"
        ;;
esac

export THRESHOLD="${THRESHOLD:-0.05}"
export SAVE="${SAVE:-False}"
export VISUALIZE="${VISUALIZE:-0}"

echo "=========================================="
echo "TEXT-VISUAL ALIGNMENT PROBE"
echo "=========================================="
echo "  Dataset:  ${SELECTED_DATASET}"
echo "  Args:     ${PYTHON_ARGS}"
echo "  Out:      ${OUT_DIR:-(default from YAML tag)}"
echo "=========================================="

if [ -n "$OUT_DIR" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --out-dir $OUT_DIR"
fi

python tools/probe_text_visual.py $PYTHON_ARGS

echo ""
echo "PROBE FINISHED at $(date)"
