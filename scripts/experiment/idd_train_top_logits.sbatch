#!/bin/bash
#SBATCH --job-name=idd_toplogits
#SBATCH --output=slurm_logs/analysis/probe/toplogits_%j.out
#SBATCH --error=slurm_logs/analysis/probe/toplogits_%j.err
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=00:45:00

WORK_DIR=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task2_10shot_seed1_frozenneck_restore
OUT_DIR=results/probe/idd_train_tuned_prompt_top_logits_frozenneck_restore
CONFIG=configs/test/idd_train_top_logits.yaml
TOP_K=3
MAX_BOXES=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --max-boxes-per-class) MAX_BOXES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p slurm_logs/analysis/probe "$OUT_DIR"

CKPT=$(ls -t "${WORK_DIR}"/best*.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    CKPT=$(ls -t "${WORK_DIR}"/epoch_*.pth 2>/dev/null | head -1)
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found in $WORK_DIR"
    exit 1
fi

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
export DATASET=IDD
export TASK=2
export FEWSHOT_K=10
export FEWSHOT_SEED=1
export FEWSHOT_DIR=data/OWOD/iddsplit
export THRESHOLD=0.05
export SAVE=False

echo "=========================================="
echo "IDD train top-logits probe"
echo "  checkpoint: $CKPT"
echo "  config:     $CONFIG"
echo "  out:        $OUT_DIR"
echo "=========================================="

python tools/probe_text_visual.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --top-logits \
    --top-k "$TOP_K" \
    --max-boxes-per-class "$MAX_BOXES" \
    --only-task T2 \
    --out-dir "$OUT_DIR"
