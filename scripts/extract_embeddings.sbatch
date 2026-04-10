#!/bin/bash
#SBATCH --job-name=extract_embed
#SBATCH --output=slurm_logs/extract_embed_%j.out
#SBATCH --error=slurm_logs/extract_embed_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# ============================================================================
# Extract text embeddings using YOLO-World model backbone
# ============================================================================
#
# Usage:
#   sbatch scripts/extract_embeddings.sbatch                  → all datasets
#   sbatch scripts/extract_embeddings.sbatch FOOD_VOCCOCO     → single dataset
#
# This extracts class embeddings and wildcard ('object') embedding via the
# YOLO-World-L text backbone.  object_tuned.npy must be separately obtained
# from a trained model checkpoint (use --extract_tuned flag).
# ============================================================================

cd "$SLURM_SUBMIT_DIR"

source /home/agipml/sourav.rout/miniconda3/bin/activate yolouniow

export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

DATASET="${1:-ALL}"
CKPT="pretrained/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth"

mkdir -p slurm_logs

# ─── Route logs to dataset subfolder ─────────────────────────────────────────
case "$DATASET" in
    IDD)           _LOG_SUBDIR=slurm_logs/idd ;;
    FOOD_VOCCOCO)  _LOG_SUBDIR=slurm_logs/voccoco ;;
    FOOD_VOC)      _LOG_SUBDIR=slurm_logs/voc ;;
    ALL|*)         _LOG_SUBDIR="" ;;  # ALL: no single subdir
esac
if [ -n "$_LOG_SUBDIR" ]; then
    mkdir -p "$_LOG_SUBDIR"
    ln -sf "../extract_embed_${SLURM_JOB_ID}.out" "${_LOG_SUBDIR}/extract_embed_${SLURM_JOB_ID}.out" 2>/dev/null || true
    ln -sf "../extract_embed_${SLURM_JOB_ID}.err" "${_LOG_SUBDIR}/extract_embed_${SLURM_JOB_ID}.err" 2>/dev/null || true
    echo "[Log] ${_LOG_SUBDIR}/extract_embed_${SLURM_JOB_ID}.{out,err}"
fi

if [[ "$DATASET" == "ALL" ]]; then
    for ds in FOOD_VOCCOCO FOOD_VOC IDD; do
        echo "========== Extracting: $ds =========="
        python tools/owod_scripts/extract_embeddings.py \
            --ckpt "$CKPT" \
            --dataset "$ds" \
            --task 0
    done
else
    echo "========== Extracting: $DATASET =========="
    python tools/owod_scripts/extract_embeddings.py \
        --ckpt "$CKPT" \
        --dataset "$DATASET" \
        --task 0
fi

echo "Done!"
