#!/bin/bash
#SBATCH --job-name=idd_embeds
#SBATCH --output=slurm_logs/idd_embeds_%j.out
#SBATCH --error=slurm_logs/idd_embeds_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00

# ============================================================================
# Extract IDD text embeddings using cached CLIP model
# Lightweight — only loads CLIP text encoder, no full detector
# ============================================================================

echo "=========================================="
echo "IDD EMBEDDING EXTRACTION"
echo "Job started at:  $(date)"
echo "Running on node: $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"
echo "=========================================="

source /home/agipml/sourav.rout/miniconda3/bin/activate yolouniow

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# CLIP cache — no internet on compute nodes
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW
mkdir -p slurm_logs embeddings/uniow-idd

echo ""
echo "Python: $(which python)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

python tools/owod_scripts/extract_idd_embeddings.py --device auto

echo ""
echo "Embedding files:"
ls -la embeddings/uniow-s/idd_*.npy 2>/dev/null || echo "  ERROR: no idd embeddings found!"

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
