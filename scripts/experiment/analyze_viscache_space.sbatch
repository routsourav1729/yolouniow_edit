#!/bin/bash
#SBATCH --job-name=viscache_space
#SBATCH --output=slurm_logs/experiment/viscache_space_%j.out
#SBATCH --error=slurm_logs/experiment/viscache_space_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

source /home/agipml/sourav.rout/miniconda3/bin/activate yolouniow
cd /home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW
export PYTHONPATH=/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW:$PYTHONPATH

CACHE=${1:-cache/idd/viscache_10shot_seed1.pt}
OUT_DIR=${2:-probe_out/viscache_space/idd_10shot_seed1}
mkdir -p "$OUT_DIR" slurm_logs/experiment

python tools/analyze_viscache_space.py \
    --cache "$CACHE" \
    --out-dir "$OUT_DIR"
