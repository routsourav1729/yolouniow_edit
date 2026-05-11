#!/bin/bash
#SBATCH --job-name=uniow_viscache
#SBATCH --output=slurm_logs/viscache_%j.out
#SBATCH --error=slurm_logs/viscache_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=01:00:00

# Always run from repo root so all relative paths (work_dirs/, configs/, ...) resolve.
REPO_ROOT=/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW
cd "$REPO_ROOT" || { echo "ERROR: cannot cd to $REPO_ROOT"; exit 1; }

# ============================================================================
# Visual K-shot cache (FS-OSOD) build + eval — INFERENCE-TIME ONLY.
#
# Stage 1: build (n_novel, K, 512) post-BN feature cache from K-shot GTs
#          using the existing T2 best checkpoint.
# Stage 2: run dist_test.sh with the *_viscache config + same checkpoint.
#
# Usage:
#   sbatch scripts/eval_viscache.sbatch --dataset IDD
#   sbatch scripts/eval_viscache.sbatch --dataset IDD --alpha 0.5
#   sbatch scripts/eval_viscache.sbatch --dataset IDD --reduce topk_mean
#   sbatch scripts/eval_viscache.sbatch --dataset nuOWODB --shot 10 --seed 1
#   sbatch scripts/eval_viscache.sbatch --dataset FOOD_VOCCOCO --t2-ckpt path/to/best.pth
#
# Required:
#   --dataset DATASET          IDD | nuOWODB | FOOD_VOCCOCO
# Optional:
#   --shot K                   default 10
#   --seed S                   default 1
#   --alpha A                  default 1.0
#   --reduce R                 mean | topk_mean | max  (default mean)
#   --t2-ckpt PATH             override T2 best checkpoint (auto-detect)
#   --threshold T              default 0.05
#   --skip-build               reuse existing cache file
#   --gpus N                   default 1
# ============================================================================

DATASET=""
FEWSHOT_K=10
FEWSHOT_SEED=1
ALPHA=1.0
REDUCE=mean
T2_CKPT_OVERRIDE=""
THRESHOLD=0.05
SKIP_BUILD=0
NUM_GPUS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)    DATASET="$2"; shift 2 ;;
        --shot)       FEWSHOT_K="$2"; shift 2 ;;
        --seed)       FEWSHOT_SEED="$2"; shift 2 ;;
        --alpha)      ALPHA="$2"; shift 2 ;;
        --reduce)     REDUCE="$2"; shift 2 ;;
        --t2-ckpt)    T2_CKPT_OVERRIDE="$2"; shift 2 ;;
        --threshold)  THRESHOLD="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=1; shift 1 ;;
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset required (IDD | nuOWODB | FOOD_VOCCOCO)"
    exit 1
fi
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}

STEM=yolo_uniow_l_lora_bn_1e-3
DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

case $DATASET in
    IDD)
        T2_CONFIG=configs/owod_ft/${STEM}_40e_8gpus_owod_idd_t2.py
        VC_CONFIG=configs/owod_ft/${STEM}_40e_8gpus_owod_idd_t2_viscache.py
        FEWSHOT_DIR=data/OWOD/iddsplit
        CACHE_SUBDIR=idd ;;
    nuOWODB)
        T2_CONFIG=configs/owod_ft/${STEM}_40e_8gpus_owod_nuowodb_t2.py
        VC_CONFIG=configs/owod_ft/${STEM}_40e_8gpus_owod_nuowodb_t2_viscache.py
        FEWSHOT_DIR=data/OWOD/nuowodb_fewshot_tfa
        CACHE_SUBDIR=nuowodb ;;
    FOOD_VOCCOCO)
        T2_CONFIG=configs/owod_ft/${STEM}_80e_8gpus_owod_food_voccoco_t2.py
        VC_CONFIG=configs/owod_ft/${STEM}_80e_8gpus_owod_food_voccoco_t2_viscache.py
        FEWSHOT_DIR=data/OWOD/voccocosplit
        CACHE_SUBDIR=food_voccoco ;;
    *) echo "ERROR: unsupported dataset $DATASET"; exit 1 ;;
esac

# T2 best checkpoint — same convention as train_owod_l_t2.sbatch.
T2_WORK_DIR=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_${DATASET_LOWER}_train_task2_${FEWSHOT_K}shot_seed${FEWSHOT_SEED}
if [ -n "$T2_CKPT_OVERRIDE" ]; then
    T2_BEST="$T2_CKPT_OVERRIDE"
else
    T2_BEST=$(ls -t "${T2_WORK_DIR}"/best*.pth 2>/dev/null | head -1)
fi
if [ -z "$T2_BEST" ] || [ ! -f "$T2_BEST" ]; then
    echo "ERROR: T2 best checkpoint not found."
    echo "  T2_WORK_DIR: ${T2_WORK_DIR}"
    echo "  pwd:         $(pwd)"
    echo "  Files in T2_WORK_DIR:"
    ls -lh "${T2_WORK_DIR}" 2>&1 | head -20 | sed 's/^/    /'
    echo "  Pass --t2-ckpt explicitly."
    exit 1
fi

CACHE_DIR=cache/${CACHE_SUBDIR}
CACHE_PATH=${CACHE_DIR}/viscache_${FEWSHOT_K}shot_seed${FEWSHOT_SEED}.pt
EVAL_DIR=work_dirs/eval_${DATASET}_L_viscache_${FEWSHOT_K}shot_seed${FEWSHOT_SEED}_a${ALPHA}_${REDUCE}
mkdir -p "$CACHE_DIR" "$EVAL_DIR" slurm_logs

# Snapshot T2 ckpt to the eval dir so concurrent training (max_keep_ckpts)
# can't delete it between Stage 1 and Stage 2. Cache and eval MUST come from
# the same exact weights — otherwise feature extractor and head are mismatched.
T2_CKPT_SNAPSHOT="${EVAL_DIR}/t2_ckpt.pth"
if [ ! -f "$T2_CKPT_SNAPSHOT" ] || [ "$T2_BEST" -nt "$T2_CKPT_SNAPSHOT" ]; then
    echo "[snapshot] $T2_BEST -> $T2_CKPT_SNAPSHOT"
    cp -f "$T2_BEST" "$T2_CKPT_SNAPSHOT"
fi
T2_BEST="$T2_CKPT_SNAPSHOT"

source /home/agipml/sourav.rout/miniconda3/bin/activate yolouniow
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW:$PYTHONPATH
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$HF_HOME
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))
export MASTER_PORT
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / NUM_GPUS ))

echo "=========================================="
echo "Visual K-Shot Cache  Dataset=$DATASET  K=$FEWSHOT_K  Seed=$FEWSHOT_SEED"
echo "  T2 ckpt:   $T2_BEST"
echo "  T2 cfg:    $T2_CONFIG"
echo "  VC cfg:    $VC_CONFIG"
echo "  Cache:     $CACHE_PATH"
echo "  alpha:     $ALPHA   reduce: $REDUCE"
echo "  Eval dir:  $EVAL_DIR"
echo "=========================================="

# ─── Stage 1: build cache ─────────────────────────────────────────────────────
if [ "$SKIP_BUILD" -eq 0 ] || [ ! -f "$CACHE_PATH" ]; then
    echo "[Stage 1] Building visual cache..."
    DATASET=${DATASET} TASK=2 \
        FEWSHOT_K=${FEWSHOT_K} FEWSHOT_SEED=${FEWSHOT_SEED} FEWSHOT_DIR=${FEWSHOT_DIR} \
        THRESHOLD=${THRESHOLD} \
        python tools/owod_scripts/build_visual_cache.py \
        --config "$T2_CONFIG" \
        --checkpoint "$T2_BEST" \
        --out "$CACHE_PATH" \
        --reduce-default "$REDUCE"
    if [ $? -ne 0 ]; then echo "ERROR: cache build failed"; exit 1; fi
else
    echo "[Stage 1] --skip-build set and cache exists, reusing $CACHE_PATH"
fi

# ─── Stage 2: eval with viscache config ───────────────────────────────────────
echo "[Stage 2] Running eval with visual cache fusion..."
# custom_hooks index of VisualCacheLoadHook = 2
# (EMAHook=0, mmdet.PipelineSwitchHook=1, VisualCacheLoadHook=2).
DATASET=${DATASET} TASK=2 THRESHOLD=${THRESHOLD} SAVE=False \
    FEWSHOT_K=${FEWSHOT_K} FEWSHOT_SEED=${FEWSHOT_SEED} FEWSHOT_DIR=${FEWSHOT_DIR} \
    PORT=${MASTER_PORT} \
    ./tools/dist_test.sh "$VC_CONFIG" "$T2_BEST" "$NUM_GPUS" \
    --work-dir "$EVAL_DIR" \
    --cfg-options \
        model.bbox_head.head_module.visual_alpha=${ALPHA} \
        model.bbox_head.head_module.visual_cache_cfg.reduce=${REDUCE} \
        custom_hooks.2.cache_path=${CACHE_PATH}

echo "=========================================="
echo "Done.  alpha=$ALPHA  reduce=$REDUCE"
echo "Eval dir: $EVAL_DIR"
echo "=========================================="
