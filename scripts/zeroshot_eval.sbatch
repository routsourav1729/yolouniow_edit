#!/bin/bash
#SBATCH --job-name=uniow_zs
#SBATCH --output=slurm_logs/zeroshot_%x_%j.out
#SBATCH --error=slurm_logs/zeroshot_%x_%j.err
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# ============================================================================
# YOLO-UniOW-L  Zero-Shot / T1-Checkpoint Evaluation  (all datasets)
#
# Text Encoder: HuggingCLIPLanguageBackbone (CLIPTextModelWithProjection)
#   L2-normalized 512-dim embeddings per class, frozen during eval.
#
# Two modes:
#   default (--zs-t2): T1-trained ckpt + TASK=2 → novel class zero-shot
#   --zs-t1          : pretrained ckpt + TASK=1 → base class zero-shot
#                      (pretrained/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth)
#
# Usage:
#   sbatch scripts/zeroshot_eval.sbatch --dataset IDD,FOOD_VOC,FOOD_VOCCOCO
#   sbatch scripts/zeroshot_eval.sbatch --dataset nuOWODB           # T1-ckpt, TASK=2 novel zeroshot
#   sbatch scripts/zeroshot_eval.sbatch --dataset nuOWODB --zs-t1  # pretrained, TASK=1 base zeroshot
#   sbatch scripts/zeroshot_eval.sbatch --dataset IDD --zs-t1
#   sbatch scripts/zeroshot_eval.sbatch --dataset IDD --ckpt work_dirs/.../best.pth
# ============================================================================

# ─── Parse arguments ──────────────────────────────────────────────────────────
DATASETS=""
CKPT_OVERRIDE=""
ZS_T1=0           # --zs-t1: pretrained ckpt + TASK=1 (base class zero-shot)
THRESHOLD=0.05
NUM_GPUS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)    DATASETS="$2"; shift 2 ;;
        --ckpt)       CKPT_OVERRIDE="$2"; shift 2 ;;
        --zs-t1)      ZS_T1=1; shift 1 ;;
        --threshold)  THRESHOLD="$2"; shift 2 ;;
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: sbatch scripts/zeroshot_eval.sbatch --dataset IDD[,FOOD_VOC,FOOD_VOCCOCO] [--zs-t1] [--ckpt PATH]"
            exit 1
            ;;
    esac
done

if [ -z "$DATASETS" ]; then
    echo "ERROR: --dataset is required. Available: IDD, FOOD_VOC, FOOD_VOCCOCO, nuOWODB"
    exit 1
fi

IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"

NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))
STEM=yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod
PRETRAIN_CKPT=pretrained/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth

# ─── Environment ──────────────────────────────────────────────────────────────
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
export NCCL_DEBUG=WARN
export MASTER_PORT

cd /home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW

mkdir -p slurm_logs/zeroshot

echo "=========================================="
echo "YOLO-UniOW-L  Zero-Shot Eval"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Started: $(date)"
echo "Datasets: ${DATASET_ARRAY[*]}  Threshold: $THRESHOLD  GPUs: $NUM_GPUS"
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.device_count()} GPUs')"
echo "=========================================="

# ─── Main loop ────────────────────────────────────────────────────────────────
EVAL_FAILED=0

for DATASET in "${DATASET_ARRAY[@]}"; do
    DATASET=$(echo "$DATASET" | xargs)

    case $DATASET in
        IDD)
            T1_CONFIG=configs/owod_ft/${STEM}_idd.py
            T1_WORK_DIR=work_dirs/${STEM}_idd_train_task1
            EMBED_T1=embeddings/uniow-idd/idd_t1.npy       # 8 base classes
            EMBED_T2=embeddings/uniow-idd/idd_t2.npy       # 14 classes: 8 base + 6 novel
            NUM_BASE_CLASSES=8
            NUM_NOVEL_CLASSES=6
            FEWSHOT_DIR=data/OWOD/iddsplit
            ;;
        FOOD_VOC)
            T1_CONFIG=configs/owod_ft/${STEM}_food_voc.py
            T1_WORK_DIR=work_dirs/${STEM}_food_voc_train_task1
            EMBED_T1=embeddings/uniow-food-voc/food_voc_t1.npy   # 10 base classes
            EMBED_T2=embeddings/uniow-food-voc/food_voc_t2.npy   # 15 classes: 10 base + 5 novel
            NUM_BASE_CLASSES=10
            NUM_NOVEL_CLASSES=5
            FEWSHOT_DIR=data/OWOD/vocsplit
            ;;
        FOOD_VOCCOCO)
            T1_CONFIG=configs/owod_ft/${STEM}_food_voccoco.py
            T1_WORK_DIR=work_dirs/${STEM}_food_voccoco_train_task1
            EMBED_T1=embeddings/uniow-food-voccoco/food_voccoco_t1.npy  # 20 base classes
            EMBED_T2=embeddings/uniow-food-voccoco/food_voccoco_t2.npy  # 40 classes: 20 base + 20 novel
            NUM_BASE_CLASSES=20
            NUM_NOVEL_CLASSES=20
            FEWSHOT_DIR=data/OWOD/voccocosplit
            ;;
        nuOWODB)
            # Base OWOD config resolves embedding_path from env DATASET/TASK automatically:
            #   TASK=1 → embeddings/uniow-l/nuowodb_t1.npy  (10 vehicle classes)
            #   TASK=2 → embeddings/uniow-l/nuowodb_t2.npy  (17 classes: 10 base + 7 pedestrian)
            T1_CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod.py
            T1_WORK_DIR=work_dirs/${STEM}_nuowodb_train_task1
            EMBED_T1=embeddings/uniow-l/nuowodb_t1.npy
            EMBED_T2=embeddings/uniow-l/nuowodb_t2.npy
            NUM_BASE_CLASSES=10
            NUM_NOVEL_CLASSES=7
            FEWSHOT_DIR=data/OWOD/nuowodb_fewshot_tfa
            ;;
        *)
            echo "ERROR: Unknown dataset '$DATASET'. Available: IDD, FOOD_VOC, FOOD_VOCCOCO, nuOWODB"
            EVAL_FAILED=1
            continue
            ;;
    esac

    if [ -n "$CKPT_OVERRIDE" ]; then
        CKPT="$CKPT_OVERRIDE"
        EVAL_TASK=2
        EMBED_FILE=$EMBED_T2
        MODE_DESC="custom ckpt, TASK=2 novel zero-shot"
    elif [ "$ZS_T1" -eq 1 ]; then
        # --zs-t1: pretrained model on T1 test set — no OWOD training at all
        CKPT="$PRETRAIN_CKPT"
        EVAL_TASK=1
        EMBED_FILE=$EMBED_T1
        MODE_DESC="pretrained ckpt, TASK=1 base zero-shot"
    else
        # default: T1-trained ckpt on T2 test set — novel class zero-shot
        CKPT=$(ls -t "${T1_WORK_DIR}"/best*.pth 2>/dev/null | head -1)
        if [ -z "$CKPT" ]; then
            echo "ERROR: No T1 checkpoint in ${T1_WORK_DIR}/. Use --ckpt or --zs-t1."
            EVAL_FAILED=1
            continue
        fi
        EVAL_TASK=2
        EMBED_FILE=$EMBED_T2
        MODE_DESC="T1 ckpt, TASK=2 novel zero-shot"
    fi

    CKPT_STEM=$(basename "$CKPT" .pth)
    EVAL_DIR=work_dirs/zeroshot_t${EVAL_TASK}_${DATASET}_${CKPT_STEM}
    mkdir -p "$EVAL_DIR"

    echo ""
    echo "=== $DATASET | $MODE_DESC | base=$NUM_BASE_CLASSES novel=$NUM_NOVEL_CLASSES ==="
    echo "  Config:    $T1_CONFIG"
    echo "  Ckpt:      $CKPT"
    echo "  Embeds:    $EMBED_FILE  $(python -c "import numpy as np; e=np.load('$EMBED_FILE'); print(e.shape)" 2>/dev/null)"
    echo "  Eval dir:  $EVAL_DIR"

    if [ ! -f "$CKPT" ]; then
        echo "ERROR: Checkpoint not found: $CKPT"
        EVAL_FAILED=1
        continue
    fi
    if [ ! -f "$EMBED_FILE" ]; then
        echo "ERROR: Embedding file not found: $EMBED_FILE"
        EVAL_FAILED=1
        continue
    fi

    # TASK=2 (default): T1-trained ckpt on T2 test set → novel class zero-shot
    # TASK=1 (--zs-t1): pretrained ckpt on T1 test set → base class zero-shot
    DATASET=$DATASET TASK=$EVAL_TASK THRESHOLD=$THRESHOLD SAVE=False \
        FEWSHOT_K=0 FEWSHOT_SEED=1 FEWSHOT_DIR=$FEWSHOT_DIR \
        PORT=${MASTER_PORT} \
        ./tools/dist_test.sh ${T1_CONFIG} ${CKPT} ${NUM_GPUS} \
        --work-dir ${EVAL_DIR}

    EVAL_RC=$?
    if [ $EVAL_RC -eq 0 ]; then
        echo "=== $DATASET DONE at $(date) — results in $EVAL_DIR ==="
        python -c "
import json, glob
files = sorted(glob.glob('${EVAL_DIR}/*.json'))
if files:
    with open(files[-1]) as f:
        r = json.load(f)
    for k, v in r.items():
        if any(x in k.lower() for x in ['owod', 'map', 'recall']):
            print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
" 2>/dev/null || true
    else
        echo "ERROR: $DATASET failed (exit code $EVAL_RC)"
        EVAL_FAILED=1
    fi

done

echo ""
echo "All datasets processed. Completed at: $(date)"
exit $EVAL_FAILED
