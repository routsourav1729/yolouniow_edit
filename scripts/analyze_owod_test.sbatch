#!/bin/bash
#SBATCH --job-name=test_analysis
#SBATCH --output=slurm_logs/analysis/test_%j.out
#SBATCH --error=slurm_logs/analysis/test_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00

# ============================================================================
# OWOD Test-Set Analysis — Confusion Matrix + Per-Class Anchor Metrics
#
# Usage:
#   sbatch scripts/analyze_owod_test.sbatch --dataset IDD --task 2
#   sbatch scripts/analyze_owod_test.sbatch --dataset nuOWODB --task 2
#   sbatch scripts/analyze_owod_test.sbatch --dataset nuOWODB --task 2 --method all
#
# Options:
#   --dataset         IDD | FOOD_VOCCOCO | FOOD_VOC | nuOWODB
#   --task            1 | 2
#   --method          baseline | kume | kume-kvk | all  (default: baseline)
#                     'all' = run every available checkpoint in one job (data loads once)
#   --checkpoint      Override checkpoint path
#   --config          Override config file path
#   --shot            Few-shot K (default: 10)
#   --seed            Few-shot seed (default: 1)
#   --out-dir         Output directory (default: <work_dir>/test_analysis)
#   --num-images      Limit number of images (default: all)
#   --conf-thr        Score threshold for confusion matrix (default: 0.05)
#   --batch-size      Inference batch size (default: 32)
# ============================================================================

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
mkdir -p slurm_logs/{idd,voccoco,voc,analysis}

# ─── Parse arguments ──────────────────────────────────────────────────────
DATASET="IDD"
TASK=2
METHOD="baseline"
CKPT=""
CONFIG=""
FEWSHOT_K=10
FEWSHOT_SEED=1
OUT_DIR_OVERRIDE=""
NUM_IMAGES=0
CONF_THR=0.05
BATCH_SIZE=64

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)    DATASET="$2"; shift 2;;
        --task)       TASK="$2"; shift 2;;
        --method)     METHOD="$2"; shift 2;;
        --checkpoint) CKPT="$2"; shift 2;;
        --config)     CONFIG="$2"; shift 2;;
        --shot)       FEWSHOT_K="$2"; shift 2;;
        --seed)       FEWSHOT_SEED="$2"; shift 2;;
        --out-dir)    OUT_DIR_OVERRIDE="$2"; shift 2;;
        --num-images) NUM_IMAGES="$2"; shift 2;;
        --conf-thr)   CONF_THR="$2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

export DATASET TASK

# ─── Resolve method suffix ────────────────────────────────────────────────
case "$METHOD" in
    kume)        METHOD_SUFFIX="_kume" ;;
    kume-kvk)
        # Margin from env KUME_KVK_MARGIN (default 0.5). Workdir suffix encodes margin.
        _M="${KUME_KVK_MARGIN:-0.5}"
        METHOD_SUFFIX="_kume_kvk_m${_M}"
        ;;
    baseline|"") METHOD_SUFFIX="" ;;
    all)         METHOD_SUFFIX="" ;;   # resolved per-entry in all mode
    *)           echo "ERROR: Unknown --method '$METHOD'. Use: baseline kume kume-kvk all"; exit 1 ;;
esac

# ─── Auto-detect config and checkpoint ────────────────────────────────────
DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
STEM=yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod

case $DATASET in
    IDD)
        LOG_DIR=slurm_logs/idd
        FEWSHOT_DIR=data/OWOD/iddsplit
        T1_CKPT_DIR=work_dirs/${STEM}_idd_train_task1
        if [ -z "$CONFIG" ]; then
            if [ "$TASK" -eq 1 ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd.py
            elif [ "$METHOD" = "kume" ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_kume.py
            elif [ "$METHOD" = "kume-kvk" ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_kume_kvk.py
            else
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2.py
            fi
        fi
        ;;
    FOOD_VOCCOCO)
        LOG_DIR=slurm_logs/voccoco
        FEWSHOT_DIR=data/OWOD/voccocosplit
        T1_CKPT_DIR=work_dirs/${STEM}_food_voccoco_train_task1
        if [ -z "$CONFIG" ]; then
            if [ "$TASK" -eq 1 ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voccoco.py
            else
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voccoco_t2.py
            fi
        fi
        ;;
    FOOD_VOC)
        LOG_DIR=slurm_logs/voc
        FEWSHOT_DIR=data/OWOD/vocsplit
        T1_CKPT_DIR=work_dirs/${STEM}_food_voc_train_task1
        if [ -z "$CONFIG" ]; then
            if [ "$TASK" -eq 1 ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_food_voc.py
            else
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voc_t2.py
            fi
        fi
        ;;
    nuOWODB)
        LOG_DIR=slurm_logs/nuscenes
        FEWSHOT_DIR=data/OWOD/nuowodb_fewshot_tfa
        T1_CKPT_DIR=work_dirs/${STEM}_nuowodb_train_task1
        if [ -z "$CONFIG" ]; then
            if [ "$TASK" -eq 1 ]; then
                # T1 uses base OWOD config with DATASET=nuOWODB TASK=1 env vars
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod.py
            elif [ "$METHOD" = "kume-kvk" ]; then
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_nuowodb_t2_kume_kvk.py
            else
                # T2 fewshot baseline config
                CONFIG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_nuowodb_t2.py
            fi
        fi
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'. Use IDD | FOOD_VOCCOCO | FOOD_VOC | nuOWODB"
        exit 1
        ;;
esac

# ─── Helper: resolve best checkpoint from a work dir ──────────────────────
resolve_ckpt() {
    local wdir="$1"
    local ckpt
    ckpt=$(ls -t "${wdir}"/best*.pth 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        ckpt=$(ls -t "${wdir}"/epoch_*.pth 2>/dev/null | head -1)
    fi
    echo "$ckpt"
}

# ─── T1 work dir (always the same regardless of method) ──────────────────
T1_WORK_DIR="$T1_CKPT_DIR"

# ─── Resolve config for T1 (used as reference pipeline regardless of method) ─
# (CONFIG is already set above in the dataset case block)

# Set env vars for the config
export THRESHOLD=0.05
export SAVE=False
export FEWSHOT_K=${FEWSHOT_K}
export FEWSHOT_SEED=${FEWSHOT_SEED}
export FEWSHOT_DIR
export VISUALIZE=${VISUALIZE:-0}

mkdir -p slurm_logs/nuscenes

# Symlink logs to dataset-specific dir
ln -sf "$(pwd)/slurm_logs/analysis/test_${SLURM_JOB_ID}.out" "${LOG_DIR}/test_${SLURM_JOB_ID}.out" 2>/dev/null
ln -sf "$(pwd)/slurm_logs/analysis/test_${SLURM_JOB_ID}.err" "${LOG_DIR}/test_${SLURM_JOB_ID}.err" 2>/dev/null

NUM_IMAGES_FLAG=""
if [ "$NUM_IMAGES" -gt 0 ] 2>/dev/null; then
    NUM_IMAGES_FLAG="--num-images $NUM_IMAGES"
fi

# ─── --method all: collect every available checkpoint, run in one pass ───
if [ "$METHOD" = "all" ]; then
    if [ "$TASK" -eq 1 ]; then
        echo "ERROR: --method all only makes sense for --task 2 (T1 has only one checkpoint)"
        exit 1
    fi

    # Collect all method variants for this dataset
    # Each entry: label|config|work_dir_suffix
    declare -a METHOD_DEFS
    case $DATASET in
        IDD)
            METHOD_DEFS=(
                "baseline|configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2.py|"
                "kume|configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_kume.py|_kume"
            )
            ;;
        FOOD_VOCCOCO)
            METHOD_DEFS=(
                "baseline|configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voccoco_t2.py|"
            )
            ;;
        FOOD_VOC)
            METHOD_DEFS=(
                "baseline|configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_80e_8gpus_owod_food_voc_t2.py|"
            )
            ;;
        nuOWODB)
            METHOD_DEFS=(
                "baseline|configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_nuowodb_t2.py|"
            )
            ;;
    esac

    # Build runs JSON
    RUNS_JSON_PATH="/tmp/owod_analysis_runs_${SLURM_JOB_ID:-$$}.json"
    echo "[" > "$RUNS_JSON_PATH"
    first=1
    for entry in "${METHOD_DEFS[@]}"; do
        label=$(echo "$entry" | cut -d'|' -f1)
        cfg=$(echo "$entry" | cut -d'|' -f2)
        suffix=$(echo "$entry" | cut -d'|' -f3)
        wdir=work_dirs/${STEM}_${DATASET_LOWER}_train_task2_${FEWSHOT_K}shot_seed${FEWSHOT_SEED}${suffix}
        ckpt=$(resolve_ckpt "$wdir")
        if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
            echo "  [skip] $label — no checkpoint in $wdir"
            continue
        fi
        out="${wdir}/test_analysis"
        mkdir -p "$out"
        [ "$first" -eq 0 ] && echo "," >> "$RUNS_JSON_PATH"
        printf '  {"label":"%s","config":"%s","checkpoint":"%s","out_dir":"%s"}' \
            "$label" "$cfg" "$ckpt" "$out" >> "$RUNS_JSON_PATH"
        first=0
        echo "  [queued] $label  ckpt=$(basename $ckpt)  out=$out"
    done
    echo "" >> "$RUNS_JSON_PATH"
    echo "]" >> "$RUNS_JSON_PATH"

    echo "=========================================="
    echo "OWOD TEST-SET ANALYSIS  (--method all)"
    echo "  Dataset:    $DATASET  T${TASK}"
    echo "  Runs JSON:  $RUNS_JSON_PATH"
    echo "  Num images: ${NUM_IMAGES:-all}"
    echo "  Conf thr:   $CONF_THR  Batch: $BATCH_SIZE"
    echo "  Log dir:    $LOG_DIR"
    echo "=========================================="
    cat "$RUNS_JSON_PATH"
    echo ""

    python tools/analyze_owod_test.py \
        --runs-json "$RUNS_JSON_PATH" \
        --conf-thr "$CONF_THR" \
        --batch-size "$BATCH_SIZE" \
        $NUM_IMAGES_FLAG

    rm -f "$RUNS_JSON_PATH"

else
    # ─── Single-method mode ────────────────────────────────────────────────
    if [ "$TASK" -eq 1 ]; then
        T2_WORK_DIR="$T1_WORK_DIR"
    else
        T2_WORK_DIR=work_dirs/${STEM}_${DATASET_LOWER}_train_task2_${FEWSHOT_K}shot_seed${FEWSHOT_SEED}${METHOD_SUFFIX}
    fi

    if [ -z "$CKPT" ]; then
        CKPT=$(resolve_ckpt "$T2_WORK_DIR")
    fi

    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "ERROR: No checkpoint found in ${T2_WORK_DIR}"
        echo "  Use --checkpoint /path/to/ckpt.pth to specify manually."
        exit 1
    fi

    if [ -n "$OUT_DIR_OVERRIDE" ]; then
        OUT_DIR="$OUT_DIR_OVERRIDE"
    else
        OUT_DIR="${T2_WORK_DIR}/test_analysis"
    fi
    mkdir -p "$OUT_DIR"

    echo "=========================================="
    echo "OWOD TEST-SET ANALYSIS"
    echo "  Dataset:    $DATASET"
    echo "  Task:       T${TASK}"
    echo "  Method:     $METHOD"
    echo "  Config:     $CONFIG"
    echo "  Checkpoint: $CKPT ($(du -sh "$CKPT" 2>/dev/null | cut -f1))"
    echo "  Out dir:    $OUT_DIR"
    echo "  Num images: ${NUM_IMAGES:-all}"
    echo "  Conf thr:   $CONF_THR"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Log dir:    $LOG_DIR"
    echo "=========================================="

    python tools/analyze_owod_test.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --out-dir "$OUT_DIR" \
        --conf-thr "$CONF_THR" \
        --batch-size "$BATCH_SIZE" \
        $NUM_IMAGES_FLAG
fi

echo ""
echo "Test analysis complete."
echo "  confusion_matrix.csv          — GT × predicted confusion matrix"
echo "  false_positives.csv           — unmatched predictions by class"
echo "  test_analysis.json            — full metrics + confusion data + unknown Tobj/stride"
echo "  unknown_tobj_by_stride.csv    — Tobj/Tunk mean,std,percentiles per unknown class × stride"
echo ""
echo "ANALYSIS FINISHED at $(date)"
