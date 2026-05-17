#!/bin/bash
#SBATCH --job-name=idd_tunk_thr
#SBATCH --output=slurm_logs/idd/eval/idd_tunk_thr_sweep_%j.out
#SBATCH --error=slurm_logs/idd/eval/idd_tunk_thr_sweep_%j.err
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00

set -euo pipefail

cd /home/agipml/sourav.rout/ALL_FILES/hypyolo/YOLO-UniOW
mkdir -p slurm_logs/idd/eval

FED_CFG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_fedbce_btbg_maskpuk_tunkfree_hnunk_80e_val10.py
FED_CKPT=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task2_10shot_seed1_fedbce_btbg_maskpuk_tunkfree_hnunk_80e_val10/best_owod_Both_epoch_20.pth

VAN_CFG=configs/owod_ft/yolo_uniow_l_lora_bn_1e-3_40e_8gpus_owod_idd_t2_80e_val10.py
VAN_CKPT=work_dirs/yolo_uniow_l_lora_bn_1e-3_20e_8gpus_owod_idd_train_task2_10shot_seed1_80e_val10/best_owod_Both_epoch_30.pth

echo "IDD T2 T_unk acceptance threshold sweep"
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

for thr in 0.002 0.005 0.01 0.05; do
    echo "=========================================="
    echo "FedBCE+HN best, metric T_unk threshold=${thr}"
    echo "=========================================="
    bash scripts/eval_owod.sbatch \
        --task 2 \
        --config "$FED_CFG" \
        --ckpt "$FED_CKPT" \
        --unk-threshold "$thr" \
        --tag fedhn_best

    echo "=========================================="
    echo "Vanilla best, metric T_unk threshold=${thr}"
    echo "=========================================="
    bash scripts/eval_owod.sbatch \
        --task 2 \
        --config "$VAN_CFG" \
        --ckpt "$VAN_CKPT" \
        --unk-threshold "$thr" \
        --tag vanilla_best
done

echo ""
echo "Sweep finished at: $(date)"
