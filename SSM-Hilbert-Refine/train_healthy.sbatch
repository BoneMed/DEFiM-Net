#!/bin/bash
#SBATCH --job-name=ssm_healthy_pt
#SBATCH --partition=gpu-h100          # EDIT: H100 partition (check `sinfo -s`). mamba compiled for sm_90 → MUST be H100.
#SBATCH --gres=gpu:1                  # EDIT: some clusters want --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=72:00:00               # EDIT: pretrain is longer (full data, 400 ep)
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# NB: -u dropped on purpose — conda's activate scripts reference unbound
# vars and would abort under `set -u`.
set -eo pipefail

# ============================================================
# 1. Environment  (must match what you compiled mamba_ssm with)
# ============================================================
module purge
module load GCC/12.3.0
module load CUDA/12.1.1
module load Anaconda3/2022.05          # EDIT if your conda comes from elsewhere

source activate pyansys                # EDIT: your conda env name

export CC=$(which gcc)
export CXX=$(which g++)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST="9.0"

cd "$SLURM_SUBMIT_DIR"

# ============================================================
# 2. Training  (healthy pretrain from scratch — no --pretrained)
# ============================================================
echo "===== healthy pretrain ====="

# EDIT all paths below to your HPC locations
SAVE_DIR=../Model/Healthy_Hilbert_MAE_Whole
PATCH_DIR=/mnt/parscratch/users/cov23yc/SSM-Data/Patches_Healthy_Train_256_128
SPLIT_JSON=/mnt/parscratch/users/cov23yc/SSM-Data/healthy_split.json

mkdir -p "$SAVE_DIR"

# --train_fraction/--val_fraction 1.0 = full data (default is 0.3 subsample).
# healthy keeps --lambda_yield 0.0 (yield-Dice is a tumor-only concern).
# To RESUME a crashed run: add  --resume "$SAVE_DIR/best_valloss.pth"
python healthyPreTrain.py \
    --save_dir       "$SAVE_DIR" \
    --patch_dir      "$PATCH_DIR" \
    --split_json     "$SPLIT_JSON" \
    --train_fraction 1.0 \
    --val_fraction   1.0 \
    --num_workers    10 \
    --batch_size     4 \
    --mse_weight     0.1 \
    --lambda_yield   0.0

echo "===== done ($SLURM_JOB_ID) ====="
