#!/bin/bash
#SBATCH --job-name=ssm_tumor_ft
#SBATCH --partition=gpu-h100          # EDIT: H100 partition (check `sinfo -s`). You compiled mamba for sm_90 → MUST be H100.
#SBATCH --gres=gpu:1                  # EDIT: 1 H100. Some clusters want --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12            # for DataLoader workers (matches --num_workers below-ish)
#SBATCH --mem=80G
#SBATCH --time=48:00:00               # EDIT: wall-clock limit
#SBATCH --output=logs/%x_%j.out       # %x=jobname %j=jobid ; ensure ./logs exists
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

# Triton JITs a small helper with gcc at runtime → needs the module gcc, not /usr/bin/gcc 4.8.5
export CC=$(which gcc)
export CXX=$(which g++)
# CUDA allocator: reduce fragmentation OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# (TORCH_CUDA_ARCH_LIST only matters at compile time; not needed at runtime,
#  but harmless to pin to the H100 you built for.)
export TORCH_CUDA_ARCH_LIST="9.0"


# ============================================================
# Training
# ============================================================
echo "===== training ====="

# EDIT all paths below to your HPC locations
PRETRAINED=../Model/Healthy_Hilbert_MAE_Whole/best_valmae.pth
SAVE_DIR=../Model/SSMHybrid_Tumor_yieldDice
TRAIN_DIR=/mnt/parscratch/users/cov23yc/SSM-Data/Patches_Tumor_Train_256_128
SPLIT_JSON=/mnt/parscratch/users/cov23yc/SSM-Data/tumor_split.json

mkdir -p "$SAVE_DIR"

# --resume "" starts a FRESH finetune from --pretrained.
# To RESUME a crashed run instead, set:
#   --resume "$SAVE_DIR/last.pth"
python finetune_tumor.py \
    --pretrained  "$PRETRAINED" \
    --save_dir    "$SAVE_DIR" \
    --train_dir   "$TRAIN_DIR" \
    --patch_dir   "$TRAIN_DIR" \
    --split_json  "$SPLIT_JSON" \
    --resume      "" \
    --num_workers 10 \
    --batch_size  2 \
    --lambda_yield 0.5 \
    --mse_weight   0.3

echo "===== done ($SLURM_JOB_ID) ====="
