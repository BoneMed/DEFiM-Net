#!/bin/bash
#SBATCH --job-name=precut_healthy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00             # healthy: 1725 cases × 27 patches = 46575 — give plenty of room
#SBATCH --output=logs/precut_healthy_%j.out
#SBATCH --error=logs/precut_healthy_%j.err

set -eo pipefail

module purge
module load Anaconda3/2022.05
source activate pyansys

cd "$SLURM_SUBMIT_DIR"

echo "===== precut healthy on $(hostname) ====="
date

python dataUtils/precut_patches.py \
  --data_dir       /mnt/parscratch/users/cov23yc/SSM-Data/ElementGraph_Healthy_Train \
  --train_out_dir  /mnt/parscratch/users/cov23yc/SSM-Data/Patches_Healthy_Train_256_128 \
  --val_out_dir    /mnt/parscratch/users/cov23yc/SSM-Data/_unused_val_healthy \
  --val_cases 0 --grid_res 256 --patch_size 128 \
  --train_overlap 0.5 --s3_splat min

echo "===== count check ====="
find /mnt/parscratch/users/cov23yc/SSM-Data/Patches_Healthy_Train_256_128 -name '*.pt' | wc -l
# expected: 46575

echo "===== done $(date) ====="
