#!/bin/bash
#SBATCH --job-name=precut_tumor
#SBATCH --cpus-per-task=8           # voxelisation is single-threaded numpy,
                                    # but DataLoader-style overhead helps
#SBATCH --mem=32G
#SBATCH --time=08:00:00             # tumor: ~1-2h actual, give plenty of headroom
#SBATCH --output=logs/precut_tumor_%j.out
#SBATCH --error=logs/precut_tumor_%j.err
# NOTE: no --partition / no --gres → uses Stanage default CPU partition.
# If Slurm complains, add e.g. `#SBATCH --partition=sheffield` (verify with sinfo).

set -eo pipefail

module purge
module load Anaconda3/2022.05       # EDIT if your conda module differs
source activate pyansys

cd "$SLURM_SUBMIT_DIR"

echo "===== precut tumor on $(hostname) ====="
python --version
date

# Idempotent: precut_patches.py skips cases whose last patch file already
# exists, so re-submitting after a crash resumes from where it stopped.
python dataUtils/precut_patches.py \
  --data_dir       /mnt/parscratch/users/cov23yc/SSM-Data/ElementGraph_Tumor_Train \
  --train_out_dir  /mnt/parscratch/users/cov23yc/SSM-Data/Patches_Tumor_Train_256_128 \
  --val_out_dir    /mnt/parscratch/users/cov23yc/SSM-Data/_unused_val_tumor \
  --val_cases 0 --grid_res 256 --patch_size 128 \
  --train_overlap 0.5 --s3_splat min

echo "===== count check ====="
find /mnt/parscratch/users/cov23yc/SSM-Data/Patches_Tumor_Train_256_128 -name '*.pt' | wc -l
# expected: 18900

echo "===== done $(date) ====="
