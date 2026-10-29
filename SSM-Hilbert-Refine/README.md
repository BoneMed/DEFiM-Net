# SSM-Hilbert-Refine

Sparse-Mamba hybrid network for voxel-level strain prediction in vertebrae.
The model is pretrained on healthy vertebrae and then fine-tuned on tumor
(metastasis) cases. Run commands from inside this directory so the relative
paths (`../Model`, `../Results`, `../Data`) resolve correctly.

## Structure

- `healthyPreTrain.py`: pretrain on healthy patches (stage 1).
- `finetune_tumor.py`: fine-tune the healthy checkpoint on tumor patches (stage 2).
- `inference.py`: patch-based full-volume inference and optional 3D rendering.
- `Net/`: network code (`DualEncoderUNet_SSM`).
- `dataUtils/`: npz extraction, train/val/test splits, patch pre-cutting,
  Hilbert scan table (`hilbert_R128.npy`) and dataset loaders.
- `toolkit/`: training loss (`Loss.py`), shared plotting helpers (`viz.py`)
  and figure / visualisation scripts.
- `*.sbatch`: Slurm jobs for patch pre-cutting and the two training stages.

## Network

The main model is `Net.DualEncoderUNet_SSM`.

```text
Physics branch:
    [BC, Load*u] + [dist_BC, dist_Load]
        -> SparsePhysicsEncoder -> MSGCDown x3

Geometry branch (E field):
    level0: ResBlock3D
    level1: ResDown3D
    level2: MambaDown3D
    level3: MambaDown3D

Fusion:
    physics p0..p3 -> FiLM -> modulate geometry g0..g3

Decoder:
    level2: MambaUp3D
    level1: MambaUp3D
    level0: conv Up -> 1x1 head -> strain prediction
```

Mamba blocks scan bone voxels in 3D Hilbert-curve order (`scan_key`, from
`dataUtils/hilbert_R{P}.npy`); air voxels are pushed to the tail of the
sequence by the bone mask. Mamba is only used at the low-resolution levels
(`32^3` and `16^3` for a `128^3` patch).

## Required Packages

```text
torch
mamba-ssm
causal-conv1d
torch-scatter
```

`pyvista` is only needed for the 3D rendering / visualisation scripts.

## Typical Commands

```bash
# 1. Healthy pretraining
python healthyPreTrain.py \
  --save_dir   ../Model/Healthy_Hilbert_MAE_Whole \
  --patch_dir  <data>/Patches_Healthy_Train_256_128 \
  --split_json <data>/healthy_split.json

# 2. Tumor fine-tuning
python finetune_tumor.py \
  --pretrained ../Model/Healthy_Hilbert_MAE_Whole/best_valmae.pth \
  --save_dir   ../Model/SSMHybrid_Tumor_Refine_Whole \
  --train_dir  <data>/Patches_Tumor_Train_256_128 \
  --patch_dir  <data>/Patches_Tumor_Train_256_128 \
  --split_json <data>/tumor_split.json

# 3. Inference on the tumor test set
python inference.py \
  --ckpt     ../Model/SSMHybrid_Tumor_Refine_Whole/best_val_mae.pth \
  --save_dir ../Results/SSMHybrid_inference \
  --test_dir ../Data/ElementGraph_Tumor_Test
```

To resume an interrupted fine-tuning run, pass
`--resume ../Model/SSMHybrid_Tumor_Refine_Whole/last.pth`.

## Main Architecture Options

- `--sparse_hidden`: hidden width in `SparsePhysicsEncoder`, default `16`.
- `--sparse_layers`: sparse diffusion depth, default `5`.
- `--mamba_d_state`: Mamba SSM state size, default `16`.
- `--mamba_d_conv`: Mamba local 1D conv width, default `4`.
- `--mamba_expand`: Mamba expansion factor, default `2`.

Fine-tuning inherits these values from the pretrained checkpoint, and
inference reads them from the checkpoint `args` when available.
