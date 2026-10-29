#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualise the effect of `_inmask_nearest_fill` on the EPTO3 target grid.

For a tumor case, voxelise once with the standard pipeline (trilinear /
min splat → nearest fill → mask) and once WITHOUT the nearest-fill step,
then draw 3 sagittal/coronal/axial slices side by side so you can see:

  raw  : holes inside bone where no element centroid landed in the voxel
  fill : holes filled by nearest-neighbour value from a hit voxel
  diff : `fill - raw` — shows exactly which voxels the fill touched

Tumor cases are particularly informative because the low-E tumor blob
has very few elements per voxel (sparse mesh in the destroyed region),
so the nearest-fill has a big visible effect there.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import distance_transform_edt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataUtils.DataLoader import HybridDataset, _inmask_nearest_fill, signed_sqrt


def inv_signed_sqrt(y, scale=10.0):
    return np.sign(y) * (y ** 2) / scale


def ansys_cmap():
    return cm.colors.LinearSegmentedColormap.from_list("ansys", [
        (0.0, "#0000ff"), (0.2, "#00ffff"), (0.4, "#00ff00"),
        (0.6, "#ffff00"), (0.8, "#ff8000"), (1.0, "#ff0000")])


def voxelize_with_and_without_fill(ds, idx):
    """Returns (raw_grid, filled_grid, mask) in MPa-equivalent units
    (already inv_signed_sqrt)."""
    ds_with_fill = ds  # the default behaviour
    sample = ds_with_fill[idx]
    mask = sample["mask"].squeeze().numpy() > 0.5
    s3_filled = sample["S3_grid"].squeeze().numpy()  # already has the fill

    # Recompute the raw (no-fill) version by repeating the splat step
    # without nearest_fill. Loads the npz directly to avoid touching
    # HybridDataset internals.
    npz = np.load(ds.files[idx])
    centroids = npz["centroids"].astype(np.float32)
    S3_raw = npz["S3"].astype(np.float32)

    R = ds.grid_res
    pad = ds.pad
    mins = centroids.min(0)
    maxs = centroids.max(0)
    size = maxs - mins
    size[size == 0] = 1.0
    bbox_min = (mins - pad * size).astype(np.float32)
    bbox_max = (maxs + pad * size).astype(np.float32)
    scale = bbox_max - bbox_min
    scale[scale == 0] = 1.0
    q = np.clip((centroids - bbox_min) / scale, 0, 1 - 1e-6)
    pts_grid_np = (q * R).astype(np.float32)              # numpy path for _splat_min
    pts_grid_t  = torch.from_numpy(pts_grid_np).float()   # tensor path for _splat_trilinear

    S3_log = signed_sqrt(S3_raw * 10.0)
    if ds.s3_splat == "min":
        s3_raw_grid, _ = ds._splat_min(pts_grid_np, S3_log, R)
    else:
        s3_raw_grid, _ = ds._splat_trilinear(pts_grid_t, S3_log, R)

    s3_raw_grid = s3_raw_grid * mask.astype(np.float32)  # mask but no fill

    # Convert both to physical strain
    s3_raw_mpa    = inv_signed_sqrt(s3_raw_grid)
    s3_filled_mpa = inv_signed_sqrt(s3_filled)
    return s3_raw_mpa, s3_filled_mpa, mask


def fmt_view(arr, mask, axis, s):
    """Slice along axis at index s, mask out air to NaN, transpose for imshow."""
    if axis == 0:
        a = arr[s, :, :]; m = mask[s, :, :]
    elif axis == 1:
        a = arr[:, s, :]; m = mask[:, s, :]
    else:
        a = arr[:, :, s]; m = mask[:, :, s]
    return np.where(m, a, np.nan).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
        default="/mnt/training_Data/SSM-Data/ElementGraph_Tumor_Train",
        help="ElementGraph dir with .npz files (use tumor for clearest effect).")
    ap.add_argument("--idx", type=int, default=0,
                    help="Sample index inside the dataset (sorted by filename).")
    ap.add_argument("--target", type=str, default="",
                    help="Substring match on the sample name (overrides --idx).")
    ap.add_argument("--grid_res", type=int, default=256)
    ap.add_argument("--s3_splat", choices=["min", "mean"], default="min",
                    help="Match the splat used in precut (tumor data uses 'min').")
    ap.add_argument("--out",
        default="/mnt/PhD/pythonproject/PHD_PROGRAM/GCN_Stress/Results/nearest_fill_compare.png")
    args = ap.parse_args()

    ds = HybridDataset(args.data_dir, grid_res=args.grid_res, s3_splat=args.s3_splat)
    print(f"[INFO] {len(ds)} cases at {args.data_dir}")

    if args.target:
        match = [i for i, n in enumerate(ds.names) if args.target in n]
        if not match:
            raise SystemExit(f"No case name contains '{args.target}'")
        idx = match[0]
    else:
        idx = args.idx
    print(f"[INFO] picked idx={idx}  name={ds.names[idx]}")

    s3_raw, s3_filled, mask = voxelize_with_and_without_fill(ds, idx)
    diff = s3_filled - s3_raw                              # MPa
    n_bone   = int(mask.sum())
    n_hit    = int((np.abs(s3_raw) > 1e-12).sum())
    n_filled = int(((np.abs(s3_raw) < 1e-12) & mask).sum())
    print(f"  bone voxels     : {n_bone:,}")
    print(f"  splat-hit       : {n_hit:,}  ({100*n_hit/n_bone:.1f}%)")
    print(f"  filled by fill  : {n_filled:,}  ({100*n_filled/n_bone:.1f}%)")
    print(f"  EPTO3 range (filled bone): "
          f"[{s3_filled[mask].min():.4g}, {s3_filled[mask].max():.4g}]")

    D, H, W = s3_filled.shape
    views = [(D // 2, "Axial",   0),
             (H // 2, "Coronal", 1),
             (W // 2, "Sagittal", 2)]

    cmap = ansys_cmap()
    diverging = cm.get_cmap("RdBu_r")

    vmin = float(np.percentile(s3_filled[mask], 2))
    vmax = float(np.percentile(s3_filled[mask], 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    # Diff is roughly the same scale; use symmetric
    dmax = float(np.max(np.abs(diff[mask]))) + 1e-12

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    row_data = [
        ("Raw (no fill)", s3_raw, cmap, vmin, vmax),
        ("Filled (nearest)", s3_filled, cmap, vmin, vmax),
        ("Diff (filled - raw)", diff, diverging, -dmax, dmax),
    ]
    for ri, (label, arr, cmap_r, lo, hi) in enumerate(row_data):
        for ci, (s, view_label, ax) in enumerate(views):
            im = axes[ri, ci].imshow(
                fmt_view(arr, mask, ax, s),
                origin="lower", cmap=cmap_r, vmin=lo, vmax=hi)
            axes[ri, ci].set_title(f"{label} | {view_label}")
            axes[ri, ci].axis("off")
            plt.colorbar(im, ax=axes[ri, ci], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{ds.names[idx]}   bone={n_bone:,}   "
        f"hit={n_hit:,}({100*n_hit/n_bone:.1f}%)   "
        f"filled={n_filled:,}({100*n_filled/n_bone:.1f}%)   "
        f"s3_splat={args.s3_splat}",
        fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[INFO] saved -> {out}")


if __name__ == "__main__":
    main()
