#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render a single tumor vertebra bone mask as a 3D surface mesh,
alpha = 0.7 (70% opaque). Used as the first panel of the Hilbert
pipeline figure.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataUtils import HybridDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../Data/ElementGraph_Tumor_Train")
    ap.add_argument("--case", default="MV04_L2_VOL_10_MV04_L2_VOL_10_sp17")
    ap.add_argument("--R", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--color", default="lightcoral")
    ap.add_argument("--azim", type=float, default=-55)
    ap.add_argument("--elev", type=float, default=18)
    ap.add_argument("--out", default="../Results/figure_bone_mask.png")
    args = ap.parse_args()

    R = args.R
    print(f"[INFO] Voxelising {args.case} at R={R}...")
    ds = HybridDataset(args.data_dir, grid_res=R)
    idx = ds.names.index(args.case)
    sample = ds[idx]
    mask = sample['mask'].squeeze().numpy() > 0.5
    print(f"[INFO] Bone voxels: {int(mask.sum())}")

    print(f"[INFO] Marching cubes...")
    verts, faces, _, _ = measure.marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=(1.0, 1.0, 1.0))
    verts_xyz = verts[:, [2, 1, 0]]                  # to (x, y, z)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts_xyz[faces],
                             alpha=args.alpha,
                             facecolor=args.color,
                             edgecolor=(0.4, 0.4, 0.4, 0.10),
                             linewidth=0.15)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, R); ax.set_ylim(0, R); ax.set_zlim(0, R)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_title(f'{args.case}\nbone mask (alpha = {args.alpha})',
                  fontsize=10)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved -> {out}")


if __name__ == "__main__":
    main()
