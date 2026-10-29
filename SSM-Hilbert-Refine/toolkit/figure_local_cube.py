#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render the second pipeline panel: an 8×8×8 local sub-cube with
PE-coloured bone voxels, translucent air, and the 3D Hilbert curve
threading through (red on bone, blue on air).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataUtils import HybridDataset
from Net.HybridModel_SSM import _make_3d_pos_enc


PAPER_DIV = LinearSegmentedColormap.from_list(
    'paper_div',
    ['#5B7FB0', '#BDD0E0', '#E5D5B5', '#E58368', '#A52424'], N=256)
AIR_RGBA = (0.94, 0.96, 0.98, 0.10)
BONE_RED = to_rgba('#A52424')
AIR_BLUE = to_rgba('#5B7FB0')


def find_3d_window(mask3d, P, target_frac=0.55):
    R = mask3d.shape[0]
    best = None
    stride = max(1, P // 2)
    for z0 in range(0, R - P + 1, stride):
        for y0 in range(0, R - P + 1, stride):
            for x0 in range(0, R - P + 1, stride):
                frac = mask3d[z0:z0+P, y0:y0+P, x0:x0+P].mean()
                if 0.30 <= frac <= 0.85:
                    d = abs(frac - target_frac)
                    if best is None or d < best[3]:
                        best = (z0, y0, x0, d, frac)
    if best is None:
        bp = np.argwhere(mask3d)
        c = bp.mean(axis=0).astype(int)
        z0 = max(0, min(R - P, c[0] - P // 2))
        y0 = max(0, min(R - P, c[1] - P // 2))
        x0 = max(0, min(R - P, c[2] - P // 2))
        return z0, y0, x0, mask3d[z0:z0+P, y0:y0+P, x0:x0+P].mean()
    return best[0], best[1], best[2], best[4]


def hilbert_curve_segments(H, mask, alpha_bone=0.95, alpha_air=0.85):
    R = H.shape[0]
    all_pts = np.stack(np.unravel_index(np.arange(R**3), (R, R, R)),
                        axis=1).astype(np.float32)
    sort_order = np.argsort(H.ravel())
    pts_sorted = all_pts[sort_order]
    starts = pts_sorted[:-1]
    ends   = pts_sorted[1:]
    si = starts.astype(int)
    in_bone = mask[si[:, 0], si[:, 1], si[:, 2]]
    colors = np.empty((len(starts), 4))
    colors[in_bone]  = (*BONE_RED[:3], alpha_bone)
    colors[~in_bone] = (*AIR_BLUE[:3], alpha_air)
    segs = np.stack([starts, ends], axis=1)[:, :, [2, 1, 0]]    # to xyz
    return segs, colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../Data/ElementGraph_Tumor_Train")
    ap.add_argument("--case", default="MV04_L2_VOL_10_MV04_L2_VOL_10_sp17")
    ap.add_argument("--R", type=int, default=128,
                    help="Vertebra grid resolution (only used to "
                         "locate the window and compute PE).")
    ap.add_argument("--P", type=int, default=8,
                    help="Window edge length (must be 2^p).")
    ap.add_argument("--azim", type=float, default=-55)
    ap.add_argument("--elev", type=float, default=18)
    ap.add_argument("--out", default="../Results/figure_local_cube.png")
    args = ap.parse_args()

    R, P = args.R, args.P
    if (P & (P - 1)) != 0:
        raise SystemExit(f"P must be power of 2, got {P}")

    print(f"[INFO] Voxelising {args.case} at R={R}...")
    ds = HybridDataset(args.data_dir, grid_res=R)
    idx = ds.names.index(args.case)
    sample = ds[idx]
    mask3d = sample['mask'].squeeze().numpy() > 0.5
    print(f"[INFO] Bone voxels: {int(mask3d.sum())}")

    z0, y0, x0, frac = find_3d_window(mask3d, P)
    print(f"[INFO] Window @ ({z0},{y0},{x0}), bone fraction {frac:.2f}")
    win_mask = mask3d[z0:z0+P, y0:y0+P, x0:x0+P]

    print(f"[INFO] Computing 60-channel PE...")
    pe60 = _make_3d_pos_enc(R, R, R, 60, 'cpu',
                             torch.float32).squeeze(0).numpy()
    pe_sum = pe60.sum(axis=0)
    pe_lim = float(np.percentile(np.abs(pe_sum[mask3d]), 98))
    win_pe = pe_sum[z0:z0+P, y0:y0+P, x0:x0+P]

    H_win = np.load(Path(__file__).resolve().parents[1] /
                     'dataUtils' / f'hilbert_R{P}.npy')

    # ---- Render ----
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Voxel cubes
    facecolors = np.zeros((P, P, P, 4))
    for i in range(P):
        for j in range(P):
            for k in range(P):
                if win_mask[i, j, k]:
                    v = np.clip(win_pe[i, j, k] / pe_lim, -1, 1)
                    rgba = list(PAPER_DIV((v + 1) / 2))
                    rgba[3] = 0.55
                    facecolors[i, j, k] = rgba
                else:
                    facecolors[i, j, k] = AIR_RGBA

    filled = np.ones((P, P, P), dtype=bool)
    ax.voxels(filled.transpose(2, 1, 0),
               facecolors=facecolors.transpose(2, 1, 0, 3),
               edgecolor=(0.7, 0.7, 0.7, 0.25),
               linewidth=0.2)

    # Hilbert curve red/blue
    win_segs, win_seg_colors = hilbert_curve_segments(
        H_win, win_mask, alpha_bone=0.95, alpha_air=0.85)
    ax.add_collection3d(Line3DCollection(
        win_segs.tolist(), colors=win_seg_colors, linewidths=2.0))

    # Mark start
    win_coords = np.stack(np.unravel_index(np.arange(P**3), (P, P, P)),
                           axis=1)
    sort_w = np.argsort(H_win.ravel())
    start_pt = win_coords[sort_w[0]].astype(float) + 0.5
    ax.scatter(start_pt[2], start_pt[1], start_pt[0],
                color='black', s=50, zorder=10)

    ax.set_xlim(0, P); ax.set_ylim(0, P); ax.set_zlim(0, P)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=args.elev, azim=args.azim)
    n_bone = int(win_mask.sum())
    ax.set_title(
        f'{args.case}\nLocal {P}×{P}×{P} cube  '
        f'(bone={n_bone}, air={P**3 - n_bone})',
        fontsize=10)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved -> {out}")


if __name__ == "__main__":
    main()
