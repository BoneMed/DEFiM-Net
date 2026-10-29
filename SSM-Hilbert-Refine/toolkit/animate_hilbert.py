#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animate the Hilbert scan + positional encoding for paper / slide use.

Three stacked panels, animated synchronously:

    Phase 1: 3D vertebra outline + Hilbert curve drawn through bone voxels
    Phase 2: + 1D ribbon below the 3D plot (each voxel shown in scan order)
    Phase 3: + positional-encoding heatmap (a few sin/cos channels)

Output: a single .gif file for embedding in slides / paper figures.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataUtils import HybridDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../Data/ElementGraph_Tumor_Train")
    ap.add_argument("--case", default=None,
                    help="Case stem (no .npz). Default: first case in data_dir.")
    ap.add_argument("--grid_res", type=int, default=32,
                    help="Voxel grid resolution. 32 keeps the curve readable; "
                         "64 gives more detail but is slower to render.")
    ap.add_argument("--n_frames", type=int, default=90,
                    help="Total frames; split equally across the 3 phases.")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default="../Results/hilbert_animation.gif")
    args = ap.parse_args()

    # -------- Load vertebra --------
    R = args.grid_res
    print(f"[INFO] Voxelising vertebra at R={R}...")
    ds = HybridDataset(args.data_dir, grid_res=R)
    if args.case is None:
        idx = 0
    else:
        idx = ds.names.index(args.case)
    case_name = ds.names[idx]
    sample = ds[idx]
    mask = sample['mask'].squeeze().numpy() > 0.5      # (R, R, R) bool
    print(f"[INFO] Case: {case_name}  bone voxels: {int(mask.sum())}")

    # -------- Load Hilbert table at this R --------
    hilbert_path = (Path(__file__).resolve().parents[1]
                    / 'dataUtils' / f'hilbert_R{R}.npy')
    if not hilbert_path.exists():
        raise SystemExit(
            f"hilbert_R{R}.npy not found. Generate with:\n"
            f"  python dataUtils/compute_hilbert_order.py --R {R}")
    H = np.load(hilbert_path)                          # (R, R, R) int32

    # Bone voxels sorted by Hilbert position (the actual scan order)
    bone_idx = np.argwhere(mask)                       # (N, 3)  (z, y, x)
    hilbert_pos = H[bone_idx[:, 0], bone_idx[:, 1], bone_idx[:, 2]]
    sort_order = np.argsort(hilbert_pos)
    bone_in_order = bone_idx[sort_order].astype(np.float32)
    N = len(bone_in_order)
    print(f"[INFO] Bone voxels in Hilbert order: N = {N}")

    # -------- Pre-compute positional encoding rows --------
    # Show 6 channels: 3 frequency octaves × (sin + cos) on the z-axis,
    # so the heat map literally shows what the model "sees" as PE.
    coords_norm = bone_in_order / max(R - 1, 1) * 2.0 - 1.0   # to [-1, 1]
    Ls = [2, 4, 6]
    pe_rows, pe_labels = [], []
    for L in Ls:
        freq = (2.0 ** L) * math.pi
        pe_rows.append(np.sin(coords_norm[:, 0] * freq))
        pe_labels.append(rf'sin($2^{L}\pi z$)')
        pe_rows.append(np.cos(coords_norm[:, 0] * freq))
        pe_labels.append(rf'cos($2^{L}\pi z$)')
    pe_rows = np.stack(pe_rows, axis=0)                # (6, N)

    # -------- Figure layout --------
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[6, 0.45, 2.6], hspace=0.45)
    ax3d = fig.add_subplot(gs[0], projection='3d')
    ax1d = fig.add_subplot(gs[1])
    axpe = fig.add_subplot(gs[2])

    cmap = cm.viridis

    # ---- 3D: light gray bone cloud (static) ----
    ax3d.scatter(bone_in_order[:, 2], bone_in_order[:, 1], bone_in_order[:, 0],
                 c='lightgray', s=6, alpha=0.07, edgecolors='none')
    ax3d.set_xlim(0, R - 1); ax3d.set_ylim(0, R - 1); ax3d.set_zlim(0, R - 1)
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
    ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
    ax3d.view_init(elev=15, azim=-60)
    ax3d.set_title(f'{case_name}    R = {R}    bone = {N} voxels',
                   fontsize=10)

    # 3D Hilbert curve as a Line3DCollection (alpha-driven reveal)
    segs = np.stack([bone_in_order[:-1], bone_in_order[1:]], axis=1)  # (N-1, 2, 3)
    seg_colors_full = cmap(np.linspace(0, 1, len(segs)))
    seg_colors = seg_colors_full.copy()
    seg_colors[:, 3] = 0.0                              # all invisible at start
    # reorder coords to (x, y, z) for matplotlib 3D
    segs_xyz = segs[:, :, [2, 1, 0]]
    line_coll = Line3DCollection(segs_xyz.tolist(),
                                  colors=seg_colors,
                                  linewidths=1.6)
    ax3d.add_collection3d(line_coll)

    # ---- 1D ribbon (initially blank) ----
    ribbon = np.ones((1, N, 4), dtype=np.float32)       # white = hidden
    ribbon_im = ax1d.imshow(ribbon, aspect='auto', interpolation='nearest')
    ax1d.set_yticks([])
    ax1d.set_xticks([0, N - 1])
    ax1d.set_xticklabels(['scan start', 'scan end'], fontsize=9)
    ax1d.set_title('1D scan sequence  (color = Hilbert position)', fontsize=10)

    # ---- Pos-enc heatmap (initially blank) ----
    pe_disp = np.full((6, N), np.nan, dtype=np.float32)
    pe_im = axpe.imshow(pe_disp, aspect='auto', cmap='RdBu_r',
                         vmin=-1, vmax=1, interpolation='nearest')
    axpe.set_yticks(range(6))
    axpe.set_yticklabels(pe_labels, fontsize=8)
    axpe.set_xticks([0, N - 1])
    axpe.set_xticklabels(['t = 0', f't = {N}'], fontsize=9)
    axpe.set_title('3D sin/cos positional encoding (sample channels)',
                    fontsize=10)

    # Phase headers
    phase_text = fig.text(0.5, 0.97, '', ha='center', fontsize=12,
                           weight='bold')

    third = max(1, args.n_frames // 3)

    def update(frame):
        # Phase 1 frames 0..third: build 3D curve only
        # Phase 2 frames third..2*third: build 1D ribbon (curve fully drawn)
        # Phase 3 frames 2*third..n_frames: build PE heatmap
        if frame < third:
            phase_text.set_text('1.  Hilbert curve fills the bone domain')
            n_curve = int(N * (frame + 1) / third)
            n_ribbon = 0
            n_pe = 0
        elif frame < 2 * third:
            phase_text.set_text('2.  Voxels lifted into a 1D scan order')
            n_curve = N
            n_ribbon = int(N * (frame - third + 1) / third)
            n_pe = 0
        else:
            phase_text.set_text('3.  Each voxel gets a 3D sin/cos positional code')
            n_curve = N
            n_ribbon = N
            n_pe = int(N * (frame - 2 * third + 1) / third)

        # 3D curve: alpha = 1 up to n_curve, 0 beyond
        new_alphas = np.zeros(len(segs))
        new_alphas[:max(0, n_curve - 1)] = 1.0
        seg_colors[:, 3] = new_alphas
        line_coll.set_color(seg_colors)

        # 1D ribbon
        if n_ribbon > 0:
            ribbon[0, :n_ribbon] = cmap(np.arange(n_ribbon) / max(N - 1, 1))
            ribbon[0, n_ribbon:] = [1.0, 1.0, 1.0, 1.0]
            ribbon_im.set_data(ribbon)

        # PE heatmap
        if n_pe > 0:
            pe_disp[:, :n_pe] = pe_rows[:, :n_pe]
            pe_disp[:, n_pe:] = np.nan
            pe_im.set_data(pe_disp)

        return [line_coll, ribbon_im, pe_im, phase_text]

    print(f"[INFO] Rendering {args.n_frames} frames at {args.fps} fps...")
    anim = FuncAnimation(fig, update, frames=args.n_frames,
                          interval=1000 // args.fps, blit=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=args.fps)
    anim.save(out_path, writer=writer)
    plt.close()
    print(f"[INFO] Saved -> {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
