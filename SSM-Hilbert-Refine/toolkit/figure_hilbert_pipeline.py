#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-panel figure illustrating the Hilbert scan pipeline:

    (a) 3D vertebra (translucent surface) with the Hilbert curve
        threading the entire grid. Curve segments inside Ω_bone are
        red; segments in air are blue.  A red box highlights the local
        sub-cube zoomed in (b).
    (b) An 8×8×8 voxel sub-cube. Bone voxels are rendered as colored
        cubes (PE value, paper diverging palette, semi-transparent);
        air voxels stay translucent. The 3D Hilbert curve again is
        red on bone segments, blue on air segments.
    (c) The 1D scan sequence after sorting by Ψ: bone tokens
        (PE-coloured) first, air tokens (light) at the tail. An
        orange arrow points to the Bi-Mamba block.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from skimage import measure
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataUtils import HybridDataset
from Net.HybridModel_SSM import _make_3d_pos_enc


PAPER_DIV = LinearSegmentedColormap.from_list(
    'paper_div',
    ['#5B7FB0', '#BDD0E0', '#E5D5B5', '#E58368', '#A52424'], N=256)
AIR_RGBA       = (0.94, 0.96, 0.98, 0.10)
BONE_RED       = to_rgba('#A52424')
AIR_BLUE       = to_rgba('#5B7FB0')
WINDOW_COLOR   = '#C0392B'
SHELL_COLOR    = 'lightcoral'
HILBERT_COLOR  = '#E67E22'


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


def draw_box(ax, z0, y0, x0, P, color, lw=2.0):
    pts = np.array([(z, y, x)
                     for z in (z0, z0 + P)
                     for y in (y0, y0 + P)
                     for x in (x0, x0 + P)], dtype=float)
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for a, b in edges:
        ax.plot([pts[a, 2], pts[b, 2]],
                 [pts[a, 1], pts[b, 1]],
                 [pts[a, 0], pts[b, 0]],
                 color=color, lw=lw)


def hilbert_curve_segments(H, mask, alpha_bone=0.85, alpha_air=0.20):
    """Return (segments, colors) for the full-grid Hilbert curve.
    Each segment is colored red (bone) or blue (air) based on its
    starting voxel's mask value."""
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

    segs = np.stack([starts, ends], axis=1)[:, :, [2, 1, 0]]   # to xyz
    return segs, colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../Data/ElementGraph_Tumor_Train")
    ap.add_argument("--case", default="MV04_L2_VOL_10_MV04_L2_VOL_10_sp10")
    ap.add_argument("--R3", type=int, default=32,
                    help="Voxel grid for 3D panel.")
    ap.add_argument("--P", type=int, default=8,
                    help="Window size (must be power of 2).")
    ap.add_argument("--out", default="../Results/figure_hilbert_pipeline.png")
    args = ap.parse_args()

    R, P = args.R3, args.P
    if (P & (P - 1)) != 0:
        raise SystemExit(f"P must be power of 2, got {P}")

    print(f"[INFO] Voxelising {args.case} at R={R}...")
    ds = HybridDataset(args.data_dir, grid_res=R)
    idx = ds.names.index(args.case)
    sample = ds[idx]
    mask3d = sample['mask'].squeeze().numpy() > 0.5
    print(f"[INFO] Bone voxels: {int(mask3d.sum())} / {R**3}")

    H3d = np.load(Path(__file__).resolve().parents[1] /
                   'dataUtils' / f'hilbert_R{R}.npy')

    z0, y0, x0, frac = find_3d_window(mask3d, P)
    print(f"[INFO] 3D window @ ({z0},{y0},{x0})  bone fraction {frac:.2f}")
    win_mask = mask3d[z0:z0+P, y0:y0+P, x0:x0+P]

    print(f"[INFO] Computing 60-channel PE...")
    pe60 = _make_3d_pos_enc(R, R, R, 60, 'cpu',
                             torch.float32).squeeze(0).numpy()
    pe_sum = pe60.sum(axis=0)
    pe_lim = float(np.percentile(np.abs(pe_sum[mask3d]), 98))
    win_pe = pe_sum[z0:z0+P, y0:y0+P, x0:x0+P]

    H_win = np.load(Path(__file__).resolve().parents[1] /
                     'dataUtils' / f'hilbert_R{P}.npy')

    # =================================================================
    # Build figure
    # =================================================================
    fig = plt.figure(figsize=(15, 5.6))
    gs  = GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.6], wspace=0.30)

    # ---------------- Panel A ----------------
    axA = fig.add_subplot(gs[0], projection='3d')

    # Bone-mask outline as translucent surface (alpha increased so the
    # vertebra is clearly recognisable as a bone shape).
    verts, faces, _, _ = measure.marching_cubes(
        mask3d.astype(np.float32), level=0.5, spacing=(1.0, 1.0, 1.0))
    verts_xyz = verts[:, [2, 1, 0]]
    mesh = Poly3DCollection(verts_xyz[faces],
                             alpha=0.35, facecolor=SHELL_COLOR,
                             edgecolor=(0.5, 0.5, 0.5, 0.15),
                             linewidth=0.2)
    axA.add_collection3d(mesh)

    # Full-grid Hilbert curve. Heavy subsample so the curve reads as a
    # clean trace rather than a fog of segments. Bone segments stay
    # red, air segments are very faint blue (mostly so the eye can tell
    # the curve does pass through air without it dominating).
    segs, seg_colors = hilbert_curve_segments(H3d, mask3d,
                                                alpha_bone=0.85,
                                                alpha_air=0.18)
    target_n = 500
    step = max(1, len(segs) // target_n)
    axA.add_collection3d(Line3DCollection(
        segs[::step].tolist(), colors=seg_colors[::step],
        linewidths=1.2))

    draw_box(axA, z0, y0, x0, P, WINDOW_COLOR, lw=2.5)

    axA.set_xlim(0, R); axA.set_ylim(0, R); axA.set_zlim(0, R)
    axA.set_xticks([]); axA.set_yticks([]); axA.set_zticks([])
    axA.view_init(elev=18, azim=-55)
    axA.set_title('(a) Hilbert curve through vertebra\n'
                   '(red: bone; blue: air)', fontsize=10)

    # ---------------- Panel B ----------------
    axB = fig.add_subplot(gs[1], projection='3d')

    # Voxel cubes — keep both bone and air, but air is very translucent
    facecolors = np.zeros((P, P, P, 4))
    for i in range(P):
        for j in range(P):
            for k in range(P):
                if win_mask[i, j, k]:
                    v = np.clip(win_pe[i, j, k] / pe_lim, -1, 1)
                    rgba = list(PAPER_DIV((v + 1) / 2))
                    rgba[3] = 0.55                    # bone slightly translucent
                    facecolors[i, j, k] = rgba
                else:
                    facecolors[i, j, k] = AIR_RGBA     # air ≈ invisible

    filled = np.ones((P, P, P), dtype=bool)
    axB.voxels(filled.transpose(2, 1, 0),
                facecolors=facecolors.transpose(2, 1, 0, 3),
                edgecolor=(0.7, 0.7, 0.7, 0.25),
                linewidth=0.2)

    # Hilbert curve in window with red/blue coloring
    win_segs, win_seg_colors = hilbert_curve_segments(
        H_win, win_mask, alpha_bone=0.95, alpha_air=0.85)
    win_lc = Line3DCollection(win_segs.tolist(),
                                colors=win_seg_colors,
                                linewidths=2.0)
    axB.add_collection3d(win_lc)

    # Mark start
    win_coords = np.stack(np.unravel_index(np.arange(P**3), (P, P, P)),
                           axis=1)
    sort_w = np.argsort(H_win.ravel())
    start_pt = win_coords[sort_w[0]].astype(float) + 0.5
    axB.scatter(start_pt[2], start_pt[1], start_pt[0],
                 color='black', s=40, zorder=10)

    axB.set_xlim(0, P); axB.set_ylim(0, P); axB.set_zlim(0, P)
    axB.set_xticks([]); axB.set_yticks([]); axB.set_zticks([])
    axB.view_init(elev=18, azim=-55)
    axB.set_title('(b) Local cube: Hilbert traverses bone + air\n'
                   '(red: bone; blue: air)', fontsize=10)

    # ---------------- Panel C ----------------
    axC = fig.add_subplot(gs[2])

    n_total = P ** 3
    win_pe_flat = win_pe.ravel()
    win_mask_flat = win_mask.ravel()
    H_win_flat = H_win.ravel()
    BIG = n_total + 10
    scan_key = np.where(win_mask_flat, H_win_flat, BIG)
    perm = np.argsort(scan_key)
    sorted_pe   = win_pe_flat[perm]
    sorted_mask = win_mask_flat[perm]
    n_bone = int(win_mask_flat.sum())

    strip = np.empty((1, n_total, 3))
    for i in range(n_total):
        if sorted_mask[i]:
            v = np.clip(sorted_pe[i] / pe_lim, -1, 1)
            strip[0, i] = PAPER_DIV((v + 1) / 2)[:3]
        else:
            strip[0, i] = AIR_RGBA[:3]
    axC.imshow(strip, aspect='auto',
                extent=[0, n_total, 0, 1.0], interpolation='nearest')
    axC.axvline(n_bone, color='black', lw=2.5, ymin=0, ymax=1)

    axC.text(n_bone / 2, 1.18,
              f'bone tokens (n = {n_bone})\nHilbert order',
              ha='center', fontsize=10)
    axC.text((n_bone + n_total) / 2, 1.18,
              f'air tokens (n = {n_total - n_bone})',
              ha='center', fontsize=10)

    arrow_y = 0.5
    arrow_off = max(15, n_total * 0.04)
    axC.annotate('', xy=(n_total + arrow_off, arrow_y),
                  xytext=(n_total + 1, arrow_y),
                  arrowprops=dict(arrowstyle='->',
                                   color=HILBERT_COLOR, lw=2.0))
    axC.text(n_total + arrow_off + 3, arrow_y, 'Bi-\nMamba',
              fontsize=10, va='center', ha='left',
              bbox=dict(boxstyle='round', facecolor=HILBERT_COLOR,
                        edgecolor='none', pad=0.45),
              color='white')

    axC.set_xlim(-0.5, n_total + arrow_off + 35)
    axC.set_ylim(0, 1.5)
    axC.set_yticks([])
    axC.set_xticks([0, n_bone, n_total])
    axC.set_xticklabels(['0', f'{n_bone}\n(bone end)', f'{n_total}'],
                        fontsize=9)
    axC.set_title('(c) 1D scan sequence: sort by Ψ(x)', fontsize=10)
    for s in ('top', 'right', 'left'):
        axC.spines[s].set_visible(False)

    fig.suptitle(f'{args.case}', fontsize=10, y=1.00)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved -> {out}")


if __name__ == "__main__":
    main()
