#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_dense_path.py — Visualise the dense physics channels:
    dist_BC, dist_Load, u_load

These are computed analytically (Euclidean distance transform from BC / Load
voxels, plus a constant global load magnitude field). They form the
"long-range" half of the dual sparse/dense physics encoder.

Run:
    python plot_dense_path.py
    python plot_dense_path.py --npz <path>.npz --grid 64
"""

import argparse
import numpy as np
from pathlib import Path

import pyvista as pv
from scipy.ndimage import distance_transform_edt


# ============================================================
# Voxelise helpers (same convention as DataLoader)
# ============================================================
def voxelise(centroids, is_bc, is_load, D=64, pad=2.0):
    bbox_min = centroids.min(axis=0) - pad
    bbox_max = centroids.max(axis=0) + pad
    extent = bbox_max - bbox_min

    pts_norm = (centroids - bbox_min) / extent
    pts_grid = (pts_norm * (D - 1)).round().astype(np.int64).clip(0, D - 1)

    bone = np.zeros((D, D, D), dtype=bool)
    BC   = np.zeros((D, D, D), dtype=np.float32)
    LOAD = np.zeros((D, D, D), dtype=np.float32)

    bone[pts_grid[:, 0], pts_grid[:, 1], pts_grid[:, 2]] = True

    bc_idx = np.where(is_bc > 0.5)[0]
    ld_idx = np.where(is_load > 0.5)[0]
    for i in bc_idx:
        x, y, z = pts_grid[i]
        BC[x, y, z] = 1.0
    for i in ld_idx:
        x, y, z = pts_grid[i]
        LOAD[x, y, z] = 1.0

    return bone, BC, LOAD, bbox_min, bbox_max


def make_dense_fields(bone_mask, BC, LOAD, u_disp):
    """Same as DataLoader.HybridDataset:"""
    if BC.max() > 0:
        dist_BC = distance_transform_edt(BC < 0.5).astype(np.float32)
        dist_BC /= (dist_BC.max() + 1e-8)
    else:
        dist_BC = np.ones_like(BC)

    if LOAD.max() > 0:
        dist_Load = distance_transform_edt(LOAD < 0.5).astype(np.float32)
        dist_Load /= (dist_Load.max() + 1e-8)
    else:
        dist_Load = np.ones_like(LOAD)

    u_sqrt = float(np.sign(u_disp) * np.sqrt(abs(u_disp)))
    u_load_field = np.full_like(BC, u_sqrt)

    # Mask: only inside bone
    dist_BC = dist_BC * bone_mask.astype(np.float32)
    dist_Load = dist_Load * bone_mask.astype(np.float32)
    u_load_field = u_load_field * bone_mask.astype(np.float32)

    return dist_BC, dist_Load, u_load_field, u_sqrt


# ============================================================
# Plot
# ============================================================
def make_uniform_grid(field, bbox_min, bbox_max, D, name):
    extent = bbox_max - bbox_min
    spacing = (extent / (D - 1)).tolist()
    grid = pv.ImageData(
        dimensions=(D + 1, D + 1, D + 1),
        origin=bbox_min.tolist(),
        spacing=spacing,
    )
    grid.cell_data[name] = field.flatten(order='F')
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz",
        default="../Data/ElementGraph_Tumor_Train/MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp10.npz")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--screenshot", default="dense_path.png")
    args = ap.parse_args()

    npz = Path(args.npz)
    print(f"[INFO] Loading {npz.name}")
    data = np.load(npz)
    centroids = data['centroids'].astype(np.float32)
    is_bc = data['is_bc'].astype(np.float32)
    is_load = data['is_load'].astype(np.float32)
    u_disp = float(data['u_disp'])
    print(f"[INFO] {len(centroids)} elems  "
          f"BC={int(is_bc.sum())}  Load={int(is_load.sum())}  u_disp={u_disp:.4f}")

    print(f"[INFO] Voxelising to {args.grid}^3 ...")
    bone, BC, LOAD, bbox_min, bbox_max = voxelise(
        centroids, is_bc, is_load, D=args.grid)
    print(f"[INFO] Bone voxels: {bone.sum()} / {args.grid**3}")
    print(f"[INFO] BC sparse voxels:   {(BC > 0).sum()}")
    print(f"[INFO] Load sparse voxels: {(LOAD > 0).sum()}")

    print("[INFO] Computing distance transforms ...")
    dist_BC, dist_Load, u_load, u_sqrt = make_dense_fields(
        bone, BC, LOAD, u_disp)
    print(f"[INFO] dist_BC   range: [{dist_BC[bone].min():.3f}, {dist_BC[bone].max():.3f}]")
    print(f"[INFO] dist_Load range: [{dist_Load[bone].min():.3f}, {dist_Load[bone].max():.3f}]")
    print(f"[INFO] u_load constant : {u_sqrt:+.4f}")

    # ---- 3-panel plot ----
    pl = pv.Plotter(shape=(1, 3),
                    window_size=(420 * 3, 750),
                    off_screen=True)

    bone_cloud = pv.PolyData(centroids)
    panels = [
        ('dist_BC',   dist_BC,   'viridis',
         'dist_BC\n(distance to nearest BC)'),
        ('dist_Load', dist_Load, 'plasma',
         'dist_Load\n(distance to nearest Load)'),
        ('u_load',    u_load,    'coolwarm',
         f'u_load\n(constant = {u_sqrt:+.4f})'),
    ]

    for col, (name, field, cmap, title) in enumerate(panels):
        pl.subplot(0, col)
        pl.set_background('white')

        # Faint bone reference
        pl.add_mesh(bone_cloud, color='lightgrey',
                    point_size=2.0, opacity=0.07,
                    render_points_as_spheres=True)

        # Dense field as voxel cubes (only bone interior)
        masked = field.copy()
        masked[~bone] = np.nan
        grid = make_uniform_grid(masked, bbox_min, bbox_max, args.grid, name)
        bone_cells = grid.threshold(value=-1e9, scalars=name, invert=False)

        clim = (float(np.nanmin(masked)), float(np.nanmax(masked)))
        if clim[0] == clim[1]:
            clim = (clim[0] - 1e-3, clim[1] + 1e-3)

        pl.add_mesh(bone_cells,
                    scalars=name,
                    cmap=cmap,
                    clim=clim,
                    opacity=1.0,
                    show_edges=False,
                    show_scalar_bar=True,
                    scalar_bar_args={
                        'title': name,
                        'n_labels': 4,
                        'fmt': '%.2f',
                        'color': 'black',
                        'vertical': True,
                        'position_x': 0.86,
                        'position_y': 0.18,
                        'width':  0.04,
                        'height': 0.55,
                    })

        pl.add_text(title, position='upper_left',
                    font_size=11, color='black')
        pl.camera_position = 'iso'
        pl.reset_camera()
        pl.camera.zoom(0.85)

    pl.link_views()
    pl.show(screenshot=args.screenshot, auto_close=True)
    print(f"[INFO] Saved: {args.screenshot}")


if __name__ == '__main__':
    main()
