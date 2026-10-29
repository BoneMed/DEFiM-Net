#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_bc_load_static.py — Static voxel-cube visualization of the two
dense distance fields used by the physics dense path:
    dist_BC, dist_Load

Two side-by-side panels:
    (a) dist_BC      (viridis)  - distance to nearest BC voxel
    (b) dist_Load    (plasma)   - distance to nearest Load voxel

Both fields are normalized to [0, 1] inside the bone, and rendered as
voxel cubes (only bone-interior cells), matching the DataLoader pipeline.

Run:
    python plot_bc_load_static.py
    python plot_bc_load_static.py --npz <path>.npz --grid 64
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


def make_dense_fields(bone_mask, BC, LOAD):
    """Same EDT computation as DataLoader.HybridDataset (after our cleanup)."""
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

    bm = bone_mask.astype(np.float32)
    return dist_BC * bm, dist_Load * bm


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
    ap.add_argument("--screenshot", default="dist_static.png")
    args = ap.parse_args()

    npz = Path(args.npz)
    print(f"[INFO] Loading {npz.name}")
    data = np.load(npz)
    centroids = data['centroids'].astype(np.float32)
    is_bc = data['is_bc'].astype(np.float32)
    is_load = data['is_load'].astype(np.float32)
    print(f"[INFO] {len(centroids)} elems  "
          f"BC={int(is_bc.sum())}  Load={int(is_load.sum())}")

    print(f"[INFO] Voxelising to {args.grid}^3 ...")
    bone, BC, LOAD, bbox_min, bbox_max = voxelise(
        centroids, is_bc, is_load, D=args.grid)
    print(f"[INFO] Bone voxels: {bone.sum()} / {args.grid**3}")

    print("[INFO] Computing distance transforms ...")
    dist_BC, dist_Load = make_dense_fields(bone, BC, LOAD)
    print(f"[INFO] dist_BC   range (in bone): "
          f"[{dist_BC[bone].min():.3f}, {dist_BC[bone].max():.3f}]")
    print(f"[INFO] dist_Load range (in bone): "
          f"[{dist_Load[bone].min():.3f}, {dist_Load[bone].max():.3f}]")

    bone_cloud = pv.PolyData(centroids)

    # Mask air voxels with NaN before building grids
    db = dist_BC.copy(); db[~bone] = np.nan
    dl = dist_Load.copy(); dl[~bone] = np.nan
    grid_bc = make_uniform_grid(db, bbox_min, bbox_max, args.grid, 'dist_BC')
    grid_ld = make_uniform_grid(dl, bbox_min, bbox_max, args.grid, 'dist_Load')

    # Threshold to keep only bone-interior cells
    cells_bc = grid_bc.threshold(value=-1e9, scalars='dist_BC')
    cells_ld = grid_ld.threshold(value=-1e9, scalars='dist_Load')

    clim_bc = (float(np.nanmin(db)), float(np.nanmax(db)))
    clim_ld = (float(np.nanmin(dl)), float(np.nanmax(dl)))

    # ---- 2-panel side-by-side ----
    pl = pv.Plotter(shape=(1, 2),
                    window_size=(1700, 900),
                    off_screen=True)

    panels = [
        (cells_bc, 'dist_BC', 'viridis', clim_bc,
         f"(a) dist_BC  - distance to nearest BC voxel\n"
         f"range [{clim_bc[0]:.2f}, {clim_bc[1]:.2f}]"),
        (cells_ld, 'dist_Load', 'plasma', clim_ld,
         f"(b) dist_Load - distance to nearest Load voxel\n"
         f"range [{clim_ld[0]:.2f}, {clim_ld[1]:.2f}]"),
    ]

    for col, (cells, name, cmap, clim, title) in enumerate(panels):
        pl.subplot(0, col)
        pl.set_background('white')

        pl.add_mesh(bone_cloud, color='lightgrey',
                    point_size=2.0, opacity=0.07,
                    render_points_as_spheres=True)

        pl.add_mesh(cells,
                    scalars=name,
                    cmap=cmap,
                    clim=clim,
                    opacity=1.0,
                    show_edges=False,
                    show_scalar_bar=True,
                    scalar_bar_args={
                        'title': name,
                        'n_labels': 5,
                        'fmt': '%.2f',
                        'color': 'black',
                        'vertical': True,
                        'position_x': 0.86,
                        'position_y': 0.18,
                        'width':  0.04,
                        'height': 0.55,
                    })

        pl.add_text(title, position='upper_left',
                    font_size=12, color='black')
        pl.camera_position = 'iso'
        pl.reset_camera()
        pl.camera.zoom(0.85)

    pl.link_views()
    pl.show(screenshot=args.screenshot, auto_close=True)
    print(f"[INFO] Saved: {args.screenshot}")


if __name__ == '__main__':
    main()
