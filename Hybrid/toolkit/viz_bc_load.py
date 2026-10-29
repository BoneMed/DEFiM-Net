#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_bc_load.py — Visualize BC / Load elements from an ElementGraph .npz file
using PyVista.

Run:
    python viz_bc_load.py
    python viz_bc_load.py --npz <path>.npz --screenshot out.png
"""

import argparse
import numpy as np
from pathlib import Path

import pyvista as pv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz",
        default="../Data/ElementGraph_Tumor_Train/MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp10.npz",
        help="Path to ElementGraph .npz file"
    )
    ap.add_argument("--screenshot", default="bc_load_viz.png",
                    help="Output PNG screenshot path (also opens window)")
    ap.add_argument("--off_screen", action="store_true",
                    help="Render without opening a window (only save screenshot)")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")

    print(f"[INFO] Loading {npz_path.name}")
    data = np.load(npz_path)
    print(f"[INFO] Available keys: {list(data.keys())}")

    centroids   = data['centroids'].astype(np.float32)   # (N, 3)
    is_bc       = data['is_bc'].astype(np.float32)        # (N,)
    is_load     = data['is_load'].astype(np.float32)      # (N,)
    E           = data['E'].astype(np.float32)            # (N,)

    n_total = len(centroids)
    n_bc = int((is_bc > 0.5).sum())
    n_load = int((is_load > 0.5).sum())

    print(f"[INFO] Total elements:  {n_total}")
    print(f"[INFO] BC elements:     {n_bc}")
    print(f"[INFO] Load elements:   {n_load}")
    print(f"[INFO] Centroid range:  "
          f"x[{centroids[:,0].min():.1f},{centroids[:,0].max():.1f}] "
          f"y[{centroids[:,1].min():.1f},{centroids[:,1].max():.1f}] "
          f"z[{centroids[:,2].min():.1f},{centroids[:,2].max():.1f}]")

    # ---- PyVista point clouds ----
    bone_cloud = pv.PolyData(centroids)
    bone_cloud['E_log'] = np.log1p(E)

    bc_cloud   = pv.PolyData(centroids[is_bc > 0.5])
    load_cloud = pv.PolyData(centroids[is_load > 0.5])

    # ---- Plotter ----
    pl = pv.Plotter(off_screen=args.off_screen, window_size=(1400, 900))
    pl.set_background('white')

    # Bone: light grey, transparent, colored by log(E)
    pl.add_mesh(
        bone_cloud,
        scalars='E_log',
        cmap='gray',
        point_size=2.0,
        opacity=0.18,
        render_points_as_spheres=True,
        show_scalar_bar=False,
        label=f'Bone elements ({n_total})',
    )

    # BC: blue cubes
    if n_bc > 0:
        pl.add_mesh(
            bc_cloud,
            color='#1f77ff',
            point_size=14,
            render_points_as_spheres=True,
            label=f'BC ({n_bc})',
        )

    # Load: red cubes
    if n_load > 0:
        pl.add_mesh(
            load_cloud,
            color='#ff2222',
            point_size=14,
            render_points_as_spheres=True,
            label=f'Load ({n_load})',
        )

    pl.add_legend(bcolor='white', face='circle', size=(0.18, 0.12))
    pl.add_text(npz_path.stem, position='upper_left',
                font_size=10, color='black')
    pl.add_axes(line_width=3)
    pl.camera_position = 'iso'

    if args.screenshot:
        pl.show(screenshot=args.screenshot, auto_close=args.off_screen)
        print(f"[INFO] Screenshot saved: {args.screenshot}")
    else:
        pl.show()


if __name__ == '__main__':
    main()
