#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_voxel_vertebra.py — Render a voxelised vertebra as a solid mesh.

Just the bone voxels, no cut, no wireframe overlay, no positional
encoding colouring. Use to show the input voxel grid for paper /
slides.

Run with any env that has pyvista installed (BoneMed / pyansys).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataUtils import HybridDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../Data/ElementGraph_Tumor_Train")
    ap.add_argument("--case", default=None,
                    help="Case stem (no .npz). Default: first case.")
    ap.add_argument("--grid_res", type=int, default=256)
    ap.add_argument("--bone_color", default="ivory",
                    help="Bone fill colour (any matplotlib colour name).")
    ap.add_argument("--scalar", default="none",
                    choices=["none", "E", "z", "y", "x"],
                    help="Optional per-voxel scalar to colour by. "
                         "E=elastic modulus (log scale), "
                         "z/y/x=coordinate gradient.")
    ap.add_argument("--screenshot",
                    default="../Results/voxel_vertebra.png")
    ap.add_argument("--off_screen", action="store_true")
    args = ap.parse_args()

    R = args.grid_res
    print(f"[INFO] Voxelising at R={R}...")
    ds = HybridDataset(args.data_dir, grid_res=R)
    idx = 0 if args.case is None else ds.names.index(args.case)
    case_name = ds.names[idx]
    sample = ds[idx]
    mask = sample['mask'].squeeze().numpy() > 0.5
    print(f"[INFO] Case: {case_name}, bone voxels: {int(mask.sum())}")

    grid = pv.ImageData(dimensions=(R + 1, R + 1, R + 1),
                         spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0))
    grid.cell_data['mask'] = mask.astype(np.float32).ravel(order='F')

    # Optional scalar field
    if args.scalar == "E":
        E = sample['geo_input'].squeeze().numpy()           # (R, R, R)
        grid.cell_data['scalar'] = np.log1p(E).ravel(order='F')
        cmap = 'viridis'
        scalar_title = 'log(1 + E)'
    elif args.scalar in ("z", "y", "x"):
        z = np.linspace(0, 1, R, dtype=np.float32)
        zz, yy, xx = np.meshgrid(z, z, z, indexing='ij')
        coord = {'z': zz, 'y': yy, 'x': xx}[args.scalar]
        grid.cell_data['scalar'] = coord.ravel(order='F')
        cmap = 'turbo'
        scalar_title = f'{args.scalar} coordinate'
    else:
        scalar_title = None

    solid = grid.threshold(0.5, scalars='mask')

    pl = pv.Plotter(off_screen=args.off_screen, window_size=(1100, 900))
    pl.set_background('white')

    if args.scalar == "none":
        pl.add_mesh(solid, color=args.bone_color, show_edges=False,
                    smooth_shading=False)
    else:
        pl.add_mesh(solid, scalars='scalar', cmap=cmap,
                    show_edges=False, show_scalar_bar=True,
                    scalar_bar_args={'title': scalar_title, 'fmt': '%.2f'})

    pl.add_axes()
    pl.add_text(f'{case_name}\nR = {R}, {int(mask.sum())} bone voxels',
                position='upper_left', font_size=10, color='black')

    out = Path(args.screenshot)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _save_screenshot():
        pl.screenshot(str(out))
        print(f"[INFO] Screenshot -> {out}")
    pl.add_key_event('s', _save_screenshot)
    pl.add_text('Press [s] to save current view as PNG',
                position='lower_left', font_size=8, color='gray')

    if args.off_screen:
        pl.screenshot(str(out))
        print(f"[INFO] Off-screen screenshot -> {out}")
    else:
        pl.show()


if __name__ == "__main__":
    main()
