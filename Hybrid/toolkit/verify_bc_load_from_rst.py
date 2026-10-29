#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_bc_load_from_rst.py — Dry-run: read BC/Load directly from RST
using nodal_boundary_conditions, and visualize them with PyVista.

Does NOT modify extract_rst_to_npz.py or any npz file.

Run:
    python verify_bc_load_from_rst.py
    python verify_bc_load_from_rst.py --rst <path>.rst
"""

import argparse
import numpy as np
from pathlib import Path

import pyvista as pv
from ansys.mapdl.reader import read_binary


def read_bc_load_from_rst(rst_path, rnum=None):
    """
    Read BC and Load (prescribed displacement) directly from RST.

    Returns:
        node_coords:  (N_total, 3)
        bc_coords:    (N_bc, 3)    fixed (D=0) nodes
        load_coords:  (N_load, 3)  prescribed-displacement (D=u_disp) nodes
        u_disp:       float        the prescribed disp value (most common nonzero)
    """
    rst = read_binary(str(rst_path))
    grid = rst.grid

    node_coords = np.array(grid.points, dtype=np.float64)
    nnum_all = rst.mesh.nnum   # ANSYS node numbers (1-based usually)
    nnum_to_idx = {int(n): i for i, n in enumerate(nnum_all)}

    if rnum is None:
        rnum = 0  # use first result set — BC/Load definition is the same in all sets

    print(f"[INFO] RST has {rst.nsets} result sets, using rnum={rnum}")
    print(f"[INFO] result_dof: {rst.result_dof(rnum)}")

    nnum_bc, dof_bc, bc_val = rst.nodal_boundary_conditions(rnum)
    print(f"[INFO] Total D-command entries: {len(nnum_bc)}")
    print(f"[INFO] Unique nodes with D:     {len(np.unique(nnum_bc))}")
    print(f"[INFO] DOF distribution:        "
          f"{dict(zip(*np.unique(dof_bc, return_counts=True)))}")
    print(f"[INFO] BC value range:          "
          f"[{bc_val.min():.6f}, {bc_val.max():.6f}]")

    fixed_mask = np.abs(bc_val) < 1e-9
    loaded_mask = ~fixed_mask

    fixed_nodes = np.unique(nnum_bc[fixed_mask])
    loaded_nodes = np.unique(nnum_bc[loaded_mask])

    print(f"[INFO] Fixed   (D=0)  unique nodes: {len(fixed_nodes)}")
    print(f"[INFO] Loaded  (D!=0) unique nodes: {len(loaded_nodes)}")

    if loaded_mask.any():
        unique_vals, counts = np.unique(np.round(bc_val[loaded_mask], 6),
                                        return_counts=True)
        print(f"[INFO] Unique nonzero D values "
              f"(top 5 by count):")
        order = np.argsort(counts)[::-1][:5]
        for i in order:
            print(f"        {unique_vals[i]:+.6f} — {counts[i]} entries")
        # Most common = u_disp
        u_disp = float(unique_vals[order[0]])

        # Distribution of which DOF is loaded
        loaded_dofs = dof_bc[loaded_mask]
        print(f"[INFO] Loaded DOF distribution: "
              f"{dict(zip(*np.unique(loaded_dofs, return_counts=True)))}")
    else:
        u_disp = 0.0

    # Map node numbers → 0-based index in node_coords
    bc_idx = np.array(
        [nnum_to_idx[int(n)] for n in fixed_nodes if int(n) in nnum_to_idx],
        dtype=np.int64)
    load_idx = np.array(
        [nnum_to_idx[int(n)] for n in loaded_nodes if int(n) in nnum_to_idx],
        dtype=np.int64)

    bc_coords = node_coords[bc_idx].astype(np.float32)
    load_coords = node_coords[load_idx].astype(np.float32)
    return node_coords, bc_coords, load_coords, u_disp


def visualize(node_coords, bc_coords, load_coords, title, screenshot):
    n_total = len(node_coords)
    n_bc = len(bc_coords)
    n_load = len(load_coords)

    bone = pv.PolyData(node_coords)
    pl = pv.Plotter(off_screen=True, window_size=(1400, 900))
    pl.set_background('white')

    pl.add_mesh(bone, color='lightgrey', point_size=2.0,
                opacity=0.18, render_points_as_spheres=True,
                label=f'Bone nodes ({n_total})')

    if n_bc > 0:
        pl.add_mesh(pv.PolyData(bc_coords),
                    color='#1f77ff', point_size=14,
                    render_points_as_spheres=True,
                    label=f'BC ({n_bc})')

    if n_load > 0:
        pl.add_mesh(pv.PolyData(load_coords),
                    color='#ff2222', point_size=14,
                    render_points_as_spheres=True,
                    label=f'Load ({n_load})')

    pl.add_legend(bcolor='white', face='circle', size=(0.18, 0.12))
    pl.add_text(title, position='upper_left', font_size=10, color='black')
    pl.add_axes(line_width=3)
    pl.camera_position = 'iso'
    pl.show(screenshot=screenshot, auto_close=True)
    print(f"[INFO] Saved screenshot: {screenshot}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rst",
        default=r"J:\Metastasis_Rob\MV04\L2\VOL_10\sphere_17\MV04_L2_VOL10_SS25.rst",
        help="Path to one .rst file")
    ap.add_argument("--rnum", type=int, default=0,
                    help="Result set index (default 0 — BC defined in all sets)")
    ap.add_argument("--screenshot", default="bc_load_from_rst.png")
    args = ap.parse_args()

    rst_path = Path(args.rst)
    if not rst_path.exists():
        # Fallback: search for any .rst that pairs with the npz we used before
        candidates = list(Path(r"J:\Metastasis_Rob").rglob("*.rst"))
        if not candidates:
            raise FileNotFoundError(f"No RST found at {rst_path} and "
                                    f"no fallback in J:\\Metastasis_Rob")
        rst_path = candidates[0]
        print(f"[WARN] Using fallback RST: {rst_path}")

    print(f"[INFO] RST: {rst_path}")
    node_coords, bc_coords, load_coords, u_disp = read_bc_load_from_rst(
        rst_path, rnum=args.rnum)
    print(f"[INFO] Detected u_disp ≈ {u_disp:.6f}")

    visualize(node_coords, bc_coords, load_coords,
              title=rst_path.stem + f"  (u_disp={u_disp:+.4f})",
              screenshot=args.screenshot)


if __name__ == '__main__':
    main()
