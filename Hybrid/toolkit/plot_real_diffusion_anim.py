#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_real_diffusion_anim.py — Animated visualisation of sparse diffusion.

Loads one .npz, voxelises BC/Load, expands by 26-neighbour diffusion N layers,
and writes an animated GIF that loops through layer 0 -> N.

Run:
    python plot_real_diffusion_anim.py
    python plot_real_diffusion_anim.py --grid 64 --n_layers 5 \
        --gif real_diffusion.gif --fps 1
"""

import argparse
import numpy as np
from pathlib import Path
import pyvista as pv


# ---- 26 offsets ----
_OFFSETS_26 = np.array([(di, dj, dk)
                         for di in (-1, 0, 1)
                         for dj in (-1, 0, 1)
                         for dk in (-1, 0, 1)
                         if not (di == 0 and dj == 0 and dk == 0)],
                        dtype=np.int64)


def voxelise_bc_load(centroids, is_bc, is_load, D=64, pad=2.0):
    bbox_min = centroids.min(axis=0) - pad
    bbox_max = centroids.max(axis=0) + pad
    extent = bbox_max - bbox_min
    pts_norm = (centroids - bbox_min) / extent
    pts_grid = (pts_norm * (D - 1)).round().astype(np.int64).clip(0, D - 1)

    active = set()
    for mask in (is_bc > 0.5, is_load > 0.5):
        for i in np.where(mask)[0]:
            x, y, z = pts_grid[i]
            active.add((int(x), int(y), int(z)))
    return np.array(sorted(active), dtype=np.int64), bbox_min, bbox_max


def run_diffusion(active_init, n_layers, D):
    print(f"  Initial active: {len(active_init)}")
    sets = [active_init.copy()]
    current = active_init
    for li in range(1, n_layers + 1):
        expanded = current[:, None, :] + _OFFSETS_26[None, :, :]
        expanded = expanded.reshape(-1, 3)
        in_bounds = ((expanded >= 0) & (expanded < D)).all(axis=1)
        expanded = expanded[in_bounds]
        merged = np.concatenate([current, expanded], axis=0)
        h = merged[:, 0] * (D * D) + merged[:, 1] * D + merged[:, 2]
        _, idx = np.unique(h, return_index=True)
        current = merged[np.sort(idx)]
        print(f"  After layer {li}: {len(current)}")
        sets.append(current.copy())
    return sets


def voxel_to_world(vox, bbox_min, bbox_max, D):
    extent = bbox_max - bbox_min
    return bbox_min + (vox.astype(np.float32) / (D - 1)) * extent


def layer_color(p, n_layers):
    if p == 0:
        return (1.00, 0.10, 0.10)
    t = (p - 1) / max(n_layers - 1, 1)
    return (1.00, 0.30 + 0.65 * t, 0.05 + 0.15 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz",
        default="../Data/ElementGraph_Tumor_Train/MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp10.npz")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=5)
    ap.add_argument("--gif", default="real_diffusion.gif")
    ap.add_argument("--fps", type=int, default=1,
                    help="Frames per second (default 1 = 1s per layer)")
    ap.add_argument("--n_loops", type=int, default=2,
                    help="How many full passes to record")
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
    active_init, bbox_min, bbox_max = voxelise_bc_load(
        centroids, is_bc, is_load, D=args.grid)
    print(f"[INFO] Initial non-zero: {len(active_init)}")

    # Build a "bone occupancy" mask: which voxels actually contain bone
    extent = bbox_max - bbox_min
    pts_norm = (centroids - bbox_min) / extent
    pts_grid = (pts_norm * (args.grid - 1)).round().astype(np.int64).clip(0, args.grid - 1)
    bone_mask = np.zeros((args.grid,) * 3, dtype=bool)
    bone_mask[pts_grid[:, 0], pts_grid[:, 1], pts_grid[:, 2]] = True
    print(f"[INFO] Bone-occupied voxels: {bone_mask.sum()} / {args.grid**3}")

    print(f"[INFO] Diffusing {args.n_layers} layers ...")
    active_sets = run_diffusion(active_init, args.n_layers, args.grid)

    # Report how much of each layer leaked into non-bone voxels
    print("[INFO] Bone-overlap per layer (raw):")
    for li, s in enumerate(active_sets):
        in_bone = bone_mask[s[:, 0], s[:, 1], s[:, 2]].sum()
        print(f"  Layer {li}: {in_bone}/{len(s)}  "
              f"({100*in_bone/max(len(s),1):.1f}% in bone)")

    # ---- Mask out air voxels for visualisation ----
    # Show only the activations that actually fall inside the bone.
    print("[INFO] Masking air voxels for visualisation...")
    masked_sets = []
    for s in active_sets:
        if len(s) == 0:
            masked_sets.append(s)
            continue
        keep = bone_mask[s[:, 0], s[:, 1], s[:, 2]]
        masked_sets.append(s[keep])
    active_sets = masked_sets

    # ---- Build voxel cube grid for each layer ----
    # Each frame uses an ImageData with cell scalar = 1 if active, 0 otherwise.
    D = args.grid
    extent = bbox_max - bbox_min
    spacing = (extent / (D - 1)).tolist()

    def make_active_grid(active):
        vol = np.zeros((D, D, D), dtype=np.float32)
        if len(active) > 0:
            vol[active[:, 0], active[:, 1], active[:, 2]] = 1.0
        grid = pv.ImageData(
            dimensions=(D + 1, D + 1, D + 1),
            origin=bbox_min.tolist(),
            spacing=spacing,
        )
        grid.cell_data['active'] = vol.flatten(order='F')
        return grid.threshold(value=0.5, scalars='active')

    layer_grids = [make_active_grid(s) for s in active_sets]

    # ---- Plotter with GIF writer ----
    pl = pv.Plotter(window_size=(1400, 1100), off_screen=True)
    pl.set_background('white')

    bone_cloud = pv.PolyData(centroids)
    pl.add_mesh(bone_cloud,
                color='lightgrey',
                point_size=2.0,
                opacity=0.12,
                render_points_as_spheres=True)

    # Initial frame
    actor = pl.add_mesh(layer_grids[0],
                        color=layer_color(0, args.n_layers),
                        opacity=1.0,
                        show_edges=False)
    pl.add_text(
        f"Layer 0  ({len(active_sets[0])} voxels)",
        position='upper_left', font_size=14, color='black',
        name='title')

    # ---- Camera: front-upper-left ----
    # Compute focus and an offset position
    focus = ((bbox_min + bbox_max) / 2).tolist()
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    # Front-upper-left: -X (left), -Y (front), +Z (up)
    cam_pos = (
        focus[0] - diag * 0.55,   # left
        focus[1] - diag * 0.85,   # front
        focus[2] + diag * 0.55,   # up
    )
    pl.camera_position = [cam_pos, focus, (0, 0, 1)]
    # First fit the bone fully, then back off a touch for margin
    pl.reset_camera()
    pl.camera.zoom(0.80)

    # Open GIF
    gif_path = Path(args.gif)
    print(f"[INFO] Writing GIF: {gif_path}")
    pl.open_gif(str(gif_path), fps=args.fps)

    n_panels = args.n_layers + 1
    for loop in range(args.n_loops):
        for p in range(n_panels):
            pl.remove_actor(actor)
            actor = pl.add_mesh(
                layer_grids[p],
                color=layer_color(p, args.n_layers),
                opacity=1.0,
                show_edges=False)
            label = ("Layer 0  (initial BC/Load)"
                     if p == 0 else f"Layer {p}  (after diffusion)")
            pl.add_text(
                f"{label}\nactive: {len(active_sets[p])}",
                position='upper_left', font_size=14, color='black',
                name='title')
            pl.write_frame()

    pl.close()
    print(f"[INFO] Saved animated GIF: {gif_path}")
    print(f"       {n_panels * args.n_loops} frames at {args.fps} fps")


if __name__ == '__main__':
    main()
