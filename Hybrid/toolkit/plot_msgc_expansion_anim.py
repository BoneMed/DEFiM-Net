#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_msgc_expansion_anim.py — Animated visualisation of how the receptive
field expands AFTER sparse + dense fusion, going through the large-kernel
MSGC encoder.

Pipeline simulated on a real bone:
  Frame 0: sparse-path output (5 layers of 26-neighbour diffusion from BC/Load)
  Frame 1: + MSGCDown level 1   (7^3 depthwise conv, radius +3)
  Frame 2: + MSGCDown level 2   (stride 2 -> effective radius +6)
  Frame 3: + MSGCDown level 3   (stride 2 -> effective radius +12)
  Frame 4: bottleneck — receptive field covers ~entire bone

Each frame is rendered as voxel cubes inside the bone, coloured by which
"layer" first activated the voxel. All operations are masked to bone
interior so signals don't leak into air (matches the bone_mask change in
SparseDiffuseConv).

Run:
    python plot_msgc_expansion_anim.py
    python plot_msgc_expansion_anim.py --grid 64 --fps 1
"""

import argparse
import numpy as np
from pathlib import Path

import pyvista as pv
from scipy.ndimage import binary_dilation, generate_binary_structure


# ============================================================
# Voxelise BC/Load (DataLoader convention)
# ============================================================
def voxelise_bc_load(centroids, is_bc, is_load, D=64, pad=2.0):
    bbox_min = centroids.min(axis=0) - pad
    bbox_max = centroids.max(axis=0) + pad
    extent = bbox_max - bbox_min
    pts_norm = (centroids - bbox_min) / extent
    pts_grid = (pts_norm * (D - 1)).round().astype(np.int64).clip(0, D - 1)

    bone = np.zeros((D, D, D), dtype=bool)
    bone[pts_grid[:, 0], pts_grid[:, 1], pts_grid[:, 2]] = True

    active = set()
    for mask in (is_bc > 0.5, is_load > 0.5):
        for i in np.where(mask)[0]:
            x, y, z = pts_grid[i]
            active.add((int(x), int(y), int(z)))

    init_active = np.zeros((D, D, D), dtype=bool)
    for x, y, z in active:
        init_active[x, y, z] = True
    return bone, init_active, bbox_min, bbox_max


# ============================================================
# Sparse 26-neighbour diffusion (n_layers steps)
# ============================================================
def sparse_diffuse(active_init, bone, n_layers=5):
    """Each step expands by 1 voxel via 26-neighbour dilation, masked to bone."""
    struct = generate_binary_structure(rank=3, connectivity=3)  # 26-nbr
    current = active_init.copy()
    for _ in range(n_layers):
        current = binary_dilation(current, structure=struct, iterations=1)
        current &= bone
    return current


# ============================================================
# Large-kernel conv expansion (MSGCDown layers)
# ============================================================
def msgc_expand(active, bone, radius):
    """One MSGCDown layer = dilation by `radius` voxels (effective rf), masked."""
    # Build a structuring ball of given radius
    rr = int(radius)
    z, y, x = np.ogrid[-rr:rr+1, -rr:rr+1, -rr:rr+1]
    ball = (z*z + y*y + x*x) <= rr*rr
    out = binary_dilation(active, structure=ball, iterations=1)
    out &= bone
    return out


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz",
        default="../Data/ElementGraph_Tumor_Train/MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp10.npz")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--sparse_layers", type=int, default=5,
                    help="How many sparse 26-nbr diffusion layers")
    ap.add_argument("--gif", default="msgc_expansion.gif")
    ap.add_argument("--fps", type=int, default=1)
    ap.add_argument("--n_loops", type=int, default=2)
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
    bone, init_active, bbox_min, bbox_max = voxelise_bc_load(
        centroids, is_bc, is_load, D=args.grid)
    print(f"[INFO] Bone voxels:  {bone.sum()} / {args.grid**3}")
    print(f"[INFO] Init active:  {init_active.sum()}")

    # Frame 0: sparse-path output (5 layers of 26-nbr diffusion, masked)
    print(f"[INFO] Sparse path: {args.sparse_layers} diffusion layers ...")
    sparse_out = sparse_diffuse(init_active, bone, args.sparse_layers)
    print(f"  active = {sparse_out.sum()}")

    # Frames 1..3: MSGCDown layers, effective radius doubles each level
    # Layer 1: 7^3 kernel at full res -> radius 3
    # Layer 2: stride 2, so original-space radius doubles -> 6
    # Layer 3: stride 4, doubles again -> 12
    radii = [3, 6, 12]
    frames = [sparse_out]
    for li, r in enumerate(radii, start=1):
        print(f"[INFO] MSGC level {li}: dilate radius {r} (masked) ...")
        nxt = msgc_expand(frames[-1], bone, r)
        frames.append(nxt)
        print(f"  active = {nxt.sum()}")

    # Frame 4: bottleneck (whole bone)
    frames.append(bone.copy())
    print(f"[INFO] Bottleneck: full bone ({bone.sum()})")

    n_frames = len(frames)   # = 5
    titles = [
        f"Stage 0  Sparse path  ({args.sparse_layers}-layer 26-nbr diffusion)",
        "Stage 1  + MSGC level 1  (7^3 kernel, r=3)",
        "Stage 2  + MSGC level 2  (stride 2, eff. r=6)",
        "Stage 3  + MSGC level 3  (stride 2, eff. r=12)",
        "Bottleneck  (RF ~ entire patch)",
    ]

    # ---- Build voxel-cube grids for each frame ----
    D = args.grid
    extent = bbox_max - bbox_min
    spacing = (extent / (D - 1)).tolist()

    def make_grid(active):
        vol = active.astype(np.float32)
        grid = pv.ImageData(
            dimensions=(D + 1, D + 1, D + 1),
            origin=bbox_min.tolist(),
            spacing=spacing,
        )
        grid.cell_data['active'] = vol.flatten(order='F')
        return grid.threshold(value=0.5, scalars='active')

    layer_grids = [make_grid(f) for f in frames]

    # ---- Plotter ----
    pl = pv.Plotter(window_size=(1400, 1100), off_screen=True)
    pl.set_background('white')

    bone_cloud = pv.PolyData(centroids)
    pl.add_mesh(bone_cloud, color='lightgrey',
                point_size=2.0, opacity=0.10,
                render_points_as_spheres=True)

    # Colour ramp: red -> orange -> yellow (per stage)
    def stage_color(i):
        if i == 0:
            return (0.95, 0.10, 0.10)
        t = (i - 1) / max(n_frames - 2, 1)
        r = 1.00
        g = 0.30 + 0.65 * t
        b = 0.05 + 0.25 * t
        return (r, g, b)

    actor = pl.add_mesh(layer_grids[0],
                        color=stage_color(0),
                        opacity=1.0, show_edges=False)
    pl.add_text(f"{titles[0]}\nactive: {int(frames[0].sum())}",
                position='upper_left', font_size=14, color='black',
                name='title')

    # Camera: front-upper-left
    focus = ((bbox_min + bbox_max) / 2).tolist()
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    cam_pos = (
        focus[0] - diag * 0.55,
        focus[1] - diag * 0.85,
        focus[2] + diag * 0.55,
    )
    pl.camera_position = [cam_pos, focus, (0, 0, 1)]
    pl.reset_camera()
    pl.camera.zoom(0.80)

    gif_path = Path(args.gif)
    print(f"[INFO] Writing GIF: {gif_path}")
    pl.open_gif(str(gif_path), fps=args.fps)

    for loop in range(args.n_loops):
        for fi in range(n_frames):
            pl.remove_actor(actor)
            actor = pl.add_mesh(
                layer_grids[fi],
                color=stage_color(fi),
                opacity=1.0,
                show_edges=False)
            pl.add_text(
                f"{titles[fi]}\nactive: {int(frames[fi].sum())}",
                position='upper_left', font_size=14, color='black',
                name='title')
            pl.write_frame()

    pl.close()
    print(f"[INFO] Saved animated GIF: {gif_path}")
    print(f"       {n_frames * args.n_loops} frames at {args.fps} fps")


if __name__ == '__main__':
    main()
