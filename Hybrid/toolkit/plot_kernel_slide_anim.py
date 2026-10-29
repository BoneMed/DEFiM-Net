#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_kernel_slide_anim.py — Animated visualization of the 7x7x7 large
depthwise kernel sliding over the bone, with progressive downsampling.

Three "scenes" play in sequence:
  Scene A — kernel sliding at full resolution (D voxels), 7^3 yellow box
            traverses the volume along a snake path.
  Scene B — stride-2 downsample, kernel sliding at D/2 resolution.
  Scene C — another stride-2 downsample, kernel sliding at D/4 resolution.

At each scene, the bone is rendered as a coarse voxel grid (showing
the *current* feature-map resolution), and a wireframe 7^3 cube highlights
the kernel position. As the kernel reaches each voxel, that voxel "lights
up" red, showing how the receptive field grows.

Run:
    python plot_kernel_slide_anim.py
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


# ============================================================
# Helpers
# ============================================================
def voxelise_bone(centroids, D=32, pad=2.0):
    bbox_min = centroids.min(axis=0) - pad
    bbox_max = centroids.max(axis=0) + pad
    extent = bbox_max - bbox_min
    pts_norm = (centroids - bbox_min) / extent
    pts_grid = (pts_norm * (D - 1)).round().astype(np.int64).clip(0, D - 1)

    bone = np.zeros((D, D, D), dtype=bool)
    bone[pts_grid[:, 0], pts_grid[:, 1], pts_grid[:, 2]] = True
    return bone, bbox_min, bbox_max


def downsample(bone):
    """Stride-2 max-pool downsample, returns bone at half resolution."""
    D = bone.shape[0]
    D2 = D // 2
    out = np.zeros((D2, D2, D2), dtype=bool)
    for i in range(D2):
        for j in range(D2):
            for k in range(D2):
                out[i, j, k] = bone[2*i:2*i+2, 2*j:2*j+2, 2*k:2*k+2].any()
    return out


def make_voxel_cubes(active_mask, bbox_min, bbox_max, D, name='active'):
    extent = bbox_max - bbox_min
    spacing = (extent / (D - 1)).tolist()
    grid = pv.ImageData(
        dimensions=(D + 1, D + 1, D + 1),
        origin=bbox_min.tolist(),
        spacing=spacing,
    )
    grid.cell_data[name] = active_mask.astype(np.float32).flatten(order='F')
    return grid.threshold(value=0.5, scalars=name)


def voxel_to_world(coord, bbox_min, bbox_max, D):
    """Map an integer voxel coord (i,j,k) to its center in world space."""
    extent = bbox_max - bbox_min
    spacing = extent / (D - 1)
    origin = bbox_min + spacing / 2  # cell center
    return origin + np.array(coord) * spacing, spacing


def make_kernel_box(center_world, spacing, kernel_voxels=7):
    """Build a wireframe cube of size kernel_voxels in voxel units."""
    half = (kernel_voxels * spacing) / 2.0
    cube = pv.Cube(
        center=center_world.tolist(),
        x_length=2 * half[0],
        y_length=2 * half[1],
        z_length=2 * half[2],
    )
    return cube


def raster_path_x(bone, n_steps):
    """Continuous raster scan along the x-axis at the most populated (z, y).

    Returns a list of (z, y, x) voxel positions where x increments by a
    fixed step from the bone's x_min to x_max along the chosen row.
    """
    if bone.sum() == 0:
        return []

    # Find the (z, y) row that contains the most bone voxels (so the box
    # actually walks through bone for as long as possible).
    counts = bone.sum(axis=2)             # (D, D) over (z, y)
    z0, y0 = np.unravel_index(np.argmax(counts), counts.shape)

    # x range: full bone bbox along x at this row, with a small margin
    xs = np.where(bone[z0, y0])[0]
    if len(xs) == 0:
        # fallback: scan the full x extent of the whole bone
        xs_all = np.where(bone.any(axis=(0, 1)))[0]
        xs = xs_all
    x_min, x_max = int(xs.min()), int(xs.max())

    # Build n_steps evenly-spaced positions, contiguous (each step ~1-2 voxels apart)
    if n_steps <= 1:
        x_positions = [x_min]
    else:
        x_positions = np.linspace(x_min, x_max, n_steps).round().astype(int).tolist()

    return [(int(z0), int(y0), int(x)) for x in x_positions]


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz",
        default="../Data/ElementGraph_Tumor_Train/MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp10.npz")
    ap.add_argument("--grid", type=int, default=32,
                    help="Initial coarse grid size (e.g. 32, downsampled to 16, 8)")
    ap.add_argument("--steps_scene1", type=int, default=16,
                    help="Raster steps in scene 1 (full res)")
    ap.add_argument("--steps_scene2", type=int, default=8,
                    help="Raster steps in scene 2 (D/2)")
    ap.add_argument("--steps_scene3", type=int, default=4,
                    help="Raster steps in scene 3 (D/4)")
    ap.add_argument("--kernel", type=int, default=7,
                    help="Large kernel side length in voxels")
    ap.add_argument("--gif", default="kernel_slide.gif")
    ap.add_argument("--fps", type=int, default=2)
    args = ap.parse_args()

    npz = Path(args.npz)
    print(f"[INFO] Loading {npz.name}")
    data = np.load(npz)
    centroids = data['centroids'].astype(np.float32)

    print(f"[INFO] Voxelising to {args.grid}^3 ...")
    bone1, bbox_min, bbox_max = voxelise_bone(centroids, D=args.grid)
    print(f"  bone1 (D={args.grid}): {bone1.sum()} voxels")
    bone2 = downsample(bone1)
    print(f"  bone2 (D={args.grid//2}): {bone2.sum()} voxels")
    bone3 = downsample(bone2)
    print(f"  bone3 (D={args.grid//4}): {bone3.sum()} voxels")

    scenes = [
        (bone1, args.grid,       "Scene 1   full resolution",
         "#ff8a3d", args.steps_scene1),
        (bone2, args.grid // 2,  "Scene 2   stride 2 (D/2)",
         "#ff5722", args.steps_scene2),
        (bone3, args.grid // 4,  "Scene 3   stride 2 (D/4)",
         "#d32f2f", args.steps_scene3),
    ]

    pl = pv.Plotter(window_size=(1400, 1100), off_screen=True)
    pl.set_background('white')

    bone_cloud = pv.PolyData(centroids)

    # Camera once, fit the whole bone
    focus = ((bbox_min + bbox_max) / 2).tolist()
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    cam_pos = (
        focus[0] - diag * 0.55,
        focus[1] - diag * 0.85,
        focus[2] + diag * 0.55,
    )
    pl.camera_position = [cam_pos, focus, (0, 0, 1)]
    pl.reset_camera()
    pl.camera.zoom(0.78)

    # Open GIF
    gif_path = Path(args.gif)
    print(f"[INFO] Writing GIF: {gif_path}")
    pl.open_gif(str(gif_path), fps=args.fps)

    for scene_idx, (bone, D, scene_title, voxel_color, n_steps) in enumerate(scenes):
        print(f"[INFO] {scene_title} (n_steps={n_steps})")

        # Build the voxel cube grid for this scene's bone
        bone_grid = make_voxel_cubes(bone, bbox_min, bbox_max, D, 'bone')

        # Continuous raster scan along x at the row with most bone voxels
        path = raster_path_x(bone, n_steps=n_steps)
        if len(path) == 0:
            continue

        # The receptive field accumulates as kernel slides
        rf = np.zeros_like(bone)

        for step_i, vox in enumerate(path):
            # Mark the kernel coverage at this position
            ks = args.kernel
            half = ks // 2
            zs = slice(max(vox[0] - half, 0), min(vox[0] + half + 1, D))
            ys = slice(max(vox[1] - half, 0), min(vox[1] + half + 1, D))
            xs = slice(max(vox[2] - half, 0), min(vox[2] + half + 1, D))
            rf[zs, ys, xs] |= bone[zs, ys, xs]

            # Build the "covered" cubes mesh
            covered_grid = make_voxel_cubes(rf, bbox_min, bbox_max, D, 'rf')

            # Build the kernel wireframe box centered at vox
            center_world, spacing = voxel_to_world(vox, bbox_min, bbox_max, D)
            kernel_box = make_kernel_box(center_world, spacing, ks)

            # ---- Draw frame ----
            pl.clear_actors()
            pl.set_background('white')

            # Faint bone reference
            pl.add_mesh(bone_cloud, color='lightgrey',
                        point_size=2.0, opacity=0.08,
                        render_points_as_spheres=True,
                        name='bone_cloud')

            # Bone voxels at current resolution (light)
            pl.add_mesh(bone_grid,
                        color='#cfd8dc',
                        opacity=0.40,
                        show_edges=False,
                        name='bone_vox')

            # Cells already touched by the kernel = receptive field so far
            try:
                pl.add_mesh(covered_grid,
                            color=voxel_color,
                            opacity=0.95,
                            show_edges=False,
                            name='covered')
            except Exception:
                pass

            # The kernel itself: yellow wireframe cube
            pl.add_mesh(kernel_box,
                        color='#ffd700',
                        style='wireframe',
                        line_width=5,
                        name='kernel')

            cov_n = int(rf.sum())
            tot_n = int(bone.sum())
            pl.add_text(
                f"{scene_title}\n"
                f"D={D}^3   kernel={ks}^3 depthwise\n"
                f"step {step_i+1}/{n_steps}   "
                f"covered {cov_n}/{tot_n} ({100*cov_n/tot_n:.0f}%)",
                position='upper_left', font_size=14, color='black',
                name='title')

            pl.write_frame()

    pl.close()
    print(f"[INFO] Saved animated GIF: {gif_path}")
    total_frames = (args.steps_scene1 + args.steps_scene2 + args.steps_scene3)
    print(f"       {total_frames} frames at {args.fps} fps")


if __name__ == '__main__':
    main()
