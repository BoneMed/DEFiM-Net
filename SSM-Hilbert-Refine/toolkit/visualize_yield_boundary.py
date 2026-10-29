#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive yield-boundary comparison: GT | Pred, both coloured by strain
with the same clip plane, AND both -0.019 yield iso-surfaces overlaid on
each viewport so failure-region overlap is visually obvious.

- Left  viewport: GT strain on the clipped cross-section
- Right viewport: Pred strain on the clipped cross-section
- BOTH viewports also show:
    * GT yield boundary (cyan, semi-transparent 3D iso-surface at strain = -0.019)
    * Pred yield boundary (red, semi-transparent 3D iso-surface at strain = -0.019)
  Drag the clip plane in the left viewport, both views re-slice together;
  the floating cyan / red shells stay in place so you can rotate and see
  overlap.

Run with the pytorch-gpu env (pyvista installed there):

  python visualize_yield_boundary.py \\
      --results ../../Results/SSMHybrid_Tumor_finetune_mre_voxel \\
      --target MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp45
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


def inv_signed_log(y, scale=100.0):
    """Inverse of signed_log(strain * 100) → decimal strain."""
    return np.sign(y) * np.expm1(np.abs(y)) / scale


# Backward-compat alias: older eval_data.npz files were named back when
# the encoding was signed_sqrt(*10). Current files are signed_log(*100).
inv_signed_sqrt = inv_signed_log


def resolve_sample_dir(args) -> Path:
    if args.sample_dir is not None:
        sd = Path(args.sample_dir).resolve()
        if not (sd / "eval_data.npz").exists():
            raise SystemExit(f"{sd}/eval_data.npz not found.")
        return sd

    results = Path(args.results).resolve()
    if not results.exists():
        raise SystemExit(f"--results dir does not exist: {results}")

    candidates = sorted(d for d in results.iterdir()
                        if d.is_dir() and (d / "eval_data.npz").exists())
    if not candidates:
        raise SystemExit(f"No <case>/eval_data.npz under {results}.")

    if args.target:
        for d in candidates:
            if args.target in d.name:
                return d
        print(f"[WARN] no case matched '{args.target}'; using {candidates[0].name}")
    return candidates[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir", nargs="?", default=None)
    ap.add_argument("--results",
                    default="../../Results/SSMHybrid_Tumor_finetune_mre_voxel")
    ap.add_argument("--target",
                    default="MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp45")
    ap.add_argument("--yield_strain", type=float, default=-0.019,
                    help="Failure threshold (compressive strain). Both GT "
                         "and Pred iso-surfaces are drawn at this value. "
                         "Default -0.019 ≈ -1.9%% compressive yield.")
    ap.add_argument("--boundary_opacity", type=float, default=0.35,
                    help="Opacity of the two boundary iso-surfaces (0=invisible, 1=solid).")
    args = ap.parse_args()

    sdir = resolve_sample_dir(args)
    name = sdir.name
    print(f"Loading: {sdir}")

    d = np.load(sdir / "eval_data.npz")
    mask = d["mask"] > 0.5
    bbox_min = d["bbox_min"]
    bbox_max = d["bbox_max"]
    pred = np.where(mask, inv_signed_log(d["pred_raw"]), 0).astype(np.float32)
    gt = np.where(mask, inv_signed_log(d["gt_raw"]), 0).astype(np.float32)
    R = mask.shape[0]

    valid = mask
    clim = [float(np.percentile(gt[valid], 2)),
            float(np.percentile(gt[valid], 98))]
    mae = float(np.mean(np.abs(pred - gt)[valid]))
    ss_res = np.sum((pred[valid] - gt[valid]) ** 2)
    ss_tot = np.sum((gt[valid] - gt[valid].mean()) ** 2) + 1e-12
    r2 = float(1 - ss_res / ss_tot)

    # Failure-volume comparison: how many bone voxels are below the yield
    # threshold in each, and how well they overlap.
    fail_gt = (gt <= args.yield_strain) & valid
    fail_pred = (pred <= args.yield_strain) & valid
    n_gt = int(fail_gt.sum())
    n_pred = int(fail_pred.sum())
    n_intersect = int((fail_gt & fail_pred).sum())
    n_union = int((fail_gt | fail_pred).sum())
    iou = n_intersect / max(n_union, 1)
    dice = 2 * n_intersect / max(n_gt + n_pred, 1)

    print(f"  GT clim:  [{clim[0]:.4f}, {clim[1]:.4f}] (strain)")
    print(f"  R²={r2:.4f}  MAE={mae:.4f} (strain)")
    print(f"  Yield threshold: {args.yield_strain}")
    print(f"  Failure voxels:  GT={n_gt}  Pred={n_pred}  intersect={n_intersect}")
    print(f"  Overlap:         IoU={iou:.3f}  Dice={dice:.3f}")

    # Build the volume. Cell data is what the threshold/clip operate on
    # naturally; for `contour` we convert to point data inside.
    spacing = (bbox_max - bbox_min) / R
    grid = pv.ImageData(dimensions=(R + 1, R + 1, R + 1),
                        spacing=spacing, origin=bbox_min)
    grid.cell_data["GT"] = gt.ravel(order="F")
    grid.cell_data["Pred"] = pred.ravel(order="F")
    grid.cell_data["mask"] = mask.astype(np.float32).ravel(order="F")
    solid = grid.threshold(0.5, scalars="mask")

    # Iso-surface extraction needs point-data scalars. Convert once.
    # Also use the point-data version for clipping so scalars survive the
    # widget's repeated clip operations (cell_data sometimes gets dropped
    # across clip() in newer pyvista versions, breaking the redraw).
    solid_pt = solid.cell_data_to_point_data()
    gt_boundary = solid_pt.contour(isosurfaces=[args.yield_strain], scalars="GT")
    pred_boundary = solid_pt.contour(isosurfaces=[args.yield_strain], scalars="Pred")

    # ===== Plotter: 2 viewports + ONE shared clip plane =====
    pl = pv.Plotter(shape=(1, 2), window_size=(1800, 900),
                    title=f"{name}  R²={r2:.4f}  MAE={mae:.4f}  "
                          f"Yield={args.yield_strain}  IoU={iou:.3f}")

    viewports = [
        (0, "GT",   "GT",   "jet", "GT (strain)"),
        (1, "Pred", "Pred", "jet", "Pred (strain)"),
    ]

    # Faint shell of the bone for context in each viewport
    shell = solid.extract_surface()
    for col, label, *_ in viewports:
        pl.subplot(0, col)
        pl.add_mesh(shell, opacity=0.04, color="gray", show_scalar_bar=False,
                    name=f"shell_{col}")
        pl.add_text(label, font_size=16, color="black", name=f"title_{col}")

    # Initial clip plane: axial-ish through the centre
    plane_origin = np.array(solid.center, dtype=float)
    plane_normal = np.array([0.0, 1.0, 0.0], dtype=float)

    def update_all(normal, origin):
        """Re-clip the strain solid; re-draw both viewports with boundaries."""
        normal_arr = np.asarray(normal, dtype=float)
        origin_arr = np.asarray(origin, dtype=float)
        # Clip the point-data version so the GT / Pred scalars survive
        # the clip filter across pyvista versions.
        clipped = solid_pt.clip(normal=normal_arr, origin=origin_arr, invert=True)

        for col, _label, scalars, cmap, sbar_title in viewports:
            pl.subplot(0, col)
            # Strain cross-section
            pl.add_mesh(
                clipped, scalars=scalars, cmap=cmap, clim=clim,
                show_edges=False, name=f"cut_{col}",
                scalar_bar_args={"title": sbar_title, "fmt": "%.4f"})

            # BOTH boundaries on BOTH views — that's what makes overlap
            # immediately visible. Cyan = where GT says it fails. Red =
            # where Pred says it fails. Where they agree the colours mix.
            pl.add_mesh(
                gt_boundary, color="cyan",
                opacity=args.boundary_opacity,
                name=f"gt_bnd_{col}", show_scalar_bar=False,
                label=f"GT yield ({args.yield_strain})")
            pl.add_mesh(
                pred_boundary, color="red",
                opacity=args.boundary_opacity,
                name=f"pred_bnd_{col}", show_scalar_bar=False,
                label=f"Pred yield ({args.yield_strain})")
        pl.render()

    # Initial draw
    update_all(plane_normal, plane_origin)

    # Legend on the right viewport (so it's not over the boundary widgets)
    pl.subplot(0, 1)
    pl.add_legend(
        [(f"GT yield",   "cyan"),
         (f"Pred yield", "red")],
        bcolor="white", face="rectangle", size=(0.18, 0.08),
        loc="upper right")

    # Plane widget on the GT viewport — its callback redraws both
    pl.subplot(0, 0)
    pl.add_plane_widget(
        callback=update_all,
        normal=plane_normal,
        origin=plane_origin,
        implicit=False,
        normal_rotation=True,
        outline_translation=True,
    )

    pl.link_views()
    pl.show()


if __name__ == "__main__":
    main()
