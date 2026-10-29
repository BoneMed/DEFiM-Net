#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive pyvista 3-panel visualisation for an inference result.
GT | Pred | |Error| with ONE shared clip-plane: dragging the plane in
the left viewport re-slices all three identically, cameras are linked.

Run with the pytorch-gpu env (pyvista 0.48 installed there):

  python visualize_inference.py \
      --results ../../Results/SSMHybrid_Tumor_finetune_mre_voxel \
      --target MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp18
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


def inv_signed_sqrt(y, scale=100.0):
    """Inverse of signed_log(strain * 100). Name kept for back-compat
    even though the actual transform is now log-based."""
    return np.sign(y) * np.expm1(np.abs(y)) / scale


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
    ap.add_argument("--target", default="MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp45")
    args = ap.parse_args()

    sdir = resolve_sample_dir(args)
    name = sdir.name
    print(f"Loading: {sdir}")

    d = np.load(sdir / "eval_data.npz")
    mask = d["mask"] > 0.5
    bbox_min = d["bbox_min"]
    bbox_max = d["bbox_max"]
    pred_mpa = np.where(mask, inv_signed_sqrt(d["pred_raw"]), 0).astype(np.float32)
    gt_mpa = np.where(mask, inv_signed_sqrt(d["gt_raw"]), 0).astype(np.float32)
    R = mask.shape[0]

    valid = mask
    clim = [float(np.percentile(gt_mpa[valid], 2)),
            float(np.percentile(gt_mpa[valid], 98))]
    eclim = [0.0, float(np.percentile(np.abs(pred_mpa - gt_mpa)[valid], 95))]
    mae = float(np.mean(np.abs(pred_mpa - gt_mpa)[valid]))
    ss_res = np.sum((pred_mpa[valid] - gt_mpa[valid]) ** 2)
    ss_tot = np.sum((gt_mpa[valid] - gt_mpa[valid].mean()) ** 2) + 1e-12
    r2 = float(1 - ss_res / ss_tot)
    print(f"  GT clim:  [{clim[0]:.4f}, {clim[1]:.4f}] MPa")
    print(f"  Err clim: [{eclim[0]:.4f}, {eclim[1]:.4f}] MPa")
    print(f"  R²={r2:.4f}  MAE={mae:.4f} MPa")

    spacing = (bbox_max - bbox_min) / R
    grid = pv.ImageData(dimensions=(R + 1, R + 1, R + 1),
                        spacing=spacing, origin=bbox_min)
    grid.cell_data["GT"] = gt_mpa.ravel(order="F")
    grid.cell_data["Pred"] = pred_mpa.ravel(order="F")
    grid.cell_data["Error"] = np.abs(pred_mpa - gt_mpa).ravel(order="F")
    grid.cell_data["mask"] = mask.astype(np.float32).ravel(order="F")
    solid = grid.threshold(0.5, scalars="mask")

    # ===== Plotter with 3 viewports + ONE shared clip plane =====
    pl = pv.Plotter(shape=(1, 3), window_size=(2400, 900),
                    title=f"{name}  R²={r2:.4f}  MAE={mae:.4f} MPa")

    # Per-viewport mesh config
    viewports = [
        (0, "GT",     "GT",     "jet", clim,  "GT (MPa)"),
        (1, "Pred",   "Pred",   "jet", clim,  "Pred (MPa)"),
        (2, "Error", "Error",   "hot", eclim, "|Error| (MPa)"),
    ]

    # Faint transparent shell of the full bone in each viewport for context
    shell = solid.extract_surface()
    for col, label, *_ in viewports:
        pl.subplot(0, col)
        pl.add_mesh(shell, opacity=0.05, color="gray", show_scalar_bar=False,
                    name=f"shell_{col}")
        pl.add_text(label, font_size=16, color="black", name=f"title_{col}")

    # Initial plane (axial-ish through the centre)
    plane_origin = np.array(solid.center, dtype=float)
    plane_normal = np.array([0.0, 1.0, 0.0], dtype=float)
    plane_state = {"normal": plane_normal, "origin": plane_origin}

    def update_all(normal, origin):
        """Re-clip the full solid and re-add the coloured cut to every viewport."""
        plane_state["normal"] = np.asarray(normal, dtype=float)
        plane_state["origin"] = np.asarray(origin, dtype=float)
        clipped = solid.clip(normal=plane_state["normal"],
                             origin=plane_state["origin"],
                             invert=True)
        for col, _label, scalars, cmap, climv, title in viewports:
            pl.subplot(0, col)
            pl.add_mesh(
                clipped, scalars=scalars, cmap=cmap, clim=climv,
                show_edges=False, name=f"cut_{col}",
                scalar_bar_args={"title": title, "fmt": "%.4f"})
        pl.render()

    # Initial draw
    update_all(plane_normal, plane_origin)

    # Single plane widget on the GT viewport — its callback redraws all 3
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
