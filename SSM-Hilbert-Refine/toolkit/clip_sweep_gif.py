#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Off-screen 3-viewport GIF: a clip plane sweeps along an axis.

Same layout as visualize_inference.py (GT | Pred | |Error|), but the
plane moves automatically from one end of the bone bbox to the other,
writing one frame per position into an animated GIF.

Usage:
  python clip_sweep_gif.py \
      --target MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp45 \
      --axis y --n_frames 60 --fps 20 \
      --out ../../Results/sp45_sweep.gif
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
    ap.add_argument("--target",
                    default="MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp45")
    ap.add_argument("--axis", default="y", choices=["x", "y", "z"],
                    help="Axis along which the clip plane sweeps.")
    ap.add_argument("--reverse", action="store_true",
                    help="Sweep from max → min instead of min → max.")
    ap.add_argument("--n_frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--window_size", default="1800,700",
                    help="W,H of each frame. Larger = nicer but bigger GIF.")
    ap.add_argument("--margin_frac", type=float, default=0.05,
                    help="Skip this fraction near each end of the bbox "
                         "(avoids empty cuts at the boundary).")
    ap.add_argument("--slice_view", action="store_true", default=True,
                    help="Position camera looking straight down the sweep "
                         "axis so the cut face is shown head-on. Default ON.")
    ap.add_argument("--no_slice_view", dest="slice_view",
                    action="store_false")
    ap.add_argument("--zoom", type=float, default=1.4,
                    help="Camera zoom factor after fitting (>1 = closer).")
    ap.add_argument("--out", required=True,
                    help="Output .gif path.")
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
    print(f"  R²={r2:.4f}  MAE={mae:.4f} MPa")

    spacing = (bbox_max - bbox_min) / R
    grid = pv.ImageData(dimensions=(R + 1, R + 1, R + 1),
                        spacing=spacing, origin=bbox_min)
    grid.cell_data["GT"] = gt_mpa.ravel(order="F")
    grid.cell_data["Pred"] = pred_mpa.ravel(order="F")
    grid.cell_data["Error"] = np.abs(pred_mpa - gt_mpa).ravel(order="F")
    grid.cell_data["mask"] = mask.astype(np.float32).ravel(order="F")
    solid = grid.threshold(0.5, scalars="mask")

    # ---- Sweep range ----
    bounds = solid.bounds                    # (xmin, xmax, ymin, ymax, zmin, zmax)
    axis_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    a_min = bounds[2 * axis_idx]
    a_max = bounds[2 * axis_idx + 1]
    margin = args.margin_frac * (a_max - a_min)
    t_lo, t_hi = a_min + margin, a_max - margin
    t_vals = np.linspace(t_lo, t_hi, args.n_frames)
    if args.reverse:
        t_vals = t_vals[::-1]
    print(f"  Sweep {args.axis}: {t_lo:.2f} → {t_hi:.2f} "
          f"({'reverse' if args.reverse else 'forward'}), "
          f"{args.n_frames} frames @ {args.fps} fps")

    normal = np.zeros(3)
    normal[axis_idx] = 1.0
    center = np.array(solid.center, dtype=float)

    # ---- Off-screen plotter ----
    w, h = (int(x) for x in args.window_size.split(","))
    pl = pv.Plotter(shape=(1, 3), window_size=(w, h), off_screen=True,
                    title=f"{name}  R²={r2:.4f}  MAE={mae:.4f} MPa")

    viewports = [
        (0, "GT",    "GT",    "jet", clim,  "GT (MPa)"),
        (1, "Pred",  "Pred",  "jet", clim,  "Pred (MPa)"),
        (2, "Error", "Error", "hot", eclim, "|Error| (MPa)"),
    ]
    shell = solid.extract_surface()
    for col, label, *_ in viewports:
        pl.subplot(0, col)
        pl.add_mesh(shell, opacity=0.05, color="gray", show_scalar_bar=False,
                    name=f"shell_{col}")
        pl.add_text(label, font_size=16, color="black", name=f"title_{col}")

    # Anchor cameras with an initial clip so reset_camera fits everything
    init_clip = solid.clip(normal=normal, origin=center, invert=True)
    for col, _label, scalars, cmap, climv, title in viewports:
        pl.subplot(0, col)
        pl.add_mesh(init_clip, scalars=scalars, cmap=cmap, clim=climv,
                    show_edges=False, name=f"cut_{col}",
                    scalar_bar_args={"title": title, "fmt": "%.4f"})

    if args.slice_view:
        # Camera looks straight down +axis at the cut face; up = next axis
        cam_offset = 2.5 * max(bounds[2*i+1] - bounds[2*i] for i in range(3))
        cam_pt = center.copy()
        cam_pt[axis_idx] += cam_offset
        up = np.zeros(3)
        up[(axis_idx + 1) % 3] = 1.0   # next axis as up
        for col, *_ in viewports:
            pl.subplot(0, col)
            pl.camera_position = [cam_pt.tolist(), center.tolist(), up.tolist()]
            pl.reset_camera()
            pl.camera.zoom(args.zoom)
    else:
        for col, *_ in viewports:
            pl.subplot(0, col)
            pl.reset_camera()
    pl.link_views()
    cam_pos = pl.camera_position   # remember the post-fit camera

    # ---- Animate ----
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.open_gif(str(out_path), fps=args.fps)

    for fi, t in enumerate(t_vals):
        origin = center.copy()
        origin[axis_idx] = float(t)
        clipped = solid.clip(normal=normal, origin=origin, invert=True)
        for col, _label, scalars, cmap, climv, title in viewports:
            pl.subplot(0, col)
            pl.add_mesh(clipped, scalars=scalars, cmap=cmap, clim=climv,
                        show_edges=False, name=f"cut_{col}",
                        scalar_bar_args={"title": title, "fmt": "%.4f"})
            pl.camera_position = cam_pos        # keep camera fixed
        pl.write_frame()
        if (fi + 1) % 10 == 0 or fi + 1 == len(t_vals):
            print(f"  frame {fi+1}/{len(t_vals)}")

    pl.close()
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Saved: {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
