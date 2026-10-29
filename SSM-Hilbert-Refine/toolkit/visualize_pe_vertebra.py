#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_pe_vertebra.py — Render a vertebra with the upper half shown as a
semi-transparent shell and the lower half as a solid mesh coloured by 3D
sinusoidal positional encoding values.

Three coloring modes:

    --mode single   one PE channel (sin/cos at chosen axis & octave)
    --mode rgb      RGB triplet from sin(2^k π u_z), sin(2^k π u_y), sin(2^k π u_x)
    --mode hsv      hue from PE phase, value from PE magnitude (decorative)

Run inside any env with pyvista installed (BoneMed / pyansys / pytorch-gpu).
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataUtils import HybridDataset
from Net.HybridModel_SSM import _make_3d_pos_enc, _POS_ENC_MAX_L


def compute_pe_field(R, mode, axis, freq_l, phase):
    """Return (R,R,R) float32 array (single channel) or (R,R,R,3) (rgb)."""
    z = np.linspace(-1, 1, R, dtype=np.float32)
    y = np.linspace(-1, 1, R, dtype=np.float32)
    x = np.linspace(-1, 1, R, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')

    if mode == 'single':
        coord = {'z': zz, 'y': yy, 'x': xx}[axis]
        freq = (2.0 ** freq_l) * math.pi
        pe = (np.sin if phase == 'sin' else np.cos)(coord * freq).astype(np.float32)
        return pe                                                    # (R,R,R)

    if mode == 'rgb':
        freq = (2.0 ** freq_l) * math.pi
        # Map sin in [-1,1] to channel intensity in [0,1]
        r = (np.sin(zz * freq) + 1.0) * 0.5
        g = (np.sin(yy * freq) + 1.0) * 0.5
        b = (np.sin(xx * freq) + 1.0) * 0.5
        return np.stack([r, g, b], axis=-1).astype(np.float32)        # (R,R,R,3)

    if mode == 'model':
        # Use the EXACT same encoding the model applies during training.
        # Computes all 6·L = 60 channels (3 axes × {sin, cos} × 10 octaves),
        # then sums over (sin, cos, all frequencies) per axis to get one
        # superimposed "axis signature", and stacks the three axes as RGB.
        L = _POS_ENC_MAX_L                                           # 10
        C_pe = 6 * L                                                  # 60
        pe60 = _make_3d_pos_enc(R, R, R, C_pe, 'cpu',
                                 torch.float32).squeeze(0).numpy()     # (60, R, R, R)
        per_axis = 2 * L                                              # 20
        pe_z = pe60[0:per_axis].sum(axis=0)                           # (R,R,R)
        pe_y = pe60[per_axis:2*per_axis].sum(axis=0)
        pe_x = pe60[2*per_axis:].sum(axis=0)

        def _norm01(a):
            mn, mx = float(a.min()), float(a.max())
            return (a - mn) / max(mx - mn, 1e-8)

        rgb = np.stack([_norm01(pe_z), _norm01(pe_y), _norm01(pe_x)],
                        axis=-1).astype(np.float32)                    # (R,R,R,3)
        return rgb

    if mode == 'model_sum':
        # Single-scalar version: collapse all 60 PE channels into one scalar
        # per voxel by summing them. Use a single diverging colormap to
        # render. The result captures ALL channels with one number, at the
        # cost of mixing the per-axis structure.
        L = _POS_ENC_MAX_L
        C_pe = 6 * L
        pe60 = _make_3d_pos_enc(R, R, R, C_pe, 'cpu',
                                 torch.float32).squeeze(0).numpy()
        return pe60.sum(axis=0).astype(np.float32)                     # (R,R,R)

    if mode == 'hsv':
        freq = (2.0 ** freq_l) * math.pi
        # Hue from sin(z) phase, value from |sin(y)| etc.
        h = (np.arctan2(np.sin(yy * freq), np.sin(xx * freq)) / (2 * math.pi) + 0.5)
        s = np.ones_like(h)
        v = (np.sin(zz * freq) + 1.0) * 0.5
        # Convert HSV->RGB without external dep
        i = np.floor(h * 6).astype(int) % 6
        f = h * 6 - np.floor(h * 6)
        p = v * (1 - s); q = v * (1 - s * f); t = v * (1 - s * (1 - f))
        rgb = np.empty(h.shape + (3,), dtype=np.float32)
        for k, (rr, gg, bb) in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                          (p, q, v), (t, p, v), (v, p, q)]):
            sel = i == k
            rgb[sel] = np.stack([rr[sel], gg[sel], bb[sel]], axis=-1)
        return rgb

    raise ValueError(f"unknown mode {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../Data/ElementGraph_Tumor_Train")
    ap.add_argument("--case", default=None,
                    help="Case stem (no .npz). Default: first case.")
    ap.add_argument("--grid_res", type=int, default=64)
    ap.add_argument("--mode", default="model",
                    choices=['single', 'rgb', 'hsv', 'model', 'model_sum'],
                    help="model: 60-channel PE summed per axis -> RGB; "
                         "model_sum: 60-channel PE summed all -> single "
                         "scalar with diverging colormap")
    ap.add_argument("--axis", default="z", choices=['z', 'y', 'x'],
                    help="single mode: which axis the encoding lives on.")
    ap.add_argument("--phase", default="sin", choices=['sin', 'cos'],
                    help="single mode: sin or cos.")
    ap.add_argument("--freq_l", type=int, default=3,
                    help="Octave L: frequency = 2^L * pi.")
    ap.add_argument("--cut_axis", default="z", choices=['z', 'y', 'x'],
                    help="Which axis the cut plane is normal to.")
    ap.add_argument("--cut_pos", type=float, default=0.0,
                    help="Cut plane position in normalized [-1, 1] coords. "
                         "Voxels with axis-coord <= cut_pos are kept solid.")
    ap.add_argument("--shell_opacity", type=float, default=0.12)
    ap.add_argument("--shell_color", default="lightcoral")
    ap.add_argument("--cmap", default="custom",
                    choices=["custom", "RdBu_r", "coolwarm", "seismic"],
                    help="Colormap for single-channel PE. 'custom' matches "
                         "the diverging palette in the paper figure: "
                         "blue–cream–red.")
    ap.add_argument("--full", action="store_true",
                    help="Skip cut + wireframe, render entire bone solid "
                         "coloured by PE.")
    ap.add_argument("--screenshot",
                    default="../Results/pe_vertebra.png")
    ap.add_argument("--off_screen", action="store_true",
                    help="Render without window (just save PNG).")
    args = ap.parse_args()

    # ---- Load vertebra ----
    R = args.grid_res
    print(f"[INFO] Voxelising at R={R}...")
    ds = HybridDataset(args.data_dir, grid_res=R)
    idx = 0 if args.case is None else ds.names.index(args.case)
    case_name = ds.names[idx]
    sample = ds[idx]
    mask = sample['mask'].squeeze().numpy() > 0.5
    print(f"[INFO] Case: {case_name}, bone voxels: {int(mask.sum())}")

    # ---- Compute PE field ----
    pe = compute_pe_field(R, args.mode, args.axis, args.freq_l, args.phase)

    # ---- Build PyVista grid + extract bone solid ----
    grid = pv.ImageData(dimensions=(R + 1, R + 1, R + 1),
                         spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0))
    if pe.ndim == 3:                        # single-channel scalar field
        grid.cell_data['PE'] = pe.ravel(order='F')
        is_scalar = True
    else:                                    # RGB / HSV / model: (R,R,R,3)
        rgb_uint8 = (pe * 255).clip(0, 255).astype(np.uint8)
        N = pe.shape[0] * pe.shape[1] * pe.shape[2]
        rgb_flat = np.empty((N, 3), dtype=np.uint8)
        for c in range(3):
            rgb_flat[:, c] = rgb_uint8[..., c].ravel(order='F')
        grid.cell_data['PE_rgb'] = rgb_flat
        is_scalar = False
    grid.cell_data['mask'] = mask.astype(np.float32).ravel(order='F')

    solid = grid.threshold(0.5, scalars='mask')

    # ---- Choose render target: full bone or cut-half + wireframe ----
    if args.full:
        target = solid
    else:
        shell = solid.extract_surface()
        centers = solid.cell_centers().points
        # PyVista ImageData orders cells X-fastest, Z-slowest
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[args.cut_axis]
        coord_min, coord_max = centers[:, axis_idx].min(), centers[:, axis_idx].max()
        cut_voxel = coord_min + (args.cut_pos + 1) * 0.5 * (coord_max - coord_min)
        keep_ids = np.where(centers[:, axis_idx] <= cut_voxel)[0]
        target = solid.extract_cells(keep_ids)

    # ---- Build colormap ----
    if args.cmap == 'custom':
        from matplotlib.colors import LinearSegmentedColormap
        # Diverging palette extracted from the paper-figure colour blocks:
        # strong-blue → light-blue → cream → coral → strong-red
        cmap = LinearSegmentedColormap.from_list(
            'paper_div',
            ['#5B7FB0', '#BDD0E0', '#E5D5B5', '#E58368', '#A52424'],
            N=256)
    else:
        cmap = args.cmap

    # ---- Render ----
    pl = pv.Plotter(off_screen=args.off_screen, window_size=(1100, 900))
    pl.set_background('white')

    if not args.full:
        # Outer wireframe over the WHOLE vertebra
        pl.add_mesh(shell, opacity=args.shell_opacity, color=args.shell_color,
                    style='wireframe', line_width=0.4, show_scalar_bar=False)

    # Solid coloured by PE
    if is_scalar:
        # Auto-set clim from data percentiles for model_sum (range varies),
        # use [-1, 1] for single-channel sin/cos.
        if args.mode == 'single':
            clim = [-1, 1]
            bar_title = f'PE: {args.phase}(2^{args.freq_l} pi u_{args.axis})'
        else:
            arr = np.asarray(grid.cell_data['PE'])[mask.ravel(order='F')]
            lim = float(np.percentile(np.abs(arr), 98))
            clim = [-lim, lim]
            bar_title = f'PE sum (60 channels)'
        pl.add_mesh(
            target, scalars='PE', cmap=cmap, clim=clim,
            show_scalar_bar=True,
            scalar_bar_args={'title': bar_title, 'fmt': '%.2f'})
    else:
        # rgb / hsv / model — direct color array
        pl.add_mesh(target, scalars='PE_rgb', rgb=True,
                    show_scalar_bar=False)

    pl.add_axes()
    pl.add_text(
        f'{case_name}\n3D Positional Encoding ({args.mode})',
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
