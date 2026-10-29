#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive PyVista viewer for tumor prediction results.
Semi-transparent shell + plane widget + toggle GT/Pred/Error.

Run with pyansys env:
  C:/Users/Administrator/anaconda3/envs/pyansys/python.exe vis_interactive.py
"""

import numpy as np
from pathlib import Path
import pyvista as pv


# ---- Find best test sample ----
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--eval_dir", default="../../Results/Tumor_directly_4ch_val_full_valmae")
ap.add_argument("--target",   default="MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp18")
args = ap.parse_args()

eval_dir = Path(args.eval_dir)
sample_dirs = sorted([d for d in eval_dir.iterdir() if d.is_dir() and (d/'eval_data.npz').exists()])

if not sample_dirs:
    print(f"No eval_data.npz found under {eval_dir}. Run inference.py --step infer first.")
    exit()

sdir = None
for d in sample_dirs:
    if args.target in d.name:
        sdir = d
        break
if sdir is None:
    sdir = sample_dirs[0]  # fallback to first

def inv_signed_sqrt(y):
    """Inverse of signed_sqrt(S3 * 10) -> MPa. Matches current training target."""
    return np.sign(y) * (y ** 2) / 10.0

print(f"Loading: {sdir.name}")
d = np.load(sdir / 'eval_data.npz')
mask = d['mask']
bbox_min = d['bbox_min']
bbox_max = d['bbox_max']

if 'pred_raw' in d:
    pred_mpa = np.where(mask, inv_signed_sqrt(d['pred_raw']), 0).astype(np.float32)
    gt_mpa = np.where(mask, inv_signed_sqrt(d['gt_raw']), 0).astype(np.float32)
else:
    pred_mpa = np.nan_to_num(d['pred_mpa'], 0).astype(np.float32)
    gt_mpa = np.nan_to_num(d['gt_mpa'], 0).astype(np.float32)
R = mask.shape[0]

# ---- Build grid (cell data, 避免表面插值到0) ----
spacing = (bbox_max - bbox_min) / R
grid = pv.ImageData(dimensions=(R+1, R+1, R+1), spacing=spacing, origin=bbox_min)
grid.cell_data['GT (MPa)'] = gt_mpa.ravel(order='F')
grid.cell_data['Pred (MPa)'] = pred_mpa.ravel(order='F')
grid.cell_data['|Error| (MPa)'] = np.abs(pred_mpa - gt_mpa).ravel(order='F')
grid.cell_data['mask'] = mask.astype(np.float32).ravel(order='F')

solid = grid.threshold(0.5, scalars='mask')
shell = solid.extract_surface()
cell_centers = solid.cell_centers().points

valid = mask > 0.5
clim_s3 = [float(np.percentile(gt_mpa[valid], 25)),
           float(np.percentile(gt_mpa[valid], 75))]
clim_err = [0, float(np.percentile(np.abs(pred_mpa - gt_mpa)[valid], 95))]

# Metrics
ss_res = np.sum((pred_mpa[valid] - gt_mpa[valid])**2)
ss_tot = np.sum((gt_mpa[valid] - gt_mpa[valid].mean())**2) + 1e-12
r2 = 1 - ss_res / ss_tot
mae = np.mean(np.abs(pred_mpa[valid] - gt_mpa[valid]))

print(f"R2={r2:.4f}, MAE={mae:.4f} MPa")
print(f"S3 clim: {clim_s3}")

# ---- Interactive Plotter ----
mode = ["Pred"]  # "GT", "Pred", "Error"
plane_state = [np.array([1., 0., 0.]), np.array(solid.center)]
current_actors = []

plotter = pv.Plotter(title=f"{sdir.name}  R2={r2:.4f}  MAE={mae:.4f} MPa")

# Semi-transparent shell
plotter.add_mesh(shell, opacity=0.08, color='gray', show_scalar_bar=False)


def update_clip(normal=None, origin=None):
    if normal is not None:
        plane_state[0] = np.array(normal)
    if origin is not None:
        plane_state[1] = np.array(origin)

    dist = (cell_centers - plane_state[1]) @ plane_state[0]
    ids = np.where(dist >= 0)[0]
    clipped = solid.extract_cells(ids)

    for a in current_actors:
        plotter.remove_actor(a)
    current_actors.clear()
    for k in list(plotter.scalar_bars.keys()):
        plotter.remove_scalar_bar(k)

    if mode[0] == "GT":
        actor = plotter.add_mesh(
            clipped, scalars='GT (MPa)', cmap='jet', clim=clim_s3,
            show_scalar_bar=True,
            scalar_bar_args={'title': 'GT S3 (MPa)', 'fmt': '%.2f'})
    elif mode[0] == "Pred":
        actor = plotter.add_mesh(
            clipped, scalars='Pred (MPa)', cmap='jet', clim=clim_s3,
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Pred S3 (MPa)', 'fmt': '%.2f'})
    else:
        actor = plotter.add_mesh(
            clipped, scalars='|Error| (MPa)', cmap='hot', clim=clim_err,
            show_scalar_bar=True,
            scalar_bar_args={'title': '|Error| (MPa)', 'fmt': '%.4f'})

    current_actors.append(actor)
    plotter.render()


# Initial clip
update_clip(np.array([1., 0., 0.]), np.array(solid.center))

# Plane widget
plotter.add_plane_widget(
    callback=update_clip,
    implicit=False,
    normal="x",
    origin=solid.center,
)

# ---- Mode toggle buttons ----
def on_gt(flag):
    if flag:
        mode[0] = "GT"
        update_clip()

def on_pred(flag):
    if flag:
        mode[0] = "Pred"
        update_clip()

def on_err(flag):
    if flag:
        mode[0] = "Error"
        update_clip()

# GT button (blue)
plotter.add_checkbox_button_widget(
    on_gt, value=False, position=(10, 10), size=30,
    border_size=3, color_on='deepskyblue', color_off='lightgray')
plotter.add_text("GT", position=(45, 16), font_size=9, color='black')

# Pred button (green)
plotter.add_checkbox_button_widget(
    on_pred, value=True, position=(10, 50), size=30,
    border_size=3, color_on='limegreen', color_off='lightgray')
plotter.add_text("Pred", position=(45, 56), font_size=9, color='black')

# Error button (red)
plotter.add_checkbox_button_widget(
    on_err, value=False, position=(10, 90), size=30,
    border_size=3, color_on='tomato', color_off='lightgray')
plotter.add_text("|Error|", position=(45, 96), font_size=9, color='black')

plotter.add_text(
    f"{sdir.name}\nR2={r2:.4f}  MAE={mae:.4f} MPa",
    position='upper_right', font_size=10, color='black')

plotter.add_axes()
plotter.show()
