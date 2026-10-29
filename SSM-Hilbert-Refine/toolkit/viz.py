#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared visualisation helpers for the training and finetuning loops.

Both healthyPreTrain.py and finetune_tumor.py import these — they used to
maintain near-duplicate copies which drifted between scripts. Single
source of truth lives here.

Functions:
    inv_signed_log(y)         — invert signed_log(strain * 100) → strain.
    ansys_cmap()              — ANSYS-style diverging colormap.
    save_scatter(...)         — GT vs Pred scatter with peak-preserving
                                subsample.
    save_curve(...)           — GT/Pred sorted curve + error band.
    save_slices(...)          — Full-volume 3-view (axial/coronal/sagittal)
                                slice comparison in MPa-equivalent units.
    save_loss_curve(csv, png) — Multi-panel epoch curve from train log.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


def inv_signed_log(y, scale=100.0):
    """Inverse of signed_log(strain * scale) → physical strain.

    Default scale=100 matches the current target encoding. Returns
    strain in decimal units (e.g. -0.005 for 0.5% compressive strain).
    """
    return np.sign(y) * np.expm1(np.abs(y)) / scale


def ansys_cmap():
    return cm.colors.LinearSegmentedColormap.from_list("ansys", [
        (0.0, "#0000ff"), (0.2, "#00ffff"), (0.4, "#00ff00"),
        (0.6, "#ffff00"), (0.8, "#ff8000"), (1.0, "#ff0000")])


def save_scatter(gt, pred, path, title, max_pts=200000, peak_frac=0.05):
    """Scatter plot of GT vs Pred.

    When subsampling is required, guarantees the most extreme `peak_frac`
    points (by |gt|) survive so the long tail stays visible — uniform
    random subsampling would drop them.

    R² and MAE are computed on the FULL (unsubsampled) arrays so the
    title reflects the real metric.
    """
    full_gt, full_pred = gt, pred
    if len(gt) > max_pts:
        order = np.argsort(np.abs(gt))               # ascending |gt|
        n_peak = max(1, int(max_pts * peak_frac))
        peak_idx = order[-n_peak:]                   # top |gt|
        rest = order[:-n_peak]
        n_rest = max_pts - n_peak
        if len(rest) > n_rest:
            rest = np.random.default_rng(0).choice(rest, n_rest, replace=False)
        idx = np.concatenate([peak_idx, rest])
        gt, pred = gt[idx], pred[idx]
    plt.figure(figsize=(6, 6))
    plt.scatter(gt, pred, s=1, alpha=0.15, c="steelblue")
    lo, hi = min(gt.min(), pred.min()), max(gt.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], "r-", lw=1.5)
    r2 = 1 - np.sum((full_gt - full_pred) ** 2) / (
        np.sum((full_gt - full_gt.mean()) ** 2) + 1e-12)
    mae = np.mean(np.abs(full_gt - full_pred))
    plt.title(f"{title}\nR²={r2:.4f} MAE={mae:.4f} MPa")
    plt.xlabel("GT (MPa)"); plt.ylabel("Pred (MPa)")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def save_curve(gt, pred, path, title, max_pts=80000):
    if len(gt) > max_pts:
        idx = np.random.choice(len(gt), max_pts, replace=False)
        gt, pred = gt[idx], pred[idx]
    order = np.argsort(gt)
    fig, axes = plt.subplots(2, 1, figsize=(16, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(gt[order], label="GT", lw=0.5, alpha=0.8)
    axes[0].plot(pred[order], label="Pred", lw=0.5, alpha=0.8)
    axes[0].set_ylabel("S3 (MPa)"); axes[0].set_title(title)
    axes[0].legend(); axes[0].grid(alpha=0.3)
    err = pred[order] - gt[order]
    axes[1].fill_between(range(len(err)), err, 0, alpha=0.4, color="red")
    axes[1].set_ylabel("Error"); axes[1].set_xlabel("Voxel (sorted)")
    axes[1].set_title(f"MAE={np.mean(np.abs(err)):.4f} MPa")
    axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close()


def save_slices(pred_vol, gt_vol, mask_vol, path, epoch):
    """Full-volume 3-view (axial / coronal / sagittal) slice comparison.
    pred_vol / gt_vol / mask_vol are torch tensors (any device, any dtype).
    """
    p = inv_signed_log(pred_vol.squeeze().cpu().numpy())
    g = inv_signed_log(gt_vol.squeeze().cpu().numpy())
    m = mask_vol.squeeze().cpu().numpy()
    D, H, W = p.shape
    cmap = ansys_cmap()
    valid = m > 0.5
    if valid.sum() == 0:
        return
    vmin = float(np.percentile(g[valid], 2))
    vmax = float(np.percentile(g[valid], 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, (s, t) in enumerate([(D // 2, "Axial"),
                                (H // 2, "Coronal"),
                                (W // 2, "Sagittal")]):
        gs = [g[s, :, :], g[:, s, :], g[:, :, s]][i]
        ps = [p[s, :, :], p[:, s, :], p[:, :, s]][i]
        ms = [m[s, :, :], m[:, s, :], m[:, :, s]][i]
        axes[0, i].imshow(np.where(ms > 0.5, gs, np.nan).T, origin="lower",
                          cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"GT {t}"); axes[0, i].axis("off")
        axes[1, i].imshow(np.where(ms > 0.5, ps, np.nan).T, origin="lower",
                          cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f"Pred {t}"); axes[1, i].axis("off")
    fig.suptitle(f"Epoch {epoch}", fontsize=14)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def save_loss_curve(csv_path, path):
    """Plot epoch / train_loss / val_r2 / val_mae from a training CSV."""
    epochs, losses, r2s, maes = [], [], [], []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            losses.append(float(row["train_loss"]))
            r = row.get("val_r2", "nan")
            r2s.append(float(r) if r != "nan" else np.nan)
            m = row.get("val_mae", "nan")
            maes.append(float(m) if m != "nan" else np.nan)
    if len(epochs) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, losses, lw=1)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Train Loss"); axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)
    vr = [(e, r) for e, r in zip(epochs, r2s) if not np.isnan(r)]
    if vr:
        axes[1].plot([e for e, _ in vr], [r for _, r in vr],
                     "o-", lw=1, ms=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("R²")
    axes[1].set_title("Val R²"); axes[1].set_ylim(-0.1, 1.05)
    axes[1].grid(alpha=0.3)
    vm = [(e, m) for e, m in zip(epochs, maes) if not np.isnan(m)]
    if vm:
        axes[2].plot([e for e, _ in vm], [m for _, m in vm],
                     "o-", lw=1, ms=3, color="orange")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("MAE (MPa)")
    axes[2].set_title("Val MAE"); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
