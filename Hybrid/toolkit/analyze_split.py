#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train vs Val feature-distribution audit.

For every npz in train+val, builds a fixed-length per-case feature
vector (geometry, BC/Load counts, E and S3 statistics, etc.), then:

  1. Per-feature: report train min..max and flag val cases that fall
     OUTSIDE train range (with a 1% margin).
  2. PCA on standardized features (fit on TRAIN only). Project val cases
     into the same space and rank them by Mahalanobis-like distance from
     the train mean to find val cases that look "novel" relative to train.
  3. Saves a 2D PCA scatter (train cloud + val points) and a CSV with
     each val case's per-feature z-score + PCA distance for later filtering.

Usage:
  python toolkit/analyze_split.py \
    --split ../Data/tumor_split.json \
    --data_dir ../Data/ElementGraph_Tumor_Train \
    --out_dir ../Results/split_audit
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Feature extraction
# ============================================================
FEATURE_NAMES = [
    'u_disp',                                       # 1
    'n_bc', 'n_load', 'n_elem',                     # 3
    'E_mean', 'E_std', 'E_min', 'E_max', 'E_p99',   # 5
    'S3_mean', 'S3_std', 'S3_min', 'S3_p1', 'S3_p99_abs',  # 5
    'bbox_dx', 'bbox_dy', 'bbox_dz',                # 3
    'centroid_std',                                  # 1
    'frac_bc', 'frac_load',                          # 2
]


def case_features(npz_path):
    d = np.load(npz_path)
    centroids = d['centroids']  # (N, 3)
    E         = d['E']
    S3        = d['S3']
    is_bc     = d['is_bc']
    is_load   = d['is_load']
    u_disp    = float(d['u_disp'][0])
    n_elem    = len(E)

    bbox_min = centroids.min(0)
    bbox_max = centroids.max(0)
    dx, dy, dz = (bbox_max - bbox_min).tolist()

    feats = [
        u_disp,
        int((is_bc > 0.5).sum()),
        int((is_load > 0.5).sum()),
        n_elem,
        float(E.mean()), float(E.std()), float(E.min()), float(E.max()),
        float(np.percentile(E, 99)),
        float(S3.mean()), float(S3.std()), float(S3.min()),
        float(np.percentile(S3, 1)),
        float(np.percentile(np.abs(S3), 99)),
        dx, dy, dz,
        float(centroids.std(axis=0).mean()),
        float((is_bc > 0.5).sum()) / n_elem,
        float((is_load > 0.5).sum()) / n_elem,
    ]
    return np.array(feats, dtype=np.float64)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",    default="../Data/tumor_split.json")
    ap.add_argument("--data_dir", default="../Data/ElementGraph_Tumor_Train")
    ap.add_argument("--out_dir",  default="../Results/split_audit")
    ap.add_argument("--n_pca",    type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    with open(args.split, 'r') as f:
        split = json.load(f)

    # ---- Build per-case feature matrix ----
    train_names, train_X = [], []
    val_names, val_X = [], []
    for c in split['train_cases']:
        f = data_dir / f"{c}.npz"
        if not f.exists():
            continue
        train_names.append(c)
        train_X.append(case_features(f))
    for c in split['val_cases']:
        f = data_dir / f"{c}.npz"
        if not f.exists():
            continue
        val_names.append(c)
        val_X.append(case_features(f))

    train_X = np.stack(train_X)
    val_X   = np.stack(val_X)
    print(f"[INFO] train: {train_X.shape}  val: {val_X.shape}")

    # ---- 1. Per-feature OOD check ----
    train_lo = train_X.min(axis=0)
    train_hi = train_X.max(axis=0)
    margin = np.maximum(np.abs(train_hi) * 0.01, 1e-6)

    val_lo_below = val_X < (train_lo - margin)         # (V, F) bool
    val_hi_above = val_X > (train_hi + margin)
    val_ood_any  = (val_lo_below | val_hi_above).any(axis=1)

    print()
    print("=== Per-feature train range vs val ===")
    print(f"{'feature':18s}  {'train_min':>10s}  {'train_max':>10s}  "
          f"{'val_min':>10s}  {'val_max':>10s}  {'#val<lo':>7s}  {'#val>hi':>7s}  flag")
    print("-" * 110)
    for i, name in enumerate(FEATURE_NAMES):
        v_lo = val_X[:, i].min()
        v_hi = val_X[:, i].max()
        n_lo = int(val_lo_below[:, i].sum())
        n_hi = int(val_hi_above[:, i].sum())
        flag = ('' if (n_lo == 0 and n_hi == 0) else 'OOD')
        print(f"{name:18s}  {train_lo[i]:10.4g}  {train_hi[i]:10.4g}  "
              f"{v_lo:10.4g}  {v_hi:10.4g}  {n_lo:7d}  {n_hi:7d}  {flag}")

    print()
    print(f"Val cases with ANY OOD feature: {int(val_ood_any.sum())}/{len(val_X)}")

    # ---- 2. Standardize + PCA (fit on TRAIN) ----
    mean = train_X.mean(axis=0)
    std  = train_X.std(axis=0).clip(min=1e-12)
    Xtr = (train_X - mean) / std
    Xva = (val_X   - mean) / std

    # SVD-based PCA
    U, S, Vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False)
    explained = (S ** 2) / np.sum(S ** 2)
    print()
    print("=== PCA (fit on train) explained variance ===")
    print("  " + "  ".join(f"PC{i+1}={explained[i]*100:.1f}%" for i in range(min(8, len(explained)))))

    # Project
    train_pc = Xtr @ Vt.T
    val_pc   = Xva @ Vt.T

    # Distance from train centroid in standardized PC space
    train_centroid = train_pc.mean(axis=0)
    train_cov = np.cov(train_pc.T)
    train_cov_inv = np.linalg.pinv(train_cov)
    diff_v = val_pc - train_centroid
    mahal_v = np.sqrt(np.einsum('vi,ij,vj->v', diff_v, train_cov_inv, diff_v))
    diff_t = train_pc - train_centroid
    mahal_t = np.sqrt(np.einsum('ti,ij,tj->t', diff_t, train_cov_inv, diff_t))
    train_d99 = float(np.percentile(mahal_t, 99))

    n_val_above_train_d99 = int((mahal_v > train_d99).sum())
    print()
    print(f"=== Mahalanobis distance from train centroid (full {Xtr.shape[1]}-D) ===")
    print(f"  train mean={mahal_t.mean():.2f}  p99={train_d99:.2f}  max={mahal_t.max():.2f}")
    print(f"  val   mean={mahal_v.mean():.2f}  p99={float(np.percentile(mahal_v,99)):.2f}  max={mahal_v.max():.2f}")
    print(f"  Val cases beyond train p99: {n_val_above_train_d99}/{len(val_X)}")

    # Top 10 most-novel val cases
    rank = np.argsort(-mahal_v)
    print()
    print("Top 10 most-novel val cases (high Mahalanobis distance):")
    for r in rank[:10]:
        print(f"  d={mahal_v[r]:6.2f}  {val_names[r]}")

    # ---- 3. Save 2D PCA scatter ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.scatter(train_pc[:, 0], train_pc[:, 1], s=8, alpha=0.4, c='steelblue', label=f'Train (n={len(train_pc)})')
    ax.scatter(val_pc[:, 0],   val_pc[:, 1],   s=20, alpha=0.8, c='red', edgecolors='k', linewidth=0.4, label=f'Val (n={len(val_pc)})')
    # Mark top-3 most novel val cases
    for r in rank[:3]:
        ax.annotate(val_names[r].split('_')[-1], (val_pc[r, 0], val_pc[r, 1]),
                    fontsize=7, color='darkred')
    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)')
    ax.set_title('Per-case feature PCA  (red dots = val)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'pca_scatter.png', dpi=200); plt.close()

    # Histograms of each feature
    n_feat = len(FEATURE_NAMES)
    fig, axes = plt.subplots((n_feat + 3) // 4, 4, figsize=(18, 3 * ((n_feat + 3) // 4)))
    axes = axes.flatten()
    for i, name in enumerate(FEATURE_NAMES):
        axes[i].hist(train_X[:, i], bins=30, alpha=0.5, label='train', color='steelblue')
        axes[i].hist(val_X[:, i],   bins=30, alpha=0.5, label='val',   color='red')
        axes[i].set_title(name); axes[i].grid(alpha=0.3)
        if i == 0: axes[i].legend()
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / 'feature_histograms.png', dpi=180); plt.close()

    # CSV with per-val-case zscores and Mahal distance
    with open(out_dir / 'val_audit.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case', 'mahal_dist'] + [f'z_{n}' for n in FEATURE_NAMES] + ['n_ood_features'])
        for j, name in enumerate(val_names):
            z = (val_X[j] - mean) / std
            n_ood = int((np.abs(z) > 3).sum())   # |z|>3 = OOD-ish
            w.writerow([name, f'{mahal_v[j]:.3f}'] + [f'{zz:.3f}' for zz in z] + [n_ood])

    print()
    print(f"[DONE] Outputs:")
    print(f"  {out_dir / 'pca_scatter.png'}")
    print(f"  {out_dir / 'feature_histograms.png'}")
    print(f"  {out_dir / 'val_audit.csv'}")


if __name__ == "__main__":
    main()
