#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Violin plot of test-set metrics (R², MAE, RMSE) per case.

Reads metrics.csv produced by inference.py, drops the worst-R² outlier,
and produces a 1×3 violin figure with per-case scatter points overlaid.

Usage:
    python plot_test_violin.py \
        --metrics_csv ../../Results/SSMHybrid_Tumor_finetune_mre_voxel/metrics.csv \
        --out ../../Results/SSMHybrid_Tumor_finetune_mre_voxel/violin_metrics.png
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv",
                    default="../../Results/SSMHybrid_Tumor_finetune_mre_voxel/metrics.csv")
    ap.add_argument("--out",
                    default="../../Results/SSMHybrid_Tumor_finetune_mre_voxel/violin_metrics.png")
    ap.add_argument("--n_drop", type=int, default=1,
                    help="Drop this many worst-R² cases (default 1).")
    args = ap.parse_args()

    rows = []
    with open(args.metrics_csv) as f:
        for r in csv.DictReader(f):
            if r["name"] in ("MEAN", "GLOBAL") or not r["name"]:
                continue
            try:
                rows.append({
                    "name": r["name"],
                    "r2": float(r["r2_vox"]),
                    "mae": float(r["mae_vox"]),
                    "rmse": float(r["rmse_vox"]),
                })
            except (ValueError, KeyError):
                continue

    n_total = len(rows)
    print(f"Total per-case rows: {n_total}")

    # ---- Drop worst-R² ----
    rows_sorted = sorted(rows, key=lambda x: x["r2"])
    dropped = rows_sorted[:args.n_drop]
    kept = rows_sorted[args.n_drop:]

    print(f"\nDropped {len(dropped)} worst-R² case(s):")
    for r in dropped:
        print(f"  R²={r['r2']:+.4f}  MAE={r['mae']:.4f}  {r['name']}")

    r2 = np.array([r["r2"] for r in kept])
    mae = np.array([r["mae"] for r in kept])
    rmse = np.array([r["rmse"] for r in kept])

    print(f"\nKept {len(kept)} cases:")
    print(f"  R²   mean={r2.mean():.4f}  median={np.median(r2):.4f}  std={r2.std():.4f}")
    print(f"  MAE  mean={mae.mean():.4f}  median={np.median(mae):.4f}")
    print(f"  RMSE mean={rmse.mean():.4f}  median={np.median(rmse):.4f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))

    panels = [
        (axes[0], r2,   "R²",   "tab:blue"),
        (axes[1], mae,  "MAE",  "tab:orange"),
        (axes[2], rmse, "RMSE", "tab:red"),
    ]

    rng = np.random.default_rng(0)

    for ax, vals, label, color in panels:
        # Violin
        vp = ax.violinplot(vals, positions=[0], widths=0.7,
                           showmeans=False, showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.35)
            body.set_edgecolor("k")
            body.set_linewidth(1)

        # Scatter (jittered)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(jitter, vals, s=22, alpha=0.7, c=color, edgecolor="k",
                   linewidth=0.4, zorder=3)

        # Mean + median lines
        mean_v = float(vals.mean())
        median_v = float(np.median(vals))
        ax.axhline(mean_v, color="k", lw=1.2, ls="--",
                   label=f"mean = {mean_v:.4f}")
        ax.axhline(median_v, color="k", lw=1.2, ls=":",
                   label=f"median = {median_v:.4f}")

        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="best", fontsize=9, frameon=True)

    drop_str = ", ".join(d["name"] for d in dropped) or "none"
    fig.suptitle(
        f"Test set metrics  (n={len(kept)},  dropped worst-R² = {drop_str})",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
