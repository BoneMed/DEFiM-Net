#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_E_train_fem.py — Statistics of element-level Elastic Modulus (E)
from N:/SecondYear/CT_MASK_E_PHANTOMLESS/Train_FEM (33 subjects).

Outputs:
    E_train_fem_histogram.png    pooled E histogram
    E_train_fem_cdf.png          cumulative distribution
    E_train_fem_boxplot.png      per-subject box plot
    E_train_fem_per_subject.png  per-subject range (sorted)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})


def load_all(root):
    subjects = sorted([d for d in Path(root).iterdir() if d.is_dir()])
    all_E = []
    stats = []
    for subj in subjects:
        csvs = list(subj.glob('*_element_E_*.csv'))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        E = df['E_MPa'].values.astype(np.float32)
        all_E.append(E)
        stats.append({
            'name': subj.name,
            'n': len(E),
            'mean': float(E.mean()),
            'median': float(np.median(E)),
            'min': float(E.min()),
            'max': float(E.max()),
            'p5': float(np.percentile(E, 5)),
            'p25': float(np.percentile(E, 25)),
            'p75': float(np.percentile(E, 75)),
            'p95': float(np.percentile(E, 95)),
        })
    pooled = np.concatenate(all_E)
    return pooled, stats, all_E


def main():
    root = r"N:\SecondYear\CT_MASK_E_PHANTOMLESS\Train_FEM"
    out_dir = Path(".")

    print("[INFO] Loading ...")
    pooled, stats, per_subj = load_all(root)
    N = len(stats)
    print(f"[INFO] {N} subjects, {len(pooled):,} elements")
    print(f"[INFO] Pooled E: [{pooled.min():.1f}, {pooled.max():.1f}] MPa")
    print(f"[INFO] Mean={pooled.mean():.1f}, Median={np.median(pooled):.1f}")

    # ===== 1. Pooled histogram =====
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pooled, bins=200, range=(0, 18000), color='#3498DB',
            edgecolor='none', alpha=0.85)
    ax.axvline(np.median(pooled), color='red', lw=2.5, ls='--',
               label=f'Median = {np.median(pooled):.0f} MPa')
    ax.axvline(pooled.mean(), color='orange', lw=2.5, ls='--',
               label=f'Mean = {pooled.mean():.0f} MPa')
    ax.set_xlabel('Elastic Modulus E (MPa)', fontsize=22)
    ax.set_ylabel('Element count', fontsize=22)
    ax.set_title(f'Distribution of Element E  (N={N} subjects, {len(pooled):,} elements)',
                 fontsize=22)
    ax.legend(fontsize=18)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    fig.savefig(out_dir / 'E_train_fem_histogram.png', dpi=150)
    plt.close(fig)
    print("[INFO] Saved E_train_fem_histogram.png")

    # ===== 2. CDF =====
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_E = np.sort(pooled)
    pct = np.linspace(0, 100, len(sorted_E))
    step = max(1, len(sorted_E) // 5000)
    ax.plot(sorted_E[::step], pct[::step], '-', color='#3498DB', lw=2.5)
    for p_val in [5, 25, 50, 75, 95]:
        e_val = float(np.percentile(pooled, p_val))
        ax.axhline(p_val, color='gray', lw=0.8, ls=':', alpha=0.5)
        ax.axvline(e_val, color='gray', lw=0.8, ls=':', alpha=0.5)
        ax.plot(e_val, p_val, 'o', color='#E74C3C', markersize=10, zorder=5)
        ax.text(e_val + 300, p_val + 2,
                f'P{p_val}={e_val:.0f}', fontsize=16, color='#E74C3C')
    ax.set_xlabel('Elastic Modulus E (MPa)', fontsize=22)
    ax.set_ylabel('Cumulative %', fontsize=22)
    ax.set_title('Cumulative Distribution of Element E', fontsize=22)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    fig.savefig(out_dir / 'E_train_fem_cdf.png', dpi=150)
    plt.close(fig)
    print("[INFO] Saved E_train_fem_cdf.png")

    # ===== 3. Per-subject box plot =====
    medians = np.array([s['median'] for s in stats])
    order = np.argsort(medians)
    ordered_E = [per_subj[i] for i in order]
    ordered_names = [stats[i]['name'] for i in order]

    fig, ax = plt.subplots(figsize=(16, 7))
    bp = ax.boxplot(ordered_E, showfliers=False, patch_artist=True,
                    widths=0.6,
                    boxprops=dict(facecolor='#AED6F1', linewidth=1.5),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    ax.set_xticklabels(ordered_names, rotation=60, ha='right', fontsize=14)
    ax.set_ylabel('E (MPa)', fontsize=22)
    ax.set_title(f'Per-subject E distribution  (N={N}, sorted by median)',
                 fontsize=22)
    ax.grid(alpha=0.3, axis='y')
    ax.tick_params(axis='y', labelsize=18)
    fig.tight_layout()
    fig.savefig(out_dir / 'E_train_fem_boxplot.png', dpi=150)
    plt.close(fig)
    print("[INFO] Saved E_train_fem_boxplot.png")

    # ===== 4. Per-subject range =====
    p5s = np.array([s['p5'] for s in stats])
    p95s = np.array([s['p95'] for s in stats])

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(N)
    ax.fill_between(x, p5s[order], p95s[order], alpha=0.3,
                    color='#3498DB', label='P5–P95 range')
    ax.plot(x, medians[order], 'o-', color='#E74C3C', lw=2, ms=6,
            label='Median E')
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_names, rotation=60, ha='right', fontsize=12)
    ax.set_ylabel('E (MPa)', fontsize=22)
    ax.set_title(f'Per-subject E range  (N={N})', fontsize=22)
    ax.legend(fontsize=18, loc='upper left')
    ax.grid(alpha=0.3)
    ax.tick_params(axis='y', labelsize=18)
    fig.tight_layout()
    fig.savefig(out_dir / 'E_train_fem_per_subject.png', dpi=150)
    plt.close(fig)
    print("[INFO] Saved E_train_fem_per_subject.png")

    # ===== Summary table =====
    print(f"\n{'='*60}")
    print(f"{'Subject':>8}  {'N':>8}  {'Min':>8}  {'Median':>8}  "
          f"{'Mean':>8}  {'P95':>8}  {'Max':>8}")
    print(f"{'-'*60}")
    for s in sorted(stats, key=lambda s: s['median']):
        print(f"{s['name']:>8}  {s['n']:>8,}  {s['min']:>8.1f}  "
              f"{s['median']:>8.1f}  {s['mean']:>8.1f}  "
              f"{s['p95']:>8.1f}  {s['max']:>8.1f}")
    print(f"{'-'*60}")
    print(f"{'POOLED':>8}  {len(pooled):>8,}  {pooled.min():>8.1f}  "
          f"{np.median(pooled):>8.1f}  {pooled.mean():>8.1f}  "
          f"{np.percentile(pooled,95):>8.1f}  {pooled.max():>8.1f}")


if __name__ == '__main__':
    main()
