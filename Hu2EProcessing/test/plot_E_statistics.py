#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_E_statistics.py — Statistics of element-level Elastic Modulus (E)
across all tumor train + test samples.

Generates 4 figures suitable for PPT:
    1. E histogram (all elements pooled, log-x)
    2. Box plot of per-sample median E (shows inter-subject variability)
    3. E cumulative distribution (percentile curve)
    4. E range per sample (min-max bars, sorted by median)

Run:
    python plot_E_statistics.py
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})


def load_all_E(data_dirs):
    """Load E from all .npz files, return pooled array + per-sample stats."""
    all_E = []
    sample_stats = []
    for d in data_dirs:
        for f in sorted(Path(d).glob("*.npz")):
            data = np.load(f)
            E = data['E'].astype(np.float32)
            E_pos = E[E > 0]
            all_E.append(E_pos)
            sample_stats.append({
                'name': f.stem,
                'n_elem': len(E),
                'n_pos': len(E_pos),
                'n_zero': int((E == 0).sum()),
                'mean': float(E_pos.mean()) if len(E_pos) > 0 else 0,
                'median': float(np.median(E_pos)) if len(E_pos) > 0 else 0,
                'p5': float(np.percentile(E_pos, 5)) if len(E_pos) > 0 else 0,
                'p95': float(np.percentile(E_pos, 95)) if len(E_pos) > 0 else 0,
                'min': float(E_pos.min()) if len(E_pos) > 0 else 0,
                'max': float(E_pos.max()) if len(E_pos) > 0 else 0,
            })
    pooled = np.concatenate(all_E)
    return pooled, sample_stats


def main():
    data_dirs = [
        "../../Data/ElementGraph_Tumor_Train",
        "../../Data/ElementGraph_Tumor_Test",
    ]
    out_dir = Path(".")

    print("[INFO] Loading E from all samples ...")
    pooled, stats = load_all_E(data_dirs)
    print(f"[INFO] {len(stats)} samples, {len(pooled):,} elements (E > 0)")
    print(f"[INFO] Pooled E: [{pooled.min():.2f}, {pooled.max():.2f}] MPa")
    print(f"[INFO] Pooled mean={pooled.mean():.1f}, median={np.median(pooled):.1f}")

    medians = np.array([s['median'] for s in stats])
    means   = np.array([s['mean'] for s in stats])
    p5s     = np.array([s['p5'] for s in stats])
    p95s    = np.array([s['p95'] for s in stats])
    mins    = np.array([s['min'] for s in stats])
    maxs    = np.array([s['max'] for s in stats])

    # ===== Figure 1: Pooled histogram (log-x) =====
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(pooled, bins=200, range=(0, 20000), color='#3498DB',
             edgecolor='none', alpha=0.85)
    ax1.axvline(np.median(pooled), color='red', lw=2, ls='--',
                label=f'Median = {np.median(pooled):.0f} MPa')
    ax1.axvline(pooled.mean(), color='orange', lw=2, ls='--',
                label=f'Mean = {pooled.mean():.0f} MPa')
    ax1.set_xlabel('Elastic Modulus E (MPa)', fontsize=20)
    ax1.set_ylabel('Element count', fontsize=20)
    ax1.set_title('Distribution of Element E (all samples pooled)', fontsize=22)
    ax1.legend(fontsize=16)
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=16)
    fig1.tight_layout()
    fig1.savefig(out_dir / 'E_histogram.png', dpi=150)
    plt.close(fig1)
    print(f"[INFO] Saved E_histogram.png")

    # ===== Figure 2: Box plot of per-sample median E =====
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    bp = ax2.boxplot([medians, means, p5s, p95s],
                     labels=['Median', 'Mean', 'P5', 'P95'],
                     patch_artist=True, widths=0.5,
                     boxprops=dict(facecolor='#AED6F1', linewidth=2),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2))
    colors = ['#3498DB', '#E67E22', '#2ECC71', '#E74C3C']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel('E (MPa)', fontsize=20)
    ax2.set_title('Per-sample E statistics (N={})'.format(len(stats)), fontsize=22)
    ax2.grid(alpha=0.3, axis='y')
    ax2.tick_params(labelsize=16)
    fig2.tight_layout()
    fig2.savefig(out_dir / 'E_boxplot.png', dpi=150)
    plt.close(fig2)
    print(f"[INFO] Saved E_boxplot.png")

    # ===== Figure 3: CDF (percentile curve) =====
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sorted_E = np.sort(pooled)
    pct = np.linspace(0, 100, len(sorted_E))
    # Subsample for plot speed
    step = max(1, len(sorted_E) // 5000)
    ax3.plot(sorted_E[::step], pct[::step], '-', color='#3498DB', lw=2.5)
    # Mark key percentiles
    for p_val in [5, 25, 50, 75, 95]:
        e_val = float(np.percentile(pooled, p_val))
        ax3.axhline(p_val, color='gray', lw=0.8, ls=':', alpha=0.5)
        ax3.axvline(e_val, color='gray', lw=0.8, ls=':', alpha=0.5)
        ax3.plot(e_val, p_val, 'o', color='#E74C3C', markersize=10, zorder=5)
        ax3.text(e_val + 200, p_val + 2,
                 f'P{p_val}={e_val:.0f}', fontsize=14, color='#E74C3C')
    ax3.set_xlabel('Elastic Modulus E (MPa)', fontsize=20)
    ax3.set_ylabel('Cumulative %', fontsize=20)
    ax3.set_title('Cumulative Distribution of Element E', fontsize=22)
    ax3.grid(alpha=0.3)
    ax3.tick_params(labelsize=16)
    fig3.tight_layout()
    fig3.savefig(out_dir / 'E_cdf.png', dpi=150)
    plt.close(fig3)
    print(f"[INFO] Saved E_cdf.png")

    # ===== Figure 4: Per-sample range (sorted by median) =====
    order = np.argsort(medians)
    fig4, ax4 = plt.subplots(figsize=(14, 6))
    x = np.arange(len(stats))
    ax4.fill_between(x, p5s[order], p95s[order], alpha=0.3,
                     color='#3498DB', label='P5–P95 range')
    ax4.plot(x, medians[order], '-', color='#E74C3C', lw=2,
             label='Median E')
    ax4.set_xlabel('Sample index (sorted by median E)', fontsize=20)
    ax4.set_ylabel('E (MPa)', fontsize=20)
    ax4.set_title(f'Per-sample E range (N={len(stats)})', fontsize=22)
    ax4.legend(fontsize=16, loc='upper left')
    ax4.grid(alpha=0.3)
    ax4.tick_params(labelsize=16)
    fig4.tight_layout()
    fig4.savefig(out_dir / 'E_per_sample.png', dpi=150)
    plt.close(fig4)
    print(f"[INFO] Saved E_per_sample.png")

    # ===== Summary stats =====
    print("\n===== SUMMARY =====")
    print(f"Samples:       {len(stats)}")
    print(f"Total elements (E>0): {len(pooled):,}")
    print(f"Pooled E range:       [{pooled.min():.1f}, {pooled.max():.1f}] MPa")
    print(f"Pooled mean / median: {pooled.mean():.1f} / {np.median(pooled):.1f} MPa")
    print(f"Pooled P5 / P95:      {np.percentile(pooled,5):.1f} / {np.percentile(pooled,95):.1f} MPa")
    print(f"Per-sample median E:  [{medians.min():.1f}, {medians.max():.1f}] MPa  "
          f"(mean={medians.mean():.1f})")
    print(f"Per-sample E>0 count: [{min(s['n_pos'] for s in stats):,}, "
          f"{max(s['n_pos'] for s in stats):,}]")
    n_zero_total = sum(s['n_zero'] for s in stats)
    print(f"Total E==0 elements:  {n_zero_total}")


if __name__ == '__main__':
    main()
