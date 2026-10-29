#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_E_max_mean_median.py — Per-subject Max / Mean / Median E
from Train_FEM (33 subjects), in one clean figure.

Run:
    python plot_E_max_mean_median.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})


def main():
    root = Path(r"N:\SecondYear\CT_MASK_E_PHANTOMLESS\Train_FEM")
    subjects = sorted([d for d in root.iterdir() if d.is_dir()])

    names, maxs, means, medians, per_subj = [], [], [], [], []
    for subj in subjects:
        csvs = list(subj.glob('*_element_E_*.csv'))
        if not csvs:
            continue
        E = pd.read_csv(csvs[0])['E_MPa'].values.astype(np.float32)
        names.append(subj.name)
        maxs.append(float(E.max()))
        means.append(float(E.mean()))
        medians.append(float(np.median(E)))
        per_subj.append(E)

    maxs = np.array(maxs)
    means = np.array(means)
    medians = np.array(medians)
    N = len(names)

    # Sort by median
    order = np.argsort(medians)
    names_s   = [names[i] for i in order]
    maxs_s    = maxs[order]
    means_s   = means[order]
    medians_s = medians[order]

    # ===== Single violin: pooled E distribution + key stats annotated =====
    pooled = np.concatenate(per_subj)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Violin (only E > 0)
    parts = ax.violinplot([pooled], positions=[0], showmeans=False,
                          showmedians=False, showextrema=False,
                          widths=0.7)
    for pc in parts['bodies']:
        pc.set_facecolor('#3498DB')
        pc.set_edgecolor('#2C3E50')
        pc.set_alpha(0.6)
        pc.set_linewidth(2)

    # Box plot inside violin
    bp = ax.boxplot([pooled], positions=[0], widths=0.15,
                    showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor='white', linewidth=2),
                    medianprops=dict(color='red', linewidth=3),
                    whiskerprops=dict(linewidth=2),
                    capprops=dict(linewidth=2))

    # Annotate key stats
    med = float(np.median(pooled))
    mn  = float(pooled.mean())
    mx  = float(pooled.max())
    p5  = float(np.percentile(pooled, 5))
    p95 = float(np.percentile(pooled, 95))

    stats_lines = [
        (mx,  'Max',    '#E74C3C'),
        (p95, 'P95',    '#E67E22'),
        (mn,  'Mean',   '#F39C12'),
        (med, 'Median', '#E74C3C'),
        (p5,  'P5',     '#2ECC71'),
    ]
    for val, label, color in stats_lines:
        ax.axhline(val, color=color, lw=1.5, ls='--', alpha=0.7, xmin=0.55)
        ax.text(0.62, val, f'  {label} = {val:.0f} MPa',
                fontsize=18, fontweight='bold', color=color,
                va='center', transform=ax.get_yaxis_transform())

    ax.set_xlim(-0.6, 1.5)
    ax.set_xticks([0])
    ax.set_xticklabels([f'All subjects (N={N})'],
                        fontsize=18)
    ax.set_ylabel('E (MPa)', fontsize=22)
    ax.set_title('Distribution of Element Elastic Modulus', fontsize=24,
                 fontweight='bold')
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    out = 'E_max_mean_median.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out}")

    # Print table
    print(f"\n{'Subject':>8}  {'Max E':>10}  {'Mean E':>10}  {'Median E':>10}")
    print("-" * 45)
    for i in order:
        print(f"{names[i]:>8}  {maxs[i]:>10.1f}  {means[i]:>10.1f}  {medians[i]:>10.1f}")
    print("-" * 45)
    print(f"{'MEAN':>8}  {maxs.mean():>10.1f}  {means.mean():>10.1f}  {medians.mean():>10.1f}")
    print(f"{'STD':>8}  {maxs.std():>10.1f}  {means.std():>10.1f}  {medians.std():>10.1f}")


if __name__ == '__main__':
    main()
