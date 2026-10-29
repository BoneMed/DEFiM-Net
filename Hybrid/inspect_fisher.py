#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect a Fisher Information snapshot for sanity / debugging."""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fisher", default="../Model/DualEncoder_Healthy/fisher.pt")
    args = ap.parse_args()

    blob = torch.load(args.fisher, map_location='cpu', weights_only=False)
    F = blob['fisher']
    theta = blob['theta_healthy']
    meta = blob.get('meta', {})

    print(f"[META] {meta}")
    print(f"[INFO] {len(F)} parameter tensors, "
          f"total {sum(t.numel() for t in F.values()):,} scalars")
    print()

    # ---- Basic sanity ----
    bad_neg = [n for n, t in F.items() if (t < 0).any()]
    bad_nan = [n for n, t in F.items() if torch.isnan(t).any() or torch.isinf(t).any()]
    if bad_neg:
        print(f"[FAIL] Negative entries in: {bad_neg[:5]}")
    if bad_nan:
        print(f"[FAIL] NaN/Inf entries in: {bad_nan[:5]}")
    if not bad_neg and not bad_nan:
        print("[OK] Non-negative, no NaN/Inf")
    print()

    # ---- Group by module name pattern ----
    groups = {
        'sparse_encoder':    re.compile(r'^sparse_encoder\.'),
        'stem (geo)':        re.compile(r'^stem_g\.'),
        'encoder (geo)':     re.compile(r'^e[1-3]_g\.'),
        'encoder (phys)':    re.compile(r'^e[1-3]_p\.'),
        'FiLM':              re.compile(r'^film\d\.'),
        'skip':              re.compile(r'^s[0-2]\.'),
        'decoder':           re.compile(r'^u[0-2]\.'),
        'head + DS':         re.compile(r'^(head|ds[12])\.'),
    }

    grp_stats = defaultdict(lambda: {'sum': 0.0, 'count': 0, 'max': 0.0, 'params': 0})
    for n, t in F.items():
        for gname, pat in groups.items():
            if pat.match(n):
                arr = t.numpy().ravel()
                grp_stats[gname]['sum']   += arr.sum()
                grp_stats[gname]['count'] += arr.size
                grp_stats[gname]['max']    = max(grp_stats[gname]['max'], arr.max())
                grp_stats[gname]['params'] += t.numel()
                break

    # ---- Print per-group ----
    print(f"{'group':24s}  {'mean Fisher':>14s}  {'max Fisher':>14s}  {'params':>10s}")
    print("-" * 70)
    for g in groups:
        s = grp_stats[g]
        if s['count'] == 0:
            continue
        mean = s['sum'] / s['count']
        print(f"{g:24s}  {mean:14.3e}  {s['max']:14.3e}  {s['params']:>10,}")
    print()

    # ---- Top 10 highest-Fisher params ----
    pmean = sorted([(n, t.mean().item()) for n, t in F.items()],
                   key=lambda x: -x[1])
    print("Top 10 most-important params (highest Fisher mean):")
    for n, m in pmean[:10]:
        print(f"  {m:11.3e}   {n}")
    print()

    # ---- Bottom 10 (least important) ----
    print("Bottom 10 least-important params (lowest Fisher mean):")
    for n, m in pmean[-10:]:
        print(f"  {m:11.3e}   {n}")
    print()

    # ---- Theta sanity: not all zero, no NaN ----
    th_norms = [t.norm().item() for t in theta.values()]
    print(f"theta_healthy norms: min={min(th_norms):.3f}  median={np.median(th_norms):.3f}  max={max(th_norms):.3f}")
    if any(np.isnan(n) or np.isinf(n) for n in th_norms):
        print("[FAIL] NaN/Inf in theta_healthy")
    else:
        print("[OK] theta_healthy snapshot looks valid")


if __name__ == "__main__":
    main()
