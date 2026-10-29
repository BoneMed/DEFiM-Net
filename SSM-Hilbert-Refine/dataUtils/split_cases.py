#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a train/val split JSON from a merged patch pool directory.

Reads all unique case names from a patch dir (case = stem minus `_pN`),
deterministically splits by case (not by patch) at the requested ratio,
and writes a manifest JSON consumed by the training scripts.

Why a JSON manifest:
  - The split is decoupled from directory layout —no need to move files
    when you change ratios or swap cases between train/val.
  - Reproducible: same seed -> same split.
  - Easy to version-control alongside experiments.

Usage:
  python split_cases.py \
    --pool_dir ../../Data/Patches_Tumor_Train_256_128 \
    --out      ../../Data/tumor_split.json \
    --val_ratio 0.2 --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def case_from_patch(pt_path):
    return pt_path.stem.rsplit('_p', 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dir", required=True,
                    help="Directory containing all train+val patches.")
    ap.add_argument("--out", required=True,
                    help="Output JSON path.")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool_dir = Path(args.pool_dir)
    files = sorted(pool_dir.glob("*.pt"))
    if not files:
        raise SystemExit(f"[ERR] no .pt files in {pool_dir}")

    cases_map = defaultdict(list)
    for f in files:
        cases_map[case_from_patch(f)].append(f.name)
    cases = sorted(cases_map.keys())
    n_total = len(cases)
    n_val = max(1, round(n_total * args.val_ratio))

    rng = random.Random(args.seed)
    shuffled = cases.copy()
    rng.shuffle(shuffled)
    val_cases = sorted(shuffled[:n_val])
    train_cases = sorted(shuffled[n_val:])

    # Sanity: patches per case distribution
    patch_counts = [len(cases_map[c]) for c in cases]
    min_p, max_p = min(patch_counts), max(patch_counts)
    total_train_patches = sum(len(cases_map[c]) for c in train_cases)
    total_val_patches = sum(len(cases_map[c]) for c in val_cases)

    manifest = {
        "pool_dir": str(pool_dir),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "n_total_cases": n_total,
        "n_train_cases": len(train_cases),
        "n_val_cases": len(val_cases),
        "n_train_patches": total_train_patches,
        "n_val_patches": total_val_patches,
        "patches_per_case_min": min_p,
        "patches_per_case_max": max_p,
        "train_cases": train_cases,
        "val_cases": val_cases,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Pool dir: {pool_dir}")
    print(f"[INFO] Total cases: {n_total}  (patches/case: {min_p}..{max_p})")
    print(f"[INFO] Train: {len(train_cases)} cases, {total_train_patches} patches")
    print(f"[INFO] Val:   {len(val_cases)} cases, {total_val_patches} patches")
    print(f"[INFO] Split ratio: {len(val_cases)/n_total:.3f} (target {args.val_ratio})")
    print(f"[INFO] Written: {out_path}")


if __name__ == "__main__":
    main()
