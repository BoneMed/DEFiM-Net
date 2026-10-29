#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental pre-cut: process ONLY cases missing from train/val patch dirs.

Rules:
- Existing train/val patch dirs are the source of truth for split.
  Cases already cut into either dir are left untouched.
- Missing cases are split into train/val preserving the CURRENT val ratio
  (n_val / (n_train + n_val)). Sampling is deterministic (seed).
- train overlap = 0.3, val overlap = 0.3 (same as existing).
"""
import argparse
import random
import time
from pathlib import Path

import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from DataLoader import HybridDataset
from precut_patches import get_patch_positions, save_patches


def case_names_in(out_dir):
    return set(f.stem.rsplit('_p', 1)[0] for f in Path(out_dir).glob('*.pt'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",      default="../../Data/ElementGraph_Tumor_Train")
    ap.add_argument("--train_out_dir", default="../../Data/Patches_Tumor_Train_256_128")
    ap.add_argument("--val_out_dir",   default="../../Data/Patches_Tumor_Val_256_128")
    ap.add_argument("--grid_res",      type=int, default=256)
    ap.add_argument("--patch_size",    type=int, default=128)
    ap.add_argument("--train_overlap", type=float, default=0.3)
    ap.add_argument("--val_overlap",   type=float, default=0.3)
    ap.add_argument("--s3_splat",      default="mean", choices=["mean", "min"])
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()

    R, P = args.grid_res, args.patch_size
    train_out = Path(args.train_out_dir)
    val_out   = Path(args.val_out_dir)
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    train_done = case_names_in(train_out)
    val_done   = case_names_in(val_out)
    n_train_done = len(train_done)
    n_val_done   = len(val_done)
    total_done = n_train_done + n_val_done
    val_ratio = n_val_done / max(total_done, 1)
    print(f"[INFO] Existing train cases: {n_train_done}")
    print(f"[INFO] Existing val   cases: {n_val_done}")
    print(f"[INFO] Current val ratio:    {val_ratio:.4f}")

    ds = HybridDataset(args.data_dir, grid_res=R, s3_splat=args.s3_splat)
    all_names = ds.names
    print(f"[INFO] Total cases in ElementGraph dir: {len(all_names)}")

    missing = [i for i, n in enumerate(all_names)
               if n not in train_done and n not in val_done]
    print(f"[INFO] Missing cases: {len(missing)}")

    if not missing:
        print("Nothing to do.")
        return

    # Preserve current val ratio for new cases
    n_new_val = round(len(missing) * val_ratio)
    rng = random.Random(args.seed)
    new_val_idx = sorted(rng.sample(missing, n_new_val))
    new_val_set = set(new_val_idx)
    new_train_idx = [i for i in missing if i not in new_val_set]
    print(f"[INFO] New split: {len(new_train_idx)} train + {n_new_val} val")

    train_positions = get_patch_positions(R, P, args.train_overlap)
    val_positions   = get_patch_positions(R, P, args.val_overlap)
    print(f"[INFO] Patches/case: train={len(train_positions)} "
          f"(overlap={args.train_overlap}), val={len(val_positions)} "
          f"(overlap={args.val_overlap})")

    def process(indices, positions, out_dir, label):
        t0 = time.time()
        for rank, i in enumerate(indices):
            name = all_names[i]
            t_case = time.time()
            sample = ds[i]
            save_patches(sample, name, positions, P, out_dir)
            elapsed = time.time() - t_case
            total = time.time() - t0
            rate = (rank + 1) / max(total, 1e-3)
            eta = (len(indices) - rank - 1) / max(rate, 1e-3)
            print(f"  [{label}] [{rank+1}/{len(indices)}] {name}: "
                  f"{elapsed:.1f}s | ETA {eta/60:.0f}min", flush=True)
        print(f"  [{label}] Done. {len(indices)} cases in {time.time()-t0:.0f}s")

    if new_train_idx:
        process(new_train_idx, train_positions, train_out, "TRAIN")
    if new_val_idx:
        process(new_val_idx,   val_positions,   val_out,   "VAL")

    print(f"\nAll done.")
    print(f"  Train patches -> {train_out} (added {len(new_train_idx)})")
    print(f"  Val   patches -> {val_out}   (added {n_new_val})")


if __name__ == "__main__":
    main()
