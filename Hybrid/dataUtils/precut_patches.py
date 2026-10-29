#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline pre-cut patches with case-level train/val split.

Train cases (default 115): overlap 0.3  -> --train_out_dir
Val   cases (default  12): overlap 0.0  -> --val_out_dir  (same as test)

Usage:
  python precut_patches.py \
    --data_dir  ../Data/ElementGraph_Tumor_Train \
    --train_out_dir ../Data/Patches_Tumor_Train_256_128 \
    --val_out_dir   ../Data/Patches_Tumor_Val_256_128 \
    --grid_res 256 --patch_size 128 --val_cases 12 --seed 42
"""

import random
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from DataLoader import HybridDataset


def get_patch_positions(R, P, overlap):
    stride = max(1, int(P * (1 - overlap)))
    starts = list(range(0, R - P + 1, stride))
    if starts[-1] + P < R:
        starts.append(R - P)
    return [(sz, sy, sx) for sz in starts for sy in starts for sx in starts]


def save_patches(sample, name, patch_positions, P, out_dir):
    for pi, (sz, sy, sx) in enumerate(patch_positions):
        sl = (slice(sz, sz+P), slice(sy, sy+P), slice(sx, sx+P))
        patch = {k: sample[k][:, sl[0], sl[1], sl[2]] for k in
                 ('geo_input', 'phys_input', 'S3_grid', 'mask')}
        torch.save({k: v.half() for k, v in patch.items()},
                   out_dir / f"{name}_p{pi}.pt")
    return len(patch_positions)


def process_cases(ds, indices, patch_positions, P, out_dir, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_patches = len(patch_positions)
    n_saved = n_skipped = 0
    t_start = time.time()

    for rank, i in enumerate(indices):
        name = ds.names[i]
        last = out_dir / f"{name}_p{n_patches-1}.pt"
        if last.exists():
            n_skipped += 1
            continue

        t0 = time.time()
        sample = ds[i]
        save_patches(sample, name, patch_positions, P, out_dir)
        n_saved += 1

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        rate = n_saved / max(total_elapsed, 1e-3)
        remaining = (len(indices) - rank - 1) / max(rate, 1e-3)
        print(f"  [{label}] [{rank+1}/{len(indices)}] {name}: "
              f"{elapsed:.1f}s | ETA {remaining/60:.0f}min")

    print(f"  [{label}] Done. saved={n_saved}, skipped={n_skipped}, "
          f"patches/case={n_patches}, total={time.time()-t_start:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",      default="../Data/ElementGraph_Tumor_Train")
    ap.add_argument("--train_out_dir", default="../Data/Patches_Tumor_Train_256_128")
    ap.add_argument("--val_out_dir",   default="../Data/Patches_Tumor_Val_256_128")
    ap.add_argument("--grid_res",   type=int,   default=256)
    ap.add_argument("--patch_size", type=int,   default=128)
    ap.add_argument("--train_overlap", type=float, default=0.3)
    ap.add_argument("--val_overlap",   type=float, default=0.0)
    ap.add_argument("--val_cases", type=int, default=12,
                    help="Number of cases held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--s3_splat", default="mean", choices=["mean", "min"])
    ap.add_argument("--only_val", action="store_true",
                    help="Only generate val patches (skip train, useful when train patches already exist)")
    args = ap.parse_args()

    R, P = args.grid_res, args.patch_size

    ds = HybridDataset(args.data_dir, grid_res=R, s3_splat=args.s3_splat)
    all_indices = list(range(len(ds)))
    print(f"[INFO] Total cases: {len(all_indices)}")

    rng = random.Random(args.seed)
    val_indices = sorted(rng.sample(all_indices, args.val_cases))
    val_set = set(val_indices)
    train_indices = [i for i in all_indices if i not in val_set]
    print(f"[INFO] Train: {len(train_indices)} cases, Val: {len(val_indices)} cases")
    print(f"[INFO] Val case names: {[ds.names[i] for i in val_indices]}")

    val_positions = get_patch_positions(R, P, args.val_overlap)
    print(f"[INFO] Val patches/case: {len(val_positions)} (overlap={args.val_overlap})")

    if not args.only_val:
        train_positions = get_patch_positions(R, P, args.train_overlap)
        print(f"[INFO] Train patches/case: {len(train_positions)} (overlap={args.train_overlap})")
        process_cases(ds, train_indices, train_positions, P,
                      Path(args.train_out_dir), "TRAIN")

    process_cases(ds, val_indices, val_positions, P,
                  Path(args.val_out_dir), "VAL")

    print("\nAll done.")
    print(f"  Train patches -> {args.train_out_dir}")
    print(f"  Val   patches -> {args.val_out_dir}")


if __name__ == "__main__":
    main()
