#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-cut existing train + val patches at a new overlap ratio.

Preserves the current train/val case assignment (read from existing
patch directories), wipes all existing .pt files, then re-splats with
the new overlap.

Default: overlap=0.5 (stride = 64 for patch_size=128).
Patches/case stays at 27 for R=256, P=128.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from DataLoader import HybridDataset
from precut_patches import get_patch_positions, save_patches


def case_names_in(out_dir):
    return set(f.stem.rsplit('_p', 1)[0] for f in Path(out_dir).glob('*.pt'))


def wipe(out_dir):
    out_dir = Path(out_dir)
    n = 0
    for f in out_dir.glob('*.pt'):
        f.unlink()
        n += 1
    print(f"[INFO] Wiped {n} files from {out_dir}")


def process(ds, keep_names, positions, P, out_dir, label):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    indices = [i for i, n in enumerate(ds.names) if n in keep_names]
    missing = keep_names - set(ds.names[i] for i in indices)
    if missing:
        print(f"[WARN] {len(missing)} recorded {label} case(s) missing from "
              f"ElementGraph dir (will be skipped)")

    n_patches = len(positions)
    t0 = time.time()
    n_done = 0
    for rank, i in enumerate(indices):
        name = ds.names[i]
        # Skip cases whose last patch already exists (resumable)
        last = out_dir / f"{name}_p{n_patches-1}.pt"
        if last.exists():
            continue
        t_case = time.time()
        sample = ds[i]
        save_patches(sample, name, positions, P, out_dir)
        n_done += 1
        elapsed = time.time() - t_case
        total = time.time() - t0
        rate = n_done / max(total, 1e-3)
        eta = (len(indices) - rank - 1) / max(rate, 1e-3)
        print(f"  [{label}] [{rank+1}/{len(indices)}] {name}: "
              f"{elapsed:.1f}s | ETA {eta/60:.0f}min", flush=True)

    print(f"  [{label}] Done. {len(indices)} cases / {len(indices)*len(positions)} "
          f"patches in {(time.time()-t0)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../../Data/ElementGraph_Tumor_Train")
    ap.add_argument("--train_out_dir", default="../../Data/Patches_Tumor_Train_256_128")
    ap.add_argument("--val_out_dir",   default="../../Data/Patches_Tumor_Val_256_128")
    ap.add_argument("--grid_res",   type=int,   default=256)
    ap.add_argument("--patch_size", type=int,   default=128)
    ap.add_argument("--overlap",    type=float, default=0.5,
                    help="Overlap ratio (0.5 -> stride 64 for P=128)")
    ap.add_argument("--s3_splat",   default="mean", choices=["mean", "min"])
    ap.add_argument("--skip_wipe",  action="store_true",
                    help="Do not delete existing .pt (for debug)")
    args = ap.parse_args()

    # Record current split from existing patch files
    train_cases = case_names_in(args.train_out_dir)
    val_cases   = case_names_in(args.val_out_dir)
    overlap_cases = train_cases & val_cases
    if overlap_cases:
        print(f"[FATAL] {len(overlap_cases)} cases appear in BOTH train and val!")
        for n in sorted(overlap_cases)[:5]:
            print(f"  {n}")
        return

    # If val dir was wiped, derive val_cases from (all elem cases) - train_cases
    # so we don't lose the original assignment just because val got emptied.
    ds_all_names = set(f.stem for f in Path(args.data_dir).glob('*.npz'))
    missing_val = (ds_all_names - train_cases) - val_cases
    if missing_val:
        print(f"[INFO] {len(missing_val)} case(s) present in data_dir but not in "
              f"train/val patch dirs. Adding them to val_cases.")
        val_cases = val_cases | missing_val

    print(f"[INFO] Recorded assignment: "
          f"TRAIN={len(train_cases)}, VAL={len(val_cases)}")

    stride = int(args.patch_size * (1 - args.overlap))
    print(f"[INFO] New overlap={args.overlap} -> stride={stride}")

    # Wipe existing .pt
    if not args.skip_wipe:
        wipe(args.train_out_dir)
        wipe(args.val_out_dir)

    # Build patch positions
    R, P = args.grid_res, args.patch_size
    positions = get_patch_positions(R, P, args.overlap)
    print(f"[INFO] Patches/case: {len(positions)}")

    # Re-voxelize + save
    ds = HybridDataset(args.data_dir, grid_res=R, s3_splat=args.s3_splat)
    print(f"[INFO] Source ElementGraph cases: {len(ds.names)}")

    process(ds, train_cases, positions, P, args.train_out_dir, "TRAIN")
    process(ds, val_cases,   positions, P, args.val_out_dir,   "VAL")

    print("\nAll done.")


if __name__ == "__main__":
    main()
