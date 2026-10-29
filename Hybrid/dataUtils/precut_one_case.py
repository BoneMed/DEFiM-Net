#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slice ONE case into patches at arbitrary grid_res / patch_size.

Re-uses the same voxelization pipeline (HybridDataset) and overlap
geometry (get_patch_positions) as the main precut script, but writes
only one case's patches — for quick overfit experiments at different
resolutions without having to regenerate all 424 cases.

Usage:
  python precut_one_case.py \
    --data_dir ../../Data/ElementGraph_Tumor_Train \
    --out_dir  ../../Data/Patches_Overfit_128_64 \
    --case MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp10 \
    --grid_res 128 --patch_size 64 --overlap 0.5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from DataLoader import HybridDataset
from precut_patches import get_patch_positions, save_patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--case",     required=True,
                    help="Case stem (without .npz).")
    ap.add_argument("--grid_res",   type=int,   default=128)
    ap.add_argument("--patch_size", type=int,   default=64)
    ap.add_argument("--overlap",    type=float, default=0.5)
    ap.add_argument("--s3_splat",   default="mean", choices=["mean", "min"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = HybridDataset(args.data_dir, grid_res=args.grid_res,
                       s3_splat=args.s3_splat)
    try:
        idx = ds.names.index(args.case)
    except ValueError:
        print(f"[ERR] case not found in {args.data_dir}: {args.case}")
        print(f"[HINT] first few available: {ds.names[:5]}")
        raise SystemExit(2)

    positions = get_patch_positions(args.grid_res, args.patch_size, args.overlap)
    print(f"[INFO] Patches to cut: {len(positions)}  "
          f"(R={args.grid_res}, P={args.patch_size}, overlap={args.overlap})")

    sample = ds[idx]
    save_patches(sample, args.case, positions, args.patch_size, out_dir)
    print(f"[INFO] Wrote {len(positions)} patches to {out_dir}")


if __name__ == "__main__":
    main()
