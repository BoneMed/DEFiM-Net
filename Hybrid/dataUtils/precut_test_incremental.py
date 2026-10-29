#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental pre-cut for TEST set. Uses overlap=0 (8 patches/case).

Skips cases already present in the test patch dir; only cuts missing ones.
"""
import argparse
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from DataLoader import HybridDataset
from precut_patches import get_patch_positions, save_patches


def case_names_in(out_dir):
    return set(f.stem.rsplit('_p', 1)[0] for f in Path(out_dir).glob('*.pt'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../../Data/ElementGraph_Tumor_Test")
    ap.add_argument("--out_dir",  default="../../Data/Patches_Tumor_Test_256_128")
    ap.add_argument("--grid_res",   type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--overlap",    type=float, default=0.0)
    ap.add_argument("--s3_splat", default="mean", choices=["mean", "min"])
    args = ap.parse_args()

    R, P = args.grid_res, args.patch_size
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    done = case_names_in(out)
    ds = HybridDataset(args.data_dir, grid_res=R, s3_splat=args.s3_splat)
    missing = [i for i, n in enumerate(ds.names) if n not in done]
    print(f"[INFO] Existing test cases: {len(done)}")
    print(f"[INFO] Total in ElementGraph: {len(ds.names)}")
    print(f"[INFO] Missing: {len(missing)}")

    if not missing:
        print("Nothing to do.")
        return

    positions = get_patch_positions(R, P, args.overlap)
    print(f"[INFO] Patches/case: {len(positions)} (overlap={args.overlap})")

    t0 = time.time()
    for rank, i in enumerate(missing):
        name = ds.names[i]
        t_case = time.time()
        sample = ds[i]
        save_patches(sample, name, positions, P, out)
        elapsed = time.time() - t_case
        total = time.time() - t0
        rate = (rank + 1) / max(total, 1e-3)
        eta = (len(missing) - rank - 1) / max(rate, 1e-3)
        print(f"  [{rank+1}/{len(missing)}] {name}: {elapsed:.1f}s | "
              f"ETA {eta/60:.0f}min", flush=True)

    print(f"\nAll done in {(time.time()-t0)/60:.1f} min. "
          f"Added {len(missing)} cases to {out}")


if __name__ == "__main__":
    main()
