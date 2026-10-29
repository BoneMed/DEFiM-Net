#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-compute the 3D Hilbert curve permutation on an R^3 voxel grid.

Output: hilbert_R{R}.npy of shape (R, R, R), int32, where
    hilbert_R[i, j, k] = position of voxel (i, j, k) on the Hilbert curve
                         (in [0, R^3)).

The HilbertScanMamba block uses argsort over this field (with air voxels
pushed to the tail via bone-mask) to obtain a 1D scan order that is
locality-preserving in 3D — adjacent positions on the curve are nearly
always adjacent in 3D, regardless of which 3 axes you look along.

Algorithm: Skilling 2004, "Programming the Hilbert curve",
vectorised in numpy. Generates R=128 (16M points) in ~5 seconds.
"""

import argparse
import time
from pathlib import Path

import numpy as np


def coords_to_hilbert(X, p):
    """Convert (N, n) integer coordinates to Hilbert distances.

    Args
    ----
    X : (N, n) int array, each value in [0, 2**p)
    p : int, number of bits per coordinate (curve order)

    Returns
    -------
    H : (N,) int64 array, Hilbert distance in [0, 2**(n*p))
    """
    X = X.astype(np.int64).copy()
    n = X.shape[1]
    M = 1 << (p - 1)

    # Step 1: inverse undo (axis transposition based on top bits)
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(n):
            mask = (X[:, i] & Q) != 0
            X[mask, 0] ^= P
            t = (X[~mask, 0] ^ X[~mask, i]) & P
            X[~mask, 0] ^= t
            X[~mask, i] ^= t
        Q >>= 1

    # Step 2: Gray-encode
    for i in range(1, n):
        X[:, i] ^= X[:, i - 1]
    t = np.zeros(X.shape[0], dtype=np.int64)
    Q = M
    while Q > 1:
        mask = (X[:, n - 1] & Q) != 0
        t[mask] ^= Q - 1
        Q >>= 1
    for i in range(n):
        X[:, i] ^= t

    # Step 3: pack the n*p transposed bits into a single integer
    H = np.zeros(X.shape[0], dtype=np.int64)
    for j in range(p):
        for i in range(n):
            bit = (X[:, i] >> (p - 1 - j)) & 1
            H = (H << 1) | bit
    return H


def build_hilbert_table(R):
    """Return (R, R, R) int32 with hilbert_index for every voxel."""
    p = int(np.log2(R))
    if (1 << p) != R:
        raise ValueError(f"R must be a power of 2, got R={R}")

    zz, yy, xx = np.meshgrid(np.arange(R), np.arange(R), np.arange(R),
                              indexing='ij')
    coords = np.stack([zz.ravel(), yy.ravel(), xx.ravel()],
                      axis=1)                      # (R^3, 3)
    H = coords_to_hilbert(coords, p)               # (R^3,)

    if H.max() >= 2**31:
        raise OverflowError(f"Hilbert max {H.max()} > int32 range; "
                            f"use int64 storage.")
    table = H.reshape(R, R, R).astype(np.int32)
    return table


def verify(table, R):
    """Sanity-check the table is a valid permutation and locality-preserving."""
    flat = table.ravel()
    assert flat.min() == 0 and flat.max() == R**3 - 1
    assert len(np.unique(flat)) == R**3, "not a permutation"

    # Adjacent distances on the curve should be 1 voxel in 3D
    inv = np.empty(R**3, dtype=np.int64)
    coords = np.stack(np.unravel_index(np.arange(R**3), (R, R, R)), axis=1)
    inv[flat] = np.arange(R**3)
    # inv[d] = flat-index where Hilbert position d sits; recover (z,y,x)
    zyx_at_d = coords[inv]
    diffs = np.abs(np.diff(zyx_at_d, axis=0)).sum(axis=1)
    assert (diffs == 1).all(), \
        f"Hilbert step is not 1-voxel: max step = {diffs.max()}"
    print(f"[OK] hilbert_R{R} is a valid Hilbert permutation "
          f"(all R^3-1 = {R**3-1} steps are 1-voxel jumps).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=128,
                    help="Grid resolution (must be power of 2). "
                         "128 matches the patch size used by HybridDataset.")
    ap.add_argument("--out_dir", default=str(Path(__file__).parent),
                    help="Where to save hilbert_R{R}.npy. Default: this file's "
                         "directory, so the model can find it via a relative "
                         "path.")
    ap.add_argument("--no_verify", action="store_true",
                    help="Skip the slow permutation/locality check.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hilbert_R{args.R}.npy"

    t0 = time.time()
    table = build_hilbert_table(args.R)
    print(f"[INFO] built hilbert_R{args.R} in {time.time()-t0:.1f}s, "
          f"shape={table.shape}, dtype={table.dtype}, "
          f"size={table.nbytes/1024/1024:.1f} MB")

    if not args.no_verify:
        verify(table, args.R)

    np.save(out_path, table)
    print(f"[INFO] saved -> {out_path}")


if __name__ == "__main__":
    main()
