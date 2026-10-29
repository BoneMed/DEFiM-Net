#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare train/val feature ranges for pre-cut Hybrid patches.

By default this follows the real training split:
  patch_dir + split_json(train_cases/val_cases)

It reports masked voxel statistics for geo_input, S3_grid in transformed space,
S3 converted back to physical space, mask occupancy, and every phys_input
channel present in the patch files.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PHYS_NAMES = ("BC", "Load*u", "dist_BC", "dist_Load")


def case_of(path):
    return path.stem.rsplit("_p", 1)[0]


def inv_signed_sqrt_np(y, scale=10.0):
    return np.sign(y) * (y ** 2) / scale


def summarize_array(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    qs = np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(qs[0]),
        "p01": float(qs[1]),
        "p05": float(qs[2]),
        "p25": float(qs[3]),
        "p50": float(qs[4]),
        "p75": float(qs[5]),
        "p95": float(qs[6]),
        "p99": float(qs[7]),
        "max": float(qs[8]),
    }


def print_stats(name, train_stats, val_stats):
    keys = ("mean", "std", "min", "p01", "p05", "p50", "p95", "p99", "max")
    print(f"\n[{name}]")
    print(f"  {'metric':<8} {'train':>14} {'val':>14} {'val/train':>12}")
    for k in keys:
        tr = train_stats[k]
        va = val_stats[k]
        ratio = va / tr if abs(tr) > 1e-12 else float("nan")
        print(f"  {k:<8} {tr:>14.6g} {va:>14.6g} {ratio:>12.4g}")


def choose_files(patch_dir, split_json):
    files = sorted(Path(patch_dir).glob("*.pt"))
    by_case = defaultdict(list)
    for f in files:
        by_case[case_of(f)].append(f)

    with open(split_json, "r", encoding="utf-8") as f:
        split = json.load(f)
    train_cases = set(split["train_cases"])
    val_cases = set(split["val_cases"])

    train_files = [f for c in sorted(train_cases) for f in by_case.get(c, [])]
    val_files = [f for c in sorted(val_cases) for f in by_case.get(c, [])]
    return train_files, val_files, train_cases, val_cases


def scan(files, tag, max_patches=0, sample_voxels=2000, seed=42, scale=10.0):
    rng = np.random.RandomState(seed)
    if max_patches and len(files) > max_patches:
        idx = rng.choice(len(files), size=max_patches, replace=False)
        files = [files[i] for i in sorted(idx)]

    buckets = defaultdict(list)
    occ = []
    phys_ch_count = 0
    phys_nonzero_patch = defaultdict(int)
    n_valid_patches = 0

    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        geo = d["geo_input"].float().squeeze(0).numpy()
        phys = d["phys_input"].float().numpy()
        s3 = d["S3_grid"].float().squeeze(0).numpy()
        mask = d["mask"].float().squeeze(0).numpy() > 0.5
        if mask.sum() < 10:
            continue

        n_valid_patches += 1
        occ.append(float(mask.mean()))
        phys_ch_count = max(phys_ch_count, phys.shape[0])

        idx = np.flatnonzero(mask.reshape(-1))
        if sample_voxels and idx.size > sample_voxels:
            idx = rng.choice(idx, size=sample_voxels, replace=False)

        geo_v = geo.reshape(-1)[idx]
        s3_v = s3.reshape(-1)[idx]
        s3_phys = inv_signed_sqrt_np(s3_v, scale=scale)

        buckets["geo_input"].append(geo_v)
        buckets["S3_sqrt"].append(s3_v)
        buckets["S3_phys"].append(s3_phys)

        mask_flat = mask.reshape(-1)
        for c in range(phys.shape[0]):
            ch_all = phys[c].reshape(-1)
            ch_masked_full = ch_all[mask_flat]
            if np.any(np.abs(ch_masked_full) > 1e-8):
                phys_nonzero_patch[c] += 1
            buckets[f"phys[{c}]"].append(ch_all[idx])

    stats = {
        "tag": tag,
        "n_files": len(files),
        "n_valid_patches": n_valid_patches,
        "occ": summarize_array(occ),
        "phys_nonzero_patch": dict(phys_nonzero_patch),
        "phys_ch_count": phys_ch_count,
    }
    for k, xs in buckets.items():
        stats[k] = summarize_array(np.concatenate(xs)) if xs else None
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_dir", default="../Data/Patches_Tumor_Train_256_128")
    ap.add_argument("--split_json", default="../Data/tumor_split.json")
    ap.add_argument("--max_patches", type=int, default=0,
                    help="0 means scan all patches. Use a number for quick sampling.")
    ap.add_argument("--sample_voxels", type=int, default=2000,
                    help="Masked voxels sampled per patch for value distributions.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=10.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    patch_dir = Path(args.patch_dir)
    split_json = Path(args.split_json)
    if not patch_dir.is_absolute():
        patch_dir = (root / patch_dir).resolve()
    if not split_json.is_absolute():
        split_json = (root / split_json).resolve()

    train_files, val_files, train_cases, val_cases = choose_files(patch_dir, split_json)
    print(f"[INFO] patch_dir={patch_dir}")
    print(f"[INFO] split_json={split_json}")
    print(f"[INFO] train cases={len(train_cases)}, files={len(train_files)}")
    print(f"[INFO] val   cases={len(val_cases)}, files={len(val_files)}")
    print(f"[INFO] max_patches={args.max_patches}, sample_voxels={args.sample_voxels}")

    train = scan(train_files, "TRAIN", args.max_patches,
                 args.sample_voxels, args.seed, args.scale)
    val = scan(val_files, "VAL", args.max_patches,
               args.sample_voxels, args.seed + 1, args.scale)

    print("\n[PATCH COUNTS]")
    print(f"  train valid patches: {train['n_valid_patches']}/{train['n_files']}")
    print(f"  val   valid patches: {val['n_valid_patches']}/{val['n_files']}")
    print_stats("mask occupancy", train["occ"], val["occ"])
    print_stats("geo_input", train["geo_input"], val["geo_input"])
    print_stats("S3 signed_sqrt", train["S3_sqrt"], val["S3_sqrt"])
    print_stats("S3 physical", train["S3_phys"], val["S3_phys"])

    max_ch = max(train["phys_ch_count"], val["phys_ch_count"])
    for c in range(max_ch):
        key = f"phys[{c}]"
        name = PHYS_NAMES[c] if c < len(PHYS_NAMES) else f"extra_ch{c}"
        if train.get(key) is not None and val.get(key) is not None:
            print_stats(f"{key} {name}", train[key], val[key])
            print(f"  patches with nonzero {key}: "
                  f"train={train['phys_nonzero_patch'].get(c, 0)}/"
                  f"{train['n_valid_patches']}, "
                  f"val={val['phys_nonzero_patch'].get(c, 0)}/"
                  f"{val['n_valid_patches']}")


if __name__ == "__main__":
    main()
