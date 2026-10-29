#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_incremental.py — Incremental extraction of new RST files.

Reads existing tumor_split.json, finds RST files NOT yet in the split,
assigns them to train/test (90/10), appends to json, and extracts npz
only for the new ones.

Existing npz are never touched.

Usage:
    C:/Users/Administrator/anaconda3/envs/pyansys/python.exe extract_incremental.py
    C:/Users/Administrator/anaconda3/envs/pyansys/python.exe extract_incremental.py \
        --rst_root I:/Metastasis_Rob
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path

from extract_rst_to_npz import extract_one_rst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rst_root", default=r"I:\Metastasis_Rob")
    ap.add_argument("--out_train",
                    default=r"H:\pythonproject\PHD_PROGRAM\GCN_Stress\Data\ElementGraph_Tumor_Train")
    ap.add_argument("--out_test",
                    default=r"H:\pythonproject\PHD_PROGRAM\GCN_Stress\Data\ElementGraph_Tumor_Test")
    ap.add_argument("--split_json",
                    default=r"H:\pythonproject\PHD_PROGRAM\GCN_Stress\Data\tumor_split.json")
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rst_root = Path(args.rst_root)
    out_train = Path(args.out_train)
    out_test = Path(args.out_test)
    split_json = Path(args.split_json)
    out_train.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    # ---- Load existing split ----
    if split_json.exists():
        with open(split_json) as f:
            existing = json.load(f)
        existing_names = set(existing['train'] + existing['test'])
        print(f"[INFO] Existing split: {len(existing['train'])} train + "
              f"{len(existing['test'])} test = {len(existing_names)} total")
    else:
        existing = {'train': [], 'test': []}
        existing_names = set()
        print("[INFO] No existing split found, starting fresh")

    # ---- Scan RST files ----
    rst_files = sorted(rst_root.rglob("*.rst"))
    print(f"[INFO] Found {len(rst_files)} RST files in {rst_root}")

    rst_info = []
    for f in rst_files:
        parts = f.relative_to(rst_root).parts
        name = '_'.join(parts[:-1]).replace('sphere_', 'sp')
        rst_info.append({'path': f, 'name': name})

    # ---- Find NEW RSTs (not in existing split) ----
    new_info = [r for r in rst_info if r['name'] not in existing_names]
    print(f"[INFO] New RSTs not in split: {len(new_info)}")

    if len(new_info) == 0:
        print("[INFO] Nothing new to extract. Done.")
        return

    # ---- Split new ones 90/10 ----
    rng = np.random.RandomState(args.seed + len(existing_names))  # different seed so no collision
    indices = list(range(len(new_info)))
    rng.shuffle(indices)
    n_test = max(1, int(len(indices) * args.test_ratio))
    new_test_idx = set(indices[:n_test])

    new_train_names = []
    new_test_names = []
    for i, info in enumerate(new_info):
        if i in new_test_idx:
            new_test_names.append(info['name'])
        else:
            new_train_names.append(info['name'])

    print(f"[INFO] New split: {len(new_train_names)} train + "
          f"{len(new_test_names)} test")

    # ---- Extract npz for new ones ----
    failed_log = out_train.parent / 'failed_samples.txt'

    def _record_failure(name, split_tag, rst_path, reason):
        with open(failed_log, 'a', encoding='utf-8') as fh:
            fh.write(f"{name}\t{split_tag}\t{rst_path}\t{reason}\n")

    n_ok, n_fail = 0, 0
    t_start = time.time()

    for i, info in enumerate(new_info):
        name = info['name']
        is_test = name in new_test_names
        out_dir = out_test if is_test else out_train
        out_path = out_dir / f"{name}.npz"
        split_tag = "TEST" if is_test else "TRAIN"

        if out_path.exists():
            print(f"[SKIP] {name}", flush=True)
            n_ok += 1
            continue

        print(f"\n[{i+1}/{len(new_info)}] {name} ({split_tag})", flush=True)
        try:
            ok = extract_one_rst(info['path'], out_path)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                _record_failure(name, split_tag, info['path'],
                                "extract_one_rst returned False")
        except Exception as e:
            print(f"    [ERROR] {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            n_fail += 1
            _record_failure(name, split_tag, info['path'],
                            f"{type(e).__name__}: {e}")

    elapsed = time.time() - t_start
    print(f"\n[INFO] Extraction done. Success: {n_ok}, Failed: {n_fail}, "
          f"Time: {elapsed:.0f}s")

    # ---- Update split json (append new names) ----
    existing['train'] = sorted(existing['train'] + new_train_names)
    existing['test'] = sorted(existing['test'] + new_test_names)

    with open(split_json, 'w') as f:
        json.dump(existing, f, indent=2)

    print(f"[INFO] Updated {split_json}:")
    print(f"  train: {len(existing['train'])}")
    print(f"  test:  {len(existing['test'])}")
    print(f"  total: {len(existing['train']) + len(existing['test'])}")

    # Print new names for reference
    print(f"\n[INFO] New train samples ({len(new_train_names)}):")
    for n in sorted(new_train_names):
        print(f"  {n}")
    print(f"\n[INFO] New test samples ({len(new_test_names)}):")
    for n in sorted(new_test_names):
        print(f"  {n}")


if __name__ == '__main__':
    main()
