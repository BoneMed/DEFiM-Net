#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split healthy ElementGraph npz files into train and test directories.

NPZ stage only splits train/test (matches the tumor convention 90/10).
Validation split is decided later, at the patch-cutting stage, via
`split_cases.py` operating on the train pool.

Incremental-safe:
  - Each invocation looks at npz files in `--src_dir` only.
  - Cases already present in `--train_dir` or `--test_dir` are skipped
    (the previous assignment is preserved).
  - Only NEW cases from this batch are split per the requested ratio.

So:  download a batch -> generate npz into a staging dir ->
     run this script -> staging cases are partitioned, dirs grow.

Usage:
  python split_healthy.py \
    --src_dir   ../../Data/ElementGraph_Healthy_All \
    --train_dir ../../Data/ElementGraph_Healthy_Train \
    --test_dir  ../../Data/ElementGraph_Healthy_Test \
    --test_ratio 0.10 --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir",    required=True,
                    help="Staging dir holding newly-generated *.npz.")
    ap.add_argument("--train_dir",  required=True,
                    help="Destination dir for train+val pool. "
                         "Existing files preserved.")
    ap.add_argument("--test_dir",   required=True,
                    help="Destination dir for test/inference. "
                         "Existing files preserved.")
    ap.add_argument("--test_ratio", type=float, default=0.10,
                    help="Fraction of NEW cases assigned to test "
                         "(default 0.10 = 10%, matches tumor).")
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--copy",  action="store_true",
                    help="Copy instead of move (default: move).")
    args = ap.parse_args()

    src_dir   = Path(args.src_dir)
    train_dir = Path(args.train_dir); train_dir.mkdir(parents=True, exist_ok=True)
    test_dir  = Path(args.test_dir);  test_dir.mkdir(parents=True, exist_ok=True)

    # Existing (already partitioned) cases —never touched.
    existing_train = {f.stem for f in train_dir.glob("*.npz")}
    existing_test  = {f.stem for f in test_dir.glob("*.npz")}
    existing_all   = existing_train | existing_test
    print(f"[INFO] Existing  train={len(existing_train)}  test={len(existing_test)}")

    # New cases in src_dir that are not yet partitioned.
    src_files = sorted(src_dir.glob("*.npz"))
    new_files = [f for f in src_files if f.stem not in existing_all]
    print(f"[INFO] Src dir total npz : {len(src_files)}")
    print(f"[INFO] New (unassigned)  : {len(new_files)}")
    if not new_files:
        print("[DONE] Nothing new to split.")
        return

    # Deterministic split of NEW cases.
    rng = random.Random(args.seed)
    new_names = [f.stem for f in new_files]
    rng.shuffle(new_names)
    n_test = max(1, round(len(new_names) * args.test_ratio))
    new_test  = sorted(new_names[:n_test])
    new_train = sorted(new_names[n_test:])

    op = shutil.copy2 if args.copy else shutil.move
    op_label = 'COPY' if args.copy else 'MOVE'
    for c in new_train:
        op(str(src_dir / f"{c}.npz"), str(train_dir / f"{c}.npz"))
    for c in new_test:
        op(str(src_dir / f"{c}.npz"), str(test_dir / f"{c}.npz"))

    # Final totals after this batch
    final_train = len(existing_train) + len(new_train)
    final_test  = len(existing_test)  + len(new_test)
    final_total = final_train + final_test
    print(f"[{op_label}]  train += {len(new_train)}  -> {train_dir}  (now {final_train})")
    print(f"[{op_label}]  test  += {len(new_test)}   -> {test_dir}   (now {final_test})")
    print(f"[INFO] Cumulative ratio: {final_train}/{final_total} train "
          f"= {final_train/final_total*100:.1f}%, test = {final_test/final_total*100:.1f}%")


if __name__ == "__main__":
    main()
