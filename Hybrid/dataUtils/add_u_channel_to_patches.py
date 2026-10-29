#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augment existing 4-channel patches with a 5th channel
`u_grid = sign(u) * sqrt(|u|) * mask`.

Same transform as the sparse `Load*u` channel, so the two u-carrying
channels live in the same numeric space.

Looks up u_disp for each case from the original ElementGraph npz files.
Safe against re-runs: skips patches already at 5ch unless --overwrite.
Atomic per-file write (tmp + rename) so crashing mid-run never leaves
a corrupt patch on disk.

Usage:
  # Dry run — just report missing u_disp lookups, don't touch files
  python add_u_channel_to_patches.py \
      --patch_dirs ../../Data/Patches_Tumor_Train_256_128 \
                    ../../Data/Patches_Tumor_Val_256_128 \
                    ../../Data/Patches_Tumor_Test_256_128 \
      --elem_dirs  ../../Data/ElementGraph_Tumor_Train \
                    ../../Data/ElementGraph_Tumor_Test \
      --dry_run

  # Real run
  python add_u_channel_to_patches.py \
      --patch_dirs ../../Data/Patches_Tumor_Train_256_128 \
                    ../../Data/Patches_Tumor_Val_256_128 \
                    ../../Data/Patches_Tumor_Test_256_128 \
      --elem_dirs  ../../Data/ElementGraph_Tumor_Train \
                    ../../Data/ElementGraph_Tumor_Test
"""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm


def build_u_map(elem_dirs):
    """case_name -> u_disp (float). case_name matches npz stem."""
    u_map = {}
    for d in elem_dirs:
        d = Path(d)
        if not d.exists():
            print(f"[WARN] elem dir does not exist: {d}")
            continue
        files = sorted(d.glob("*.npz"))
        for f in files:
            try:
                with np.load(f) as raw:
                    u = float(raw['u_disp'][0])
            except KeyError:
                print(f"[WARN] no u_disp in {f.name}")
                continue
            except Exception as e:
                print(f"[WARN] failed to read {f.name}: {e}")
                continue
            u_map[f.stem] = u
    return u_map


def case_from_patch(pt_path):
    """patch file name -> case name (strip '_pNN')."""
    return pt_path.stem.rsplit('_p', 1)[0]


def process_patch(pt_path, u_map, overwrite=False, revert=False):
    """Returns status tag."""
    data = torch.load(pt_path, weights_only=False)
    phys = data['phys_input']

    if revert:
        if phys.shape[0] == 4:
            return 'skip_already_4ch'
        if phys.shape[0] != 5:
            return f'bad_shape_{tuple(phys.shape)}'
        data['phys_input'] = phys[:4].contiguous()
        tmp = pt_path.with_suffix('.pt.tmp')
        torch.save(data, tmp)
        tmp.replace(pt_path)
        return 'ok'

    if phys.shape[0] == 5 and not overwrite:
        return 'skip_already_5ch'
    if phys.shape[0] not in (4, 5):
        return f'bad_shape_{tuple(phys.shape)}'

    case = case_from_patch(pt_path)
    u = u_map.get(case)
    if u is None:
        return 'no_u_disp'

    u_sqrt = float(np.sign(u) * np.sqrt(abs(u)))   # match sparse Load*u transform

    mask = data['mask']                            # (1, P, P, P) fp16
    u_ch = (mask.float() * u_sqrt).to(phys.dtype)

    if phys.shape[0] == 5:
        # Overwrite existing ch4 in-place (keep first 4 channels untouched).
        new_phys = phys.clone()
        new_phys[4:5] = u_ch
    else:
        new_phys = torch.cat([phys, u_ch], dim=0)  # (5, P, P, P)
    data['phys_input'] = new_phys

    # Atomic replace: write tmp then rename
    tmp = pt_path.with_suffix('.pt.tmp')
    torch.save(data, tmp)
    tmp.replace(pt_path)
    return 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_dirs", nargs='+', required=True)
    ap.add_argument("--elem_dirs",  nargs='+', required=True,
                    help="ElementGraph npz dirs holding u_disp")
    ap.add_argument("--dry_run", action='store_true',
                    help="Only validate case->u lookup; don't touch files")
    ap.add_argument("--overwrite", action='store_true',
                    help="Re-compute ch4 even if patch is already 5ch "
                         "(use when switching transform, e.g. raw -> sqrt)")
    ap.add_argument("--revert_to_4ch", action='store_true',
                    help="Opposite of augment: drop ch4 from any 5ch patch. "
                         "Ignores --overwrite and u_map.")
    ap.add_argument("--verify_n", type=int, default=2,
                    help="Load N patches after processing to sanity-check 5ch")
    args = ap.parse_args()

    print(f"[INFO] Building u_disp map from {len(args.elem_dirs)} elem dir(s)")
    u_map = build_u_map(args.elem_dirs)
    print(f"[INFO] {len(u_map)} cases with u_disp found")
    if u_map:
        samples = list(u_map.items())[:5]
        for k, v in samples:
            u_sqrt = np.sign(v) * np.sqrt(abs(v))
            print(f"    {k:60s}  u={v:+.4f}  -> sign*sqrt(|u|)={u_sqrt:+.4f}")

    all_counts = Counter()

    for pd in args.patch_dirs:
        pd = Path(pd)
        files = sorted(pd.glob("*.pt"))
        if not files:
            print(f"\n[WARN] {pd}: no .pt files")
            continue
        print(f"\n[INFO] {pd.name}: {len(files)} patches")

        # Check coverage first
        cases = {case_from_patch(f) for f in files}
        missing = [c for c in cases if c not in u_map]
        if missing:
            print(f"  [WARN] {len(missing)} / {len(cases)} cases have no u_disp lookup, e.g.:")
            for c in missing[:5]:
                print(f"    - {c}")

        if args.dry_run:
            ok = sum(1 for f in files if case_from_patch(f) in u_map)
            missing_n = len(files) - ok
            print(f"  [DRY] would_ok={ok}  would_skip_no_u={missing_n}")
            all_counts['would_ok'] += ok
            all_counts['would_skip_no_u'] += missing_n
            continue

        counts = Counter()
        for f in tqdm(files, ncols=80, desc=pd.name):
            tag = process_patch(f, u_map, overwrite=args.overwrite,
                                revert=args.revert_to_4ch)
            counts[tag] += 1
        print(f"  result: {dict(counts)}")
        all_counts.update(counts)

        # Verify a few
        if args.verify_n > 0 and counts.get('ok', 0) + counts.get('skip_already_5ch', 0) > 0:
            for f in files[:args.verify_n]:
                d = torch.load(f, weights_only=False)
                shape = tuple(d['phys_input'].shape)
                u_ch_mean = float(
                    d['phys_input'][4][d['mask'].squeeze() > 0.5].float().mean()
                ) if shape[0] >= 5 else None
                print(f"  verify {f.name}: phys_input={shape}, "
                      f"u_ch bone-mean={u_ch_mean}")

    print(f"\n[DONE] totals: {dict(all_counts)}")


if __name__ == "__main__":
    main()
