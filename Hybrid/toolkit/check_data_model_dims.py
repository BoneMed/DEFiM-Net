#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check patch tensor dimensions against the current Hybrid model.

This script is intentionally read-only.  It reports tensor shapes, channel
counts, basic masked statistics, and optionally runs one forward pass through
DualEncoderUNet_Sparse.
"""

import argparse
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED = {
    "geo_input": 1,
    "phys_input": 4,
    "S3_grid": 1,
    "mask": 1,
}

PHYS_NAMES = ("BC", "Load*u", "dist_BC", "dist_Load")


def _fmt_shape(x):
    return "x".join(str(v) for v in tuple(x.shape))


def _as_float_tensor(x):
    if not isinstance(x, torch.Tensor):
        return None
    return x.float() if x.dtype == torch.float16 else x


def _tensor_stats(x, mask=None):
    x = _as_float_tensor(x)
    if x is None:
        return "not a tensor"
    vals = x
    if mask is not None and x.ndim >= 4:
        m = (mask > 0.5).bool()
        if x.shape[0] == 1:
            vals = x[:, m.squeeze(0)]
        elif x.shape[1:] == m.shape[1:]:
            vals = x[:, m.squeeze(0)]
    vals = vals.detach().cpu().reshape(-1)
    if vals.numel() == 0:
        return "empty"
    finite = torch.isfinite(vals)
    finite_count = int(finite.sum().item())
    if finite_count == 0:
        return f"finite=0/{vals.numel()}"
    v = vals[finite]
    nz = int((v.abs() > 1e-8).sum().item())
    return (f"min={v.min().item():+.5g}, max={v.max().item():+.5g}, "
            f"mean={v.mean().item():+.5g}, nonzero={nz}/{v.numel()}")


def _load_patch(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for k, v in data.items():
        out[k] = _as_float_tensor(v) if isinstance(v, torch.Tensor) else v
    return out


def _check_sample(path, data, verbose=True):
    if verbose:
        print(f"\n[PATCH] {path.name}")
    missing = [k for k in EXPECTED if k not in data]
    if missing:
        if verbose:
            print(f"  [ERROR] missing keys: {missing}")
        return False, None

    ok = True
    info = {
        "active": 0,
        "total": 0,
        "phys_nonzero": [],
        "phys_has_signal": [],
        "phys_channels": None,
    }
    mask = data["mask"]
    for key, expected_ch in EXPECTED.items():
        x = data[key]
        if not isinstance(x, torch.Tensor):
            if verbose:
                print(f"  [ERROR] {key}: not a tensor ({type(x).__name__})")
            ok = False
            continue
        shape = tuple(x.shape)
        ch = shape[0] if x.ndim >= 1 else None
        status = "OK" if ch == expected_ch else "MISMATCH"
        if status != "OK":
            ok = False
        if verbose:
            print(f"  {key:10s} shape={_fmt_shape(x):16s} dtype={str(x.dtype):12s} "
                  f"channels={ch} expected={expected_ch} [{status}]")

    geo = data["geo_input"]
    phys = data["phys_input"]
    tgt = data["S3_grid"]
    spatial = tuple(geo.shape[1:])
    for key in ("phys_input", "S3_grid", "mask"):
        if tuple(data[key].shape[1:]) != spatial:
            if verbose:
                print(f"  [ERROR] spatial shape mismatch: {key} has "
                      f"{tuple(data[key].shape[1:])}, geo has {spatial}")
            ok = False

    active = int((mask > 0.5).sum().item())
    total = int(mask.numel())
    info["active"] = active
    info["total"] = total
    if verbose:
        print(f"  mask active: {active}/{total} ({active / max(total, 1):.4%})")
        print(f"  geo stats in mask: { _tensor_stats(geo, mask) }")
        print(f"  S3  stats in mask: { _tensor_stats(tgt, mask) }")

    phys_ch = phys.shape[0] if isinstance(phys, torch.Tensor) and phys.ndim >= 1 else 0
    info["phys_channels"] = phys_ch
    for i in range(phys_ch):
        name = PHYS_NAMES[i] if i < len(PHYS_NAMES) else f"extra_ch{i}"
        m = (mask > 0.5).squeeze(0)
        nz = int((phys[i][m].abs() > 1e-8).sum().item()) if active else 0
        info["phys_nonzero"].append(nz)
        info["phys_has_signal"].append(nz > 0)
        note = "" if i < 4 else "  [IGNORED by current model]"
        if verbose:
            print(f"  phys[{i}] {name:9s}: {_tensor_stats(phys[i:i + 1], mask)}{note}")

    if phys_ch > 4:
        if verbose:
            print("  [WARN] phys_input has extra channels. Current model uses only "
                  "channels 0..3; channels >=4 are ignored.")
    return ok, info


def _run_forward(data, device):
    from net import DualEncoderUNet_Sparse

    geo = data["geo_input"].unsqueeze(0).to(device)
    phys = data["phys_input"].unsqueeze(0).to(device)
    mask = data["mask"].unsqueeze(0).to(device)

    model = DualEncoderUNet_Sparse().to(device).eval()
    with torch.no_grad():
        out = model(geo, phys, bone_mask=mask)

    if isinstance(out, tuple):
        shapes = [tuple(t.shape) for t in out]
        print(f"\n[FORWARD] OK tuple outputs: {shapes}")
    else:
        print(f"\n[FORWARD] OK output: {tuple(out.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_dir", default="../Data/Patches_Tumor_Train_256_128")
    ap.add_argument("--glob", default="*.pt")
    ap.add_argument("--num", type=int, default=5,
                    help="Number of patches to inspect.")
    ap.add_argument("--random", action="store_true",
                    help="Randomly sample patches instead of taking sorted first N.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summary_only", action="store_true",
                    help="Only print aggregate shape/channel summary.")
    ap.add_argument("--run_forward", action="store_true",
                    help="Also instantiate DualEncoderUNet_Sparse and run one patch.")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    patch_dir = Path(args.patch_dir)
    if not patch_dir.is_absolute():
        patch_dir = (ROOT / patch_dir).resolve()

    files = sorted(patch_dir.glob(args.glob))
    print(f"[INFO] patch_dir: {patch_dir}")
    print(f"[INFO] found: {len(files)} files matching {args.glob}")
    print("[INFO] model expects geo=1ch, phys=4ch, target=1ch, mask=1ch")
    print("[INFO] phys channel meaning: 0=BC, 1=Load*u, 2=dist_BC, 3=dist_Load")

    if not files:
        raise FileNotFoundError(f"No patch files found in {patch_dir}")

    if args.random:
        rng = random.Random(args.seed)
        files = rng.sample(files, len(files))

    all_ok = True
    first_data = None
    summaries = []
    checked = files[:max(args.num, 1)]
    for path in checked:
        data = _load_patch(path)
        if first_data is None:
            first_data = data
        ok, info = _check_sample(path, data, verbose=not args.summary_only)
        all_ok = ok and all_ok
        if info is not None:
            summaries.append(info)

    if summaries:
        active = sum(s["active"] for s in summaries)
        total = sum(s["total"] for s in summaries)
        max_ch = max(s["phys_channels"] or 0 for s in summaries)
        print(f"\n[AGGREGATE] checked={len(summaries)} patches")
        print(f"  mask active: {active}/{total} ({active / max(total, 1):.4%})")
        for i in range(max_ch):
            name = PHYS_NAMES[i] if i < len(PHYS_NAMES) else f"extra_ch{i}"
            patches_with = sum(1 for s in summaries
                               if i < len(s["phys_has_signal"]) and s["phys_has_signal"][i])
            nonzero = sum(s["phys_nonzero"][i] for s in summaries
                          if i < len(s["phys_nonzero"]))
            ratio = nonzero / max(active, 1)
            note = "" if i < 4 else "  [IGNORED by current model]"
            print(f"  phys[{i}] {name:9s}: patches_with_signal={patches_with}/"
                  f"{len(summaries)}, nonzero_voxels={nonzero}, "
                  f"ratio_in_mask={ratio:.6f}{note}")

    if args.run_forward:
        device = torch.device(args.device)
        _run_forward(first_data, device)

    print("\n[SUMMARY] " + ("OK: dimensions match expected model inputs"
                           if all_ok else
                           "CHECK WARNINGS/ERRORS ABOVE: possible mismatch"))


if __name__ == "__main__":
    main()
