#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Fisher Information Matrix on the HEALTHY task and snapshot
healthy-pretrained parameters for EWC continual learning.

Output (one .pt file):
    {
        'fisher':         {param_name -> tensor of squared-gradient mean (same shape as param)},
        'theta_healthy':  {param_name -> snapshot tensor},
        'meta': {ckpt: ..., n_samples: ..., split_key: 'train_cases', ...}
    }

Run once (offline) before launching tumor fine-tune:
    python compute_ewc_fisher.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from net import DualEncoderUNet_Sparse
from dataUtils import PreCutPatchDataset
from toolkit import StrainFailLoss


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="../Model/DualEncoder_Healthy/best_valloss.pth",
                    help="Healthy-pretrained checkpoint to compute Fisher on.")
    ap.add_argument("--patch_dir",
                    default="../Data/Patches_Healthy_Train_256_128",
                    help="Pre-cut healthy patches.")
    ap.add_argument("--split_json",
                    default="../Data/healthy_split.json")
    ap.add_argument("--split_key", default="train_cases",
                    choices=["train_cases", "val_cases"],
                    help="Which case list to use for Fisher computation.")
    ap.add_argument("--n_samples", type=int, default=500,
                    help="Number of patches to back-prop over. More = "
                         "more accurate Fisher but slower. 500 is usually enough.")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    # Loss params (must match training config)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--loss_scale", type=float, default=10.0)
    ap.add_argument("--fail_threshold", type=float, default=-0.019)
    ap.add_argument("--lambda_fail", type=float, default=0.3)
    ap.add_argument("--ds_weight", type=float, default=0.3)

    ap.add_argument("--out",
                    default="../Model/DualEncoder_Healthy/fisher.pt",
                    help="Where to save the Fisher + theta_healthy snapshot.")
    return ap.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ---- Load model + ckpt ----
    model = DualEncoderUNet_Sparse().to(device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    print(f"[INFO] Loaded {ckpt_path.name} (epoch {ckpt.get('epoch','?')})")

    # ---- Snapshot theta_healthy BEFORE doing anything else ----
    theta_healthy = {n: p.detach().clone() for n, p in model.named_parameters()}

    # ---- Loader over healthy train_cases ----
    with open(args.split_json, 'r', encoding='utf-8') as f:
        split = json.load(f)
    cases = split[args.split_key]
    print(f"[INFO] Using {len(cases)} {args.split_key} from {args.split_json}")

    ds = PreCutPatchDataset(Path(args.patch_dir), case_filter=cases)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    # ---- Loss ----
    criterion = StrainFailLoss(beta=args.beta, scale=args.loss_scale,
                               fail_threshold=args.fail_threshold,
                               lambda_fail=args.lambda_fail,
                               print_every=int(1e9))
    ds_w = args.ds_weight

    # ---- Initialize Fisher accumulator (zeros, same shape as each param) ----
    fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters()}

    # ---- Compute Fisher: average squared gradient across n_samples patches ----
    # The model's MSGCDown / ResBlock3D wrap their forward in
    # torch.utils.checkpoint when (self.training and x.requires_grad).
    # That re-entrant graph saves activations multiple times and trips up
    # the kind of repeated backward we do here, raising
    # "Attempt to retrieve a tensor saved by autograd multiple times".
    #
    # Fix: disable checkpointing for the duration of Fisher computation so
    # the forward graph is the standard one. Two pieces:
    #   1. Flip every submodule's `use_checkpoint` flag to False
    #      (handles MSGCDown).
    #   2. Monkey-patch torch.utils.checkpoint.checkpoint to a passthrough
    #      (handles ResBlock3D, which gates checkpoint on `self.training`
    #      not on a flag, so flipping a flag doesn't help it).
    # Train mode is still used so the model returns the (pred, ds2, ds1)
    # 3-tuple and we can compute the same DS-aware loss as training.
    import torch.utils.checkpoint as _ckpt
    _orig_checkpoint = _ckpt.checkpoint
    _ckpt.checkpoint = lambda fn, *a, **kw: fn(*a)

    for m in model.modules():
        if hasattr(m, 'use_checkpoint'):
            m.use_checkpoint = False

    model.train()                    # so forward returns DS tuple
    for p in model.parameters():
        p.requires_grad = True

    n_done = 0
    t0 = time.time()
    pbar = tqdm(total=args.n_samples, desc="Fisher", ncols=100)
    try:
        for batch in loader:
            if n_done >= args.n_samples:
                break

            geo  = batch['geo_input'].to(device)
            phys = batch['phys_input'].to(device)
            tgt  = batch['S3_grid'].to(device)
            msk  = batch['mask'].to(device)
            if (msk > 0.5).sum() < 200:
                continue

            pred, ds2, ds1 = model(geo, phys, bone_mask=msk)
            loss = (criterion(pred, tgt, msk) +
                    ds_w * (criterion(ds2, tgt, msk) + criterion(ds1, tgt, msk)))

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            # Zero grads, backward
            model.zero_grad(set_to_none=True)
            loss.backward()

            # Accumulate squared grads
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        fisher[n] += p.grad.detach() ** 2

            n_done += geo.shape[0]   # count patches, not steps
            pbar.update(geo.shape[0])
    finally:
        pbar.close()
        # Restore the original checkpoint function so subsequent code in the
        # same process (or imports of this module) gets normal behavior.
        _ckpt.checkpoint = _orig_checkpoint

    elapsed = time.time() - t0
    print(f"[INFO] Fisher accumulation done. n_samples={n_done}  "
          f"time={elapsed/60:.1f} min")

    # ---- Average over n_samples ----
    with torch.no_grad():
        for n in fisher:
            fisher[n] = fisher[n] / max(1, n_done)

    # ---- Stats ----
    fisher_cpu = {n: t.detach().cpu() for n, t in fisher.items()}
    theta_cpu  = {n: t.detach().cpu() for n, t in theta_healthy.items()}
    means = [t.mean().item() for t in fisher_cpu.values() if t.numel() > 0]
    print(f"[INFO] Fisher per-param mean: "
          f"min={min(means):.3e}  median={np.median(means):.3e}  max={max(means):.3e}")

    # ---- Save ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'fisher': fisher_cpu,
        'theta_healthy': theta_cpu,
        'meta': {
            'ckpt': str(ckpt_path),
            'n_samples': n_done,
            'split_json': str(args.split_json),
            'split_key': args.split_key,
            'beta': args.beta,
            'fail_threshold': args.fail_threshold,
            'lambda_fail': args.lambda_fail,
        }
    }, out_path)
    print(f"[INFO] Saved Fisher + theta to {out_path} "
          f"({out_path.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
