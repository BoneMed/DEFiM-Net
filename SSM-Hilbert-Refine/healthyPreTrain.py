#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSM-Hybrid full-data training.

Geometry path:  [E_grid] (1ch)
Physics path:   [BC, Load*u, dist_BC, dist_Load] (4ch)
                -> geometry-conditioned state-space propagation
FiLM fusion at every encoder level.

Validation: fixed-patch full-volume reconstruction.
"""

import os
import csv
import time
import random
import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Net import DualEncoderUNet_SSM as DualEncoderUNet
from dataUtils import PreCutPatchDataset
from toolkit import (StrainFailLoss,
                     inv_signed_log,
                     save_scatter, save_curve, save_slices, save_loss_curve)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Patch-based validation (no npz needed)
# ============================================================
def _reconstruct_case(model, files, patch_positions, P, R, device, use_amp,
                      scan_key=None, amp_dtype=torch.bfloat16):
    """Reconstruct full volume with Gaussian-weighted blending (same as inference).

    scan_key: (1, 1, P, P, P) Hilbert position field on `device`. Same
              tensor reused across every patch (Hilbert is patch-local).
              Required when the model is DualEncoderUNet_SSM.
    """
    pred_vol   = torch.zeros(1, 1, R, R, R)
    weight_vol = torch.zeros(1, 1, R, R, R)
    gt_vol     = torch.zeros(1, 1, R, R, R)
    mask_vol   = torch.zeros(1, 1, R, R, R)

    # Gaussian window for overlap blending
    coords = torch.arange(P, dtype=torch.float32) - (P - 1) / 2.0
    sigma = P / 4.0
    w1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    w3d = (w1d[:, None, None] * w1d[None, :, None] * w1d[None, None, :]).unsqueeze(0).unsqueeze(0)

    for f in files:
        idx = int(f.stem.rsplit('_p', 1)[1])
        data = torch.load(f, weights_only=False)
        if 'patch_origin' in data:
            origin = data['patch_origin']
            if torch.is_tensor(origin):
                origin = origin.cpu().numpy()
            sz, sy, sx = (int(origin[0]), int(origin[1]), int(origin[2]))
        else:
            if idx >= len(patch_positions):
                continue
            sz, sy, sx = patch_positions[idx]
        sl = (slice(sz, sz+P), slice(sy, sy+P), slice(sx, sx+P))

        data = {k: v.float() if v.dtype == torch.float16 else v for k, v in data.items()}
        geo  = data['geo_input'].unsqueeze(0).to(device)
        phys = data['phys_input'].unsqueeze(0).to(device)
        msk  = data['mask'].unsqueeze(0).to(device)
        kwargs = {'bone_mask': msk}
        if scan_key is not None:
            kwargs['scan_key'] = scan_key
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            p = model(geo, phys, **kwargs)

        pred_vol[:, :, sl[0], sl[1], sl[2]]   += p.float().squeeze().cpu() * w3d.squeeze()
        weight_vol[:, :, sl[0], sl[1], sl[2]]  += w3d.squeeze()
        # GT on disk is in the OLD signed_sqrt(strain*10) encoding;
        # convert to the current signed_log(strain*100) encoding so
        # downstream inv_signed_log gives correct physical strain.
        # Identity: y_new = sign(y_old) * log1p(y_old**2 * 10).
        y_old = data['S3_grid'].squeeze().float()
        gt_vol[0, 0, sl[0], sl[1], sl[2]] = torch.sign(y_old) * torch.log1p(y_old.pow(2) * 10.0)
        mask_vol[0, 0, sl[0], sl[1], sl[2]] = data['mask'].squeeze()

    pred_vol = pred_vol / weight_vol.clamp(min=1e-8)
    return pred_vol, gt_vol, mask_vol


@torch.no_grad()
def validate_patches(model, val_patch_ds, criterion, device, use_amp,
                     epoch, save_dir, grid_res=256, save_vis=False,
                     amp_dtype=torch.bfloat16,
                     yield_threshold=-0.019):
    """Validate on all pre-cut val patches.

    Reports per-case mean MAE/R2 and global MAE/R2 (single statistic on the
    concatenated voxel pool, same as inference.py's "Global"), plus two
    flavours of relative error (MRE):
        mre_case  — per-case Relative L2 = ||err||₂ / ||tgt||₂, averaged.
                    Engineering standard, robust to bulk-near-zero voxels.
        mre_voxel — per-voxel MAPE = |err|/(|tgt|+eps), averaged within case
                    then across cases.  eps=1e-3 MPa floors tiny denominators.

    Yield-region overlap (failure boundary at strain ≤ yield_threshold):
        yield_dice_case — mean per-case hard Dice of (gt_fail) vs (pred_fail)
        yield_dice_glb  — global Dice over all cases' voxels concatenated

    Returns: per_case_mae, per_case_r2, global_mae, global_r2, val_loss,
             mre_case, mre_voxel, yield_dice_case, yield_dice_glb
    """
    model.eval()
    vis_dir = Path(save_dir) / "val_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Streaming accumulators (global): identical to concat+mean but O(1) memory.
    g_sum_ae = 0.0
    g_sse      = 0.0
    g_sum_gt   = 0.0
    g_sum_gt_sq = 0.0
    g_n = 0
    loss_sum, loss_n = 0.0, 0
    # Streaming counts for global yield-region Dice.
    g_fail_gt = 0
    g_fail_pred = 0
    g_fail_inter = 0
    per_case_yield_dice = []

    # Per-case metrics are computed after full-volume reconstruction, so
    # overlapping validation patches do not count the same voxel multiple times.
    from collections import defaultdict
    case_map = defaultdict(list)
    for f in val_patch_ds.files:
        case_map[f.stem.rsplit('_p', 1)[0]].append(f)

    # Bounded reservoir for scatter/curve visualization (only filled when save_vis).
    # Stratified per-patch: a handful of guaranteed extreme voxels + random bulk.
    # Uniform random sampling alone misses the long tail (peaks are <0.01% of
    # bone voxels) and gives the false impression that pred saturates at the
    # bulk's range, even when the model actually reaches deep into the tail.
    VIS_PEAK_PER_PATCH = 10   # most compressive (smallest signed-sqrt) per patch
    VIS_TENS_PER_PATCH = 5    # most tensile / least negative (largest)
    VIS_BULK_PER_PATCH = 35   # random middle
    res_gt, res_pred = [], []
    vis_rng = np.random.RandomState(0)

    files = val_patch_ds.files       # indexed access for loss-only patch loop

    # Hilbert table is patch-local and identical for every validation patch.
    # Cache it once on device instead of copying it for every sample.
    sk_full = None
    if getattr(val_patch_ds, '_scan_key', None) is not None:
        sk_full = (val_patch_ds._scan_key
                   .unsqueeze(0).unsqueeze(0).to(device))

    for i in range(len(val_patch_ds)):
        sample = val_patch_ds[i]
        if int((sample['mask'] > 0.5).sum().item()) < 200:
            continue

        geo  = sample['geo_input'].unsqueeze(0).to(device, non_blocking=True)
        phys = sample['phys_input'].unsqueeze(0).to(device, non_blocking=True)
        tgt  = sample['S3_grid'].unsqueeze(0).to(device, non_blocking=True)
        msk  = sample['mask'].unsqueeze(0).to(device, non_blocking=True)
        sk = sk_full if sk_full is not None else sample['scan_key'].unsqueeze(0).to(
            device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            pred = model(geo, phys, bone_mask=msk, scan_key=sk)
            loss = criterion(pred, tgt, msk)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss_sum += loss.item()
            loss_n += 1

        if save_vis:
            m_np = sample['mask'].squeeze().numpy() > 0.5
            gt_mpa = inv_signed_log(sample['S3_grid'].squeeze().numpy()[m_np])
            pr_mpa = inv_signed_log(pred.float().squeeze().cpu().numpy()[m_np])
            valid = np.isfinite(gt_mpa) & np.isfinite(pr_mpa)
            gt_v = gt_mpa[valid]
            pr_v = pr_mpa[valid]
            n = len(gt_v)
            if n == 0:
                continue
            # Stratified pick: peaks (most compressive) + tail (most tensile) +
            # random bulk. argpartition is O(n), cheaper than full sort.
            k_peak = min(VIS_PEAK_PER_PATCH, n)
            k_tens = min(VIS_TENS_PER_PATCH, max(0, n - k_peak))
            k_bulk = min(VIS_BULK_PER_PATCH, max(0, n - k_peak - k_tens))
            picked = set()
            if k_peak > 0:
                picked.update(np.argpartition(gt_v, k_peak - 1)[:k_peak].tolist())
            if k_tens > 0:
                picked.update(np.argpartition(-gt_v, k_tens - 1)[:k_tens].tolist())
            if k_bulk > 0:
                remaining = np.setdiff1d(np.arange(n), np.fromiter(picked, dtype=np.int64),
                                          assume_unique=True)
                if len(remaining) > 0:
                    chosen = vis_rng.choice(remaining,
                                            size=min(k_bulk, len(remaining)),
                                            replace=False)
                    picked.update(chosen.tolist())
            idx = np.fromiter(picked, dtype=np.int64)
            res_gt.append(gt_v[idx])
            res_pred.append(pr_v[idx])

    # Full-volume validation metrics (same overlap blending as inference).
    # MRE definitions:
    #   per_case_rel_l2 = ||pred - target||_2 / ||target||_2  (engineering standard,
    #       robust to bulk-near-zero voxels, scale-free per case)
    #   per_case_mape   = mean(|pred - target| / (|target| + eps))  per voxel,
    #       eps = 1e-3 MPa floors tiny denominators.  Reported per case then
    #       averaged across cases.
    MAPE_EPS = 1e-3
    per_case_mae_list, per_case_r2_list = [], []
    per_case_rel_l2_list, per_case_mape_list = [], []
    patch_positions = getattr(val_patch_ds, '_positions', None)
    if patch_positions is None:
        from dataUtils.precut_patches import get_patch_positions
        patch_positions = get_patch_positions(grid_res, val_patch_ds.patch_size,
                                              val_patch_ds.overlap)

    for case_name, case_files in sorted(case_map.items()):
        sample0 = torch.load(case_files[0], weights_only=False)
        P = int(sample0.get('patch_size', sample0['S3_grid'].shape[-1]))
        R = int(sample0.get('grid_res', grid_res))

        pred_vol, gt_vol, mask_vol = _reconstruct_case(
            model, case_files, patch_positions, P, R, device, use_amp,
            scan_key=sk_full, amp_dtype=amp_dtype)
        m_np = mask_vol.squeeze().numpy() > 0.5
        if not m_np.any():
            continue
        gt_v = inv_signed_log(gt_vol.squeeze().numpy()[m_np])
        pr_v = inv_signed_log(pred_vol.squeeze().numpy()[m_np])
        valid = np.isfinite(gt_v) & np.isfinite(pr_v)
        gt_v = gt_v[valid]
        pr_v = pr_v[valid]
        if len(gt_v) == 0:
            continue

        diff = gt_v - pr_v
        ae = np.abs(diff).sum()
        se = (diff ** 2).sum()
        sg = gt_v.sum()
        sg2 = (gt_v ** 2).sum()
        nv = len(gt_v)

        g_sum_ae += ae
        g_sse += se
        g_sum_gt += sg
        g_sum_gt_sq += sg2
        g_n += nv

        tss_c = sg2 - nv * (sg / nv) * (sg / nv)
        per_case_mae_list.append(ae / nv)
        per_case_r2_list.append(1.0 - se / max(tss_c, 1e-12))

        # MRE: per-case relative L2 + per-case mean per-voxel MAPE
        gt_norm  = float(np.linalg.norm(gt_v))
        err_norm = float(np.linalg.norm(diff))
        per_case_rel_l2_list.append(err_norm / max(gt_norm, 1e-9))
        per_case_mape_list.append(
            float(np.mean(np.abs(diff) / (np.abs(gt_v) + MAPE_EPS))))

        # Yield-region Dice (hard threshold at -0.019 strain). Skip cases
        # where neither GT nor Pred has any failure voxels (Dice undefined).
        fail_gt = gt_v <= yield_threshold
        fail_pred = pr_v <= yield_threshold
        n_gt_c = int(fail_gt.sum())
        n_pr_c = int(fail_pred.sum())
        n_int_c = int((fail_gt & fail_pred).sum())
        g_fail_gt += n_gt_c
        g_fail_pred += n_pr_c
        g_fail_inter += n_int_c
        if n_gt_c + n_pr_c > 0:
            per_case_yield_dice.append(2.0 * n_int_c / (n_gt_c + n_pr_c))

    if g_n == 0 or not per_case_mae_list:
        nan = float('nan')
        return nan, nan, nan, nan, nan, nan, nan, nan, nan

    # Global metrics (matches inference.py's "Global")
    mean_gt = g_sum_gt / g_n
    tss = g_sum_gt_sq - g_n * mean_gt * mean_gt
    global_mae = float(g_sum_ae / g_n)
    global_r2  = float(1.0 - g_sse / max(tss, 1e-12))
    val_loss   = loss_sum / max(loss_n, 1)

    mae = float(np.mean(per_case_mae_list))
    r2  = float(np.mean(per_case_r2_list))
    mre_case  = float(np.mean(per_case_rel_l2_list))   # 0.15 = 15% relative L2 error
    mre_voxel = float(np.mean(per_case_mape_list))     # mean per-voxel MAPE (averaged across cases)

    # Yield overlap metrics
    yield_dice_case = (float(np.mean(per_case_yield_dice))
                       if per_case_yield_dice else float('nan'))
    yield_dice_glb = (2.0 * g_fail_inter / (g_fail_gt + g_fail_pred)
                      if g_fail_gt + g_fail_pred > 0 else float('nan'))

    # ---- Visualisation (only when save_vis=True, every N epochs) ----
    if save_vis and res_gt:
        gt_vis = np.concatenate(res_gt)
        pr_vis = np.concatenate(res_pred)
        save_scatter(gt_vis, pr_vis, vis_dir / f"ep{epoch:04d}_scatter.png",
                     f"Epoch {epoch} Val ({len(val_patch_ds)} patches, "
                     f"vis subsample ~{len(gt_vis)})")
        save_curve(gt_vis, pr_vis, vis_dir / f"ep{epoch:04d}_curve.png",
                   f"Epoch {epoch} GT vs Pred (subsample)")

        if case_map:
            # Pick a random case for slice visualisation each epoch (seeded
            # by epoch so the same epoch's vis stays consistent if recomputed,
            # but different epochs will pick different cases).
            case_keys = sorted(case_map.keys())
            vis_case = np.random.RandomState(epoch).choice(case_keys)
            case_files = case_map[vis_case]

            sample0 = torch.load(case_files[0], weights_only=False)
            P = sample0['S3_grid'].shape[-1]  # 128
            R = grid_res                        # 256 (same as precut_patches)
            n_patches = len(case_files)
            n_per_dim = round(n_patches ** (1/3))
            # Must match precut_recut_overlap.py default (overlap=0.5 -> stride = P/2).
            # 27 patches/case => n_per_dim=3 => stride=64 for P=128.
            stride = max(1, int(P * 0.5)) if n_per_dim >= 3 else P
            starts = list(range(0, R - P + 1, stride))
            if starts[-1] + P < R:
                starts.append(R - P)
            patch_positions = [(sz, sy, sx)
                               for sz in starts for sy in starts for sx in starts]

            pred_vol, gt_vol, mask_vol = _reconstruct_case(
                model, case_files, patch_positions, P, R, device, use_amp,
                scan_key=sk_full, amp_dtype=amp_dtype)
            save_slices(pred_vol, gt_vol, mask_vol,
                        vis_dir / f"ep{epoch:04d}_slice_{vis_case}.png", epoch)

    return (mae, r2, global_mae, global_r2, val_loss,
            mre_case, mre_voxel, yield_dice_case, yield_dice_glb)


# ============================================================
# Args
# ============================================================
def get_args():
    ap = argparse.ArgumentParser(description="Train SSM-Hybrid on healthy data")

    # ---- I/O ----
    ap.add_argument("--data_dir",   default="/mnt/training_Data/SSM-Data/ElementGraph_Healthy_Train")
    ap.add_argument("--save_dir",   default="../Model/Healthy_Hilbert_MAE_Whole/")
    ap.add_argument("--patch_dir",  default="/mnt/training_Data/SSM-Data/Patches_Healthy_Train_256_128")
    ap.add_argument("--split_json", default="/mnt/training_Data/SSM-Data/healthy_split.json")
    ap.add_argument("--resume",     type=str, default=None,
                    help="Checkpoint to resume (model + optimizer + schedulers + trackers).")

    # ---- Grid / patches ----
    ap.add_argument("--grid_res", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="Overlap used when patches were cut (stride = P*(1-overlap)).")

    # ---- Training ----
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--train_fraction", type=float, default=0.3,
                    help="Fraction of train cases to keep (0 < f <= 1). "
                         "Sampled deterministically by --seed.")
    ap.add_argument("--val_fraction", type=float, default=0.3,
                    help="Fraction of val cases to keep (0 < f <= 1). "
                         "Validation does per-case full-volume "
                         "reconstruction so this scales runtime linearly.")
    ap.add_argument("--num_workers", type=int, default=10)
    ap.add_argument("--amp_dtype", choices=['bf16', 'fp16'], default='bf16')

    # ---- Architecture ----
    ap.add_argument("--sparse_hidden", type=int, default=16)
    ap.add_argument("--sparse_layers", type=int, default=5)
    ap.add_argument("--mamba_d_state", type=int, default=16)
    ap.add_argument("--mamba_d_conv", type=int, default=4)
    ap.add_argument("--mamba_expand", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="nn.Dropout3d rate applied in every ResBlock, "
                         "Mamba block, Up fuse, and before the head.")

    # ---- Loss (L1 main + magnitude-grad L1 on extreme tail) ----
    ap.add_argument("--lambda_grad", type=float, default=0.5,
                    help="L1 of magnitude gradient |∂pred-∂target| on "
                         "voxel pairs touching the top-k compressive tail.")
    ap.add_argument("--extreme_tail_fraction", type=float, default=0.2)
    ap.add_argument("--extreme_min_voxels", type=int, default=64)
    ap.add_argument("--tail_weight", type=float, default=0.0,
                    help="Per-voxel weight on the compressive tail of "
                         "main loss. Up to ×(1+tail_weight) ≈ ×4 for "
                         "most negative target voxels when set to 3.")
    ap.add_argument("--tail_power", type=float, default=1.0)
    ap.add_argument("--max_tail_weight", type=float, default=8.0)
    ap.add_argument("--mse_weight", type=float, default=0.1,
                    help="Small MSE component added to main L1 (per-voxel "
                         "|err| + mse_weight·err²). 0 = pure L1 (oscillates "
                         "near optimum). 0.1 = mild quadratic basin for "
                         "smooth convergence. Try 0.3 if still hard.")
    # ---- Yield-region soft Dice loss + validation overlap metric ----
    ap.add_argument("--lambda_yield", type=float, default=0.0,
                    help="Weight on soft Dice yield loss. 0 = off "
                         "(healthy training default). Try 0.5 on tumor "
                         "finetune to push failure-region overlap.")
    ap.add_argument("--yield_threshold", type=float, default=-0.019,
                    help="Compressive strain threshold for failure region. "
                         "Used by both the soft Dice loss and the validation "
                         "Dice metric. Negative = compression.")
    ap.add_argument("--yield_temp", type=float, default=0.005,
                    help="Sigmoid bandwidth (in physical strain units) for "
                         "the soft Dice loss. Smaller = sharper transition "
                         "but weaker gradient outside the band.")

    return ap.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Dataset ----
    patch_dir  = Path(args.patch_dir)
    split_json = Path(args.split_json)
    if not split_json.exists():
        raise SystemExit(f"--split_json missing: {split_json}")
    if not patch_dir.exists() or not list(patch_dir.glob("*.pt")):
        raise SystemExit(f"--patch_dir empty: {patch_dir}")

    import json
    with open(split_json, 'r', encoding='utf-8') as f:
        split = json.load(f)
    train_cases = split['train_cases']
    val_cases   = split['val_cases']
    print(f"[INFO] Split: {len(train_cases)} train / {len(val_cases)} val cases")

    if not (0.0 < args.train_fraction <= 1.0):
        raise SystemExit(f"--train_fraction must be in (0, 1], got {args.train_fraction}")
    if args.train_fraction < 1.0:
        n_full = len(train_cases)
        n_keep = max(1, int(round(n_full * args.train_fraction)))
        rng = random.Random(args.seed)
        train_cases = sorted(rng.sample(train_cases, n_keep))
        print(f"[INFO] --train_fraction={args.train_fraction}: "
              f"using {n_keep}/{n_full} train cases (seed={args.seed})")

    if not (0.0 < args.val_fraction <= 1.0):
        raise SystemExit(f"--val_fraction must be in (0, 1], got {args.val_fraction}")
    if args.val_fraction < 1.0:
        n_full = len(val_cases)
        n_keep = max(1, int(round(n_full * args.val_fraction)))
        rng = random.Random(args.seed + 1)
        val_cases = sorted(rng.sample(val_cases, n_keep))
        print(f"[INFO] --val_fraction={args.val_fraction}: "
              f"using {n_keep}/{n_full} val cases (seed={args.seed + 1})")

    train_ds = PreCutPatchDataset(
        patch_dir, case_filter=train_cases,
        npz_dir=args.data_dir, grid_res=args.grid_res,
        patch_size=args.patch_size, overlap=args.overlap)
    val_patch_ds = PreCutPatchDataset(
        patch_dir, case_filter=val_cases,
        npz_dir=args.data_dir, grid_res=args.grid_res,
        patch_size=args.patch_size, overlap=args.overlap)
    print(f"[INFO] Train: {len(train_ds)} patches  |  Val: {len(val_patch_ds)} patches")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # ---- Model ----
    model = DualEncoderUNet(
        sparse_hidden=args.sparse_hidden,
        sparse_layers=args.sparse_layers,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] DualEncoderUNet_SSM: {n_params:,} ({n_params/1e6:.2f}M)  "
          f"sparse(h={args.sparse_hidden}, L={args.sparse_layers})  "
          f"mamba(state={args.mamba_d_state}, conv={args.mamba_d_conv}, "
          f"expand={args.mamba_expand})")

    start_epoch = 1
    resume_ckpt = None
    if args.resume and os.path.isfile(args.resume):
        resume_ckpt = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(
            resume_ckpt['model_state_dict'], strict=False)
        if missing or unexpected:
            print(f"[WARN] Resume loaded with missing={len(missing)} "
                  f"unexpected={len(unexpected)} keys")
        start_epoch = int(resume_ckpt.get('epoch', 0)) + 1

    if resume_ckpt is not None:
        print(f"[INFO] Resumed from epoch {start_epoch - 1}")

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    warmup_epochs = 5
    eta_min = 1e-5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=eta_min)
    print(f"[INFO] Sched: warmup {warmup_epochs}ep -> cosine (eta_min={eta_min:.1e})")

    if resume_ckpt is not None:
        if 'optimizer_state_dict' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        if 'warmup_scheduler_state_dict' in resume_ckpt:
            warmup_scheduler.load_state_dict(resume_ckpt['warmup_scheduler_state_dict'])
        if 'cosine_scheduler_state_dict' in resume_ckpt:
            cosine_scheduler.load_state_dict(resume_ckpt['cosine_scheduler_state_dict'])

    criterion = StrainFailLoss(lambda_grad=args.lambda_grad,
                               extreme_tail_fraction=args.extreme_tail_fraction,
                               extreme_min_voxels=args.extreme_min_voxels,
                               tail_weight=args.tail_weight,
                               tail_power=args.tail_power,
                               max_tail_weight=args.max_tail_weight,
                               mse_weight=args.mse_weight)
    print(f"[INFO] loss: (L1 + {args.mse_weight}·MSE)(tail_w={args.tail_weight}) + "
          f"λ_grad={args.lambda_grad}·∂L1 on top "
          f"{args.extreme_tail_fraction*100:.1f}% tail")

    csv_path = Path(args.save_dir) / "train_log.csv"
    if not csv_path.exists() or start_epoch == 1:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss",
                                    "val_mae", "val_r2",
                                    "val_global_mae", "val_global_r2",
                                    "val_mre_case", "val_mre_voxel",
                                    "yield_dice_case", "yield_dice_global",
                                    "lr", "sec"])

    use_amp = (device.type == 'cuda')
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16  # only fp16 needs scaling
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    print(f"[INFO] AMP={use_amp} dtype={amp_dtype} scaler={use_scaler}  "
          f"steps/epoch={len(train_loader)}  bs={args.batch_size}")

    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_train_loss = float('inf')
    patience_ctr = 0
    if resume_ckpt is not None:
        best_val_loss = float(resume_ckpt.get('best_val_loss',
                              resume_ckpt.get('val_loss', best_val_loss)))
        best_val_mae = float(resume_ckpt.get('best_val_mae',
                             resume_ckpt.get('val_mae', best_val_mae)))
        best_train_loss = float(resume_ckpt.get('best_train_loss',
                                resume_ckpt.get('train_loss', best_train_loss)))
        patience_ctr = int(resume_ckpt.get('patience_ctr', 0))

    # ---- Training ----
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        ep_loss, n_steps = 0.0, 0
        t0 = time.time()

        loop = tqdm(train_loader,
                    desc=f"Ep {epoch}/{args.epochs}",
                    ncols=100)
        for batch in loop:
            geo = batch['geo_input'].to(device, non_blocking=True)
            phys = batch['phys_input'].to(device, non_blocking=True)
            tgt = batch['S3_grid'].to(device)
            msk = batch['mask'].to(device)
            sk  = batch['scan_key'].to(device)

            if (msk > 0.5).sum() < 200:
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                pred = model(geo, phys, bone_mask=msk, scan_key=sk)
                loss = criterion(pred, tgt, msk)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            ep_loss += loss.item()
            n_steps += 1
            loop.set_postfix(loss=f"{loss.item():.4f}")

        elapsed = time.time() - t0
        avg_loss = ep_loss / max(1, n_steps)
        lr = optimizer.param_groups[0]['lr']

        # ---- Validation ----
        mae = r2 = global_mae = global_r2 = val_loss = float('nan')
        mre_case = mre_voxel = float('nan')
        yield_dice_case = yield_dice_glb = float('nan')
        (mae, r2, global_mae, global_r2, val_loss,
         mre_case, mre_voxel,
         yield_dice_case, yield_dice_glb) = validate_patches(
            model, val_patch_ds, criterion, device, use_amp,
            epoch, args.save_dir, grid_res=args.grid_res, save_vis=True,
            yield_threshold=args.yield_threshold)
        print(f"  [Val] loss={val_loss:.5f}  per-case MAE={mae:.4f} R2={r2:.4f}  "
              f"global MAE={global_mae:.4f} R2={global_r2:.4f}  "
              f"MRE(L2)={mre_case:.4f} MRE(MAPE)={mre_voxel:.4f}  "
              f"Yield-Dice case={yield_dice_case:.3f} global={yield_dice_glb:.3f}")

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        print(f"Ep {epoch:03d} | Train {avg_loss:.5f} | Val {val_loss:.5f} | "
              f"R2={r2:.4f} | LR {lr:.2e} | {elapsed:.1f}s")

        with open(csv_path, "a", newline="") as f:
            def _fmt(x, d=4):
                return f"{x:.{d}f}" if not np.isnan(x) else "nan"
            csv.writer(f).writerow([
                epoch, f"{avg_loss:.6f}",
                _fmt(val_loss, 6), _fmt(mae), _fmt(r2),
                _fmt(global_mae), _fmt(global_r2),
                _fmt(mre_case), _fmt(mre_voxel),
                _fmt(yield_dice_case), _fmt(yield_dice_glb),
                f"{lr:.6e}", f"{elapsed:.1f}"])

        def _save_ckpt(path_name):
            p = Path(args.save_dir) / path_name
            state = {'model_state_dict':            model.state_dict(),
                     'optimizer_state_dict':        optimizer.state_dict(),
                     'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                     'cosine_scheduler_state_dict': cosine_scheduler.state_dict(),
                     'epoch': epoch, 'train_loss': avg_loss,
                     'val_loss': val_loss, 'val_mae': mae, 'val_r2': r2,
                     'val_global_mae': global_mae, 'val_global_r2': global_r2,
                     'best_val_loss':   best_val_loss,
                     'best_val_mae':    best_val_mae,
                     'best_train_loss': best_train_loss,
                     'patience_ctr':    patience_ctr,
                     'args': vars(args)}
            torch.save(state, p)
            return p

        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            p = _save_ckpt("best_valloss.pth")
            print(f"  -> New best val_loss={val_loss:.5f} @ ep{epoch} -> {p.name}")
        else:
            patience_ctr += 1

        if not np.isnan(mae) and mae < best_val_mae:
            best_val_mae = mae
            p = _save_ckpt("best_valmae.pth")
            print(f"  -> New best val_mae={mae:.5f} @ ep{epoch} -> {p.name}")

        if not np.isnan(avg_loss) and avg_loss < best_train_loss:
            best_train_loss = avg_loss
            p = _save_ckpt("best_trainloss.pth")
            print(f"  -> New best train_loss={avg_loss:.5f} @ ep{epoch} -> {p.name}")

        if patience_ctr >= args.patience:
            print("Early stop."); break

        if epoch % 10 == 0:
            save_loss_curve(csv_path, Path(args.save_dir) / "loss_curve.png")

    _save_ckpt("final.pth")
    save_loss_curve(csv_path, Path(args.save_dir) / "loss_curve.png")
    print("Done.")


if __name__ == "__main__":
    main()
