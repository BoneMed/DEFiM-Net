#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DualEncoder CNN Full-Data Training.

Geometry path:  [E_grid] (1ch)
Physics path:   [BC, Load*u, dist_BC, dist_Load] (4ch)
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from net import DualEncoderUNet_Sparse as DualEncoderUNet
from dataUtils import PreCutPatchDataset
from toolkit import StrainFailLoss


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def inv_signed_sqrt(y):
    """inverse of signed_sqrt(S3*10) -> S3 MPa"""
    return np.sign(y) * (y ** 2) / 10.0


def ansys_cmap():
    return cm.colors.LinearSegmentedColormap.from_list('ansys', [
        (0.0, '#0000ff'), (0.2, '#00ffff'), (0.4, '#00ff00'),
        (0.6, '#ffff00'), (0.8, '#ff8000'), (1.0, '#ff0000')])


# ============================================================
# Visualization
# ============================================================
def save_scatter(gt, pred, path, title, max_pts=100000):
    if len(gt) > max_pts:
        idx = np.random.choice(len(gt), max_pts, replace=False)
        gt, pred = gt[idx], pred[idx]
    plt.figure(figsize=(6, 6))
    plt.scatter(gt, pred, s=1, alpha=0.15, c='steelblue')
    lo, hi = min(gt.min(), pred.min()), max(gt.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], 'r-', lw=1.5)
    r2 = 1 - np.sum((gt-pred)**2) / (np.sum((gt-gt.mean())**2)+1e-12)
    mae = np.mean(np.abs(gt-pred))
    plt.title(f'{title}\nR²={r2:.4f} MAE={mae:.4f} MPa')
    plt.xlabel('GT (MPa)'); plt.ylabel('Pred (MPa)')
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def save_slices(pred_vol, gt_vol, mask_vol, path, epoch):
    """Full-volume 3-view slices in MPa."""
    p = inv_signed_sqrt(pred_vol.squeeze().cpu().numpy())
    g = inv_signed_sqrt(gt_vol.squeeze().cpu().numpy())
    m = mask_vol.squeeze().cpu().numpy()
    D, H, W = p.shape
    cmap = ansys_cmap()
    valid = m > 0.5
    if valid.sum() == 0:
        return
    # Auto vmin/vmax from GT (p2..p98) for robust colormap
    vmin = float(np.percentile(g[valid], 2))
    vmax = float(np.percentile(g[valid], 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, (s, t) in enumerate([(D//2, 'Axial'), (H//2, 'Coronal'), (W//2, 'Sagittal')]):
        gs = [g[s,:,:], g[:,s,:], g[:,:,s]][i]
        ps = [p[s,:,:], p[:,s,:], p[:,:,s]][i]
        ms = [m[s,:,:], m[:,s,:], m[:,:,s]][i]
        axes[0,i].imshow(np.where(ms>0.5, gs, np.nan).T, origin='lower',
                          cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0,i].set_title(f'GT {t}'); axes[0,i].axis('off')
        axes[1,i].imshow(np.where(ms>0.5, ps, np.nan).T, origin='lower',
                          cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1,i].set_title(f'Pred {t}'); axes[1,i].axis('off')
    fig.suptitle(f'Epoch {epoch}', fontsize=14)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def save_curve(gt, pred, path, title, max_pts=80000):
    if len(gt) > max_pts:
        idx = np.random.choice(len(gt), max_pts, replace=False)
        gt, pred = gt[idx], pred[idx]
    order = np.argsort(gt)
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})
    axes[0].plot(gt[order], label='GT', lw=0.5, alpha=0.8)
    axes[0].plot(pred[order], label='Pred', lw=0.5, alpha=0.8)
    axes[0].set_ylabel('S3 (MPa)'); axes[0].set_title(title)
    axes[0].legend(); axes[0].grid(alpha=0.3)
    err = pred[order] - gt[order]
    axes[1].fill_between(range(len(err)), err, 0, alpha=0.4, color='red')
    axes[1].set_ylabel('Error'); axes[1].set_xlabel('Voxel (sorted)')
    axes[1].set_title(f'MAE={np.mean(np.abs(err)):.4f} MPa'); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close()


def save_loss_curve(csv_path, path):
    epochs, losses, r2s, maes = [], [], [], []
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            losses.append(float(row['train_loss']))
            r = row['val_r2']
            r2s.append(float(r) if r != 'nan' else np.nan)
            m = row['val_mae']
            maes.append(float(m) if m != 'nan' else np.nan)
    if len(epochs) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, losses, lw=1)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Train Loss'); axes[0].set_yscale('log'); axes[0].grid(alpha=0.3)
    vr = [(e, r) for e, r in zip(epochs, r2s) if not np.isnan(r)]
    if vr:
        axes[1].plot([e for e,_ in vr], [r for _,r in vr], 'o-', lw=1, ms=3)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('R²')
    axes[1].set_title('Val R²'); axes[1].set_ylim(-0.1, 1.05); axes[1].grid(alpha=0.3)
    vm = [(e, m) for e, m in zip(epochs, maes) if not np.isnan(m)]
    if vm:
        axes[2].plot([e for e,_ in vm], [m for _,m in vm], 'o-', lw=1, ms=3, color='orange')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('MAE (MPa)')
    axes[2].set_title('Val MAE'); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


# ============================================================
# Patch-based validation (no npz needed)
# ============================================================
def _reconstruct_case(model, files, patch_positions, P, R, device, use_amp):
    """Reconstruct full volume with Gaussian-weighted blending (same as inference)."""
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
        if idx >= len(patch_positions):
            continue
        sz, sy, sx = patch_positions[idx]
        sl = (slice(sz, sz+P), slice(sy, sy+P), slice(sx, sx+P))

        data = torch.load(f, weights_only=False)
        data = {k: v.float() if v.dtype == torch.float16 else v for k, v in data.items()}
        geo  = data['geo_input'].unsqueeze(0).to(device)
        phys = data['phys_input'].unsqueeze(0).to(device)
        msk  = data['mask'].unsqueeze(0).to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            p = model(geo, phys, bone_mask=msk)

        pred_vol[:, :, sl[0], sl[1], sl[2]]   += p.float().squeeze().cpu() * w3d.squeeze()
        weight_vol[:, :, sl[0], sl[1], sl[2]]  += w3d.squeeze()
        # GT and mask: direct overwrite (overlap regions have same values)
        gt_vol[0, 0, sl[0], sl[1], sl[2]]   = data['S3_grid'].squeeze()
        mask_vol[0, 0, sl[0], sl[1], sl[2]] = data['mask'].squeeze()

    pred_vol = pred_vol / weight_vol.clamp(min=1e-8)
    return pred_vol, gt_vol, mask_vol


@torch.no_grad()
def validate_patches(model, val_patch_ds, criterion, device, use_amp,
                     epoch, save_dir, grid_res=256, save_vis=False):
    """Validate on all pre-cut val patches. No npz required.

    Reports both:
      - per-case mean MAE / R²  (paper-friendly, less sensitive to outliers)
      - global  MAE / R²        (single statistic on the concatenated voxel
                                  pool — same as inference.py's "Global")

    Args:
        save_vis: if True, save scatter / curve / slice visualisation.
                  Called every vis_interval epochs to avoid I/O overhead.

    Returns:
        per_case_mae, per_case_r2, global_mae, global_r2, val_loss
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

    # Per-case accumulators: keyed by case name (stem before "_pN").
    # Patches arrive in sorted order so all patches of a case appear
    # consecutively, but using a dict is robust to any iteration order.
    from collections import defaultdict
    pc = defaultdict(lambda: {'sum_ae': 0.0, 'sse': 0.0,
                              'sum_gt': 0.0, 'sum_gt_sq': 0.0, 'n': 0})

    # Bounded reservoir for scatter/curve visualization (only filled when save_vis).
    VIS_VOXELS_PER_PATCH = 50
    res_gt, res_pred = [], []
    vis_rng = np.random.RandomState(0)

    files = val_patch_ds.files       # indexed access for per-case grouping

    for i in range(len(val_patch_ds)):
        sample = val_patch_ds[i]
        case_name = files[i].stem.rsplit('_p', 1)[0]

        geo  = sample['geo_input'].unsqueeze(0).to(device)
        phys = sample['phys_input'].unsqueeze(0).to(device)
        tgt  = sample['S3_grid'].unsqueeze(0).to(device)
        msk  = sample['mask'].unsqueeze(0).to(device)

        if (msk > 0.5).sum() < 200:
            continue

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(geo, phys, bone_mask=msk)
            loss = criterion(pred, tgt, msk)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss_sum += loss.item()
            loss_n += 1

        m_np = sample['mask'].squeeze().numpy() > 0.5
        gt_mpa = inv_signed_sqrt(sample['S3_grid'].squeeze().numpy()[m_np])
        pr_mpa = inv_signed_sqrt(pred.float().squeeze().cpu().numpy()[m_np])
        valid = np.isfinite(gt_mpa) & np.isfinite(pr_mpa)
        gt_v = gt_mpa[valid]
        pr_v = pr_mpa[valid]
        if len(gt_v) == 0:
            continue

        diff = gt_v - pr_v
        ae   = np.abs(diff).sum()
        se   = (diff ** 2).sum()
        sg   = gt_v.sum()
        sg2  = (gt_v ** 2).sum()
        nv   = len(gt_v)

        # Global
        g_sum_ae    += ae
        g_sse       += se
        g_sum_gt    += sg
        g_sum_gt_sq += sg2
        g_n         += nv

        # Per-case
        d = pc[case_name]
        d['sum_ae']    += ae
        d['sse']       += se
        d['sum_gt']    += sg
        d['sum_gt_sq'] += sg2
        d['n']         += nv

        if save_vis:
            k = min(VIS_VOXELS_PER_PATCH, len(gt_v))
            idx = vis_rng.choice(len(gt_v), size=k, replace=False)
            res_gt.append(gt_v[idx])
            res_pred.append(pr_v[idx])

    if g_n == 0 or not pc:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    # Global metrics (matches inference.py's "Global")
    mean_gt = g_sum_gt / g_n
    tss = g_sum_gt_sq - g_n * mean_gt * mean_gt
    global_mae = float(g_sum_ae / g_n)
    global_r2  = float(1.0 - g_sse / max(tss, 1e-12))
    val_loss   = loss_sum / max(loss_n, 1)

    # Per-case mean
    per_case_mae_list, per_case_r2_list = [], []
    for d in pc.values():
        if d['n'] == 0:
            continue
        mg_c  = d['sum_gt'] / d['n']
        tss_c = d['sum_gt_sq'] - d['n'] * mg_c * mg_c
        per_case_mae_list.append(d['sum_ae'] / d['n'])
        per_case_r2_list.append(1.0 - d['sse'] / max(tss_c, 1e-12))
    mae = float(np.mean(per_case_mae_list))
    r2  = float(np.mean(per_case_r2_list))

    # ---- Visualisation (only when save_vis=True, every N epochs) ----
    if save_vis and res_gt:
        gt_vis = np.concatenate(res_gt)
        pr_vis = np.concatenate(res_pred)
        save_scatter(gt_vis, pr_vis, vis_dir / f"ep{epoch:04d}_scatter.png",
                     f"Epoch {epoch} Val ({len(val_patch_ds)} patches, "
                     f"vis subsample ~{len(gt_vis)})")
        save_curve(gt_vis, pr_vis, vis_dir / f"ep{epoch:04d}_curve.png",
                   f"Epoch {epoch} GT vs Pred (subsample)")

        all_files = val_patch_ds.files
        if all_files:
            from collections import defaultdict
            case_map = defaultdict(list)
            for f in all_files:
                case_map[f.stem.rsplit('_p', 1)[0]].append(f)
            first_case = sorted(case_map.keys())[0]
            case_files = case_map[first_case]

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
                model, case_files, patch_positions, P, R, device, use_amp)
            save_slices(pred_vol, gt_vol, mask_vol,
                        vis_dir / f"ep{epoch:04d}_slice_{first_case}.png", epoch)

    return mae, r2, global_mae, global_r2, val_loss


# ============================================================
# Args
# ============================================================
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../Data/ElementGraph_Healthy_Train")
    ap.add_argument("--save_dir", default="../Model/DualEncoder_Healthy/")
    ap.add_argument("--grid_res", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--max_cache", type=int, default=50)
    ap.add_argument("--patch_dir", default="../Data/Patches_Healthy_Train_256_128",
                    help="Merged train+val pool dir (all patches live here).")
    ap.add_argument("--split_json", default="../Data/healthy_split.json",
                    help="Manifest JSON with train_cases/val_cases lists. "
                         "If missing, falls back to --val_patch_dir.")
    ap.add_argument("--val_patch_dir", default="",
                    help="Legacy fallback val dir used only when --split_json "
                         "is absent or unreadable.")

    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_amp", action="store_true", default=True)

    ap.add_argument("--beta", type=float, default=0.02,
                    help="SmoothL1 beta in signed-sqrt target space.")
    ap.add_argument("--loss_scale", type=float, default=10.0,
                    help="Scale used by target transform signed_sqrt(S3 * scale).")
    ap.add_argument("--fail_threshold", type=float, default=-0.019,
                    help="Physical-space threshold for compressive failure region.")
    ap.add_argument("--lambda_fail", type=float, default=0.3,
                    help="Weight for physical-space failure-region MAE.")
    ap.add_argument("--lambda_lap", type=float, default=0.0,
                    help=argparse.SUPPRESS)
    ap.add_argument("--gamma", type=float, default=0.0,
                    help=argparse.SUPPRESS)
    ap.add_argument("--ds_weight", type=float, default=0.3)

    ap.add_argument("--vis_interval", type=int, default=5,
                    help="Save val scatter/curve/slice every N epochs")

    ap.add_argument("--resume", type=str,
                    default="")
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
    patch_dir     = Path(args.patch_dir)
    val_patch_dir = Path(args.val_patch_dir)
    split_json    = Path(args.split_json) if args.split_json else None

    # Preferred path: case-level split driven by a JSON manifest, both
    # train and val read from the same merged pool (patch_dir).
    if split_json is not None and split_json.exists():
        import json
        with open(split_json, 'r', encoding='utf-8') as f:
            split = json.load(f)
        train_cases = split['train_cases']
        val_cases   = split['val_cases']
        print(f"[INFO] Split: {split_json}  "
              f"({len(train_cases)} train / {len(val_cases)} val cases)")

        train_ds = PreCutPatchDataset(patch_dir, case_filter=train_cases)
        val_patch_ds = PreCutPatchDataset(patch_dir, case_filter=val_cases)
        print(f"[INFO] Train: {len(train_ds)} patches")
        print(f"[INFO] Val:   {len(val_patch_ds)} patches")
    else:
        if split_json is not None:
            print(f"[WARN] split_json not found: {split_json}  (falling back to --val_patch_dir)")

        # Legacy: val patches in a separate directory
        if val_patch_dir.exists() and len(list(val_patch_dir.glob("*.pt"))) > 0:
            val_patch_ds = PreCutPatchDataset(val_patch_dir)
            print(f"[INFO] Val: {len(val_patch_ds)} patches from {val_patch_dir}")
        else:
            val_patch_ds = None
            print(f"[INFO] Val: disabled (no patches at {val_patch_dir})")

        # Train patches from patch_dir (may contain the full pool including val cases)
        if patch_dir.exists() and len(list(patch_dir.glob("*.pt"))) > 0:
            train_ds = PreCutPatchDataset(patch_dir)
            print(f"[INFO] Train: {len(train_ds)} patches from {patch_dir}")
        else:
            raise FileNotFoundError(f"No .pt files found in {patch_dir}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # ---- Model ----
    model = DualEncoderUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] DualEncoderUNet: {n_params:,} ({n_params/1e6:.2f}M)")

    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        print(f"[INFO] Resumed from epoch {start_epoch - 1}")

    # ---- Optimizer & Scheduler ----
    # Warmup: LinearLR from lr/10 to lr over 5 epochs
    # Then: CosineAnnealingLR to eta_min over remaining epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    warmup_epochs = 5
    eta_min = 1e-5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=eta_min)
    print(f"[INFO] Scheduler: LinearLR warmup ({args.lr/10:.1e}->{args.lr:.1e}, "
          f"{warmup_epochs}ep) + CosineAnnealingLR (eta_min={eta_min:.1e})")

    criterion = StrainFailLoss(beta=args.beta, scale=args.loss_scale,
                               fail_threshold=args.fail_threshold,
                               lambda_fail=args.lambda_fail,
                               print_every=500)
    ds_w = args.ds_weight

    # ---- Log ----
    # Columns:
    #   train_loss : mean per-batch loss this epoch
    #   val_loss   : signed_sqrt-space loss on val patches
    #   val_mae / val_r2               : per-CASE mean (paper-friendly)
    #   val_global_mae / val_global_r2 : single statistic on all voxels
    csv_path = Path(args.save_dir) / "train_log.csv"
    if not csv_path.exists() or start_epoch == 1:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss",
                                     "val_mae", "val_r2",
                                     "val_global_mae", "val_global_r2",
                                     "lr", "sec"])

    use_amp = device.type == 'cuda' and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[INFO] AMP={use_amp}, Steps/epoch={len(train_loader)}, BS={args.batch_size}")
    print(f"[INFO] loss=StrainFailLoss beta={args.beta}, scale={args.loss_scale}, "
          f"fail_threshold={args.fail_threshold}, lambda_fail={args.lambda_fail}, "
          f"ds_weight={ds_w}")

    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_train_loss = float('inf')
    patience_ctr = 0

    # ---- Training ----
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        ep_loss, n_steps = 0.0, 0
        t0 = time.time()

        loop = tqdm(train_loader, desc=f"Ep {epoch}/{args.epochs}", ncols=100)
        for batch in loop:
            geo = batch['geo_input'].to(device)
            phys = batch['phys_input'].to(device)
            tgt = batch['S3_grid'].to(device)
            msk = batch['mask'].to(device)

            if (msk > 0.5).sum() < 200:
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred, ds2, ds1 = model(geo, phys, bone_mask=msk)
                loss = (criterion(pred, tgt, msk) +
                        ds_w * (criterion(ds2, tgt, msk) + criterion(ds1, tgt, msk)))

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item()
            n_steps += 1
            loop.set_postfix(loss=f"{loss.item():.4f}")

        elapsed = time.time() - t0
        avg_loss = ep_loss / max(1, n_steps)
        lr = optimizer.param_groups[0]['lr']

        # ---- Validation (every epoch; vis every vis_interval epochs) ----
        mae = r2 = global_mae = global_r2 = val_loss = float('nan')
        if val_patch_ds is not None:
            do_vis = (epoch % args.vis_interval == 0) or epoch == 1
            mae, r2, global_mae, global_r2, val_loss = validate_patches(
                model, val_patch_ds, criterion, device, use_amp,
                epoch, args.save_dir, grid_res=args.grid_res,
                save_vis=do_vis)
            print(f"  [Val] loss={val_loss:.5f}  per-case MAE={mae:.4f} R2={r2:.4f}  "
                  f"global MAE={global_mae:.4f} R2={global_r2:.4f}"
                  + ("  (+ vis)" if do_vis else ""))

        # Scheduler: warmup first N epochs, then CosineAnnealingLR
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        print(f"Ep {epoch:03d} | Train {avg_loss:.5f} | Val {val_loss:.5f} | "
              f"R2={r2:.4f} | LR {lr:.2e} | {elapsed:.1f}s")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{avg_loss:.6f}",
                f"{val_loss:.6f}"   if not np.isnan(val_loss)   else "nan",
                f"{mae:.4f}"        if not np.isnan(mae)        else "nan",
                f"{r2:.4f}"         if not np.isnan(r2)         else "nan",
                f"{global_mae:.4f}" if not np.isnan(global_mae) else "nan",
                f"{global_r2:.4f}"  if not np.isnan(global_r2)  else "nan",
                f"{lr:.6e}", f"{elapsed:.1f}"])

        # Best by val_loss (overwrite single file)
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            p = Path(args.save_dir) / "best_valloss.pth"
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch, 'train_loss': avg_loss,
                        'val_loss': val_loss, 'val_mae': mae, 'val_r2': r2,
                        'val_global_mae': global_mae, 'val_global_r2': global_r2,
                        'args': vars(args)}, p)
            print(f"  -> New best val_loss={val_loss:.5f} @ ep{epoch} -> {p.name}")
        else:
            patience_ctr += 1

        # Best by val_mae (overwrite single file)
        if not np.isnan(mae) and mae < best_val_mae:
            best_val_mae = mae
            p = Path(args.save_dir) / "best_valmae.pth"
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch, 'train_loss': avg_loss,
                        'val_loss': val_loss, 'val_mae': mae, 'val_r2': r2,
                        'val_global_mae': global_mae, 'val_global_r2': global_r2,
                        'args': vars(args)}, p)
            print(f"  -> New best val_mae={mae:.5f} @ ep{epoch} -> {p.name}")

        # Best by train_loss (overwrite single file)
        if not np.isnan(avg_loss) and avg_loss < best_train_loss:
            best_train_loss = avg_loss
            p = Path(args.save_dir) / "best_trainloss.pth"
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch, 'train_loss': avg_loss,
                        'val_loss': val_loss, 'val_mae': mae, 'val_r2': r2,
                        'val_global_mae': global_mae, 'val_global_r2': global_r2,
                        'args': vars(args)}, p)
            print(f"  -> New best train_loss={avg_loss:.5f} @ ep{epoch} -> {p.name}")

        if patience_ctr >= args.patience:
            print("Early stop."); break

        if epoch % 10 == 0:
            save_loss_curve(csv_path, Path(args.save_dir) / "loss_curve.png")

    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                'args': vars(args)}, Path(args.save_dir) / "final.pth")
    save_loss_curve(csv_path, Path(args.save_dir) / "loss_curve.png")
    print("Done.")


if __name__ == "__main__":
    main()
