#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overfit_one_patch.py — Single-CASE overfit test for the bone-masked
DualEncoderUNet_Sparse pipeline.

Loads ALL patches of one case (e.g. 27 patches for grid_res=256, P=128,
overlap=0.3) and overfits the network on them for N epochs.

Outputs (under --save_dir):
    overfit_log.csv         per-epoch train_loss / R2 / MAE / lr / sec
    overfit_curve.png       loss / R2 / MAE curves
    overfit_scatter.png     final GT vs Pred scatter (whole case, all patches)
    slices/ep{N}_{patch}.png  3-axis slice (GT vs Pred) every --vis_every epochs
                              for the central / first patch

Run:
    python overfit_one_patch.py --case MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp10 \
        --epochs 500 --lr 5e-4 --vis_every 25
"""

import argparse
import csv
import time
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from net import DualEncoderUNet_Sparse as DualEncoderUNet
from toolkit import StrainFailLoss


# ============================================================
# Helpers
# ============================================================
def inv_signed_sqrt(y):
    """inverse of signed_sqrt(S3*10) -> S3 in MPa"""
    return np.sign(y) * (y ** 2) / 10.0


def ansys_cmap():
    return cm.colors.LinearSegmentedColormap.from_list('ansys', [
        (0.0, '#0000ff'), (0.2, '#00ffff'), (0.4, '#00ff00'),
        (0.6, '#ffff00'), (0.8, '#ff8000'), (1.0, '#ff0000')])


def save_slices(pred, gt, mask, path, epoch, vmin=None, vmax=None):
    """3-view slices (Axial / Coronal / Sagittal), GT on top, Pred on bottom."""
    p = inv_signed_sqrt(pred.squeeze().cpu().numpy())
    g = inv_signed_sqrt(gt.squeeze().cpu().numpy())
    m = mask.squeeze().cpu().numpy()
    D, H, W = p.shape
    cmap = ansys_cmap()
    valid = m > 0.5
    if valid.sum() == 0:
        return
    if vmin is None or vmax is None:
        vmin = float(np.percentile(g[valid], 2))
        vmax = float(np.percentile(g[valid], 98))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, (s, t) in enumerate([(D//2, 'Axial'),
                                 (H//2, 'Coronal'),
                                 (W//2, 'Sagittal')]):
        gs = [g[s, :, :], g[:, s, :], g[:, :, s]][i]
        ps = [p[s, :, :], p[:, s, :], p[:, :, s]][i]
        ms = [m[s, :, :], m[:, s, :], m[:, :, s]][i]
        axes[0, i].imshow(np.where(ms > 0.5, gs, np.nan).T, origin='lower',
                          cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'GT  {t}')
        axes[0, i].axis('off')
        axes[1, i].imshow(np.where(ms > 0.5, ps, np.nan).T, origin='lower',
                          cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Pred  {t}')
        axes[1, i].axis('off')
    fig.suptitle(f'Epoch {epoch}    vmin={vmin:.4f}  vmax={vmax:.4f}',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_curves(csv_path, out_path):
    eps, losses, r2s, maes = [], [], [], []
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            eps.append(int(row['epoch']))
            losses.append(float(row['loss']))
            r2s.append(float(row['r2']))
            maes.append(float(row['mae_mpa']))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].plot(eps, losses, lw=1.2)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Train loss'); axes[0].grid(alpha=0.3)

    # Clip R² to a sensible range so the early-step crash doesn't blow the plot
    r2_clipped = np.clip(r2s, -0.5, 1.05)
    axes[1].plot(eps, r2_clipped, lw=1.2, color='#2ca02c')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('R²  (clip [-0.5, 1.05])')
    axes[1].set_title('R² (whole case)')
    axes[1].axhline(0.95, color='red', ls='--', lw=0.8, alpha=0.6)
    axes[1].set_ylim(-0.55, 1.1)
    axes[1].grid(alpha=0.3)

    axes[2].plot(eps, maes, lw=1.2, color='#ff7f0e')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('MAE (MPa)')
    axes[2].set_title('MAE'); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def list_case_patches(patch_dir, case_name=None, min_patches=1):
    """Group patches by case and return one case's files."""
    files = sorted(Path(patch_dir).glob("*.pt"))
    case_map = defaultdict(list)
    for f in files:
        case = '_'.join(f.stem.split('_')[:-1])
        case_map[case].append(f)

    if case_name is not None:
        if case_name not in case_map:
            raise ValueError(f"Case {case_name} not found in {patch_dir}. "
                             f"Available: {list(case_map.keys())[:5]} ...")
        return case_name, case_map[case_name]

    # auto-pick: first case with the most patches
    candidates = sorted(case_map.items(),
                        key=lambda kv: (-len(kv[1]), kv[0]))
    case, patches = candidates[0]
    if len(patches) < min_patches:
        raise RuntimeError(f"No case has >= {min_patches} patches in {patch_dir}")
    return case, patches


class CaseDataset(Dataset):
    """Loads all patches of one case into memory."""
    def __init__(self, files):
        self.samples = []
        for f in files:
            d = torch.load(f, weights_only=False)
            d = {k: (v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float16 else v)
                 for k, v in d.items()}
            d['_name'] = f.stem
            self.samples.append(d)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_keep_dict(batch):
    out = {}
    for k in ('geo_input', 'phys_input', 'S3_grid', 'mask'):
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out['_name'] = [b['_name'] for b in batch]
    return out


# ============================================================
# Main
# ============================================================
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_dir",
                    default="../Data/Patches_Tumor_Train_256_128",
                    help="Pre-cut patch directory")
    ap.add_argument("--case", type=str, default=None,
                    help="Case name to overfit. Auto-pick first if None.")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--loss_scale", type=float, default=10.0)
    ap.add_argument("--fail_threshold", type=float, default=-0.019)
    ap.add_argument("--lambda_fail", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--ds_weight", type=float, default=0.3)
    ap.add_argument("--no_amp", action="store_true", default=True)
    ap.add_argument("--vis_every", type=int, default=25,
                    help="Save slice viz every N epochs")
    ap.add_argument("--log_every", type=int, default=5,
                    help="Print to stdout every N epochs")
    ap.add_argument("--save_dir", default="./overfit_check_case/")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = get_args()
    save_dir = Path(args.save_dir)
    slice_dir = save_dir / "slices"
    save_dir.mkdir(parents=True, exist_ok=True)
    slice_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ---- Pick a case ----
    case_name, patch_files = list_case_patches(args.patch_dir, args.case)
    print(f"[INFO] Case: {case_name}")
    print(f"[INFO] Patches: {len(patch_files)}")

    # ---- Load all patches into memory ----
    print(f"[INFO] Loading {len(patch_files)} patches into memory...")
    ds = CaseDataset(patch_files)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_keep_dict,
                        drop_last=False)

    # Show shapes from first patch
    s0 = ds[0]
    print(f"[INFO] geo  : {tuple(s0['geo_input'].shape)}")
    print(f"[INFO] phys : {tuple(s0['phys_input'].shape)}")
    print(f"[INFO] tgt  : {tuple(s0['S3_grid'].shape)}")
    print(f"[INFO] mask : {tuple(s0['mask'].shape)}")

    # Find a "central" patch with most bone for slice viz
    best_idx = max(range(len(ds)),
                   key=lambda i: (ds[i]['mask'] > 0.5).sum().item())
    print(f"[INFO] Central viz patch: {ds[best_idx]['_name']}  "
          f"(mask={(ds[best_idx]['mask']>0.5).sum().item()})")

    # Fixed colour range from this patch's GT
    vis_sample = ds[best_idx]
    gt_mpa = inv_signed_sqrt(vis_sample['S3_grid'].squeeze().numpy())
    valid_np = vis_sample['mask'].squeeze().numpy() > 0.5
    vmin = float(np.percentile(gt_mpa[valid_np], 2))
    vmax = float(np.percentile(gt_mpa[valid_np], 98))
    print(f"[INFO] vis-patch GT MPa p2..p98: [{vmin:.4f}, {vmax:.4f}]")

    save_slices(vis_sample['S3_grid'], vis_sample['S3_grid'], vis_sample['mask'],
                slice_dir / "ep0000_GT.png", epoch=0, vmin=vmin, vmax=vmax)

    # ---- Model ----
    model = DualEncoderUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Params: {n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=1e-4)
    steps_per_epoch = max(1, len(loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        epochs=args.epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.05, anneal_strategy='cos')
    criterion = StrainFailLoss(beta=args.beta, scale=args.loss_scale,
                               fail_threshold=args.fail_threshold,
                               lambda_fail=args.lambda_fail,
                               print_every=int(1e9))
    use_amp = device.type == 'cuda' and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[INFO] AMP: {use_amp}, batch_size={args.batch_size}, "
          f"steps/epoch={steps_per_epoch}")
    print(f"[INFO] Scheduler: OneCycleLR(max_lr={args.lr}, "
          f"pct_start=0.05, cos)")

    # ---- CSV log ----
    csv_path = save_dir / "overfit_log.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "r2", "mae_mpa", "lr", "sec"])

    # ---- Overfit loop ----
    print(f"[INFO] Overfitting case {case_name} for {args.epochs} epochs ...")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        ep_loss, n_steps = 0.0, 0
        # Aggregate metrics across all patches at the end of each epoch (eval mode)
        for batch in loader:
            geo  = batch['geo_input'].to(device)
            phys = batch['phys_input'].to(device)
            tgt  = batch['S3_grid'].to(device)
            msk  = batch['mask'].to(device)

            if (msk > 0.5).sum() < 200:
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred, ds2, ds1 = model(geo, phys, bone_mask=msk)
                loss = (criterion(pred, tgt, msk) +
                        args.ds_weight *
                        (criterion(ds2, tgt, msk) + criterion(ds1, tgt, msk)))

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[ep {epoch}] NaN/Inf loss, skip step")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ep_loss += loss.item()
            n_steps += 1

        avg_loss = ep_loss / max(1, n_steps)
        sec = time.time() - t0

        # ---- Per-epoch metrics: forward all patches in eval mode, aggregate ----
        model.eval()
        all_gt, all_pr = [], []
        with torch.no_grad():
            for sample in ds:
                geo  = sample['geo_input'].unsqueeze(0).to(device)
                phys = sample['phys_input'].unsqueeze(0).to(device)
                tgt  = sample['S3_grid'].unsqueeze(0).to(device)
                msk  = sample['mask'].unsqueeze(0).to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    p = model(geo, phys, bone_mask=msk)
                if isinstance(p, tuple):
                    p = p[0]
                m_bool = (msk.squeeze() > 0.5).cpu().numpy()
                all_gt.append(inv_signed_sqrt(
                    tgt.squeeze().cpu().numpy()[m_bool]))
                all_pr.append(inv_signed_sqrt(
                    p.float().squeeze().cpu().numpy()[m_bool]))
        gt_v = np.concatenate(all_gt)
        pr_v = np.concatenate(all_pr)
        mae = float(np.mean(np.abs(gt_v - pr_v)))
        ss_res = float(np.sum((gt_v - pr_v) ** 2))
        ss_tot = float(np.sum((gt_v - gt_v.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot

        lr_now = optimizer.param_groups[0]['lr']
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{avg_loss:.6f}",
                f"{r2:.6f}", f"{mae:.6f}",
                f"{lr_now:.6e}", f"{sec:.3f}"])

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"  ep {epoch:4d}  loss={avg_loss:.5f}  "
                  f"R2={r2:+.4f}  MAE={mae:.4f} MPa  ({sec:.1f}s)")

        if epoch == 1 or epoch % args.vis_every == 0 or epoch == args.epochs:
            # Re-forward the central viz patch and dump slice
            with torch.no_grad():
                geo  = vis_sample['geo_input'].unsqueeze(0).to(device)
                phys = vis_sample['phys_input'].unsqueeze(0).to(device)
                msk  = vis_sample['mask'].unsqueeze(0).to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pv = model(geo, phys, bone_mask=msk)
                if isinstance(pv, tuple):
                    pv = pv[0]
            save_slices(pv.detach(),
                        vis_sample['S3_grid'],
                        vis_sample['mask'],
                        slice_dir / f"ep{epoch:04d}.png",
                        epoch=epoch, vmin=vmin, vmax=vmax)
            save_curves(csv_path, save_dir / "overfit_curve.png")

    elapsed = time.time() - t_start
    print(f"[INFO] Done in {elapsed:.0f}s "
          f"({args.epochs/(elapsed/60):.1f} epochs/min)")

    # ---- Final scatter (whole case) ----
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    if len(gt_v) > 100000:
        sel = np.random.choice(len(gt_v), 100000, replace=False)
        ax2.scatter(gt_v[sel], pr_v[sel], s=1.5, alpha=0.18, c='steelblue')
    else:
        ax2.scatter(gt_v, pr_v, s=1.5, alpha=0.25, c='steelblue')
    lo = float(min(gt_v.min(), pr_v.min()))
    hi = float(max(gt_v.max(), pr_v.max()))
    ax2.plot([lo, hi], [lo, hi], 'r-', lw=1.2)
    ax2.set_xlabel('GT (MPa)'); ax2.set_ylabel('Pred (MPa)')
    ax2.set_title(f'Final  R²={r2:.4f}  MAE={mae:.4f} MPa  '
                  f'({len(patch_files)} patches)')
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(save_dir / 'overfit_scatter.png', dpi=150)
    plt.close(fig2)

    save_curves(csv_path, save_dir / 'overfit_curve.png')

    if r2 > 0.9:
        print(f"[VERDICT] PASS — final R²={r2:.4f} > 0.90")
    elif r2 > 0.5:
        print(f"[VERDICT] WEAK — final R²={r2:.4f}, learns but slowly")
    else:
        print(f"[VERDICT] FAIL — final R²={r2:.4f}")
    print(f"[INFO] Logs:  {csv_path}")
    print(f"[INFO] Curve: {save_dir / 'overfit_curve.png'}")
    print(f"[INFO] Slices in: {slice_dir}")


if __name__ == '__main__':
    main()
