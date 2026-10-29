#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overfit_single.py — Overfit ONE patch with OneCycleLR and full visualisation.

Same plumbing as overfit_one_patch.py (case version) but loads exactly one
patch and runs N steps. Each "step" is one full forward/backward on the
same patch — there is no notion of epoch, just steps.

Outputs (under --save_dir):
    overfit_log.csv         per-step  loss / R2 / MAE / lr / sec
    overfit_curve.png       loss / R2 / MAE curves
    overfit_scatter.png     final GT vs Pred scatter
    slices/step{N}.png      3-axis slice every --vis_every steps

Run:
    python overfit_single.py --steps 500 --lr 1e-3 --vis_every 25
    python overfit_single.py --patch <name>.pt --steps 1000
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from net import DualEncoderUNet_Sparse as DualEncoderUNet
from toolkit import StrainFailLoss


def inv_signed_sqrt(y):
    return np.sign(y) * (y ** 2) / 10.0


def ansys_cmap():
    return cm.colors.LinearSegmentedColormap.from_list('ansys', [
        (0.0, '#0000ff'), (0.2, '#00ffff'), (0.4, '#00ff00'),
        (0.6, '#ffff00'), (0.8, '#ff8000'), (1.0, '#ff0000')])


def save_slices(pred, gt, mask, path, step, vmin=None, vmax=None):
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
    fig.suptitle(f'Step {step}    vmin={vmin:.4f}  vmax={vmax:.4f}',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_curves(csv_path, out_path):
    steps, losses, r2s, maes, lrs = [], [], [], [], []
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            steps.append(int(row['step']))
            losses.append(float(row['loss']))
            r2s.append(float(row['r2']))
            maes.append(float(row['mae_mpa']))
            lrs.append(float(row['lr']))
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    axes[0].plot(steps, losses, lw=1.2)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Train loss'); axes[0].grid(alpha=0.3)

    r2_clipped = np.clip(r2s, -0.5, 1.05)
    axes[1].plot(steps, r2_clipped, lw=1.2, color='#2ca02c')
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('R²  (clip [-0.5, 1.05])')
    axes[1].set_title('R²')
    axes[1].axhline(0.95, color='red', ls='--', lw=0.8, alpha=0.6)
    axes[1].set_ylim(-0.55, 1.1)
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, maes, lw=1.2, color='#ff7f0e')
    axes[2].set_xlabel('Step'); axes[2].set_ylabel('MAE (MPa)')
    axes[2].set_title('MAE'); axes[2].grid(alpha=0.3)

    axes[3].plot(steps, lrs, lw=1.2, color='#7a3eaf')
    axes[3].set_xlabel('Step'); axes[3].set_ylabel('LR')
    axes[3].set_title('Learning rate'); axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def find_patch_with_signal(patch_dir, min_voxels=200):
    files = sorted(Path(patch_dir).glob("*.pt"))
    for f in files:
        d = torch.load(f, weights_only=False)
        m = d['mask']
        if (m > 0.5).sum().item() >= min_voxels:
            return f, d
    raise RuntimeError(f"No patch with mask>={min_voxels} found in {patch_dir}")


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_dir",
                    default="../Data/Patches_Tumor_Train_256_128")
    ap.add_argument("--patch", type=str, default=None,
                    help="Specific patch filename (in --patch_dir). "
                         "If None, picks first patch with mask>=200.")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="OneCycleLR max_lr")
    ap.add_argument("--pct_start", type=float, default=0.15,
                    help="Fraction of total steps spent on warmup")
    ap.add_argument("--final_div_factor", type=float, default=1e3,
                    help="OneCycleLR final_div_factor (final_lr = max_lr / this)")
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--loss_scale", type=float, default=10.0)
    ap.add_argument("--fail_threshold", type=float, default=-0.019)
    ap.add_argument("--lambda_fail", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--ds_weight", type=float, default=0.3)
    ap.add_argument("--no_amp", action="store_true", default=True)
    ap.add_argument("--vis_every", type=int, default=25)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_dir", default="./overfit_single/")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = get_args()
    save_dir = Path(args.save_dir)
    slice_dir = save_dir / "slices"
    save_dir.mkdir(parents=True, exist_ok=True)
    slice_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ---- Load patch ----
    if args.patch is not None:
        patch_path = Path(args.patch_dir) / args.patch
        if not patch_path.exists():
            raise FileNotFoundError(patch_path)
        data = torch.load(patch_path, weights_only=False)
    else:
        patch_path, data = find_patch_with_signal(args.patch_dir)
    print(f"[INFO] Patch: {patch_path.name}")

    data = {k: (v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float16 else v)
            for k, v in data.items()}
    geo  = data['geo_input'].unsqueeze(0).to(device)
    phys = data['phys_input'].unsqueeze(0).to(device)
    tgt  = data['S3_grid'].unsqueeze(0).to(device)
    msk  = data['mask'].unsqueeze(0).to(device)

    print(f"[INFO] geo  : {tuple(geo.shape)}")
    print(f"[INFO] phys : {tuple(phys.shape)}  (channels={phys.shape[1]})")
    print(f"[INFO] tgt  : {tuple(tgt.shape)}  S3 in [{tgt.min():.4f}, {tgt.max():.4f}]")
    print(f"[INFO] mask : active={(msk>0.5).sum().item()}")

    bc_v = (phys[:, 0].abs() > 1e-6).sum().item()
    ld_v = (phys[:, 1].abs() > 1e-6).sum().item()
    print(f"[INFO] BC voxels:   {bc_v}")
    print(f"[INFO] Load voxels: {ld_v}")

    gt_mpa = inv_signed_sqrt(tgt.squeeze().cpu().numpy())
    valid_np = msk.squeeze().cpu().numpy() > 0.5
    vmin = float(np.percentile(gt_mpa[valid_np], 2))
    vmax = float(np.percentile(gt_mpa[valid_np], 98))
    print(f"[INFO] GT MPa  p2..p98 = [{vmin:.4f}, {vmax:.4f}]")

    save_slices(tgt, tgt, msk, slice_dir / "step0000_GT.png",
                step=0, vmin=vmin, vmax=vmax)

    # ---- Model ----
    model = DualEncoderUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Params: {n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=1e-4,
                                   betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=args.steps,
        pct_start=args.pct_start,
        anneal_strategy='cos',
        div_factor=25.0,                 # initial_lr = max_lr / 25
        final_div_factor=args.final_div_factor)
    criterion = StrainFailLoss(beta=args.beta, scale=args.loss_scale,
                               fail_threshold=args.fail_threshold,
                               lambda_fail=args.lambda_fail,
                               print_every=int(1e9))
    use_amp = device.type == 'cuda' and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[INFO] AMP={use_amp}, OneCycleLR max_lr={args.lr}, "
          f"steps={args.steps}, pct_start={args.pct_start}, "
          f"div_factor=25, final_div_factor={args.final_div_factor}")

    csv_path = save_dir / "overfit_log.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss", "r2", "mae_mpa", "lr", "sec"])

    print(f"[INFO] Overfitting one patch for {args.steps} steps ...")
    t_start = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred, ds2, ds1 = model(geo, phys, bone_mask=msk)
            loss = (criterion(pred, tgt, msk) +
                    args.ds_weight *
                    (criterion(ds2, tgt, msk) + criterion(ds1, tgt, msk)))

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[step {step}] NaN/Inf loss, abort")
            return

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        sec = time.time() - t0

        with torch.no_grad():
            m_bool = (msk.squeeze() > 0.5).cpu().numpy()
            gt_v = inv_signed_sqrt(tgt.squeeze().cpu().numpy()[m_bool])
            pr_v = inv_signed_sqrt(pred.float().squeeze().cpu().numpy()[m_bool])
            mae = float(np.mean(np.abs(gt_v - pr_v)))
            ss_res = float(np.sum((gt_v - pr_v) ** 2))
            ss_tot = float(np.sum((gt_v - gt_v.mean()) ** 2)) + 1e-12
            r2 = 1.0 - ss_res / ss_tot

        lr_now = optimizer.param_groups[0]['lr']
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                step, f"{loss.item():.6f}",
                f"{r2:.6f}", f"{mae:.6f}",
                f"{lr_now:.6e}", f"{sec:.3f}"])

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"  step {step:4d}  loss={loss.item():.5f}  "
                  f"R2={r2:+.4f}  MAE={mae:.4f} MPa  "
                  f"lr={lr_now:.2e}  ({sec*1000:.0f}ms)",
                  flush=True)

        if step == 1 or step % args.vis_every == 0 or step == args.steps:
            save_slices(pred.detach(), tgt, msk,
                        slice_dir / f"step{step:04d}.png",
                        step=step, vmin=vmin, vmax=vmax)
            save_curves(csv_path, save_dir / "overfit_curve.png")

    elapsed = time.time() - t_start
    print(f"[INFO] Done in {elapsed:.0f}s ({args.steps/elapsed:.1f} steps/s)")

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
    ax2.set_title(f'Final  R²={r2:.4f}  MAE={mae:.4f} MPa')
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(save_dir / 'overfit_scatter.png', dpi=150)
    plt.close(fig2)

    save_curves(csv_path, save_dir / 'overfit_curve.png')

    if r2 > 0.9:
        print(f"[VERDICT] PASS — R²={r2:.4f} > 0.90")
    elif r2 > 0.5:
        print(f"[VERDICT] WEAK — R²={r2:.4f}")
    else:
        print(f"[VERDICT] FAIL — R²={r2:.4f}")
    print(f"[INFO] Logs:  {csv_path}")
    print(f"[INFO] Curve: {save_dir / 'overfit_curve.png'}")
    print(f"[INFO] Slices in: {slice_dir}")


if __name__ == '__main__':
    main()
