#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overfit_npz.py — In-memory overfit from a single ElementGraph .npz.

Loads ONE case, voxelizes it once, pre-cuts all patches into CPU memory,
then overfits for N epochs without any disk I/O. Every --val_every epochs
it reconstructs the full volume with Gaussian-weighted patch blending and
records scatter/curve/slice visualisations alongside voxel-level MAE/R².

Run:
  python toolkit/overfit_npz.py \
    --data_dir ../Data/ElementGraph_Tumor_Train \
    --case MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp10 \
    --grid_res 128 --patch_size 64 --overlap 0.5 \
    --epochs 300 --val_every 50 --lr 5e-4 \
    --save_dir ../Results/overfit_npz_128_64
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from net import DualEncoderUNet_Sparse as DualEncoderUNet
from dataUtils import HybridDataset
from dataUtils.precut_patches import get_patch_positions
from toolkit import StrainFailLoss


# ============================================================
# Helpers
# ============================================================
def inv_signed_sqrt(y):
    return np.sign(y) * (y ** 2) / 10.0


def ansys_cmap():
    return cm.colors.LinearSegmentedColormap.from_list('ansys', [
        (0.0, '#0000ff'), (0.2, '#00ffff'), (0.4, '#00ff00'),
        (0.6, '#ffff00'), (0.8, '#ff8000'), (1.0, '#ff0000')])


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Visualisation
# ============================================================
def save_slices(pred_vol, gt_vol, mask_vol, path, epoch):
    p = inv_signed_sqrt(pred_vol.squeeze().cpu().numpy())
    g = inv_signed_sqrt(gt_vol.squeeze().cpu().numpy())
    m = mask_vol.squeeze().cpu().numpy()
    D, H, W = p.shape
    cmap = ansys_cmap()
    valid = m > 0.5
    if valid.sum() == 0:
        return
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


def save_scatter(gt, pred, path, title):
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


def save_log_curves(csv_path, out_path):
    ep, loss, r2, mae = [], [], [], []
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            ep.append(int(row['epoch']))
            loss.append(float(row['train_loss']))
            r2.append(float(row['val_r2']) if row['val_r2'] != 'nan' else np.nan)
            mae.append(float(row['val_mae']) if row['val_mae'] != 'nan' else np.nan)
    if len(ep) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(ep, loss, lw=1); axes[0].set_yscale('log')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Train loss'); axes[0].grid(alpha=0.3)
    vr = [(e, v) for e, v in zip(ep, r2) if not np.isnan(v)]
    if vr:
        axes[1].plot([e for e,_ in vr], [v for _,v in vr], 'o-', ms=4)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Val R²')
    axes[1].set_ylim(-0.1, 1.05); axes[1].grid(alpha=0.3)
    vm = [(e, v) for e, v in zip(ep, mae) if not np.isnan(v)]
    if vm:
        axes[2].plot([e for e,_ in vm], [v for _,v in vm], 'o-', ms=4, color='orange')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Val MAE (MPa)'); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


# ============================================================
# Reconstruction for validation
# ============================================================
@torch.no_grad()
def reconstruct_full(model, patches, positions, R, P, device, use_amp):
    coords = torch.arange(P, dtype=torch.float32) - (P - 1) / 2.0
    sigma = P / 4.0
    w1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    w3d = (w1d[:, None, None] * w1d[None, :, None] * w1d[None, None, :]).unsqueeze(0).unsqueeze(0)

    pred_vol   = torch.zeros(1, 1, R, R, R)
    weight_vol = torch.zeros(1, 1, R, R, R)
    gt_vol     = torch.zeros(1, 1, R, R, R)
    mask_vol   = torch.zeros(1, 1, R, R, R)

    for (geo, phys, s3, mask), (sz, sy, sx) in zip(patches, positions):
        sl = (slice(sz, sz+P), slice(sy, sy+P), slice(sx, sx+P))
        with torch.cuda.amp.autocast(enabled=use_amp):
            p = model(geo.unsqueeze(0).to(device),
                      phys.unsqueeze(0).to(device),
                      bone_mask=mask.unsqueeze(0).to(device)).float().cpu()
        pred_vol[:, :, sl[0], sl[1], sl[2]]   += p.squeeze(0) * w3d.squeeze(0)
        weight_vol[:, :, sl[0], sl[1], sl[2]] += w3d.squeeze(0)
        gt_vol[0, 0, sl[0], sl[1], sl[2]]   = s3.squeeze()
        mask_vol[0, 0, sl[0], sl[1], sl[2]] = mask.squeeze()

    pred_vol = pred_vol / weight_vol.clamp(min=1e-8)
    return pred_vol, gt_vol, mask_vol


# ============================================================
# Args
# ============================================================
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../Data/ElementGraph_Tumor_Train")
    ap.add_argument("--case",     required=True,
                    help="Case stem (no .npz). Comma-separate multiple: c1,c2,c3")
    ap.add_argument("--grid_res",   type=int,   default=128)
    ap.add_argument("--patch_size", type=int,   default=64)
    ap.add_argument("--overlap",    type=float, default=0.5)
    ap.add_argument("--s3_splat",   default="mean", choices=["mean", "min"])

    ap.add_argument("--epochs",    type=int,   default=300)
    ap.add_argument("--val_every", type=int,   default=50)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr",         type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--beta",        type=float, default=0.02)
    ap.add_argument("--loss_scale",  type=float, default=10.0)
    ap.add_argument("--fail_threshold", type=float, default=-0.019)
    ap.add_argument("--lambda_fail", type=float, default=0.3)
    ap.add_argument("--gamma",       type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--lambda_lap",  type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--ds_weight",   type=float, default=0.3)
    ap.add_argument("--no_amp",      action="store_true", default=True)

    ap.add_argument("--save_dir", default="../Results/overfit_npz/")
    ap.add_argument("--seed",     type=int, default=42)
    return ap.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = save_dir / "slices"; vis_dir.mkdir(exist_ok=True)

    # ---- Load + voxelise one or more cases ----
    case_list = [c.strip() for c in args.case.split(',') if c.strip()]
    ds = HybridDataset(args.data_dir, grid_res=args.grid_res, s3_splat=args.s3_splat)

    R, P = args.grid_res, args.patch_size
    positions = get_patch_positions(R, P, args.overlap)

    # patches: flat pool, each entry tagged with its case index for val reconstruction.
    # per_case_patches: list[list] — patches grouped by case for full-volume val.
    patches = []
    per_case_patches = []
    for c_idx, case in enumerate(case_list):
        try:
            idx = ds.names.index(case)
        except ValueError:
            print(f"[ERR] case '{case}' not found. first few: {ds.names[:5]}")
            raise SystemExit(2)
        sample = ds[idx]
        print(f"[INFO] Loaded case [{c_idx}]: {case}")

        case_patches = []
        for (sz, sy, sx) in positions:
            sl = (slice(sz, sz+P), slice(sy, sy+P), slice(sx, sx+P))
            tup = (
                sample['geo_input'][:,  sl[0], sl[1], sl[2]].clone(),
                sample['phys_input'][:, sl[0], sl[1], sl[2]].clone(),
                sample['S3_grid'][:,    sl[0], sl[1], sl[2]].clone(),
                sample['mask'][:,       sl[0], sl[1], sl[2]].clone(),
            )
            case_patches.append(tup)
            patches.append(tup)
        per_case_patches.append(case_patches)

    print(f"[INFO] Cached {len(patches)} patches from {len(case_list)} case(s)  "
          f"(R={R}, P={P}, overlap={args.overlap})")

    # ---- Model ----
    model = DualEncoderUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] DualEncoderUNet: {n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    # Warmup + cosine — same schedule as train_hybrid.py
    warmup_epochs = 5
    eta_min = 1e-5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=eta_min)
    print(f"[INFO] Scheduler: LinearLR warmup ({args.lr/10:.1e}->{args.lr:.1e}, "
          f"{warmup_epochs} ep) + CosineAnnealingLR (eta_min={eta_min:.1e})")

    criterion = StrainFailLoss(beta=args.beta, scale=args.loss_scale,
                               fail_threshold=args.fail_threshold,
                               lambda_fail=args.lambda_fail,
                               print_every=int(1e9))

    use_amp = device.type == 'cuda' and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- Log ----
    csv_path = save_dir / "overfit_log.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_mae", "val_r2", "lr", "sec"])

    # ---- Train ----
    indices = list(range(len(patches)))
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(indices)
        t0 = time.time()
        ep_loss, n_steps = 0.0, 0

        # Mini-batch over cached patches
        for bi in range(0, len(indices), args.batch_size):
            batch_idx = indices[bi:bi + args.batch_size]
            geo  = torch.stack([patches[i][0] for i in batch_idx]).to(device)
            phys = torch.stack([patches[i][1] for i in batch_idx]).to(device)
            s3   = torch.stack([patches[i][2] for i in batch_idx]).to(device)
            mask = torch.stack([patches[i][3] for i in batch_idx]).to(device)

            if (mask > 0.5).sum() < 200:
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred, ds2, ds1 = model(geo, phys, bone_mask=mask)
                loss = (criterion(pred, s3, mask) +
                        args.ds_weight * (criterion(ds2, s3, mask) + criterion(ds1, s3, mask)))
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item(); n_steps += 1

        elapsed = time.time() - t0
        avg_loss = ep_loss / max(1, n_steps)
        lr_now = optimizer.param_groups[0]['lr']

        # Scheduler step (warmup first N epochs, then cosine).
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        # ---- Periodic full-volume val (per case, averaged) ----
        val_mae, val_r2, val_loss = float('nan'), float('nan'), float('nan')
        if epoch % args.val_every == 0 or epoch == args.epochs or epoch == 1:
            model.eval()
            mae_list, r2_list, loss_list = [], [], []
            all_g, all_p = [], []
            for c_idx, case_patches in enumerate(per_case_patches):
                pred_vol, gt_vol, mask_vol = reconstruct_full(
                    model, case_patches, positions, R, P, device, use_amp)

                mnp = mask_vol.squeeze().numpy() > 0.5
                gt_mpa   = inv_signed_sqrt(gt_vol.squeeze().numpy())
                pred_mpa = inv_signed_sqrt(pred_vol.squeeze().numpy())
                g_valid = gt_mpa[mnp]; p_valid = pred_mpa[mnp]
                mae_c = float(np.mean(np.abs(g_valid - p_valid)))
                r2_c  = float(1 - np.sum((g_valid - p_valid)**2) /
                                 (np.sum((g_valid - g_valid.mean())**2) + 1e-12))
                mae_list.append(mae_c); r2_list.append(r2_c)
                all_g.append(g_valid); all_p.append(p_valid)

                # Per-case loss in signed_sqrt space
                with torch.no_grad():
                    vl = criterion(pred_vol.to(device), gt_vol.to(device),
                                    mask_vol.to(device))
                    loss_list.append(float(vl.item()))

                # Save per-case slice + scatter
                tag = f"ep{epoch:04d}_c{c_idx}"
                save_slices(pred_vol, gt_vol, mask_vol,
                            vis_dir / f"{tag}.png", epoch)
                save_scatter(g_valid, p_valid,
                             save_dir / f"scatter_{tag}.png",
                             f"{case_list[c_idx]}  ep{epoch}")

            # Aggregate across cases
            val_mae  = float(np.mean(mae_list))
            val_r2   = float(np.mean(r2_list))
            val_loss = float(np.mean(loss_list))

            # Per-case breakdown in log
            per_case_str = "  ".join(
                f"c{i}(R2={r:.3f},MAE={m:.4f})"
                for i, (r, m) in enumerate(zip(r2_list, mae_list)))

            print(f"Ep {epoch:04d} | train {avg_loss:.5f} | "
                  f"val loss={val_loss:.5f} MAE={val_mae:.4f} R2={val_r2:.4f} | "
                  f"{per_case_str} | lr {lr_now:.2e} | {elapsed:.1f}s")
        else:
            print(f"Ep {epoch:04d} | train {avg_loss:.5f} | "
                  f"lr {lr_now:.2e} | {elapsed:.1f}s")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{avg_loss:.6f}",
                f"{val_loss:.6f}" if not np.isnan(val_loss) else "nan",
                f"{val_mae:.4f}"  if not np.isnan(val_mae)  else "nan",
                f"{val_r2:.4f}"   if not np.isnan(val_r2)   else "nan",
                f"{lr_now:.6e}", f"{elapsed:.1f}"])

        # refresh log curve image every val
        if epoch % args.val_every == 0:
            save_log_curves(csv_path, save_dir / "log_curves.png")

    save_log_curves(csv_path, save_dir / "log_curves.png")
    torch.save({'model_state_dict': model.state_dict(),
                'epoch': args.epochs, 'args': vars(args)},
               save_dir / "final.pth")
    print(f"[DONE] Results -> {save_dir}")


if __name__ == "__main__":
    main()
