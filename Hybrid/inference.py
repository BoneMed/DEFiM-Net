#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified inference script for DualEncoderUNet_Sparse.

Works for both pretrained (healthy bone) and fine-tuned (tumor) checkpoints.

Two steps:
  Step 1 (--step infer): pytorch-gpu env — inference + metrics + matplotlib plots
  Step 2 (--step render): pyansys env   — pyvista 3D renders from saved npz

Usage:
  # Best by val_loss
  python inference.py --step infer \
    --test_dir ../Data/ElementGraph_Tumor_Test \
    --ckpt ../Model/DualEncoder_Local_half/best_valloss.pth \
    --save_dir ../Results/Local_half_valloss \
    --grid_res 256 --patch_size 128

  # Best by val_mae
  python inference.py --step infer \
    --test_dir ../Data/ElementGraph_Tumor_Test \
    --ckpt ../Model/DualEncoder_Local_half/best_valmae.pth \
    --save_dir ../Results/Local_half_valmae \
    --grid_res 256 --patch_size 128

  # 3D renders (pyansys env)
  python inference.py --step render --save_dir ../Results/Local_half_valloss
"""

import argparse
import csv
import time
import numpy as np
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm


# ============================================================
# Transforms
# ============================================================
def inv_signed_sqrt(y):
    """inverse of signed_sqrt(S3 * 10) -> MPa"""
    return np.sign(y) * (y ** 2) / 10.0


def ansys_cmap():
    return cm.colors.LinearSegmentedColormap.from_list('ansys', [
        (0.0, '#0000ff'), (0.2, '#00ffff'), (0.4, '#00ff00'),
        (0.6, '#ffff00'), (0.8, '#ff8000'), (1.0, '#ff0000')])


# ============================================================
# Metrics
# ============================================================
def compute_metrics(gt, pred):
    valid = np.isfinite(gt) & np.isfinite(pred)
    g, p = gt[valid], pred[valid]
    mae  = float(np.mean(np.abs(g - p)))
    rmse = float(np.sqrt(np.mean((g - p) ** 2)))
    r2   = float(1 - np.sum((g - p) ** 2) / (np.sum((g - g.mean()) ** 2) + 1e-12))
    return {'r2': r2, 'mae': mae, 'rmse': rmse, 'n': int(valid.sum()),
            'gt_min': float(g.min()), 'gt_max': float(g.max()),
            'pred_min': float(p.min()), 'pred_max': float(p.max())}


# ============================================================
# Plots
# ============================================================
def save_scatter(gt, pred, path, title):
    plt.figure(figsize=(7, 7))
    plt.scatter(gt, pred, s=0.5, alpha=0.1, c='steelblue')
    lo, hi = min(gt.min(), pred.min()), max(gt.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], 'r-', lw=1.5)
    r2  = 1 - np.sum((gt-pred)**2) / (np.sum((gt-gt.mean())**2) + 1e-12)
    mae = np.mean(np.abs(gt-pred))
    plt.title(f'{title}\nR2={r2:.4f}  MAE={mae:.4f} MPa')
    plt.xlabel('GT (MPa)'); plt.ylabel('Pred (MPa)')
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()


def save_curve(gt, pred, path, title, max_pts=80000):
    if len(gt) > max_pts:
        idx = np.random.choice(len(gt), max_pts, replace=False)
        gt, pred = gt[idx], pred[idx]
    order = np.argsort(gt)
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})
    axes[0].plot(gt[order], label='GT',   lw=0.5, alpha=0.8)
    axes[0].plot(pred[order], label='Pred', lw=0.5, alpha=0.8)
    axes[0].set_ylabel('S3 (MPa)'); axes[0].set_title(title)
    axes[0].legend(); axes[0].grid(alpha=0.3)
    err = pred[order] - gt[order]
    axes[1].fill_between(range(len(err)), err, 0, alpha=0.4, color='red')
    axes[1].set_ylabel('Error'); axes[1].set_xlabel('Voxel (sorted)')
    axes[1].set_title(f'MAE={np.mean(np.abs(err)):.4f} MPa')
    axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close()


def save_slices(pred_mpa, gt_mpa, mask, path, title):
    cmap = ansys_cmap()
    D, H, W = pred_mpa.shape
    valid = mask > 0.5
    if not valid.any():
        return
    vmin = np.percentile(gt_mpa[valid], 2)
    vmax = np.percentile(gt_mpa[valid], 98)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, (s, t) in enumerate([(D//2, 'Axial'), (H//2, 'Coronal'), (W//2, 'Sagittal')]):
        gs = [gt_mpa[s,:,:],   gt_mpa[:,s,:],   gt_mpa[:,:,s]][i]
        ps = [pred_mpa[s,:,:], pred_mpa[:,s,:], pred_mpa[:,:,s]][i]
        ms = [mask[s,:,:],     mask[:,s,:],     mask[:,:,s]][i]
        axes[0,i].imshow(np.where(ms>0.5, gs, np.nan).T, origin='lower',
                         cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0,i].set_title(f'GT {t}'); axes[0,i].axis('off')
        axes[1,i].imshow(np.where(ms>0.5, ps, np.nan).T, origin='lower',
                         cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1,i].set_title(f'Pred {t}'); axes[1,i].axis('off')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def save_error_slices(pred_mpa, gt_mpa, mask, path, title):
    D, H, W = pred_mpa.shape
    err = np.abs(pred_mpa - gt_mpa)
    valid = mask > 0.5
    if not valid.any():
        return
    emax = np.percentile(err[valid], 95)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (s, t) in enumerate([(D//2, 'Axial'), (H//2, 'Coronal'), (W//2, 'Sagittal')]):
        es = [err[s,:,:], err[:,s,:], err[:,:,s]][i]
        ms = [mask[s,:,:], mask[:,s,:], mask[:,:,s]][i]
        im = axes[i].imshow(np.where(ms>0.5, es, np.nan).T, origin='lower',
                            cmap='hot', vmin=0, vmax=emax)
        axes[i].set_title(f'|Error| {t}'); axes[i].axis('off')
    fig.colorbar(im, ax=axes[-1], fraction=0.03, label='MPa')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def save_summary_chart(all_metrics, save_dir):
    names = [m['name'] for m in all_metrics]
    r2s   = [m['r2']  for m in all_metrics]
    maes  = [m['mae'] for m in all_metrics]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(names)), 6))
    x = np.arange(len(names))
    ax1.bar(x, r2s, color='steelblue')
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('R2'); ax1.set_title(f'R2 per sample (mean={np.mean(r2s):.4f})')
    ax1.set_ylim(0, 1.05); ax1.grid(alpha=0.3, axis='y')
    ax1.axhline(np.mean(r2s), color='red', ls='--', lw=1)
    ax2.bar(x, maes, color='orange')
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('MAE (MPa)'); ax2.set_title(f'MAE per sample (mean={np.mean(maes):.4f})')
    ax2.grid(alpha=0.3, axis='y')
    ax2.axhline(np.mean(maes), color='red', ls='--', lw=1)
    plt.tight_layout()
    plt.savefig(save_dir / 'summary_chart.png', dpi=200); plt.close()


# ============================================================
# Step 1: Inference
# ============================================================
def step_infer(args):
    from net import DualEncoderUNet_Sparse
    from dataUtils import HybridDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    R, P = args.grid_res, args.patch_size
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = DualEncoderUNet_Sparse().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[INFO] Checkpoint: {args.ckpt}  (epoch {ckpt.get('epoch','?')})")
    print(f"[INFO] Grid={R}, Patch={P}, Device={device}")

    # Dataset
    ds = HybridDataset(args.test_dir, grid_res=R, s3_splat=args.s3_splat)
    n_total = len(ds)
    n_run = args.limit if args.limit > 0 else n_total
    n_run = min(n_run, n_total)
    # Optional random subset (deterministic via --random_seed)
    if args.random_seed is not None:
        rng = np.random.RandomState(args.random_seed)
        sample_indices = sorted(rng.choice(n_total, size=n_run, replace=False).tolist())
        print(f"[INFO] Test samples: {n_total} (running random {n_run}, seed={args.random_seed})")
        print(f"[INFO] Selected indices: {sample_indices}")
    else:
        sample_indices = list(range(n_run))
        print(f"[INFO] Test samples: {n_total} (running first {n_run})")

    # Patch positions (50% overlap -> stride = 0.5 * P) — match precut_recut_overlap.py
    overlap_ratio = 0.50
    stride = max(1, int(round(P * (1.0 - overlap_ratio))))
    starts = list(range(0, R - P + 1, stride))
    if starts[-1] + P < R:
        starts.append(R - P)
    patch_positions = [(sz, sy, sx) for sz in starts for sy in starts for sx in starts]
    print(f"[INFO] Patches per sample: {len(patch_positions)}  (stride={stride}, overlap={overlap_ratio*100:.0f}%)")

    # CSV log
    csv_path = save_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['name', 'r2_vox', 'mae_vox', 'rmse_vox',
                                 'r2_elem', 'mae_elem', 'n_voxels', 'n_elements',
                                 'gt_min', 'gt_max', 'pred_min', 'pred_max', 'time_s'])

    all_metrics = []
    all_gt_vox, all_pred_vox = [], []

    for rank, i in enumerate(sample_indices):
        name = ds.names[i]
        sdir = save_dir / name
        sdir.mkdir(exist_ok=True)
        t0 = time.time()
        print(f"\n[{rank+1}/{n_run}] (idx {i}) {name}")

        sample = ds[i]
        raw = np.load(ds.files[i])
        centroids = raw['centroids'].astype(np.float32)
        S3_raw    = raw['S3'].astype(np.float32)

        # Patch inference with overlap averaging (Gaussian-weighted blending)
        pred_vol = torch.zeros(1, 1, R, R, R)
        weight_vol = torch.zeros(1, 1, R, R, R)

        # Gaussian window for blending overlap (peak at center)
        coords = torch.arange(P, dtype=torch.float32) - (P - 1) / 2.0
        sigma = P / 4.0
        w1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        w3d = (w1d[:, None, None] * w1d[None, :, None] * w1d[None, None, :]).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            for sz, sy, sx in patch_positions:
                sl = (slice(sz, sz+P), slice(sy, sy+P), slice(sx, sx+P))
                geo_p  = sample['geo_input'][:,  sl[0], sl[1], sl[2]].unsqueeze(0).to(device)
                phys_p = sample['phys_input'][:, sl[0], sl[1], sl[2]].unsqueeze(0).to(device)
                msk_p  = sample['mask'][:, sl[0], sl[1], sl[2]].unsqueeze(0).to(device)
                pred_p = model(geo_p, phys_p, bone_mask=msk_p).float().cpu()
                pred_vol[:, :, sl[0], sl[1], sl[2]]  += pred_p * w3d
                weight_vol[:, :, sl[0], sl[1], sl[2]] += w3d

        pred_vol = pred_vol / weight_vol.clamp(min=1e-8)
        pred_np  = pred_vol.squeeze().numpy()
        mask     = sample['mask'].squeeze().numpy()
        gt_log   = sample['S3_grid'].squeeze().numpy()
        gt_mpa   = inv_signed_sqrt(gt_log)
        pred_mpa = inv_signed_sqrt(pred_np)

        # ---- Voxel-level metrics ----
        valid = mask > 0.5
        m_vox = compute_metrics(gt_mpa[valid], pred_mpa[valid])

        # ---- Element-level metrics (back-project to mesh centroids) ----
        pad = 0.01
        mins = centroids.min(0); maxs = centroids.max(0)
        size = maxs - mins; size[size == 0] = 1.0
        bbox_min = (mins - pad * size).astype(np.float32)
        bbox_max = (maxs + pad * size).astype(np.float32)
        scale = bbox_max - bbox_min; scale[scale == 0] = 1.0
        q  = np.clip((centroids - bbox_min) / scale, 0, 1 - 1e-6)
        gi = np.clip(np.round(q * (R - 1)).astype(np.int32), 0, R - 1)

        pred_elem = inv_signed_sqrt(pred_np[gi[:, 0], gi[:, 1], gi[:, 2]])
        r2_elem  = float(1 - np.sum((pred_elem - S3_raw)**2) /
                         (np.sum((S3_raw - S3_raw.mean())**2) + 1e-12))
        mae_elem = float(np.mean(np.abs(pred_elem - S3_raw)))

        elapsed = time.time() - t0
        print(f"  Voxel  R2={m_vox['r2']:.4f}  MAE={m_vox['mae']:.4f}  RMSE={m_vox['rmse']:.4f}")
        print(f"  Element R2={r2_elem:.4f}  MAE={mae_elem:.4f}  ({elapsed:.1f}s)")

        all_metrics.append({**m_vox, 'name': name, 'r2_elem': r2_elem, 'mae_elem': mae_elem})
        all_gt_vox.append(gt_mpa[valid])
        all_pred_vox.append(pred_mpa[valid])

        # Write CSV row
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                name,
                f"{m_vox['r2']:.4f}", f"{m_vox['mae']:.4f}", f"{m_vox['rmse']:.4f}",
                f"{r2_elem:.4f}", f"{mae_elem:.4f}",
                m_vox['n'], len(S3_raw),
                f"{m_vox['gt_min']:.2f}", f"{m_vox['gt_max']:.2f}",
                f"{m_vox['pred_min']:.2f}", f"{m_vox['pred_max']:.2f}",
                f"{elapsed:.1f}"])

        # Plots
        title = f"{name}  R2={m_vox['r2']:.4f}  MAE={m_vox['mae']:.4f} MPa"
        save_scatter(gt_mpa[valid], pred_mpa[valid], sdir / 'scatter.png', title)
        save_curve(gt_mpa[valid], pred_mpa[valid], sdir / 'curve.png', f'{name} GT vs Pred')
        save_slices(pred_mpa * mask, gt_mpa * mask, mask.astype(np.float32),
                    sdir / 'slices.png', title)
        save_error_slices(pred_mpa * mask, gt_mpa * mask, mask.astype(np.float32),
                          sdir / 'error_map.png', f'{name} |Error|')

        # Save npz for pyvista render
        np.savez_compressed(sdir / 'eval_data.npz',
                             pred_raw=pred_np.astype(np.float32),
                             gt_raw=gt_log.astype(np.float32),
                             mask=mask.astype(np.float32),
                             bbox_min=bbox_min, bbox_max=bbox_max,
                             centroids=centroids,
                             pred_elem=pred_elem.astype(np.float32),
                             gt_elem=S3_raw.astype(np.float32))
        print(f"  Saved -> {sdir}")

    # Global summary
    gt_all   = np.concatenate(all_gt_vox)
    pred_all = np.concatenate(all_pred_vox)
    gm = compute_metrics(gt_all, pred_all)
    mean_r2   = np.mean([m['r2']      for m in all_metrics])
    mean_mae  = np.mean([m['mae']     for m in all_metrics])
    mean_r2e  = np.mean([m['r2_elem'] for m in all_metrics])
    mean_maee = np.mean([m['mae_elem'] for m in all_metrics])

    print(f"\n{'='*60}")
    print(f"Test Summary ({n_run} samples)")
    print(f"  Voxel  mean R2={mean_r2:.4f}  mean MAE={mean_mae:.4f}")
    print(f"  Element mean R2={mean_r2e:.4f}  mean MAE={mean_maee:.4f}")
    print(f"  Global voxel R2={gm['r2']:.4f}  MAE={gm['mae']:.4f}  RMSE={gm['rmse']:.4f}")
    for m in all_metrics:
        print(f"    {m['name']:30s}  vox R2={m['r2']:.4f}  elem R2={m['r2_elem']:.4f}"
              f"  MAE={m['mae']:.4f}")
    print(f"{'='*60}")

    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([])
        w.writerow(['MEAN', f"{mean_r2:.4f}", f"{mean_mae:.4f}", '',
                    f"{mean_r2e:.4f}", f"{mean_maee:.4f}"])
        w.writerow(['GLOBAL', f"{gm['r2']:.4f}", f"{gm['mae']:.4f}", f"{gm['rmse']:.4f}",
                    '', '', gm['n']])

    save_summary_chart(all_metrics, save_dir)
    save_scatter(gt_all, pred_all, save_dir / 'global_scatter.png',
                 f"All  R2={gm['r2']:.4f}  MAE={gm['mae']:.4f}")
    print(f"\nResults saved to: {save_dir}")


# ============================================================
# Step 2: PyVista 3D renders (off-screen)
# ============================================================
def step_render(args):
    import pyvista as pv
    pv.OFF_SCREEN = True
    pv.global_theme.background = 'white'
    pv.global_theme.font.color = 'black'

    save_dir = Path(args.save_dir)
    sample_dirs = sorted([d for d in save_dir.iterdir()
                          if d.is_dir() and (d / 'eval_data.npz').exists()])
    print(f"Rendering {len(sample_dirs)} samples from {save_dir}")

    for sdir in sample_dirs:
        name = sdir.name
        d = np.load(sdir / 'eval_data.npz')
        mask     = d['mask'] > 0.5
        bbox_min = d['bbox_min']
        bbox_max = d['bbox_max']
        pred_mpa = np.where(mask, inv_signed_sqrt(d['pred_raw']), 0).astype(np.float32)
        gt_mpa   = np.where(mask, inv_signed_sqrt(d['gt_raw']),   0).astype(np.float32)
        R = mask.shape[0]

        valid = mask
        clim  = [float(np.percentile(gt_mpa[valid],  2)),
                 float(np.percentile(gt_mpa[valid], 98))]
        eclim = [0, float(np.percentile(np.abs(pred_mpa - gt_mpa)[valid], 95))]

        spacing = (bbox_max - bbox_min) / R
        grid = pv.ImageData(dimensions=(R+1, R+1, R+1), spacing=spacing, origin=bbox_min)
        grid.cell_data['GT']    = gt_mpa.ravel(order='F')
        grid.cell_data['Pred']  = pred_mpa.ravel(order='F')
        grid.cell_data['Error'] = np.abs(pred_mpa - gt_mpa).ravel(order='F')
        grid.cell_data['mask']  = mask.astype(np.float32).ravel(order='F')

        solid = grid.threshold(0.5, scalars='mask')
        shell = solid.extract_surface()
        center = np.array(solid.center)

        # Clip renders: Y and X axis
        for normal, view_fn, tag in [([0,1,0], 'view_xz', 'Y'), ([1,0,0], 'view_yz', 'X')]:
            pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(3000, 1000))
            for col, (sc, tt, cmap_, cl, sb) in enumerate([
                    ('GT',    'GT',      'jet', clim,  'S3 (MPa)'),
                    ('Pred',  'Pred',    'jet', clim,  'S3 (MPa)'),
                    ('Error', '|Error|', 'hot', eclim, '|Error| (MPa)')]):
                pl.subplot(0, col)
                pl.add_mesh(shell, opacity=0.08, color='gray', show_scalar_bar=False)
                pl.add_mesh(solid.clip(normal=normal, origin=center),
                            scalars=sc, cmap=cmap_, clim=cl, show_scalar_bar=True,
                            scalar_bar_args={'title': sb, 'fmt': '%.2f'})
                pl.add_text(tt, font_size=12); pl.add_axes()
                getattr(pl, view_fn)()
            pl.screenshot(str(sdir / f'{name}_3d_clip{tag}.png')); pl.close()

        # Isometric GT vs Pred
        pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(2400, 1000))
        for col, (sc, tt) in enumerate([('GT', 'GT (MPa)'), ('Pred', 'Pred (MPa)')]):
            pl.subplot(0, col)
            pl.add_mesh(solid, scalars=sc, cmap='jet', clim=clim,
                        show_scalar_bar=True, scalar_bar_args={'title': tt, 'fmt': '%.2f'})
            pl.add_mesh(shell, opacity=0.05, color='gray', show_scalar_bar=False)
            pl.add_text(tt, font_size=14); pl.add_axes(); pl.view_isometric()
        pl.screenshot(str(sdir / f'{name}_3d_iso.png')); pl.close()

        print(f"  {name}: done")

    print(f"All renders saved to {save_dir}")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Unified inference for DualEncoderUNet_Sparse")
    ap.add_argument("--step",       default="infer", choices=["infer", "render", "both"])
    ap.add_argument("--test_dir",   default="../Data/ElementGraph_Tumor_Test")
    ap.add_argument("--ckpt",       default="../Model/DualEncoder_Tumor/best.pth")
    ap.add_argument("--save_dir",   default="../Results/inference")
    ap.add_argument("--grid_res",   type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--s3_splat",   default="mean", choices=["mean", "min"])
    ap.add_argument("--limit",      type=int, default=0,
                    help="Only run first N samples (0 = all)")
    ap.add_argument("--random_seed", type=int, default=None,
                    help="Random seed for sample selection. With --limit N, "
                         "selects N random samples deterministically. "
                         "Same seed across ckpts = same set of cases.")
    args = ap.parse_args()

    if args.step in ("infer", "both"):
        step_infer(args)
    if args.step in ("render", "both"):
        step_render(args)


if __name__ == "__main__":
    main()
