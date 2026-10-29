#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
u_disp sensitivity probe.

Run inference on one test case twice:
  run A: unchanged phys_input
  run B: phys_input[:, 1] (Load*u channel) scaled by --scale (default 2.0)

If the model truly uses the u magnitude, pred should scale roughly
proportionally in the load-transfer region. If pred barely changes,
the network has learned to ignore u.

Usage (pytorch-gpu env):
  python toolkit/u_sensitivity.py \
    --ckpt ../Model/DualEncoder_Tumor_directly_4ch_val_full/best_valmae.pth \
    --test_dir ../Data/ElementGraph_Tumor_Test \
    --case MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp18 \
    --scale 2.0 \
    --save_dir ../Results/u_sensitivity
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from net import DualEncoderUNet_Sparse
from dataUtils import HybridDataset


def inv_signed_sqrt(y):
    return np.sign(y) * (y ** 2) / 10.0


def run_patches(model, sample, device, R, P, stride, scale=1.0):
    """Full-volume Gaussian-blended patch inference. scale multiplies Load*u channel."""
    starts = list(range(0, R - P + 1, stride))
    if starts[-1] + P < R:
        starts.append(R - P)
    positions = [(sz, sy, sx) for sz in starts for sy in starts for sx in starts]

    coords = torch.arange(P, dtype=torch.float32) - (P - 1) / 2.0
    sigma = P / 4.0
    w1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    w3d = (w1d[:, None, None] * w1d[None, :, None] * w1d[None, None, :]).unsqueeze(0).unsqueeze(0)

    pred_vol = torch.zeros(1, 1, R, R, R)
    weight_vol = torch.zeros(1, 1, R, R, R)

    geo_full  = sample['geo_input']
    phys_full = sample['phys_input'].clone()
    if scale != 1.0:
        phys_full[1] = phys_full[1] * scale  # only Load*u channel
    mask_full = sample['mask']

    with torch.no_grad():
        for sz, sy, sx in positions:
            sl = (slice(sz, sz+P), slice(sy, sy+P), slice(sx, sx+P))
            g = geo_full[:,  sl[0], sl[1], sl[2]].unsqueeze(0).to(device)
            p = phys_full[:, sl[0], sl[1], sl[2]].unsqueeze(0).to(device)
            m = mask_full[:, sl[0], sl[1], sl[2]].unsqueeze(0).to(device)
            out = model(g, p, bone_mask=m).float().cpu()
            pred_vol[:, :, sl[0], sl[1], sl[2]]   += out * w3d
            weight_vol[:, :, sl[0], sl[1], sl[2]] += w3d

    pred_vol = pred_vol / weight_vol.clamp(min=1e-8)
    return pred_vol.squeeze().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_dir", default="../Data/ElementGraph_Tumor_Test")
    ap.add_argument("--case", required=True)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--grid_res", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--save_dir", default="../Results/u_sensitivity")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    model = DualEncoderUNet_Sparse().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[INFO] Loaded ckpt epoch {ckpt.get('epoch','?')}")

    ds = HybridDataset(args.test_dir, grid_res=args.grid_res, s3_splat='mean')
    idx = None
    for i, n in enumerate(ds.names):
        if args.case in n:
            idx = i; break
    if idx is None:
        print(f"[ERR] {args.case} not found in {args.test_dir}"); return
    print(f"[INFO] Case: {ds.names[idx]}")

    sample = ds[idx]
    mask_np = sample['mask'].squeeze().numpy() > 0.5
    P, R = args.patch_size, args.grid_res
    stride = max(1, int(round(P * 0.5)))

    # Run A: original
    print(f"[RUN A] scale=1.0")
    predA = run_patches(model, sample, device, R, P, stride, scale=1.0)
    # Run B: scaled
    print(f"[RUN B] scale={args.scale}")
    predB = run_patches(model, sample, device, R, P, stride, scale=args.scale)

    # Convert to MPa
    predA_mpa = inv_signed_sqrt(predA)
    predB_mpa = inv_signed_sqrt(predB)
    gt_mpa    = inv_signed_sqrt(sample['S3_grid'].squeeze().numpy())

    # Stats on masked voxels
    a = predA_mpa[mask_np]; b = predB_mpa[mask_np]; g = gt_mpa[mask_np]
    ratio = b / np.where(np.abs(a) > 1e-8, a, np.nan)
    print("="*60)
    print(f"GT  range         : [{g.min():.4f}, {g.max():.4f}] MPa   mean={g.mean():.4f}")
    print(f"Pred A (×1)       : [{a.min():.4f}, {a.max():.4f}] MPa   mean={a.mean():.4f}")
    print(f"Pred B (×{args.scale})       : [{b.min():.4f}, {b.max():.4f}] MPa   mean={b.mean():.4f}")
    print(f"B / A ratio        : median={np.nanmedian(ratio):.3f}  "
          f"p25={np.nanpercentile(ratio,25):.3f}  p75={np.nanpercentile(ratio,75):.3f}")
    print(f"MAE(A, B)          : {np.mean(np.abs(a - b)):.5f} MPa  "
          f"(GT dyn range: {g.max()-g.min():.4f})")
    rel_change = np.mean(np.abs(a - b)) / max(np.mean(np.abs(a)), 1e-8)
    print(f"Mean |Δpred| / |predA|: {rel_change*100:.2f}%   "
          f"(expected ~{(args.scale-1)*100:.0f}% if fully sensitive)")
    print("="*60)

    # Save slice comparison
    D, H, W = predA_mpa.shape
    vmin = float(np.percentile(g, 2)); vmax = float(np.percentile(g, 98))
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    for i, (s, t) in enumerate([(D//2, 'Axial'), (H//2, 'Coronal'), (W//2, 'Sagittal')]):
        def sl(vol): return [vol[s,:,:], vol[:,s,:], vol[:,:,s]][i]
        def mk(vol): return [mask_np[s,:,:], mask_np[:,s,:], mask_np[:,:,s]][i]
        for row, (vol, title) in enumerate([
                (predA_mpa, f'Pred ×1 {t}'),
                (predB_mpa, f'Pred ×{args.scale} {t}'),
                (predB_mpa - predA_mpa, f'Δ (B-A) {t}')]):
            cmap = 'jet' if row < 2 else 'seismic'
            v = (vmin, vmax) if row < 2 else (-abs(vmax-vmin)*0.2, abs(vmax-vmin)*0.2)
            im = axes[row, i].imshow(
                np.where(mk(vol), sl(vol), np.nan).T, origin='lower',
                cmap=cmap, vmin=v[0], vmax=v[1])
            axes[row, i].set_title(title); axes[row, i].axis('off')
            plt.colorbar(im, ax=axes[row, i], fraction=0.03)
    fig.suptitle(f"{args.case}  |  u scale ×1 vs ×{args.scale}", fontsize=14)
    plt.tight_layout()
    out_path = save_dir / f"{args.case}_u_x{args.scale}.png"
    plt.savefig(out_path, dpi=180); plt.close()
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
