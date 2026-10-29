#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full fine-tuning on tumor (metastasis) data.

Load pretrained checkpoint from healthy bone training,
then full-parameter fine-tune on tumor RST-extracted data.

Same architecture, same loss, same pipeline.
Just different data + pretrained init.
"""

import copy
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

from net import DualEncoderUNet_Sparse
from dataUtils import HybridDataset, PatchTrainDataset, PreCutPatchDataset
from toolkit import StrainFailLoss
# Reuse the train_hybrid validate_patches so finetune reports identical metrics.
from train_hybrid import validate_patches


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def inv_signed_log1p(y):
    return np.sign(y) * (np.exp(np.abs(y)) - 1)

def inv_signed_sqrt(y):
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


def save_slices(pred_vol, gt_vol, mask_vol, path, epoch):
    p = inv_signed_sqrt(pred_vol.squeeze().cpu().numpy())
    g = inv_signed_sqrt(gt_vol.squeeze().cpu().numpy())
    m = mask_vol.squeeze().cpu().numpy()
    D, H, W = p.shape
    cmap = ansys_cmap()
    valid = m > 0.5
    if valid.sum() == 0: return
    vmin, vmax = np.percentile(g[valid], 2), np.percentile(g[valid], 98)
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
    if len(epochs) < 2: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, losses, lw=1)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Train Loss'); axes[0].set_yscale('log'); axes[0].grid(alpha=0.3)
    vr = [(e, r) for e, r in zip(epochs, r2s) if not np.isnan(r)]
    if vr: axes[1].plot([e for e,_ in vr], [r for _,r in vr], 'o-', lw=1, ms=3)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('R²')
    axes[1].set_title('Val R²'); axes[1].set_ylim(-0.1, 1.05); axes[1].grid(alpha=0.3)
    vm = [(e, m) for e, m in zip(epochs, maes) if not np.isnan(m)]
    if vm: axes[2].plot([e for e,_ in vm], [m for _,m in vm], 'o-', lw=1, ms=3, color='orange')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('MAE (MPa)')
    axes[2].set_title('Val MAE'); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


# ============================================================
# Full-volume validation (fixed patches)
# ============================================================
@torch.no_grad()
def validate_full(model, full_ds, val_indices, grid_res, patch_size,
                   device, use_amp, epoch, save_dir, n_vis=3):
    """
    Validate on ALL val cases.  Reports both:
      - per-case mean MAE / R²  (paper-friendly, less sensitive to outliers)
      - global MAE / R²         (one statistic over the concatenated voxel pool,
                                 same definition as train_hybrid.py)

    Visualisation (slices/scatter/curve) is saved only for the first
    `n_vis` cases to keep I/O bounded.

    Returns (mean_per_case_mae, mean_per_case_r2, global_mae, global_r2).
    """
    model.eval()
    vis_dir = Path(save_dir) / "val_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    R, P = grid_res, patch_size
    starts = list(range(0, R - P + 1, P))
    if starts[-1] + P < R:
        starts.append(R - P)

    # Streaming global stats (no concat)
    g_sum_ae = 0.0; g_sse = 0.0; g_sum_gt = 0.0; g_sum_gt_sq = 0.0; g_n = 0
    # Per-case stats
    per_case_mae, per_case_r2 = [], []
    # Reservoir for global scatter/curve viz (small)
    VIS_VOXELS_PER_CASE = 5000
    res_gt, res_pred = [], []
    rng = np.random.RandomState(0)

    for vi_idx, vi in enumerate(val_indices):
        sample = full_ds[vi]
        pred_vol = torch.zeros(1, 1, R, R, R)
        for sz in starts:
            for sy in starts:
                for sx in starts:
                    sl = (slice(sz, sz+P), slice(sy, sy+P), slice(sx, sx+P))
                    geo_p  = sample['geo_input'][:,  sl[0], sl[1], sl[2]].unsqueeze(0).to(device)
                    phys_p = sample['phys_input'][:, sl[0], sl[1], sl[2]].unsqueeze(0).to(device)
                    msk_p  = sample['mask'][:,       sl[0], sl[1], sl[2]].unsqueeze(0).to(device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        pred_p = model(geo_p, phys_p, bone_mask=msk_p)
                    pred_vol[:, :, sl[0], sl[1], sl[2]] = pred_p.float().cpu()

        m_np = sample['mask'].squeeze().numpy() > 0.5
        gt_mpa = inv_signed_sqrt(sample['S3_grid'].squeeze().numpy()[m_np])
        pr_mpa = inv_signed_sqrt(pred_vol.squeeze().numpy()[m_np])
        valid_mask = np.isfinite(gt_mpa) & np.isfinite(pr_mpa)
        gt_v = gt_mpa[valid_mask]; pr_v = pr_mpa[valid_mask]
        if len(gt_v) == 0:
            continue

        # Per-case metrics
        diff = gt_v - pr_v
        mae_c = float(np.abs(diff).mean())
        tss_c = float(((gt_v - gt_v.mean()) ** 2).sum())
        sse_c = float((diff ** 2).sum())
        r2_c  = float(1.0 - sse_c / max(tss_c, 1e-12))
        per_case_mae.append(mae_c)
        per_case_r2.append(r2_c)

        # Global streaming
        g_sum_ae    += np.abs(diff).sum()
        g_sse       += (diff ** 2).sum()
        g_sum_gt    += gt_v.sum()
        g_sum_gt_sq += (gt_v ** 2).sum()
        g_n         += len(gt_v)

        # Reservoir for vis (bounded)
        k = min(VIS_VOXELS_PER_CASE, len(gt_v))
        idx = rng.choice(len(gt_v), size=k, replace=False)
        res_gt.append(gt_v[idx]); res_pred.append(pr_v[idx])

        # Slice visualization for first n_vis cases only
        if vi_idx < n_vis:
            save_slices(pred_vol[0], sample['S3_grid'], sample['mask'],
                        vis_dir / f"ep{epoch:04d}_case{vi_idx}_slices.png", epoch)

    if g_n == 0 or not per_case_mae:
        return float('nan'), float('nan'), float('nan'), float('nan')

    # Per-case means
    mean_pc_mae = float(np.mean(per_case_mae))
    mean_pc_r2  = float(np.mean(per_case_r2))

    # Global stats (matches train_hybrid.py)
    mean_gt = g_sum_gt / g_n
    tss     = g_sum_gt_sq - g_n * mean_gt * mean_gt
    global_mae = float(g_sum_ae / g_n)
    global_r2  = float(1.0 - g_sse / max(tss, 1e-12))

    # Global scatter/curve from reservoir
    if res_gt:
        gt_vis = np.concatenate(res_gt)
        pr_vis = np.concatenate(res_pred)
        save_scatter(gt_vis, pr_vis,
                     vis_dir / f"ep{epoch:04d}_scatter.png",
                     f"Ep{epoch} Val ({len(per_case_mae)} cases, "
                     f"vis subsample ~{len(gt_vis)})")
        save_curve(gt_vis, pr_vis,
                   vis_dir / f"ep{epoch:04d}_curve.png",
                   f"Ep{epoch} GT vs Pred (subsample)")

    return mean_pc_mae, mean_pc_r2, global_mae, global_r2


# ============================================================
# Args
# ============================================================
def get_args():
    ap = argparse.ArgumentParser(description="Fine-tune on tumor data")
    ap.add_argument("--train_dir", default="../Data/ElementGraph_Tumor_Train",
                    help="(legacy) source ElementGraph dir.")
    ap.add_argument("--test_dir", default="../Data/ElementGraph_Tumor_Test",
                    help="(legacy) tumor test dir, NOT used for val anymore.")
    ap.add_argument("--split_json", default="../Data/tumor_split.json",
                    help="Manifest with train_cases/val_cases (held out from train pool). "
                         "Test set stays untouched for final inference.")
    ap.add_argument("--save_dir", default="../Model/DualEncoder_Tumor_finetune/")
    ap.add_argument("--pretrained",
                    default="../Model/DualEncoder_Healthy/best_valloss.pth",
                    help="Pretrained checkpoint from healthy bone training. "
                         "Architecture must match (same DualEncoderUNet_Sparse).")

    ap.add_argument("--grid_res", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--max_cache", type=int, default=30)
    ap.add_argument("--patch_dir", default="../Data/Patches_Tumor_Train_256_128",
                    help="Pre-cut patch dir. If exists, use PreCutPatchDataset.")

    ap.add_argument("--epochs", type=int, default=150,
                    help="Fine-tune typically converges within ~100 epochs.")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="Lower than from-scratch lr=5e-4. Tumor fine-tune "
                         "starts close to a good optimum; smaller steps "
                         "avoid blowing away the healthy prior.")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=20)
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
    ap.add_argument("--gamma", type=float, default=0.0,
                    help=argparse.SUPPRESS)
    ap.add_argument("--ds_weight", type=float, default=0.3)

    ap.add_argument("--val_interval", type=int, default=1,
                    help="Run validation every N epochs (default 1 = every "
                         "epoch, matches train_hybrid.py).")

    # ---- Fine-tune transfer strategy: LLRD + freeze + EMA + healthy replay ----
    ap.add_argument("--encoder_lr_mult", type=float, default=0.1,
                    help="LLRD: encoder param groups use lr * this multiplier.")
    ap.add_argument("--head_lr_mult", type=float, default=2.5,
                    help="LLRD: head + DS heads use lr * this multiplier.")
    ap.add_argument("--freeze_encoder_epochs", type=int, default=5,
                    help="Freeze encoder for first N epochs.")
    ap.add_argument("--ema_decay", type=float, default=0.999,
                    help="EMA decay (0 to disable).")
    ap.add_argument("--healthy_replay_ratio", type=float, default=0.1,
                    help="Healthy review loss weight α: total = loss_t + α*loss_h.")
    ap.add_argument("--healthy_patch_dir", default="../Data/Patches_Healthy_Train_256_128",
                    help="Pool of healthy patches for replay (only train_cases used).")
    ap.add_argument("--healthy_split_json", default="../Data/healthy_split.json",
                    help="Healthy split.json — train_cases for replay, val_cases for dual val.")
    ap.add_argument("--joint_score_alpha", type=float, default=1.0,
                    help="ckpt selection penalty coefficient: "
                         "score = tumor_val_X + α * max(0, healthy_val_X - baseline_X). "
                         "α=0 -> ignore healthy. α=1 -> equal trade. "
                         "α large -> strict healthy preservation.")

    ap.add_argument("--train_subsample_ratio", type=float, default=0.2,
                    help="Use only this fraction of tumor train_cases "
                         "(deterministic via --seed). 1.0 = use all; "
                         "0.2 = use 20% to test low-data regime.")

    # ---- Continual learning: EWC (Elastic Weight Consolidation) ----
    # No healthy data is touched at training time. A pre-computed Fisher
    # Information matrix + healthy parameter snapshot pulls important
    # parameters back toward their healthy values via a quadratic penalty.
    # ---- Legacy EWC args (no longer used in this run, kept for back-compat) ----
    ap.add_argument("--ewc_lambda", type=float, default=0.0,
                    help="EWC penalty weight: total = loss + lambda * Σ F_i (θ_i - θ_h_i)². "
                         "Sensitive to scale: typical values 1e2..1e5. Set 0 to disable.")
    ap.add_argument("--ewc_fisher_path", default="../Model/DualEncoder_Healthy/fisher.pt",
                    help="Path to fisher.pt produced by compute_ewc_fisher.py.")
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

    # ---- Datasets (split.json -> train_cases / val_cases pool) ----
    patch_dir  = Path(args.patch_dir)
    split_json = Path(args.split_json) if args.split_json else None
    if split_json is None or not split_json.exists():
        raise SystemExit(f"--split_json required for fine-tune. Missing: {split_json}")
    if not patch_dir.exists() or not list(patch_dir.glob("*.pt")):
        raise SystemExit(f"--patch_dir empty: {patch_dir}")

    import json
    with open(split_json, 'r', encoding='utf-8') as f:
        split = json.load(f)
    train_cases = split['train_cases']
    val_cases   = split['val_cases']
    print(f"[INFO] Split: {split_json}  "
          f"({len(train_cases)} train / {len(val_cases)} val cases)")

    # Optional: subsample tumor train cases (low-data regime experiment)
    if args.train_subsample_ratio < 1.0:
        rng = random.Random(args.seed)
        n_keep = max(1, int(round(len(train_cases) * args.train_subsample_ratio)))
        train_cases = sorted(rng.sample(train_cases, n_keep))
        print(f"[INFO] Subsampled to {n_keep} train cases "
              f"(ratio={args.train_subsample_ratio})")

    train_ds     = PreCutPatchDataset(patch_dir, case_filter=train_cases)
    val_patch_ds = PreCutPatchDataset(patch_dir, case_filter=val_cases)
    print(f"[INFO] Train: {len(train_ds)} patches")
    print(f"[INFO] Val:   {len(val_patch_ds)} patches  "
          f"(test_dir held out, not used during fine-tune)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # ---- Healthy replay loader (Method B: dual-batch synchronized) + healthy val ----
    use_replay = args.healthy_replay_ratio > 0
    healthy_loader = None
    healthy_iter   = None
    healthy_val_ds = None

    h_split_json = Path(args.healthy_split_json)
    h_patch_dir  = Path(args.healthy_patch_dir)
    if not h_split_json.exists() or not h_patch_dir.exists():
        print(f"[WARN] healthy paths missing -> replay & dual val disabled "
              f"({h_split_json} / {h_patch_dir})")
        use_replay = False
    else:
        with open(h_split_json, 'r', encoding='utf-8') as f:
            h_split = json.load(f)
        # Replay: only train_cases (val_cases stay clean for evaluation)
        if use_replay:
            h_train_ds = PreCutPatchDataset(h_patch_dir, case_filter=h_split['train_cases'])
            healthy_loader = DataLoader(
                h_train_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True)
            healthy_iter = iter(healthy_loader)
            print(f"[INFO] Healthy replay: {len(h_train_ds)} patches "
                  f"({len(h_split['train_cases'])} cases), α={args.healthy_replay_ratio}")
        # Dual val: healthy val_cases (held-out, never seen during replay)
        healthy_val_ds = PreCutPatchDataset(h_patch_dir, case_filter=h_split['val_cases'])
        print(f"[INFO] Healthy dual-val: {len(healthy_val_ds)} patches "
              f"({len(h_split['val_cases'])} cases)")

    def next_healthy_batch():
        nonlocal healthy_iter
        try:
            return next(healthy_iter)
        except StopIteration:
            healthy_iter = iter(healthy_loader)
            return next(healthy_iter)

    # ---- Model: load pretrained ----
    model = DualEncoderUNetSparse().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if args.pretrained and os.path.isfile(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        pre_epoch = ckpt.get('epoch', '?')
        pre_loss = ckpt.get('loss', '?')
        print(f"[INFO] Loaded pretrained: {args.pretrained}")
        print(f"  (epoch {pre_epoch}, loss {pre_loss})")
    else:
        print(f"[WARN] No pretrained checkpoint, training from scratch")

    print(f"[INFO] DualEncoderUNet_Sparse: {n_params:,} ({n_params/1e6:.2f}M)")

    # ---- LLRD param groups: encoder / body / head ----
    encoder_modules = [model.sparse_encoder, model.stem_g,
                       model.e1_g, model.e2_g, model.e3_g,
                       model.e1_p, model.e2_p, model.e3_p]
    head_modules    = [model.head, model.ds1, model.ds2]
    encoder_params = [p for m in encoder_modules for p in m.parameters()]
    head_params    = [p for m in head_modules    for p in m.parameters()]
    enc_ids = {id(p) for p in encoder_params}
    head_ids = {id(p) for p in head_params}
    body_params = [p for p in model.parameters()
                   if id(p) not in enc_ids and id(p) not in head_ids]

    enc_lr  = args.lr * args.encoder_lr_mult
    head_lr = args.lr * args.head_lr_mult
    optimizer = torch.optim.AdamW(
        [
            {'params': encoder_params, 'lr': enc_lr,  'name': 'encoder'},
            {'params': body_params,    'lr': args.lr, 'name': 'body'},
            {'params': head_params,    'lr': head_lr, 'name': 'head'},
        ],
        weight_decay=args.weight_decay)
    print(f"[INFO] LLRD lrs: encoder={enc_lr:.1e}  body={args.lr:.1e}  head={head_lr:.1e}")

    # ---- Freeze encoder helper ----
    def set_encoder_requires_grad(flag: bool):
        for p in encoder_params:
            p.requires_grad = flag
    if args.freeze_encoder_epochs > 0:
        set_encoder_requires_grad(False)
        print(f"[INFO] Freezing encoder for first {args.freeze_encoder_epochs} epochs")

    steps_per_epoch = max(1, len(train_loader))
    warmup_epochs = 5
    eta_min = 1e-5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=eta_min)
    print(f"[INFO] Scheduler: LinearLR warmup ({args.lr/10:.1e}->{args.lr:.1e}, "
          f"{warmup_epochs} ep) + CosineAnnealingLR (eta_min={eta_min:.1e})")

    # ---- EMA model for validation (smooths fine-tune fluctuations) ----
    use_ema = args.ema_decay > 0
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad = False
        print(f"[INFO] EMA enabled, decay={args.ema_decay}")
    else:
        print(f"[INFO] EMA disabled (--ema_decay 0)")

    def update_ema(ema, src, decay):
        with torch.no_grad():
            for p_e, p in zip(ema.parameters(), src.parameters()):
                p_e.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
            # Copy buffers (BN/SpatialNorm running stats) directly — no decay.
            for b_e, b in zip(ema.buffers(), src.buffers()):
                b_e.data.copy_(b.data)

    criterion = StrainFailLoss(beta=args.beta, scale=args.loss_scale,
                               fail_threshold=args.fail_threshold,
                               lambda_fail=args.lambda_fail,
                               print_every=500)
    ds_w = args.ds_weight

    # ---- Log ----
    # Columns:
    #   train_loss : mean per-batch loss this epoch (signed_sqrt space + DS)
    #   val_mae / val_r2 : per-CASE mean (paper-friendly)
    #   val_global_mae / val_global_r2 : single statistic on the concatenated
    #     voxel pool (matches train_hybrid.py's reported numbers)
    csv_path = Path(args.save_dir) / "finetune_log.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch",
            "train_loss",                 # = loss_t + α * loss_h
            "loss_tumor", "loss_healthy", # split components (replay)
            # Tumor val metrics
            "val_t_loss", "val_t_mae", "val_t_r2",
            "val_t_global_mae", "val_t_global_r2",
            # Healthy val metrics (dual val)
            "val_h_loss", "val_h_mae", "val_h_r2",
            "val_h_global_mae", "val_h_global_r2",
            # Joint score (lower = better) for ckpt selection.
            # score = tumor + α * max(0, healthy - baseline)
            "score_mae", "score_loss",
            "lr", "sec",
        ])

    use_amp = False  # FP32 to avoid AMP overflow NaN
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[INFO] AMP={use_amp}, LR={args.lr}, Steps/epoch={steps_per_epoch}")

    # ---- Healthy baseline (measured once, before any tumor training) ----
    # Used as the threshold for joint score penalty:
    #   score = tumor_val_X + α * max(0, healthy_val_X - baseline_X)
    healthy_baseline_mae  = float('nan')
    healthy_baseline_loss = float('nan')
    if healthy_val_ds is not None and len(healthy_val_ds) > 0:
        print(f"[INFO] Measuring healthy baseline (epoch 0, pretrained ckpt) ...")
        bl_mae, bl_r2, bl_gmae, bl_gr2, bl_loss = validate_patches(
            model, healthy_val_ds, criterion, device, use_amp,
            epoch=0, save_dir=str(Path(args.save_dir) / "healthy_baseline"),
            grid_res=args.grid_res, save_vis=False)
        healthy_baseline_mae  = bl_mae
        healthy_baseline_loss = bl_loss
        print(f"[INFO] Healthy baseline: MAE={bl_mae:.4f}  R2={bl_r2:.4f}  "
              f"loss={bl_loss:.5f}")

    best_train_loss = float('inf')
    best_score_mae  = float('inf')
    best_score_loss = float('inf')
    val_patience_ctr = 0   # counts VAL periods without score_mae improvement

    # ---- Training ----
    for epoch in range(1, args.epochs + 1):
        # Unfreeze encoder once warmup-freeze period ends.
        if (args.freeze_encoder_epochs > 0
                and epoch == args.freeze_encoder_epochs + 1):
            set_encoder_requires_grad(True)
            print(f"[INFO] Encoder unfrozen at epoch {epoch}")

        model.train()
        ep_loss_t, ep_loss_h, n_steps = 0.0, 0.0, 0
        t0 = time.time()

        loop = tqdm(train_loader, desc=f"FT Ep {epoch}/{args.epochs}", ncols=110)
        for batch in loop:
            geo = batch['geo_input'].to(device)
            phys = batch['phys_input'].to(device)
            tgt = batch['S3_grid'].to(device)
            msk = batch['mask'].to(device)

            if (msk > 0.5).sum() < 200:
                continue

            # ---- Tumor batch (学新): forward + backward, accumulate grads ----
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_t, ds2_t, ds1_t = model(geo, phys, bone_mask=msk)
                loss_t = (criterion(pred_t, tgt, msk) +
                          ds_w * (criterion(ds2_t, tgt, msk) + criterion(ds1_t, tgt, msk)))
            if torch.isnan(loss_t) or torch.isinf(loss_t):
                continue
            scaler.scale(loss_t).backward()

            # ---- Healthy replay batch (复习): separate forward + accumulate grads ----
            loss_h_val = 0.0
            if use_replay:
                hb = next_healthy_batch()
                geo_h  = hb['geo_input'].to(device)
                phys_h = hb['phys_input'].to(device)
                tgt_h  = hb['S3_grid'].to(device)
                msk_h  = hb['mask'].to(device)
                if (msk_h > 0.5).sum() >= 200:
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        pred_h, ds2_h, ds1_h = model(geo_h, phys_h, bone_mask=msk_h)
                        loss_h = (criterion(pred_h, tgt_h, msk_h) +
                                  ds_w * (criterion(ds2_h, tgt_h, msk_h) +
                                          criterion(ds1_h, tgt_h, msk_h)))
                    if not (torch.isnan(loss_h) or torch.isinf(loss_h)):
                        scaler.scale(args.healthy_replay_ratio * loss_h).backward()
                        loss_h_val = loss_h.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA update
            if use_ema:
                update_ema(ema_model, model, args.ema_decay)

            ep_loss_t += loss_t.item()
            ep_loss_h += loss_h_val
            n_steps += 1
            loop.set_postfix(t=f"{loss_t.item():.4f}", h=f"{loss_h_val:.4f}")

        elapsed = time.time() - t0
        avg_loss_t = ep_loss_t / max(1, n_steps)
        avg_loss_h = ep_loss_h / max(1, n_steps) if use_replay else float('nan')
        avg_loss = avg_loss_t + (args.healthy_replay_ratio * avg_loss_h
                                  if use_replay else 0.0)
        # Body group lr is the representative one (encoder/head are scaled).
        lr = next((g['lr'] for g in optimizer.param_groups
                   if g.get('name') == 'body'),
                  optimizer.param_groups[0]['lr'])

        # Scheduler: warmup first N epochs, then cosine annealing
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        # ---- Dual validation (tumor val + healthy val) ----
        do_val = (epoch % args.val_interval == 0) or epoch == 1
        # Tumor val
        t_mae = t_r2 = t_gmae = t_gr2 = t_loss = float('nan')
        # Healthy val
        h_mae = h_r2 = h_gmae = h_gr2 = h_loss = float('nan')

        if do_val:
            do_vis = (epoch % args.val_interval == 0) or epoch == 1
            eval_model = ema_model if use_ema else model

            if len(val_patch_ds) > 0:
                t_mae, t_r2, t_gmae, t_gr2, t_loss = validate_patches(
                    eval_model, val_patch_ds, criterion, device, use_amp,
                    epoch, str(Path(args.save_dir) / "tumor_val"),
                    grid_res=args.grid_res, save_vis=do_vis)
                print(f"  [Tumor Val{'-EMA' if use_ema else ''}] "
                      f"loss={t_loss:.5f}  perCase MAE={t_mae:.4f} R2={t_r2:.4f}  "
                      f"global MAE={t_gmae:.4f} R2={t_gr2:.4f}")

            if healthy_val_ds is not None and len(healthy_val_ds) > 0:
                h_mae, h_r2, h_gmae, h_gr2, h_loss = validate_patches(
                    eval_model, healthy_val_ds, criterion, device, use_amp,
                    epoch, str(Path(args.save_dir) / "healthy_val"),
                    grid_res=args.grid_res, save_vis=do_vis)
                print(f"  [Healthy Val{'-EMA' if use_ema else ''}] "
                      f"loss={h_loss:.5f}  perCase MAE={h_mae:.4f} R2={h_r2:.4f}  "
                      f"global MAE={h_gmae:.4f} R2={h_gr2:.4f}")

            # Joint score (penalty form, lower = better):
            #   score = tumor_X + α * max(0, healthy_X - baseline_X)
            # Healthy below baseline -> no penalty (no degradation).
            # Healthy above baseline -> penalty proportional to degradation.
            a = args.joint_score_alpha
            score_mae = float('nan')
            score_loss = float('nan')
            if not (np.isnan(t_mae) or np.isnan(h_mae) or np.isnan(healthy_baseline_mae)):
                score_mae = t_mae + a * max(0.0, h_mae - healthy_baseline_mae)
            if not (np.isnan(t_loss) or np.isnan(h_loss) or np.isnan(healthy_baseline_loss)):
                score_loss = t_loss + a * max(0.0, h_loss - healthy_baseline_loss)
        else:
            score_mae = float('nan')
            score_loss = float('nan')

        loss_breakdown = (f"t={avg_loss_t:.4f} h={avg_loss_h:.4f}"
                          if use_replay else f"t={avg_loss_t:.4f}")
        score_str = (f"score MAE={score_mae:.4f} loss={score_loss:.4f}"
                     if not np.isnan(score_mae) else "")
        print(f"Ep {epoch:03d} | {loss_breakdown} → total {avg_loss:.5f} | "
              f"{score_str} | LR {lr:.2e} | {elapsed:.1f}s")

        with open(csv_path, "a", newline="") as f:
            def fmt(x, d=4):
                return f"{x:.{d}f}" if not np.isnan(x) else "nan"
            csv.writer(f).writerow([
                epoch,
                f"{avg_loss:.6f}",
                f"{avg_loss_t:.6f}",
                f"{avg_loss_h:.6f}" if not np.isnan(avg_loss_h) else "nan",
                fmt(t_loss, 6), fmt(t_mae), fmt(t_r2), fmt(t_gmae), fmt(t_gr2),
                fmt(h_loss, 6), fmt(h_mae), fmt(h_r2), fmt(h_gmae), fmt(h_gr2),
                fmt(score_mae), fmt(score_loss, 6),
                f"{lr:.6e}", f"{elapsed:.1f}",
            ])

        # When EMA is enabled, val-based ckpts save EMA weights.
        save_eval_state = ema_model.state_dict() if use_ema else model.state_dict()

        def _save_ckpt(path_name):
            p = Path(args.save_dir) / path_name
            torch.save({'model_state_dict': save_eval_state,
                        'epoch': epoch,
                        'train_loss': avg_loss,
                        'val_t_loss': t_loss, 'val_t_mae': t_mae, 'val_t_r2': t_r2,
                        'val_t_global_mae': t_gmae, 'val_t_global_r2': t_gr2,
                        'val_h_loss': h_loss, 'val_h_mae': h_mae, 'val_h_r2': h_r2,
                        'val_h_global_mae': h_gmae, 'val_h_global_r2': h_gr2,
                        'healthy_baseline_mae':  healthy_baseline_mae,
                        'healthy_baseline_loss': healthy_baseline_loss,
                        'score_mae': score_mae, 'score_loss': score_loss,
                        'is_ema': use_ema,
                        'pretrained': args.pretrained,
                        'args': vars(args)}, p)
            return p

        # ---- Best by score_mae (LOWER is better, primary ckpt selection) ----
        if do_val and not np.isnan(score_mae) and score_mae < best_score_mae:
            best_score_mae = score_mae
            val_patience_ctr = 0
            p = _save_ckpt("best_score_mae.pth")
            print(f"  -> New best score_mae={score_mae:.5f} @ ep{epoch} -> {p.name}")
        elif do_val:
            val_patience_ctr += 1

        # ---- Best by score_loss (LOWER is better) ----
        if do_val and not np.isnan(score_loss) and score_loss < best_score_loss:
            best_score_loss = score_loss
            p = _save_ckpt("best_score_loss.pth")
            print(f"  -> New best score_loss={score_loss:.5f} @ ep{epoch} -> {p.name}")

        # ---- Best by train_loss: live model (train metric, not EMA-applicable) ----
        if not np.isnan(avg_loss) and avg_loss < best_train_loss:
            best_train_loss = avg_loss
            p = Path(args.save_dir) / "best_trainloss.pth"
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch, 'train_loss': avg_loss,
                        'val_t_r2': t_r2, 'val_h_r2': h_r2,
                        'score_mae': score_mae, 'score_loss': score_loss,
                        'is_ema': False,
                        'pretrained': args.pretrained,
                        'args': vars(args)}, p)
            print(f"  -> New best train_loss={avg_loss:.5f} @ ep{epoch} -> {p.name}")

        # Early stop on score_mae plateau
        if do_val and val_patience_ctr >= args.patience:
            print(f"Early stop: score_mae no improvement for "
                  f"{args.patience} val periods.")
            break

        if epoch % 10 == 0:
            save_loss_curve(csv_path, Path(args.save_dir) / "finetune_curve.png")

    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                'args': vars(args)}, Path(args.save_dir) / "final.pth")
    save_loss_curve(csv_path, Path(args.save_dir) / "finetune_curve.png")
    print("Done.")


if __name__ == "__main__":
    main()
