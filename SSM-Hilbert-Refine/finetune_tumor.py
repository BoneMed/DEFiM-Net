#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSM-Hybrid fine-tuning on tumor (metastasis) data.

Load pretrained checkpoint from healthy bone training,
then full-parameter fine-tune on tumor RST-extracted data.

Same SSM-Hybrid architecture, same loss, same pipeline.
Just different data + pretrained init.
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

from Net import DualEncoderUNet_SSM
from dataUtils import PreCutPatchDataset
from toolkit import StrainFailLoss, save_loss_curve
# Reuse the healthyPreTrain validate_patches so finetune reports identical metrics.
from healthyPreTrain import validate_patches


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrainBatchPrefetcher:
    """
    Move the next DataLoader batch to GPU on a side stream.

    This overlaps CPU->GPU copies with the current forward/backward pass and
    keeps per-batch Python checks off the CUDA stream where possible.
    """
    def __init__(self, loader, device, scan_key=None,
                 min_bone_voxels=200, enabled=True):
        self.loader = loader
        self.device = device
        self.scan_key = scan_key
        self.min_bone_voxels = min_bone_voxels
        self.enabled = enabled and device.type == "cuda"

    def __len__(self):
        return len(self.loader)

    def _prepare(self, batch, non_blocking):
        valid_voxels = None
        if self.min_bone_voxels > 0:
            valid_voxels = int((batch['mask'] > 0.5).sum().item())

        geo = batch['geo_input'].to(self.device, non_blocking=non_blocking)
        phys = batch['phys_input'].to(self.device, non_blocking=non_blocking)
        tgt = batch['S3_grid'].to(self.device, non_blocking=non_blocking)
        msk = batch['mask'].to(self.device, non_blocking=non_blocking)
        if self.scan_key is not None:
            sk = self.scan_key
        else:
            sk = batch['scan_key'].to(self.device, non_blocking=non_blocking)
        return geo, phys, tgt, msk, sk, valid_voxels

    @staticmethod
    def _record_current_stream(batch):
        stream = torch.cuda.current_stream()
        for item in batch:
            if torch.is_tensor(item):
                item.record_stream(stream)

    def __iter__(self):
        if not self.enabled:
            for batch in self.loader:
                yield self._prepare(batch, non_blocking=False)
            return

        stream = torch.cuda.Stream(device=self.device)
        iterator = iter(self.loader)

        def preload(cpu_batch):
            with torch.cuda.stream(stream):
                return self._prepare(cpu_batch, non_blocking=True)

        try:
            next_batch = preload(next(iterator))
        except StopIteration:
            return

        for cpu_batch in iterator:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            batch = next_batch
            self._record_current_stream(batch)
            next_batch = preload(cpu_batch)
            yield batch

        torch.cuda.current_stream(self.device).wait_stream(stream)
        self._record_current_stream(next_batch)
        yield next_batch



# ============================================================
# Args
# ============================================================
def get_args():
    ap = argparse.ArgumentParser(description="Fine-tune SSM-Hybrid on tumor data")

    # ---- I/O ----
    ap.add_argument("--train_dir", default="/mnt/training_Data/SSM-Data/Patches_Tumor_Train_256_128")
    ap.add_argument("--split_json", default="/mnt/training_Data/SSM-Data/tumor_split.json")
    ap.add_argument("--save_dir",   default="../Model/SSMHybrid_Tumor_Refine_Whole")
    ap.add_argument("--patch_dir",  default="/mnt/training_Data/SSM-Data/Patches_Tumor_Train_256_128")
    ap.add_argument("--pretrained",
                    default="../Model/Healthy_Hilbert_MAE_Whole/best_valmae.pth")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to last.pth to resume optimizer / scaler / trackers.")

    # ---- Grid / patches ----
    ap.add_argument("--grid_res", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="Overlap used when patches were cut (stride = P*(1-overlap)).")

    # ---- Training ----
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--train_fraction", type=float, default=1.0,
                    help="Fraction of train cases to keep (0 < f <= 1). "
                         "Sampled deterministically by --seed.")
    ap.add_argument("--val_fraction", type=float, default=1.0,
                    help="Fraction of val cases to keep (0 < f <= 1). "
                         "Validation does per-case full-volume "
                         "reconstruction so this scales runtime linearly.")
    ap.add_argument("--num_workers", type=int, default=18,
                    help="Parallel data workers (nnUNet-style). Each loads a "
                         "235MB full volume + crops; ~10 keeps the GPU fed.")
    ap.add_argument("--prefetch_factor", type=int, default=4,
                    help="Batches each worker prefetches ahead (hides 235MB IO).")
    ap.add_argument("--min_bone_voxels", type=int, default=200,
                    help="Skip batches whose mask has fewer bone voxels.")
    ap.add_argument("--amp_dtype", choices=['bf16', 'fp16'], default='bf16')

    # ---- Architecture (must match pretrained) ----
    ap.add_argument("--sparse_hidden", type=int, default=16)
    ap.add_argument("--sparse_layers", type=int, default=5)
    ap.add_argument("--mamba_d_state", type=int, default=16)
    ap.add_argument("--mamba_d_conv", type=int, default=4)
    ap.add_argument("--mamba_expand", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="nn.Dropout3d rate applied in every ResBlock, "
                         "Mamba block, Up fuse, and before the head. "
                         "0.0 = no dropout. Try 0.1 for moderate regularisation.")

    # ---- Loss (L1 main + magnitude-grad L1 on extreme tail) ----
    ap.add_argument("--lambda_grad", type=float, default=0.8,
                    help="L1 of magnitude gradient |∂pred-∂target| on "
                         "voxel pairs touching the top-k compressive "
                         "tail. Fights blob-shaped predictions in the "
                         "high-stress concentration.")
    ap.add_argument("--extreme_tail_fraction", type=float, default=0.05)
    ap.add_argument("--extreme_min_voxels", type=int, default=128)
    ap.add_argument("--tail_weight", type=float, default=3.0,
                    help="Per-voxel weight on the compressive tail.")
    ap.add_argument("--tail_power", type=float, default=1.0)
    ap.add_argument("--max_tail_weight", type=float, default=8.0)
    ap.add_argument("--mse_weight", type=float, default=0.1,
                    help="Small MSE added to main L1 (|err| + mse_weight·err²) "
                         "for smoother convergence near the optimum. Keep small "
                         "— larger MSE = more over-smoothing.")
    # ---- Yield-region soft Dice loss + validation overlap metric ----
    ap.add_argument("--lambda_yield", type=float, default=0.5,
                    help="Weight on soft Dice yield loss. Tumor default 0.5; "
                         "set 0 to disable.")
    ap.add_argument("--yield_threshold", type=float, default=-0.019,
                    help="Compressive strain threshold for failure region.")
    ap.add_argument("--yield_temp", type=float, default=0.005,
                    help="Sigmoid bandwidth (physical strain units) for soft Dice.")

    return ap.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[INFO] Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Datasets (fixed-grid pre-cut patches) ----
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
        npz_dir=args.train_dir, grid_res=args.grid_res,
        patch_size=args.patch_size, overlap=args.overlap)
    val_patch_ds = PreCutPatchDataset(
        patch_dir, case_filter=val_cases,
        npz_dir=args.train_dir, grid_res=args.grid_res,
        patch_size=args.patch_size, overlap=args.overlap)

    # Cache the (constant) Hilbert scan_key once on GPU to skip per-sample collation.
    train_scan_key = None
    if getattr(train_ds, "_scan_key", None) is not None:
        train_scan_key = train_ds._scan_key.unsqueeze(0).unsqueeze(0).to(device)
        train_ds._scan_key = None
    print(f"[INFO] Train: {len(train_ds)} patches  |  Val: {len(val_patch_ds)} patches")

    loader_kwargs = dict(batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"),
                         drop_last=False)
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True,
                             prefetch_factor=args.prefetch_factor)
    train_loader = DataLoader(train_ds, **loader_kwargs)
    train_batches = TrainBatchPrefetcher(
        train_loader, device, scan_key=train_scan_key,
        min_bone_voxels=args.min_bone_voxels)

    # ---- Model: load pretrained ----
    pretrained_ckpt = None
    if args.pretrained and os.path.isfile(args.pretrained):
        pretrained_ckpt = torch.load(args.pretrained, map_location=device)
        ckpt_args = pretrained_ckpt.get('args', {})
        arch_defaults = {
            'sparse_hidden': 16,
            'sparse_layers': 5,
            'mamba_d_state': 16,
            'mamba_d_conv': 4,
            'mamba_expand': 2,
        }
        for key, default_value in arch_defaults.items():
            if key in ckpt_args and getattr(args, key) == default_value:
                setattr(args, key, ckpt_args[key])
        print("[INFO] Architecture after pretrained inheritance: "
              f"sparse_hidden={args.sparse_hidden}, "
              f"sparse_layers={args.sparse_layers}, "
              f"mamba_d_state={args.mamba_d_state}, "
              f"mamba_d_conv={args.mamba_d_conv}, "
              f"mamba_expand={args.mamba_expand}")

    model = DualEncoderUNet_SSM(
        sparse_hidden=args.sparse_hidden,
        sparse_layers=args.sparse_layers,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if pretrained_ckpt is not None:
        # strict=False so old ckpts that still carry ds1/ds2 weights
        # (deep supervision was removed in this version) load cleanly.
        missing, unexpected = model.load_state_dict(
            pretrained_ckpt['model_state_dict'], strict=False)
        ds_unexpected = [k for k in unexpected if k.startswith(('ds1', 'ds2'))]
        other_unexpected = [k for k in unexpected if k not in ds_unexpected]
        if ds_unexpected:
            print(f"[INFO] Ignored {len(ds_unexpected)} legacy DS keys "
                  f"(ds1/ds2 removed in current arch)")
        if other_unexpected:
            print(f"[WARN] unexpected keys in ckpt: {other_unexpected}")
        if missing:
            print(f"[WARN] missing keys in current model: {missing}")
        pre_epoch = pretrained_ckpt.get('epoch', '?')
        pre_loss = pretrained_ckpt.get('loss', pretrained_ckpt.get('val_loss', '?'))
        print(f"[INFO] Loaded pretrained: {args.pretrained}")
        print(f"  (epoch {pre_epoch}, loss {pre_loss})")
    else:
        print(f"[WARN] No pretrained checkpoint, training from scratch")

    print(f"[INFO] DualEncoderUNet_SSM: {n_params:,} ({n_params/1e6:.2f}M)  "
          f"sparse(h={args.sparse_hidden}, L={args.sparse_layers})  "
          f"mamba(state={args.mamba_d_state}, conv={args.mamba_d_conv}, "
          f"expand={args.mamba_expand})")

    # ---- Optimizer (full-param fine-tune) ----
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    print(f"[INFO] AdamW (full-param fine-tune): "
          f"lr={args.lr:.1e}  wd={args.weight_decay:.1e}")

    steps_per_epoch = max(1, len(train_loader))
    # Fixed learning rate — no warmup, no cosine decay. lr stays at args.lr
    # the whole run.
    print(f"[INFO] Fixed LR = {args.lr:.1e} (no scheduler)")

    criterion = StrainFailLoss(lambda_grad=args.lambda_grad,
                               extreme_tail_fraction=args.extreme_tail_fraction,
                               extreme_min_voxels=args.extreme_min_voxels,
                               tail_weight=args.tail_weight,
                               tail_power=args.tail_power,
                               max_tail_weight=args.max_tail_weight,
                               mse_weight=args.mse_weight,
                               lambda_yield=args.lambda_yield,
                               yield_threshold_strain=args.yield_threshold,
                               yield_temp_strain=args.yield_temp)
    print(f"[INFO] loss: (L1 + {args.mse_weight}·MSE)(tail_w={args.tail_weight}) + "
          f"λ_grad={args.lambda_grad}·∂L1 on top "
          f"{args.extreme_tail_fraction*100:.1f}% tail + "
          f"λ_yield={args.lambda_yield}·SoftDice(@strain={args.yield_threshold}, "
          f"temp={args.yield_temp})")

    csv_path = Path(args.save_dir) / "finetune_log.csv"
    resuming = bool(args.resume and os.path.isfile(args.resume))
    if resuming and csv_path.exists():
        print(f"[INFO] Resuming — appending to existing log: {csv_path}")
    else:
        if resuming:
            print(f"[INFO] Resuming but log missing — creating fresh {csv_path}")
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss",
                "val_loss", "val_mae", "val_r2",
                "val_global_mae", "val_global_r2",
                "val_mre_case", "val_mre_voxel",
                "yield_dice_case", "yield_dice_global",
                "lr", "sec",
            ])

    use_amp = (device.type == 'cuda')
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16  # only fp16 needs scaling
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    print(f"[INFO] AMP={use_amp} dtype={amp_dtype} scaler={use_scaler}  "
          f"steps/epoch={steps_per_epoch}")

    best_val_mae       = float('inf')
    best_val_loss      = float('inf')
    best_val_mre_case  = float('inf')
    best_val_mre_voxel = float('inf')
    val_patience_ctr   = 0
    start_epoch = 1

    # ---- Resume: restore model + optimizer + scaler + trackers ----
    if args.resume and os.path.isfile(args.resume):
        rckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(rckpt['model_state_dict'], strict=True)
        # Optimizer / scheduler / scaler only exist in last.pth (full
        # training state); best_*.pth carry model weights + metrics only.
        # Tolerate their absence so resuming from a best ckpt still works
        # (just reinitialises the optimizer instead of crashing).
        if 'optimizer_state_dict' in rckpt:
            optimizer.load_state_dict(rckpt['optimizer_state_dict'])
            scaler.load_state_dict(rckpt['scaler_state_dict'])
            # fixed-LR run: no scheduler state to restore
        else:
            print("[WARN] resume ckpt has no optimizer state "
                  "(best_*.pth?) — weights loaded, optimizer reinitialised.")
        start_epoch        = int(rckpt.get('epoch', 0)) + 1
        best_val_mae       = float(rckpt.get('best_val_mae',       best_val_mae))
        best_val_loss      = float(rckpt.get('best_val_loss',      best_val_loss))
        best_val_mre_case  = float(rckpt.get('best_val_mre_case',  best_val_mre_case))
        best_val_mre_voxel = float(rckpt.get('best_val_mre_voxel', best_val_mre_voxel))
        val_patience_ctr   = int(rckpt.get('val_patience_ctr',     0))
        print(f"[INFO] Resume @ epoch {start_epoch}  "
              f"best_val_mae={best_val_mae:.5f}  "
              f"best_val_mre_case={best_val_mre_case:.5f}  "
              f"best_val_mre_voxel={best_val_mre_voxel:.5f}")

    LOG_EVERY = 5  # tqdm postfix refresh rate

    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        model.train()
        ep_loss = torch.zeros((), device=device)
        n_steps = 0
        t0 = time.time()

        loop = tqdm(train_batches, total=len(train_loader),
                    desc=f"FT Ep {epoch}/{args.epochs}",
                    ncols=110)
        for geo, phys, tgt, msk, sk, valid_voxels in loop:
            if valid_voxels is not None and valid_voxels < args.min_bone_voxels:
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                pred = model(geo, phys, bone_mask=msk, scan_key=sk)
                loss = criterion(pred, tgt, msk)

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

            ep_loss = ep_loss + loss.detach()
            n_steps += 1
            if n_steps % LOG_EVERY == 0:
                loop.set_postfix(loss=f"{loss.detach().item():.4f}")

        elapsed = time.time() - t0
        avg_loss = (ep_loss / max(1, n_steps)).item()
        lr = optimizer.param_groups[0]['lr']
        # fixed LR — no scheduler step

        # ---- Validation (every epoch) ----
        v_mae = v_r2 = v_gmae = v_gr2 = v_loss = float('nan')
        v_mre_case = v_mre_voxel = float('nan')
        v_yield_case = v_yield_glb = float('nan')

        if len(val_patch_ds) > 0:
            (v_mae, v_r2, v_gmae, v_gr2, v_loss,
             v_mre_case, v_mre_voxel,
             v_yield_case, v_yield_glb) = validate_patches(
                model, val_patch_ds, criterion, device, use_amp,
                epoch, str(Path(args.save_dir) / "val"),
                grid_res=args.grid_res, save_vis=True,
                amp_dtype=amp_dtype,
                yield_threshold=args.yield_threshold)
            print(f"  [Val] "
                  f"loss={v_loss:.5f}  perCase MAE={v_mae:.4f} R2={v_r2:.4f}  "
                  f"global MAE={v_gmae:.4f} R2={v_gr2:.4f}  "
                  f"MRE(L2)={v_mre_case:.4f} MRE(MAPE)={v_mre_voxel:.4f}  "
                  f"Yield-Dice case={v_yield_case:.3f} global={v_yield_glb:.3f}")

        def _fmt(x, d=4):
            return f"{x:.{d}f}" if not np.isnan(x) else "nan"
        print(f"Ep {epoch:03d} | train {avg_loss:.5f} | "
              f"val MAE={v_mae:.4f} R2={v_r2:.4f} yield_dice={v_yield_case:.3f} | "
              f"LR {lr:.2e} | {elapsed:.1f}s")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{avg_loss:.6f}",
                _fmt(v_loss, 6), _fmt(v_mae), _fmt(v_r2),
                _fmt(v_gmae), _fmt(v_gr2),
                _fmt(v_mre_case), _fmt(v_mre_voxel),
                _fmt(v_yield_case), _fmt(v_yield_glb),
                f"{lr:.6e}", f"{elapsed:.1f}",
            ])

        def _save_ckpt(path_name):
            p = Path(args.save_dir) / path_name
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'train_loss': avg_loss,
                        'val_loss': v_loss, 'val_mae': v_mae, 'val_r2': v_r2,
                        'val_global_mae': v_gmae, 'val_global_r2': v_gr2,
                        'val_mre_case': v_mre_case, 'val_mre_voxel': v_mre_voxel,
                        'pretrained': args.pretrained,
                        'args': vars(args)}, p)
            return p

        if not np.isnan(v_mae) and v_mae < best_val_mae:
            best_val_mae = v_mae
            val_patience_ctr = 0
            p = _save_ckpt("best_val_mae.pth")
            print(f"  -> New best val_mae={v_mae:.5f} @ ep{epoch} -> {p.name}")
        else:
            val_patience_ctr += 1

        if not np.isnan(v_loss) and v_loss < best_val_loss:
            best_val_loss = v_loss
            p = _save_ckpt("best_val_loss.pth")
            print(f"  -> New best val_loss={v_loss:.5f} @ ep{epoch} -> {p.name}")

        if not np.isnan(v_mre_case) and v_mre_case < best_val_mre_case:
            best_val_mre_case = v_mre_case
            p = _save_ckpt("best_val_mre_case.pth")
            print(f"  -> New best val_mre_case={v_mre_case:.5f} @ ep{epoch} -> {p.name}")

        if not np.isnan(v_mre_voxel) and v_mre_voxel < best_val_mre_voxel:
            best_val_mre_voxel = v_mre_voxel
            p = _save_ckpt("best_val_mre_voxel.pth")
            print(f"  -> New best val_mre_voxel={v_mre_voxel:.5f} @ ep{epoch} -> {p.name}")

        # last.pth: full resumable state, written every epoch
        last_state = {
            'model_state_dict':            model.state_dict(),
            'optimizer_state_dict':        optimizer.state_dict(),
            'scaler_state_dict':           scaler.state_dict(),
            'epoch':                       epoch,
            'best_val_mae':                best_val_mae,
            'best_val_loss':               best_val_loss,
            'best_val_mre_case':           best_val_mre_case,
            'best_val_mre_voxel':          best_val_mre_voxel,
            'val_patience_ctr':            val_patience_ctr,
            'train_loss':                  avg_loss,
            'val_loss': v_loss, 'val_mae': v_mae, 'val_r2': v_r2,
            'val_mre_case': v_mre_case, 'val_mre_voxel': v_mre_voxel,
            'pretrained': args.pretrained,
            'args':       vars(args),
        }
        torch.save(last_state, Path(args.save_dir) / "last.pth")

        if val_patience_ctr >= args.patience:
            print(f"Early stop: val_mae plateaued for {args.patience} val periods.")
            break

        if epoch % 10 == 0:
            save_loss_curve(csv_path, Path(args.save_dir) / "finetune_curve.png")

    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                'args': vars(args)}, Path(args.save_dir) / "final.pth")
    save_loss_curve(csv_path, Path(args.save_dir) / "finetune_curve.png")
    print("Done.")


if __name__ == "__main__":
    main()
