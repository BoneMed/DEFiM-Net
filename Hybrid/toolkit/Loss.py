#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss functions for voxel-based stress prediction.

Current training uses StrainFailLoss:

  L = SmoothL1(pred_sqrt, gt_sqrt, bone_mask)
      + lambda_fail * MAE(pred_mpa, gt_mpa, failure_region)

The failure region is selected in physical space from the ground truth:
gt_mpa <= fail_threshold.  Older Laplacian/focal losses are kept below for
comparison scripts and old checkpoints.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PRINT_EVERY = int(os.environ.get("LOSS_PRINT_EVERY", "50"))


def inv_signed_sqrt_torch(y, scale=10.0):
    """Inverse of signed_sqrt(x * scale): y -> x."""
    return y * y.abs() / scale


def smooth_l1_masked(pred, target, mask, beta=0.02):
    """
    Per-sample SmoothL1 in transformed signed-sqrt space, averaged on bone voxels.
    """
    m = (mask > 0.5).float()
    B = pred.shape[0]

    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction='none')
    voxel_count = m.view(B, -1).sum(dim=1).clamp(min=1.0)
    per_sample = (loss * m).view(B, -1).sum(dim=1) / voxel_count
    return per_sample.mean()


def strain_fail_mae_masked(pred, target, mask, scale=10.0, fail_threshold=-0.019):
    """
    MAE in physical MPa/strain space, only where GT is in the compressive
    failure region.  Samples without failure voxels do not contribute to this
    term, so batches with no failure region get a zero auxiliary loss.
    """
    m = (mask > 0.5)
    B = pred.shape[0]

    pred_phys = inv_signed_sqrt_torch(pred, scale=scale)
    target_phys = inv_signed_sqrt_torch(target, scale=scale)
    fail_mask = (target_phys <= fail_threshold) & m

    fail_count = fail_mask.view(B, -1).sum(dim=1)
    has_fail = fail_count > 0
    if not has_fail.any():
        return pred.new_zeros(())

    abs_err = (pred_phys - target_phys).abs() * fail_mask.float()
    per_sample = abs_err.view(B, -1).sum(dim=1) / fail_count.clamp(min=1.0)
    return per_sample[has_fail].mean()


class StrainFailLoss(nn.Module):
    """
    Masked SmoothL1 in signed-sqrt space plus physical-space failure MAE.

    beta is an error threshold in transformed space. fail_threshold is a
    physical-space stress/strain threshold after inverse signed-sqrt transform.
    No Laplacian smoothing is used.
    """
    def __init__(self, beta=0.02, scale=10.0, fail_threshold=-0.019,
                 lambda_fail=0.3, print_every=DEFAULT_PRINT_EVERY):
        super().__init__()
        self.beta = beta
        self.scale = scale
        self.fail_threshold = fail_threshold
        self.lambda_fail = lambda_fail
        self.print_every = print_every
        self._step = 0

    def _is_rank0(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def forward(self, pred, target, mask):
        pred_f = pred.float()
        tgt_f = target.float()
        mask_f = mask.float()

        main = smooth_l1_masked(pred_f, tgt_f, mask_f, beta=self.beta)
        fail = strain_fail_mae_masked(
            pred_f, tgt_f, mask_f, scale=self.scale,
            fail_threshold=self.fail_threshold)
        total = main + self.lambda_fail * fail

        self._step += 1
        if self.print_every and (self._step % self.print_every == 0) and self._is_rank0():
            print(f"[Loss] step={self._step} "
                  f"main={main.item():.6f} fail={fail.item():.6f} "
                  f"total={total.item():.6f}")

        return total


def focal_mse_masked(pred, target, mask, gamma=2.0):
    """
    Per-sample Focal MSE with mask support for 3D volumes.

    Each sample in the batch gets its own error_max and mean-weight
    normalization, so cases with very different stress ranges
    contribute comparably to the gradient. The old per-batch version
    let the highest-stress sample dominate and effectively zero out
    the gradient from low-stress samples.

      for each sample b:
        focal_weight_b = (|err_b| / max|err_b|) ^ gamma
        focal_weight_b /= mean(focal_weight_b on bone voxels)
        loss_b = mean_on_bone(focal_weight_b * err_b^2)
      return mean_over_batch(loss_b)

    gamma=0: standard MSE (uniform weights).
    gamma=2: strong focus on large-error voxels within each sample.
    """
    m = (mask > 0.5).float()
    B = pred.shape[0]

    abs_err = (pred - target).abs()

    # Per-sample max over bone voxels -> (B, 1, 1, 1, 1)
    masked_err = abs_err * m
    error_max = masked_err.view(B, -1).max(dim=1, keepdim=True)[0]
    error_max = error_max.view(B, 1, 1, 1, 1).detach().clamp(min=1e-6)

    focal_weight = (abs_err / error_max).detach() ** gamma

    # Per-sample mean weight -> normalize so mean=1 on each sample's bone voxels
    voxel_count = m.view(B, -1).sum(dim=1).clamp(min=1.0)        # (B,)
    fw_sum      = (focal_weight * m).view(B, -1).sum(dim=1)      # (B,)
    mean_fw = (fw_sum / voxel_count).view(B, 1, 1, 1, 1).clamp(min=1e-6)
    focal_weight = focal_weight / mean_fw

    # Per-sample MSE on bone voxels, then uniform mean over the batch
    sq_err = (pred - target) ** 2
    per_sample = ((focal_weight * sq_err * m).view(B, -1).sum(dim=1)
                  / voxel_count)                                  # (B,)
    return per_sample.mean()


def laplacian_3d_loss(pred, target, mask):
    """
    3D discrete Laplacian loss on voxel grid.

    For each voxel, Laplacian = voxel_value - mean(6-neighbors).
    Loss = mean_masked( (Lap(pred) - Lap(gt))^2 )

    Encourages pred to match GT's local curvature / high-frequency detail.
    """
    m = (mask > 0.5).float()

    # 3x3x3 Laplacian kernel: center=6, 6-neighbors=-1
    kernel = torch.zeros(1, 1, 3, 3, 3, device=pred.device, dtype=pred.dtype)
    kernel[0, 0, 1, 1, 1] = 6.0
    kernel[0, 0, 0, 1, 1] = -1.0
    kernel[0, 0, 2, 1, 1] = -1.0
    kernel[0, 0, 1, 0, 1] = -1.0
    kernel[0, 0, 1, 2, 1] = -1.0
    kernel[0, 0, 1, 1, 0] = -1.0
    kernel[0, 0, 1, 1, 2] = -1.0

    lap_pred = F.conv3d(pred, kernel, padding=1)
    lap_gt = F.conv3d(target, kernel, padding=1)

    # Erode mask by 1 voxel to avoid boundary artifacts
    mask_eroded = F.conv3d(m, torch.ones(1, 1, 3, 3, 3, device=m.device, dtype=m.dtype) / 27.0, padding=1)
    mask_eroded = (mask_eroded > 0.99).float()

    diff = ((lap_pred - lap_gt) ** 2) * mask_eroded
    return diff.sum() / (mask_eroded.sum() + 1e-8)


def huber_masked(pred, target, mask, delta=0.4359):
    """
    Per-sample Huber loss with mask support for 3D volumes.

    For |err| < delta:  loss = 0.5 * err^2          (quadratic, like MSE)
    For |err| >= delta: loss = delta*(|err| - 0.5*delta)  (linear, like MAE)

    Default delta = sqrt(0.19) ≈ 0.4359 — the transformed-space magnitude
    of the failure strain (raw -0.019 -> signed_sqrt(-0.019*10) = -0.4359).
    Errors below this stay quadratic so most voxels train like MSE; errors
    above this become linear so single-voxel outliers can't dominate the
    gradient.
    """
    m = (mask > 0.5).float()
    B = pred.shape[0]

    diff = (pred - target) * m
    abs_err = diff.abs()

    # Standard Huber per-element
    quad = 0.5 * diff * diff
    lin  = delta * (abs_err - 0.5 * delta)
    loss = torch.where(abs_err < delta, quad, lin)

    # Per-sample mean on bone voxels, then uniform batch mean
    voxel_count = m.view(B, -1).sum(dim=1).clamp(min=1.0)
    per_sample = (loss * m).view(B, -1).sum(dim=1) / voxel_count
    return per_sample.mean()


class HuberLapLoss(nn.Module):
    """
    L = Huber(pred, gt, mask, delta) + lambda_lap * Laplacian3D(pred, gt, mask)
    """
    def __init__(self, lambda_lap=0.1, delta=0.4359, print_every=DEFAULT_PRINT_EVERY):
        super().__init__()
        self.lambda_lap = lambda_lap
        self.delta = delta
        self.print_every = print_every
        self._step = 0

    def _is_rank0(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def forward(self, pred, target, mask):
        pred_f = pred.float()
        tgt_f = target.float()
        mask_f = mask.float()

        hub = huber_masked(pred_f, tgt_f, mask_f, delta=self.delta)
        lap = laplacian_3d_loss(pred_f, tgt_f, mask_f)

        total = hub + self.lambda_lap * lap

        self._step += 1
        if self.print_every and (self._step % self.print_every == 0) and self._is_rank0():
            print(f"[Loss] step={self._step} "
                  f"huber={hub.item():.6f} lap={lap.item():.6f} "
                  f"total={total.item():.6f}")

        return total


class FocalLapLoss(nn.Module):
    """
    L = FocalMSE(pred, gt, mask, gamma) + lambda_lap * Laplacian3D(pred, gt, mask)
    """
    def __init__(self, lambda_lap=0.5, gamma=2.0, print_every=DEFAULT_PRINT_EVERY):
        super().__init__()
        self.lambda_lap = lambda_lap
        self.gamma = gamma
        self.print_every = print_every
        self._step = 0

    def _is_rank0(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def forward(self, pred, target, mask):
        pred_f = pred.float()
        tgt_f = target.float()
        mask_f = mask.float()

        fmse = focal_mse_masked(pred_f, tgt_f, mask_f, gamma=self.gamma)
        lap = laplacian_3d_loss(pred_f, tgt_f, mask_f)

        total = fmse + self.lambda_lap * lap

        self._step += 1
        if self.print_every and (self._step % self.print_every == 0) and self._is_rank0():
            print(f"[Loss] step={self._step} "
                  f"focal={fmse.item():.6f} lap={lap.item():.6f} "
                  f"total={total.item():.6f}")

        return total


if __name__ == "__main__":
    B, D = 1, 32
    pred = torch.randn(B, 1, D, D, D)
    gt = torch.randn(B, 1, D, D, D)
    mask = (torch.rand(B, 1, D, D, D) > 0.3).float()

    loss_fn = FocalLapLoss(lambda_lap=0.5, gamma=2.0, print_every=1)
    loss = loss_fn(pred, gt, mask)
    print(f"[Test] Loss Output: {loss.item():.6f}")
