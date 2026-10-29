#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss for voxel-based stress prediction.

  L = weighted_L1(pred, target, bone_mask)
      + lambda_grad * L1(| |∂pred| - |∂target| |, top-k tail region)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       magnitude-gradient edge-sharpness term

Two complementary terms:
  * Main: weighted L1 over all bone voxels. Optional ``tail_weight``
    re-weights the compressive tail (per-voxel weight up to ×4 by
    default).
  * Edge sharpness: ``| |∂pred| − |∂target| |`` on voxel pairs touching
    the top-k most-negative target region. Magnitude (sign-invariant)
    form is more tolerant of one-voxel peak displacement than the
    signed form. Directly fights blob-shaped predictions in the
    high-stress concentration.

An extreme-region MSE term that used to sit alongside the gradient
term was removed — MSE is a conditional-mean estimator, so applying
it inside the top-k tail averages adjacent peaks together, working
against sharpness. Main L1 + tail_weight already supervises voxel
magnitudes there.

Everything operates in the signed-log target encoding space the
network predicts in.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def signed_log_torch(x, scale=100.0):
    """signed log1p transform: sign(x) * log(1 + |x * scale|).

    Used as target encoding for strain — preserves the extreme tail
    better than signed_sqrt because log compresses less aggressively
    near zero AND saturates more gracefully at large magnitudes.
    """
    return torch.sign(x) * torch.log1p(x.abs() * scale)


def inv_signed_log_torch(y, scale=100.0):
    """Inverse of signed_log_torch: sign(y) * (exp|y| - 1) / scale."""
    return torch.sign(y) * y.abs().expm1() / scale


def weighted_l1_masked(pred, target, mask,
                       tail_weight=0.0, tail_power=1.0,
                       max_weight=8.0,
                       mse_weight=0.0):
    """L1 (MAE) with optional small MSE component + optional extra weight
    on large compressive target magnitudes. Operates in the (signed-log)
    target encoding space.

    Per-voxel base = |err| + mse_weight · err².  Pure L1's constant ±1
    gradient is robust but oscillates near the minimum; a small MSE
    provides a quadratic basin near zero so convergence is smooth.
    """
    m = (mask > 0.5).float()
    B = pred.shape[0]

    err = pred - target
    loss = err.abs()
    if mse_weight > 0.0:
        loss = loss + mse_weight * err.pow(2)
    if tail_weight > 0.0:
        comp = (-target).clamp(min=0.0)
        comp_flat = (comp * m).view(B, -1)
        ref = comp_flat.max(dim=1).values.view(B, *([1] * (pred.dim() - 1)))
        ref = ref.clamp(min=1e-6)
        weight = 1.0 + tail_weight * (comp / ref).pow(tail_power)
        weight = weight.clamp(max=max_weight)
    else:
        weight = 1.0

    voxel_count = m.view(B, -1).sum(dim=1).clamp(min=1.0)
    per_sample = (loss * weight * m).view(B, -1).sum(dim=1) / voxel_count
    return per_sample.mean()


def strain_extreme_grad_masked(pred, target, mask,
                               tail_fraction=0.02,
                               min_tail_voxels=64):
    """Edge-sharpness L1 on the top-k compressive tail (magnitude form).

    The MSE term that used to sit alongside this has been removed —
    inside the top-k tail, MSE is mathematically a conditional-mean
    estimator and was averaging multiple peaks, working against the
    sharpness we want. The main weighted-L1 loss (with tail_weight)
    already supervises voxel magnitudes everywhere, so we don't need an
    extra magnitude term on top of it; the gradient L1 below is the
    only part of this function that targets a different statistic
    (spatial derivative), so it's the only part we keep.

    For each axis (z, y, x) and each adjacent voxel pair (a, b):
        * pair contributes iff both endpoints are bone, AND
        * at least one endpoint is in the top-k tail.
    Loss is ``| |∂pred| − |∂target| |`` averaged over contributing pairs.

    The magnitude form is more tolerant of one-voxel peak displacement
    than the signed form: a peak shifted by one voxel flips the sign of
    one pair-gradient, doubling the penalty under signed; under
    magnitude both peaks register the same edge presence.

    Returns the scalar gradient L1 term (zero if no contributing pair).
    """
    m = (mask > 0.5)
    B = pred.shape[0]
    grad_terms = []

    for b in range(B):
        valid = m[b]                                # (1, D, H, W) bool
        n = int(valid.sum().item())
        if n == 0:
            continue

        k = int(math.ceil(n * tail_fraction))
        k = max(1, min(n, max(k, int(min_tail_voxels))))

        tgt_valid = target[b][valid]
        topk_vals, _ = torch.topk(tgt_valid, k=k, largest=False)
        threshold = topk_vals.max()
        extreme_3d = valid & (target[b] <= threshold)   # (1, D, H, W)
        ext_b = extreme_3d.unsqueeze(0)                 # (1, 1, D, H, W)
        bone_b = valid.unsqueeze(0)
        tgt_b = target[b:b + 1]
        pred_b = pred[b:b + 1]

        losses = []
        for axis in (2, 3, 4):
            dp = pred_b.diff(dim=axis)
            dt = tgt_b.diff(dim=axis)
            ea = ext_b.narrow(axis, 0, ext_b.size(axis) - 1)
            eb = ext_b.narrow(axis, 1, ext_b.size(axis) - 1)
            ba = bone_b.narrow(axis, 0, bone_b.size(axis) - 1)
            bb = bone_b.narrow(axis, 1, bone_b.size(axis) - 1)
            pair = ((ea | eb) & ba & bb).float()
            denom = pair.sum().clamp(min=1.0)
            losses.append(((dp.abs() - dt.abs()).abs() * pair).sum() / denom)
        grad_terms.append(torch.stack(losses).mean())

    if not grad_terms:
        return pred.new_zeros(())
    return torch.stack(grad_terms).mean()


def soft_dice_yield_masked(pred, target, mask,
                           thr_enc, temp_enc):
    """Soft Dice on the yield-failure region (target_strain <= threshold).

    Operates entirely in the (signed-log * 100) target encoding space.
    The physical strain threshold is converted to encoded space once by
    the caller and passed in as ``thr_enc``; same for ``temp_enc``.

    Smoothed indicator per voxel: sigmoid((thr_enc - x) / temp_enc).
        x deep below threshold (strong failure) → ≈ 1
        x far above                              → ≈ 0
        x ≈ threshold                            → 0.5

    Per-sample Dice = 2·Σ(g·p) / (Σg + Σp), averaged over the batch.
    Returns ``1 - mean(Dice)``  (so minimising the loss pushes Dice toward 1).
    """
    m = (mask > 0.5).float()
    soft_gt   = torch.sigmoid((thr_enc - target) / temp_enc) * m
    soft_pred = torch.sigmoid((thr_enc - pred)   / temp_enc) * m

    B = pred.shape[0]
    g_flat = soft_gt.view(B, -1)
    p_flat = soft_pred.view(B, -1)

    inter = (g_flat * p_flat).sum(dim=1)
    union = g_flat.sum(dim=1) + p_flat.sum(dim=1)
    dice  = 2.0 * inter / (union + 1e-6)
    return 1.0 - dice.mean()


def yield_threshold_to_encoded(strain_threshold, scale=100.0):
    """Convert a physical strain threshold to the signed_log(*100) space."""
    s = float(strain_threshold)
    return float(np.sign(s)) * float(np.log1p(abs(s) * scale)) if s != 0.0 else 0.0


def yield_temp_to_encoded(strain_temp, strain_threshold, scale=100.0):
    """Convert a physical strain bandwidth to encoded units via the local
    derivative of signed_log at the threshold."""
    d_local = scale / (1.0 + scale * abs(float(strain_threshold)))
    return float(strain_temp) * d_local


class StrainFailLoss(nn.Module):
    """Weighted L1 + magnitude-gradient L1 + optional soft-Dice yield term.

    Four pieces (any can be turned off via its weight):
      * Main L1: weighted L1 over all bone voxels with optional
        compressive re-weighting (``tail_weight``).
      * Extreme grad L1: ``| |∂pred| − |∂target| |`` on voxel pairs
        touching the top-k most negative target region (sharpness).
      * Soft Dice on yield region: 1 - soft Dice between
        ``(pred_strain ≤ threshold)`` and ``(target_strain ≤ threshold)``.
        Pushes overlap of the failure region (the metric the user actually
        cares about). Active when ``lambda_yield > 0``.

    The extreme MSE term has been removed: inside the top-k tail MSE
    is a conditional-mean estimator, which averages across multiple
    peaks and works against sharpness. Main L1 + tail_weight already
    supervises voxel magnitudes in the same region.
    """
    def __init__(self,
                 lambda_grad=0.0,
                 extreme_tail_fraction=0.02,
                 extreme_min_voxels=64, tail_weight=0.0,
                 tail_power=1.0, max_tail_weight=8.0,
                 mse_weight=0.0,
                 lambda_yield=0.0,
                 yield_threshold_strain=-0.019,
                 yield_temp_strain=0.005):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.extreme_tail_fraction = extreme_tail_fraction
        self.extreme_min_voxels = extreme_min_voxels
        self.tail_weight = tail_weight
        self.tail_power = tail_power
        self.max_tail_weight = max_tail_weight
        self.mse_weight = mse_weight
        self.lambda_yield = lambda_yield
        # Pre-convert physical strain threshold / temp to encoded space
        # so the Dice term can compare directly against (pred, target)
        # without per-iter inversion.
        self.yield_thr_enc = yield_threshold_to_encoded(yield_threshold_strain)
        self.yield_temp_enc = yield_temp_to_encoded(
            yield_temp_strain, yield_threshold_strain)

    def forward(self, pred, target, mask, aux=False):
        """
        Args:
            aux: if True, skip the heavy extreme-tail grad term (topk +
                 per-sample Python loop). Used for deep-supervision
                 auxiliary heads where the main L1 signal is enough.
        """
        pred_f = pred.float()
        tgt_f = target.float()
        mask_f = mask.float()

        main = weighted_l1_masked(
            pred_f, tgt_f, mask_f,
            tail_weight=self.tail_weight, tail_power=self.tail_power,
            max_weight=self.max_tail_weight,
            mse_weight=self.mse_weight)

        extreme_grad = pred_f.new_zeros(())
        if self.lambda_grad > 0.0 and not aux:
            extreme_grad = strain_extreme_grad_masked(
                pred_f, tgt_f, mask_f,
                tail_fraction=self.extreme_tail_fraction,
                min_tail_voxels=self.extreme_min_voxels)

        yield_dice = pred_f.new_zeros(())
        if self.lambda_yield > 0.0 and not aux:
            yield_dice = soft_dice_yield_masked(
                pred_f, tgt_f, mask_f,
                thr_enc=self.yield_thr_enc,
                temp_enc=self.yield_temp_enc)

        total = (main
                 + self.lambda_grad * extreme_grad
                 + self.lambda_yield * yield_dice)
        return total
