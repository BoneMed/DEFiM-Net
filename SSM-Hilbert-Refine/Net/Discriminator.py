#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D PatchGAN discriminator + LSGAN loss helpers for adversarial training.

Conditional pix2pix-style: D takes (condition, prediction) and outputs a
patchwise real/fake score map. Conditioning on `geo_input` keeps the
discriminator focused on "is this strain field plausible given THIS
geometry" rather than "is this strain field plausible in general".

LSGAN (least-squares GAN) is used instead of BCE — empirically more
stable for image-to-image translation, smoother gradients, less likely
to suffer mode collapse.

Both predictions and ground-truth are masked by the bone mask before
being fed to D so air voxels don't give D a free signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGAN3D(nn.Module):
    """3D PatchGAN discriminator.

    Architecture: N stride-2 4×4×4 conv layers + GroupNorm + LeakyReLU,
    final 1×1 conv to a single score map. For a 128³ input with
    n_layers=4, the output is ~8³ patchwise scores (each score covers
    ~36-voxel receptive field).

    Args:
        cin: input channels (typically 2 = geo + prediction).
        base_ch: channel count of the first conv layer; doubles each
            layer up to `base_ch * 8`.
        n_layers: number of stride-2 layers (controls patch size of D's
            field of view).
    """
    def __init__(self, cin=2, base_ch=32, n_layers=4):
        super().__init__()
        layers = [
            # No norm on first layer (standard PatchGAN choice).
            nn.Conv3d(cin, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        c = base_ch
        for _ in range(1, n_layers):
            c_next = min(c * 2, base_ch * 8)
            layers.extend([
                nn.Conv3d(c, c_next, 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, c_next),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            c = c_next
        layers.append(nn.Conv3d(c, 1, 1))         # patchwise score head
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _mask_for_d(pred, target, mask):
    """Mask both pred and target by bone mask so D doesn't get a free
    discrimination signal from air voxels (where pred can be anything
    and target is 0).
    """
    m = (mask > 0.5).to(pred.dtype)
    return pred * m, target * m


def lsgan_d_loss(D, geo, target, pred, mask):
    """LSGAN discriminator loss:
        L_D = 0.5 * (MSE(D(real), 1) + MSE(D(fake), 0))

    `pred` must be detached by the caller before passing in — D should
    NOT backprop into the generator on its own step.
    Conditioning by concatenating `geo` along channel dim.
    """
    pred_m, tgt_m = _mask_for_d(pred, target, mask)
    real_pair = torch.cat([geo, tgt_m], dim=1)
    fake_pair = torch.cat([geo, pred_m], dim=1)
    score_real = D(real_pair)
    score_fake = D(fake_pair)
    loss_real = F.mse_loss(score_real, torch.ones_like(score_real))
    loss_fake = F.mse_loss(score_fake, torch.zeros_like(score_fake))
    return 0.5 * (loss_real + loss_fake)


def lsgan_g_loss(D, geo, pred, mask):
    """LSGAN generator adversarial loss:
        L_G_adv = MSE(D(fake), 1)

    The generator's pred is NOT detached here — gradients flow through
    D into the generator. D's params are typically frozen during the G
    step by not calling its optimizer (but `requires_grad` stays True).
    """
    m = (mask > 0.5).to(pred.dtype)
    fake_pair = torch.cat([geo, pred * m], dim=1)
    score_fake = D(fake_pair)
    return F.mse_loss(score_fake, torch.ones_like(score_fake))
