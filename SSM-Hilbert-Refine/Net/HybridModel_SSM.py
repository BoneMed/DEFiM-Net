#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sparse-Mamba Hybrid network — Hilbert-scan variant.

Physics front-end:
    [BC, Load*u] -> SparsePhysicsEncoder -> dense physics feature
    [dist_BC, dist_Load] are fused inside SparsePhysicsEncoder.

Backbone per branch:
    level0: ResBlock3D
    level1: ResDown3D
    level2: MambaDown3D
    level3: MambaDown3D

Geometry-branch Mamba scans bone voxels in 3D Hilbert-curve order:

  - The scan_key field is a precomputed (R, R, R) Hilbert-position table.
    Voxels close in 3D get close 1D positions, regardless of axis.
  - Air voxels are pushed to the tail by the per-patch bone_mask
    (BIG_VAL substitution before sort).
  - 3D sin/cos positional encoding is added to features so each bone
    voxel "self-localises" — Mamba can use a small input-dependent
    Δ_t at zeroed air positions and a large Δ_t when a bone token
    arrives, preserving long-range bone-to-bone dependency.

The public class name stays DualEncoderUNet_SSM so the existing training,
fine-tuning, and inference scripts can keep importing the same symbol.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    # Import the Mamba block straight from its submodule. The top-level
    # `from mamba_ssm import Mamba` pulls in MambaLMHeadModel via the
    # package __init__, which imports transformers symbols removed in
    # newer transformers (e.g. GreedySearchDecoderOnlyOutput). We don't
    # use the LM head, so go directly to the module and dodge that.
    from mamba_ssm.modules.mamba_simple import Mamba
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only used when env lacks mamba-ssm
    try:
        from mamba_ssm import Mamba
        _MAMBA_IMPORT_ERROR = None
    except Exception as exc2:
        Mamba = None
        _MAMBA_IMPORT_ERROR = exc2

from .model_MSGC import SpatialNorm, HierarchicalFeatureBlock
from .SparsePhysics import SparsePhysicsEncoder


class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        ng = 8 if cout >= 8 else 1
        self.gn1 = nn.GroupNorm(ng, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(ng, cout)
        self.act = nn.GELU()
        # Channel-wise dropout between the two convs (no learnable params,
        # so checkpoints loaded with strict=True still work).
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.proj = None
        if cin != cout or stride != 1:
            self.proj = nn.Sequential(
                nn.Conv3d(cin, cout, 1, stride=stride, bias=False),
                nn.GroupNorm(ng, cout))

    def _impl(self, x):
        identity = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.gn2(self.conv2(x))
        if self.proj is not None:
            identity = self.proj(identity)
        return self.act(x + identity)

    def forward(self, x):
        if self.training and x.requires_grad:
            return checkpoint(self._impl, x, use_reentrant=False)
        return self._impl(x)


class ResDown3D(nn.Module):
    """Stride-2 residual downsample block used as the second ResNet stage."""
    def __init__(self, cin, cout, use_checkpoint=True, dropout=0.0):
        super().__init__()
        self.block = ResBlock3D(cin, cout, stride=2, dropout=dropout)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        return self.block(x)


class MSGCDown(nn.Module):
    """Stride-2 Conv + Multi-Scale Gated Conv block (from old Hybrid).

    Used by the physics branch backbone to keep MSG's multi-scale local
    inductive bias on the FiLM modulation signal, while the geometry
    branch uses Mamba for long-range load reasoning.
    """
    def __init__(self, cin, cout, order=3, use_checkpoint=True):
        super().__init__()
        self.down_conv = nn.Conv3d(cin, cout, 3, stride=2, padding=1, bias=False)
        self.norm = SpatialNorm(cout)
        self.act = nn.GELU()
        self.msgc = HierarchicalFeatureBlock(
            conv_op=nn.Conv3d, input_channels=cout, output_channels=cout,
            transform_channels=cout // 2, kernel_size=3,
            nonlin=nn.GELU, order=order, drop_path=0.1)
        self.use_checkpoint = use_checkpoint

    def _impl(self, x):
        x = self.act(self.norm(self.down_conv(x)))
        return self.msgc(x)

    def forward(self, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(self._impl, x, use_reentrant=False)
        return self._impl(x)


_POS_ENC_MAX_L = 10  # Cap on n_freqs.
                     # Max freq = 2^(L-1)·π ≈ 1608 → fp32-safe and far
                     # below fp16 overflow (~6.5e4). Beyond L≈10 the
                     # spatial period is finer than a single voxel on
                     # the 128³ grid, so higher octaves degenerate into
                     # numerical noise rather than carrying useful
                     # positional content.


def _make_3d_pos_enc(D, H, W, C, device, dtype):
    """3D sinusoidal positional encoding of shape (1, C, D, H, W).

    NeRF-style multi-band: each axis contributes ``n_freqs`` sin and
    ``n_freqs`` cos channels at frequencies ``π, 2π, 4π, ..., 2^(n_freqs-1)·π``.
    ``n_freqs = min(C // 6, _POS_ENC_MAX_L)`` so the highest band stays
    inside fp32's usable precision range.

    The first ``6·n_freqs`` channels carry positional info; remaining
    ``C - 6·n_freqs`` channels are zero-padded so 1×1 convs / Mamba
    projections can use the spare capacity for non-positional content.

    Computed in fp32 internally and cast to ``dtype`` only at the end so
    that high-frequency bands stay numerically clean even under autocast.
    """
    n_freqs = min(C // 6, _POS_ENC_MAX_L)
    if n_freqs < 1:
        return torch.zeros(1, C, D, H, W, device=device, dtype=dtype)

    fp32 = torch.float32
    freqs = (2.0 ** torch.arange(n_freqs, device=device, dtype=fp32)
             * math.pi)                                  # (n_freqs,)
    z = torch.linspace(-1, 1, D, device=device, dtype=fp32)
    y = torch.linspace(-1, 1, H, device=device, dtype=fp32)
    x = torch.linspace(-1, 1, W, device=device, dtype=fp32)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')   # (D, H, W) each

    chunks = []
    for ax in (zz, yy, xx):
        scaled = ax.unsqueeze(0) * freqs.view(-1, 1, 1, 1)  # (n_freqs, D, H, W)
        chunks.append(torch.sin(scaled))
        chunks.append(torch.cos(scaled))
    pe = torch.cat(chunks, dim=0)                          # (6 n_freqs, D, H, W)

    if pe.shape[0] < C:
        pad = torch.zeros(C - pe.shape[0], D, H, W,
                          device=device, dtype=fp32)
        pe = torch.cat([pe, pad], dim=0)
    return pe.unsqueeze(0).to(dtype)                       # (1, C, D, H, W)


class HilbertScanMamba(nn.Module):
    """
    1D Mamba block that scans bone voxels in 3D Hilbert-curve order.

    The scan_key field carries a per-voxel position on a precomputed
    Hilbert curve (small = early on the curve). Air voxels are
    substituted with BIG_VAL via the bone_mask before argsort, so they
    fall to the tail. Their feature inputs are then zeroed (Plan B):

        * Mamba's selectivity Δ_t = softplus(W_Δ x_t + b_Δ) collapses
          to a small constant when x_t = 0, freezing the hidden state
          across long air gaps.
        * Bone tokens carry a 3D sin/cos positional encoding so each
          one self-localises — even if the state has decayed, the new
          bone token re-injects 3D context.

    Bidirectional: forward + reverse scan, mean-fused.

    Inputs
        x          : (B, C, D, H, W) feature map
        scan_key   : (B, 1, D, H, W) Hilbert position field at the same
                     spatial resolution as x. Air voxels should already
                     hold BIG_VAL (handled by the parent net via
                     bone_mask substitution).
        bone_mask  : (B, 1, D, H, W) float mask at the same resolution,
                     1.0 = bone, 0.0 = air. Used to zero air features.
    """
    BIG_VAL = float(2 ** 30)  # ≫ any Hilbert position at R ≤ 1024

    def __init__(self, channels, d_state=16, d_conv=4, expand=2,
                 bidirectional=True, pos_init_scale=0.1, dropout=0.0):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is required for HilbertScanMamba. "
                f"Original import error: {_MAMBA_IMPORT_ERROR!r}")

        self.channels = channels
        self.bidirectional = bidirectional
        self.pre_norm = SpatialNorm(channels)
        self.post_norm = SpatialNorm(channels)

        self.fwd = Mamba(d_model=channels, d_state=d_state,
                         d_conv=d_conv, expand=expand)
        if bidirectional:
            self.bwd = Mamba(d_model=channels, d_state=d_state,
                             d_conv=d_conv, expand=expand)

        self.gate = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        # Channel-wise dropout on the gated Mamba output. Regularises the
        # long-range Hilbert-scan signal before it's added back to identity.
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        # Learnable scalar so the model can dial pos-encoding influence.
        # Init small — features dominate at the start of training.
        self.pos_scale = nn.Parameter(torch.tensor(float(pos_init_scale)))

    def _impl(self, x, scan_key, bone_mask):
        identity = x
        xn = self.pre_norm(x)
        B, C, D, H, W = xn.shape

        # 1) Add 3D sin/cos positional encoding (so bone tokens self-localise)
        pe = _make_3d_pos_enc(D, H, W, C, xn.device, xn.dtype)
        xn = xn + self.pos_scale * pe

        # 2) Plan B — zero air features so they don't pollute the SSM state.
        #    bone_mask is float in {0, 1}; broadcast over channels.
        bm = bone_mask.to(xn.dtype)
        xn = xn * bm

        # 3) Sort voxels by Hilbert position; air (BIG_VAL) goes to tail.
        x_flat = xn.flatten(2).permute(0, 2, 1).contiguous()       # (B, N, C)
        sk = scan_key.flatten(2).squeeze(1)                         # (B, N)
        perm = sk.argsort(dim=1)                                    # (B, N)
        idx = perm.unsqueeze(-1).expand(-1, -1, C)                  # (B, N, C)
        x_seq = torch.gather(x_flat, 1, idx)                        # (B, N, C)

        # 4) Bidirectional Mamba.
        y_seq = self.fwd(x_seq)
        if self.bidirectional:
            y_rev = torch.flip(self.bwd(torch.flip(x_seq, dims=[1])),
                               dims=[1])
            y_seq = 0.5 * (y_seq + y_rev)

        # Mamba kernels can return a different dtype than x_flat under AMP;
        # cast back to avoid "Expected self.dtype to be equal to src.dtype"
        # in scatter_.
        y_seq = y_seq.to(x_flat.dtype)

        # 5) Scatter back to 3D.
        y_flat = torch.empty_like(x_flat)
        y_flat.scatter_(1, idx, y_seq)
        y = y_flat.permute(0, 2, 1).contiguous().view(B, C, D, H, W)

        out = identity + self.gate(identity) * self.dropout(y)
        return self.post_norm(out)

    def forward(self, x, scan_key, bone_mask):
        return self._impl(x, scan_key, bone_mask)


class MambaDown3D(nn.Module):
    """Stride-2 downsample followed by Hilbert-scan Mamba at low resolution."""
    def __init__(self, cin, cout, d_state=16, d_conv=4, expand=2,
                 bidirectional=True, use_checkpoint=True, dropout=0.0):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride=2, padding=1, bias=False),
            SpatialNorm(cout),
            nn.GELU(),
        )
        self.local = ResBlock3D(cout, cout, dropout=dropout)
        self.mamba = HilbertScanMamba(
            cout, d_state=d_state, d_conv=d_conv, expand=expand,
            bidirectional=bidirectional, dropout=dropout)
        self.use_checkpoint = use_checkpoint

    def _impl(self, x, scan_key, bone_mask):
        x = self.down(x)
        x = self.local._impl(x)
        return self.mamba(x, scan_key, bone_mask)

    def forward(self, x, scan_key, bone_mask):
        # Note: gradient checkpointing is intentionally OFF here.
        # mamba_ssm's `selective_scan_cuda` uses a custom autograd Function
        # that holds saved tensors across forward/backward; combining it with
        # torch.utils.checkpoint causes "No grad accumulator for a saved leaf"
        # errors on Windows.  The MambaDown3D operates at low spatial
        # resolution (stride-2 already applied), so the activation memory
        # without checkpointing is small.
        return self._impl(x, scan_key, bone_mask)


class FiLMFusion(nn.Module):
    """Physics features -> (scale, shift) -> modulate geometry features."""
    def __init__(self, ch):
        super().__init__()
        self.proj = nn.Conv3d(ch, ch * 2, 1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, geo_feat, phys_feat):
        scale, shift = self.proj(phys_feat).chunk(2, dim=1)
        return geo_feat * (1 + scale) + shift


class Up(nn.Module):
    def __init__(self, cin, cout, skip_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(cin, cout, 2, stride=2)
        self.fuse = ResBlock3D(cout + skip_ch, cout, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        dZ = skip.size(2) - x.size(2)
        dY = skip.size(3) - x.size(3)
        dX = skip.size(4) - x.size(4)
        x = F.pad(x, [dX // 2, dX - dX // 2,
                       dY // 2, dY - dY // 2,
                       dZ // 2, dZ - dZ // 2])
        return self.fuse(torch.cat([x, skip], dim=1))


class MambaUp3D(nn.Module):
    """ConvTranspose upsample + skip-concat ResBlock fuse + Hilbert Mamba.

    Used at the two deeper decoder levels (u1 / u2) where token count is
    small enough for Mamba's selective scan to be practical. Restores the
    long-range refinement that the encoder Hilbert-scan provides, on the
    decoder side after spatial fusion.
    """
    def __init__(self, cin, cout, skip_ch,
                 d_state=16, d_conv=4, expand=2,
                 bidirectional=True, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(cin, cout, 2, stride=2)
        self.fuse = ResBlock3D(cout + skip_ch, cout, dropout=dropout)
        self.mamba = HilbertScanMamba(
            cout, d_state=d_state, d_conv=d_conv, expand=expand,
            bidirectional=bidirectional, dropout=dropout)

    def forward(self, x, skip, scan_key, bone_mask):
        x = self.up(x)
        dZ = skip.size(2) - x.size(2)
        dY = skip.size(3) - x.size(3)
        dX = skip.size(4) - x.size(4)
        x = F.pad(x, [dX // 2, dX - dX // 2,
                      dY // 2, dY - dY // 2,
                      dZ // 2, dZ - dZ // 2])
        x = self.fuse(torch.cat([x, skip], dim=1))
        x = self.mamba(x, scan_key, bone_mask)
        return x


class DualEncoderUNet_SSM(nn.Module):
    """
    Sparse-physics + deep Mamba dual encoder U-Net.

    Args keep the old class import stable while exposing the new Mamba knobs.
    """
    def __init__(self, geo_in=1, sparse_in=2, dense_in=2,
                 sparse_hidden=16, sparse_layers=5,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 mamba_bidirectional=True, use_checkpoint=True,
                 dropout=0.0):
        super().__init__()
        C1, C2, C3, C4 = 32, 64, 128, 256
        # nn.Dropout3d has no learnable params, so flipping `dropout` between
        # train runs does not break checkpoint compatibility — strict=True
        # load works either way.

        self.sparse_encoder = SparsePhysicsEncoder(
            sparse_in=sparse_in, dense_in=dense_in,
            hidden=sparse_hidden, out_ch=C1,
            n_layers=sparse_layers, grid_size=64)

        # Geometry branch: ResBlock + ResDown + MambaDown + MambaDown.
        self.stem_g = ResBlock3D(geo_in, C1, dropout=dropout)
        self.e1_g = ResDown3D(C1, C2, use_checkpoint=use_checkpoint, dropout=dropout)
        self.e2_g = MambaDown3D(
            C2, C3, d_state=mamba_d_state, d_conv=mamba_d_conv,
            expand=mamba_expand, bidirectional=mamba_bidirectional,
            use_checkpoint=use_checkpoint, dropout=dropout)
        self.e3_g = MambaDown3D(
            C3, C4, d_state=mamba_d_state, d_conv=mamba_d_conv,
            expand=mamba_expand, bidirectional=mamba_bidirectional,
            use_checkpoint=use_checkpoint, dropout=dropout)

        # Physics branch backbone: keep the original MSG path so the
        # FiLM modulation signal carries local multi-scale structure.
        # Only the geometry branch uses Mamba for long-range load reasoning.
        self.e1_p = MSGCDown(C1, C2, use_checkpoint=use_checkpoint)
        self.e2_p = MSGCDown(C2, C3, use_checkpoint=use_checkpoint)
        self.e3_p = MSGCDown(C3, C4, use_checkpoint=use_checkpoint)

        self.film0 = FiLMFusion(C1)
        self.film1 = FiLMFusion(C2)
        self.film2 = FiLMFusion(C3)
        self.film3 = FiLMFusion(C4)

        self.s0 = ResBlock3D(C1, C1, dropout=dropout)
        self.s1 = ResBlock3D(C2, C2, dropout=dropout)
        self.s2 = ResBlock3D(C3, C3, dropout=dropout)

        # Decoder: two deep levels use MambaUp3D (Hilbert-scan refinement
        # after spatial fusion), full-res u0 stays conv-only (token count
        # at full resolution is too large for Mamba to be worth it).
        self.u2 = MambaUp3D(
            C4, C3, skip_ch=C3,
            d_state=mamba_d_state, d_conv=mamba_d_conv,
            expand=mamba_expand, bidirectional=mamba_bidirectional,
            dropout=dropout)
        self.u1 = MambaUp3D(
            C3, C2, skip_ch=C2,
            d_state=mamba_d_state, d_conv=mamba_d_conv,
            expand=mamba_expand, bidirectional=mamba_bidirectional,
            dropout=dropout)
        self.u0 = Up(C2, C1, skip_ch=C1, dropout=dropout)

        # Pre-head channel dropout. Direct regulariser on the final
        # prediction; held as a separate Module so it doesn't change
        # `self.head`'s state_dict key.
        self.head_dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv3d(C1, 1, 1)

    def forward(self, geo_input, phys_input, bone_mask=None, scan_key=None):
        if scan_key is None:
            raise ValueError(
                "DualEncoderUNet_SSM requires `scan_key` (B,1,D,H,W) at the "
                "input resolution: per-voxel Hilbert position from "
                "compute_hilbert_order.py. The DataLoader provides it.")
        if bone_mask is None:
            raise ValueError(
                "DualEncoderUNet_SSM requires `bone_mask` (B,1,D,H,W) at "
                "the input resolution. The Mamba block uses it to push "
                "air voxels to the scan tail and zero their inputs.")

        sparse_ch = phys_input[:, :2]
        dense_ch = phys_input[:, 2:4]
        p0 = self.sparse_encoder(sparse_ch, dense_ch, bone_mask=bone_mask)

        g0 = self.stem_g(geo_input)
        g0 = self.film0(g0, p0)

        g1 = self.e1_g(g0)
        p1 = self.e1_p(p0)
        g1 = self.film1(g1, p1)

        # ---- Multi-level scan_key + bone_mask ----
        # scan_key holds Hilbert positions; bone-tail substitution must
        # happen BEFORE pooling, otherwise an air child with a small
        # Hilbert position would pull the parent's key forward.
        # Pool semantics:
        #   scan_key: min-pool   -> earliest bone child wins (any-bone parent
        #             gets a real Hilbert position; all-air parents stay BIG)
        #   bone_mask: max-pool  -> any-bone child marks the parent as bone
        BIG = HilbertScanMamba.BIG_VAL
        sk_full = torch.where(bone_mask > 0.5,
                              scan_key.to(bone_mask.dtype),
                              torch.full_like(bone_mask, BIG))

        size_l1 = g1.shape[-3:]
        size_l2 = (size_l1[0] // 2, size_l1[1] // 2, size_l1[2] // 2)
        size_l3 = (size_l2[0] // 2, size_l2[1] // 2, size_l2[2] // 2)

        def _min_pool(t, target_size):
            # F.adaptive_avg_pool3d-style downsample, but min-pooled.
            # Implemented as adaptive max-pool of the negation.
            return -F.adaptive_max_pool3d(-t, output_size=target_size)

        def _max_pool(t, target_size):
            return F.adaptive_max_pool3d(t, output_size=target_size)

        # Pooled scan_key + bone_mask at every decoder level that hosts a
        # Mamba block. l1 is also needed now because u1 (decoder) restores
        # Hilbert-scan refinement at that resolution.
        sk_l1, bm_l1 = _min_pool(sk_full, size_l1), _max_pool(bone_mask, size_l1)
        sk_l2, bm_l2 = _min_pool(sk_full, size_l2), _max_pool(bone_mask, size_l2)
        sk_l3, bm_l3 = _min_pool(sk_full, size_l3), _max_pool(bone_mask, size_l3)

        g2 = self.e2_g(g1, sk_l2, bm_l2)
        p2 = self.e2_p(p1)               # MSGCDown, conv-only
        g2 = self.film2(g2, p2)

        g3 = self.e3_g(g2, sk_l3, bm_l3)
        p3 = self.e3_p(p2)               # MSGCDown, conv-only
        g3 = self.film3(g3, p3)

        a0 = self.s0(g0)
        a1 = self.s1(g1)
        a2 = self.s2(g2)

        d2 = self.u2(g3, a2, sk_l2, bm_l2)   # decoder at level 2 (MambaUp)
        d1 = self.u1(d2, a1, sk_l1, bm_l1)   # decoder at level 1 (MambaUp)
        d0 = self.u0(d1, a0)                 # full-res          (conv-only)

        out = self.head(self.head_dropout(d0))
        return out
