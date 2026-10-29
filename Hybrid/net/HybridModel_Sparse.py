#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DualEncoder with Sparse Physics path.

Same as HybridModel.DualEncoderUNet but physics path uses:
  - SparsePhysicsEncoder for [BC, Load*u] (sparse → learned diffusion → dense)
  - Concat with [dist_BC, dist_Load] (already dense)
  - Then MSGCDown at each level (same as geometry path)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model_MSGC import HierarchicalFeatureBlock, SpatialNorm
from .SparsePhysics import SparsePhysicsEncoder


# -------------------- ResBlock --------------------
class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        ng = 8 if cout >= 8 else 1
        self.gn1 = nn.GroupNorm(ng, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(ng, cout)
        self.act = nn.GELU()
        self.proj = None
        if cin != cout or stride != 1:
            self.proj = nn.Sequential(
                nn.Conv3d(cin, cout, 1, stride=stride, bias=False),
                nn.GroupNorm(ng, cout))

    def _impl(self, x):
        idt = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        if self.proj is not None:
            idt = self.proj(idt)
        return self.act(x + idt)

    def forward(self, x):
        if self.training and x.requires_grad:
            return checkpoint(self._impl, x, use_reentrant=False)
        return self._impl(x)


# -------------------- Heavy Encoder Block (Geometry) --------------------
class MSGCDown(nn.Module):
    """Stride-2 Conv + MSGC multi-scale feature block."""
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


# -------------------- FiLM Fusion --------------------
class FiLMFusion(nn.Module):
    """Physics features -> (scale, shift) -> modulate geometry features."""
    def __init__(self, ch):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv3d(ch, ch * 2, 1, bias=True))
        nn.init.zeros_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)

    def forward(self, geo_feat, phys_feat):
        params = self.proj(phys_feat)
        scale, shift = params.chunk(2, dim=1)
        return geo_feat * (1 + scale) + shift


# -------------------- Decoder --------------------
class Up(nn.Module):
    def __init__(self, cin, cout, skip_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(cin, cout, 2, stride=2)
        self.fuse = ResBlock3D(cout + skip_ch, cout)

    def forward(self, x, skip):
        x = self.up(x)
        dZ = skip.size(2) - x.size(2)
        dY = skip.size(3) - x.size(3)
        dX = skip.size(4) - x.size(4)
        x = F.pad(x, [dX // 2, dX - dX // 2,
                       dY // 2, dY - dY // 2,
                       dZ // 2, dZ - dZ // 2])
        return self.fuse(torch.cat([x, skip], dim=1))


class DualEncoderUNet_Sparse(nn.Module):
    """
    Same architecture as DualEncoderUNet, but physics path front-end
    uses SparsePhysicsEncoder to handle sparse BC/Load channels.

    Physics input split:
      sparse: [BC, Load*u] (2ch) → SparsePhysicsEncoder → 32ch dense
      dense:  [dist_BC, dist_Load] (2ch) → included in sparse encoder

    Then both paths go through MSGCDown + FiLM as before.
    """
    def __init__(self, geo_in=1, sparse_in=2, dense_in=2,
                 sparse_hidden=16, sparse_layers=5,
                 use_checkpoint=True):
        super().__init__()
        # 4 spatial levels: full, 1/2, 1/4, 1/8 (bottleneck)
        C1, C2, C3, C4 = 32, 64, 128, 256

        # ---- Sparse Physics Front-end ----
        self.sparse_encoder = SparsePhysicsEncoder(
            sparse_in=sparse_in, dense_in=dense_in,
            hidden=sparse_hidden, out_ch=C1,
            n_layers=sparse_layers, grid_size=64)

        # ---- Geometry Encoder ----
        self.stem_g = ResBlock3D(geo_in, C1)
        self.e1_g = MSGCDown(C1, C2)
        self.e2_g = MSGCDown(C2, C3)
        self.e3_g = MSGCDown(C3, C4)

        # ---- Physics Encoder (after sparse front-end) ----
        # Already starts at C1=32 from sparse encoder
        self.e1_p = MSGCDown(C1, C2)
        self.e2_p = MSGCDown(C2, C3)
        self.e3_p = MSGCDown(C3, C4)

        # ---- FiLM Fusion at each level ----
        self.film0 = FiLMFusion(C1)
        self.film1 = FiLMFusion(C2)
        self.film2 = FiLMFusion(C3)
        self.film3 = FiLMFusion(C4)

        # ---- Skip transforms ----
        self.s0 = ResBlock3D(C1, C1)
        self.s1 = ResBlock3D(C2, C2)
        self.s2 = ResBlock3D(C3, C3)

        # ---- Decoder ----
        self.u2 = Up(C4, C3, skip_ch=C3)
        self.u1 = Up(C3, C2, skip_ch=C2)
        self.u0 = Up(C2, C1, skip_ch=C1)

        self.head = nn.Conv3d(C1, 1, 1)

        # ---- Deep Supervision ----
        self.ds2 = nn.Conv3d(C3, 1, 1)
        self.ds1 = nn.Conv3d(C2, 1, 1)

    def forward(self, geo_input, phys_input, bone_mask=None):
        """
        Args:
            geo_input:  (B, 1, P, P, P)  [E_grid]
            phys_input: (B, 4, P, P, P)  [BC, Load*u, dist_BC, dist_Load]
            bone_mask:  (B, 1, P, P, P) or (B, P, P, P), optional.
                        If provided, the sparse path's diffusion is pruned to
                        bone voxels so signals never leak into air.
        """
        # Split physics into sparse and dense
        sparse_ch = phys_input[:, :2]  # (B, 2, P³) [BC, Load*u]
        dense_ch = phys_input[:, 2:4]  # (B, 2, P³) [dist_BC, dist_Load]

        # ---- Physics front-end: sparse diffusion + dense concat ----
        p0 = self.sparse_encoder(sparse_ch, dense_ch,
                                  bone_mask=bone_mask)  # (B, 32, P³)

        # ---- Geometry stem ----
        g0 = self.stem_g(geo_input)  # (B, 32, P³)

        # ---- Level 0: FiLM ----
        g0 = self.film0(g0, p0)

        # ---- Level 1 ----
        g1 = self.e1_g(g0)
        p1 = self.e1_p(p0)
        g1 = self.film1(g1, p1)

        # ---- Level 2 ----
        g2 = self.e2_g(g1)
        p2 = self.e2_p(p1)
        g2 = self.film2(g2, p2)

        # ---- Level 3 (bottleneck) ----
        g3 = self.e3_g(g2)
        p3 = self.e3_p(p2)
        g3 = self.film3(g3, p3)

        # ---- Skip ----
        a0 = self.s0(g0)
        a1 = self.s1(g1)
        a2 = self.s2(g2)

        # ---- Decoder ----
        d2 = self.u2(g3, a2)
        d1 = self.u1(d2, a1)
        d0 = self.u0(d1, a0)

        out = self.head(d0)

        if self.training:
            sz = out.shape[2:]
            ds2 = F.interpolate(self.ds2(d2), size=sz,
                                mode='trilinear', align_corners=False)
            ds1 = F.interpolate(self.ds1(d1), size=sz,
                                mode='trilinear', align_corners=False)
            return out, ds2, ds1

        return out


# ================================================================
# Test
# ================================================================
if __name__ == "__main__":
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P = 64

    model = DualEncoderUNet_Sparse().to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"DualEncoderUNet_Sparse: {n_total:,} ({n_total/1e6:.2f}M)")

    # Component breakdown
    components = {
        'SparsePhysicsEncoder': ['sparse_encoder'],
        'Geometry Encoder': ['stem_g', 'e1_g', 'e2_g', 'e3_g'],
        'Physics Encoder': ['e1_p', 'e2_p', 'e3_p'],
        'FiLM': ['film0', 'film1', 'film2', 'film3'],
        'Skip': ['s0', 's1', 's2'],
        'Decoder': ['u0', 'u1', 'u2'],
        'Head+DS': ['head', 'ds1', 'ds2'],
    }
    for label, prefixes in components.items():
        n = sum(p.numel() for name, p in model.named_parameters()
                if any(name.startswith(pf) for pf in prefixes))
        print(f"  {label:25s} {n:>10,} ({n/1e6:.2f}M) {n/n_total*100:5.1f}%")

    # Forward
    geo = torch.randn(1, 1, P, P, P, device=device)
    phys = torch.zeros(1, 4, P, P, P, device=device)
    # Sparse BC/Load
    phys[0, 0, 10:12, 10:12, 10:12] = 1.0  # BC
    phys[0, 1, 50:52, 50:52, 50:52] = -0.67  # Load
    phys[0, 2:] = torch.rand(2, P, P, P)  # dense: dist_BC, dist_Load
    mask = (torch.rand(1, 1, P, P, P, device=device) > 0.3).float()

    model.eval()
    with torch.no_grad():
        out = model(geo, phys, bone_mask=mask)
    print(f"\nOutput: {out.shape}")

    # Training
    model.train()
    out, ds2, ds1 = model(geo, phys, bone_mask=mask)
    loss = out.sum() + ds2.sum() + ds1.sum()
    loss.backward()
    print("Backward OK!")

    # Speed
    model.eval()
    for _ in range(3):
        with torch.no_grad():
            model(geo, phys)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(10):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(geo, phys)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    print(f"Inference: {sum(times)/len(times)*1000:.1f} ms")

    if device.type == 'cuda':
        print(f"GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
