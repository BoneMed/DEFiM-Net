#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sparse Physics Encoder using torch_scatter-based sparse diffusion conv.

Processes sparse BC/Load channels:
  1. Extract non-zero voxels as sparse points
  2. SparseConvLayer x N: 26-neighbor diffusion (only on active voxels)
  3. ToDense: scatter back to 3D grid
  4. Concat with dense channels (dist_BC, dist_Load, u_load)

No spconv dependency — uses torch_scatter for efficient sparse operations.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter

from .model_MSGC import SpatialNorm


# ================================================================
# 26-neighbor offsets (precomputed)
# ================================================================
_OFFSETS_26 = None

def _get_offsets(device):
    global _OFFSETS_26
    if _OFFSETS_26 is not None and _OFFSETS_26.device == device:
        return _OFFSETS_26
    offsets = []
    for d in [-1, 0, 1]:
        for h in [-1, 0, 1]:
            for w in [-1, 0, 1]:
                if d == 0 and h == 0 and w == 0:
                    continue
                offsets.append([d, h, w])
    _OFFSETS_26 = torch.tensor(offsets, device=device, dtype=torch.long)
    return _OFFSETS_26


# ================================================================
# Sparse Diffusion Conv Layer
# ================================================================
class SparseDiffuseConv(nn.Module):
    """
    One layer of sparse 26-neighbor diffusion conv.

    For each active voxel:
      1. Gather 26 neighbors (if also active or newly created)
      2. Linear transform on self + mean of neighbor features
      3. Expand active set to include neighbor coordinates

    Input:  (N, C) features + (N, 3) coords
    Output: (M, C') features + (M, 3) coords   (M >= N, expanded)
    """
    def __init__(self, in_ch, out_ch, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.self_linear = nn.Linear(in_ch, out_ch)
        self.neighbor_linear = nn.Linear(in_ch, out_ch)
        self.act = nn.GELU()
        # SpatialNorm = LayerNorm across channel dim per voxel.
        # Consistent with the main MSG encoder; stable at any batch size.
        self.norm = SpatialNorm(out_ch)

    def _hash(self, coords):
        D = self.grid_size
        return coords[:, 0] * D * D + coords[:, 1] * D + coords[:, 2]

    def forward(self, features, coords, bone_mask=None):
        """
        Args:
            features:  (N, C) active voxel features
            coords:    (N, 3) integer voxel coordinates
            bone_mask: (D, D, D) bool, optional. If given, neighbours that
                       fall in air voxels are pruned BEFORE aggregation, so
                       sparse signals never leak outside the bone.
        Returns:
            new_features: (M, C') expanded features
            new_coords:   (M, 3) expanded coordinates
        """
        device = features.device
        N, C = features.shape
        D = self.grid_size
        offsets = _get_offsets(device)  # (26, 3)

        # Expand to neighbors: (N, 26, 3)
        neighbor_coords = coords.unsqueeze(1) + offsets.unsqueeze(0)

        # Boundary check
        valid = (
            (neighbor_coords[:, :, 0] >= 0) & (neighbor_coords[:, :, 0] < D) &
            (neighbor_coords[:, :, 1] >= 0) & (neighbor_coords[:, :, 1] < D) &
            (neighbor_coords[:, :, 2] >= 0) & (neighbor_coords[:, :, 2] < D)
        )  # (N, 26)

        # Prune neighbours that land outside the bone
        if bone_mask is not None:
            nb_clamped = neighbor_coords.clamp(0, D - 1)
            in_bone = bone_mask[nb_clamped[..., 0],
                                 nb_clamped[..., 1],
                                 nb_clamped[..., 2]]   # (N, 26) bool
            valid = valid & in_bone

        # All candidate coords (original + valid neighbors)
        nb_coords = neighbor_coords[valid]  # (E, 3)
        # Source features for each neighbor edge
        src_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(valid)
        nb_src_idx = src_idx[valid]  # (E,) which original voxel this neighbor came from
        nb_features = features[nb_src_idx]  # (E, C)

        # Merge all coordinates: original + new neighbors
        all_coords = torch.cat([coords, nb_coords], dim=0)  # (N+E, 3)
        all_hashes = self._hash(all_coords)

        # Unique coordinates (deduplicate)
        unique_hashes, inv_idx = all_hashes.unique(return_inverse=True)
        M = unique_hashes.shape[0]

        # Scatter original features to unique positions
        orig_inv = inv_idx[:N]  # first N are original
        self_feat = torch.zeros(M, C, device=device)
        self_count = torch.zeros(M, 1, device=device)
        self_feat.scatter_add_(0, orig_inv.unsqueeze(1).expand(-1, C), features)
        self_count.scatter_add_(0, orig_inv.unsqueeze(1), torch.ones(N, 1, device=device))

        # Scatter neighbor features to unique positions
        nb_inv = inv_idx[N:]  # rest are neighbors
        nb_feat_sum = torch.zeros(M, C, device=device)
        nb_count = torch.zeros(M, 1, device=device)
        nb_feat_sum.scatter_add_(0, nb_inv.unsqueeze(1).expand(-1, C), nb_features)
        nb_count.scatter_add_(0, nb_inv.unsqueeze(1), torch.ones(nb_coords.shape[0], 1, device=device))

        # Self transform + neighbor mean transform
        has_self = self_count > 0
        self_mean = torch.zeros_like(self_feat)
        self_mean[has_self.expand_as(self_feat)] = (
            self_feat[has_self.expand_as(self_feat)] /
            self_count.expand_as(self_feat)[has_self.expand_as(self_feat)])

        has_nb = nb_count > 0
        nb_mean = torch.zeros_like(nb_feat_sum)
        nb_mean[has_nb.expand_as(nb_feat_sum)] = (
            nb_feat_sum[has_nb.expand_as(nb_feat_sum)] /
            nb_count.expand_as(nb_feat_sum)[has_nb.expand_as(nb_feat_sum)])

        out = self.act(self.self_linear(self_mean) + self.neighbor_linear(nb_mean))
        out = self.norm(out)

        # Recover coordinates from unique hashes
        new_coords = torch.stack([
            unique_hashes // (D * D),
            (unique_hashes % (D * D)) // D,
            unique_hashes % D,
        ], dim=1)

        return out, new_coords


# ================================================================
# Sparse Physics Encoder
# ================================================================
class SparsePhysicsEncoder(nn.Module):
    """
    Encodes sparse BC/Load channels via learned diffusion,
    then concatenates with dense distance fields.

    Sparse path: [BC, Load*u] (2ch) → 3 layers diffusion → (16ch, dense 64³)
    Dense path:  [dist_BC, dist_Load] (2ch) → passthrough
    Output:      (16+2=18ch) → Conv1x1 → (out_ch, 64³)
    """
    def __init__(self, sparse_in=2, dense_in=2, hidden=16, out_ch=32,
                 n_layers=3, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.n_layers = n_layers
        self.hidden = hidden

        # Sparse diffusion layers
        self.layers = nn.ModuleList()
        self.layers.append(SparseDiffuseConv(sparse_in, hidden, grid_size))
        for _ in range(n_layers - 1):
            self.layers.append(SparseDiffuseConv(hidden, hidden, grid_size))

        # Fuse sparse + dense
        self.fuse = nn.Sequential(
            nn.Conv3d(hidden + dense_in, out_ch, 1, bias=False),
            SpatialNorm(out_ch),
            nn.GELU(),
        )

    def _to_dense(self, features, coords, batch_size, grid_size, channels):
        """Scatter sparse features back to dense 3D grid."""
        D = grid_size
        dense = torch.zeros(batch_size, channels, D, D, D,
                             device=features.device, dtype=features.dtype)

        # Assume single batch for now
        d, h, w = coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()
        d = d.clamp(0, D-1); h = h.clamp(0, D-1); w = w.clamp(0, D-1)
        dense[0, :, d, h, w] = features.T  # (C, M) -> assign

        return dense

    def forward(self, sparse_input, dense_input, bone_mask=None):
        """
        Args:
            sparse_input: (B, 2, D, H, W) = [BC, Load*u] (mostly zeros)
            dense_input:  (B, 2, D, H, W) = [dist_BC, dist_Load]
            bone_mask:    (B, 1, D, H, W) or (B, D, H, W), optional.
                          If provided, sparse diffusion is pruned to bone voxels
                          so signals never leak into air.
        Returns:
            (B, out_ch, D, H, W)
        """
        B = sparse_input.shape[0]
        D = sparse_input.shape[-1]  # dynamic: 64 or 128
        device = sparse_input.device
        out_list = []

        # Normalize bone_mask shape: (B, D, D, D) bool
        if bone_mask is not None:
            bm = bone_mask
            if bm.dim() == 5:
                bm = bm.squeeze(1)
            bm = bm > 0.5
        else:
            bm = None

        for b in range(B):
            sp = sparse_input[b]  # (2, D, D, D)

            # Extract non-zero voxels
            nonzero_mask = sp.abs().sum(dim=0) > 1e-8  # (D, D, D)
            coords = torch.nonzero(nonzero_mask, as_tuple=False)  # (N, 3)
            features = sp[:, coords[:, 0], coords[:, 1], coords[:, 2]].T  # (N, 2)

            sample_mask = bm[b] if bm is not None else None

            # Drop initial sparse points that fall outside bone (rounding noise)
            if sample_mask is not None and coords.shape[0] > 0:
                in_bone = sample_mask[coords[:, 0], coords[:, 1], coords[:, 2]]
                coords = coords[in_bone]
                features = features[in_bone]

            if coords.shape[0] == 0:
                sparse_dense = torch.zeros(1, self.hidden, D, D, D, device=device)
            else:
                # Update grid_size for diffusion layers
                for layer in self.layers:
                    layer.grid_size = D
                x, c = features, coords
                for layer in self.layers:
                    x, c = layer(x, c, bone_mask=sample_mask)

                sparse_dense = self._to_dense(x, c, 1, D, self.hidden)

            out_list.append(sparse_dense)

        sparse_dense_batch = torch.cat(out_list, dim=0)  # (B, hidden, D, D, D)

        # Concat sparse + dense channels
        combined = torch.cat([sparse_dense_batch, dense_input], dim=1)
        return self.fuse(combined)


# ================================================================
# Test
# ================================================================
if __name__ == "__main__":
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = 64

    # Simulate sparse BC/Load
    sparse_in = torch.zeros(1, 2, D, D, D, device=device)
    # 30 BC voxels
    bc_coords = torch.randint(0, D, (30, 3))
    sparse_in[0, 0, bc_coords[:, 0], bc_coords[:, 1], bc_coords[:, 2]] = 1.0
    # 20 Load voxels
    ld_coords = torch.randint(0, D, (20, 3))
    sparse_in[0, 1, ld_coords[:, 0], ld_coords[:, 1], ld_coords[:, 2]] = -0.67

    # Dense distance fields
    dense_in = torch.rand(1, 2, D, D, D, device=device)

    n_nonzero = (sparse_in.abs().sum(dim=1) > 0).sum().item()
    print(f"Grid: {D}³, Non-zero voxels: {n_nonzero}")

    model = SparsePhysicsEncoder(
        sparse_in=2, dense_in=2, hidden=16, out_ch=32,
        n_layers=3, grid_size=D).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SparsePhysicsEncoder: {n_params:,} params")

    # Forward
    with torch.no_grad():
        out = model(sparse_in, dense_in)
    print(f"Output: {out.shape}")

    # Backward
    model.train()
    out = model(sparse_in, dense_in)
    out.sum().backward()
    print("Backward OK!")

    # Speed
    model.eval()
    for _ in range(3):
        with torch.no_grad():
            model(sparse_in, dense_in)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(20):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(sparse_in, dense_in)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    print(f"Speed: {sum(times)/len(times)*1000:.1f} ms")
