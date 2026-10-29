#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-Encoder DataLoader.

数据源: Data/ElementGraph/*.npz (单元级数据)
  → 在线体素化为 grid_res³
  → 输出两路:
    geo_input:  [E_grid] (1ch) - 几何/材料
    phys_input: [BC, Load*u, dist_BC, dist_Load] (4ch) - 物理条件
    target:     [S3_grid] (1ch)
    mask:       [occupied] (1ch)

训练时从全图切 patch_size³ 的patch。
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import distance_transform_edt, binary_fill_holes, binary_dilation


# ================================================================
# 辅助函数
# ================================================================
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def signed_power(x, p=0.3):
    return np.sign(x) * np.abs(x) ** p


def _inmask_nearest_fill(grid, mask):
    mask_bool = mask > 0.5
    valid = (grid != 0) & mask_bool
    if not valid.any():
        return grid
    _, inds = distance_transform_edt(~valid, return_indices=True)
    filled = grid.copy()
    need = mask_bool & (~valid)
    filled[need] = grid[tuple(inds)][need]
    return filled


def _clipped_dilation(mask, max_dist=2, iterations=2):
    mask = mask > 0.5
    dist = distance_transform_edt(~mask)
    dilated = binary_dilation(mask, iterations=iterations)
    result = dilated & (dist <= max_dist)
    result = binary_fill_holes(result)
    return result.astype(np.float32)


# ================================================================
# HybridDataset: 体素化 + 距离场
# ================================================================
class HybridDataset(Dataset):
    """
    加载ElementGraph npz → 在线体素化。
    输出:
      geo:  [E_grid] (1ch)
      phys: [BC, Load*u, dist_BC, dist_Load] (4ch)
      target: [S3_grid] (1ch)
      mask: (1ch)
    """
    def __init__(self, data_dir, grid_res=128, dilate_iter=2, pad=0.01, s3_splat='mean'):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npz"))
        self.names = [f.stem for f in self.files]
        self.grid_res = grid_res
        self.dilate_iter = dilate_iter
        self.pad = pad
        self.s3_splat = s3_splat  # 'mean' or 'min'

        print(f"[HybridDataset] {len(self.files)} samples, res={grid_res}, s3_splat={s3_splat}")
        print(f"  bbox: per-sample (each sample uses its own bbox)")

    @staticmethod
    def _sample_bbox(centroids, pad=0.01):
        """Compute bbox for a single sample's centroids."""
        mins = centroids.min(axis=0)
        maxs = centroids.max(axis=0)
        size = maxs - mins
        return (mins - pad * size).astype(np.float32), (maxs + pad * size).astype(np.float32)

    def _to_grid_coords(self, pts, bbox_min, bbox_max):
        R = self.grid_res
        scale = bbox_max - bbox_min
        scale[scale == 0] = 1.0
        q = (pts - bbox_min) / scale
        q = np.clip(q, 0, 1 - 1e-6)
        return q * (R - 1)

    def _splat_trilinear(self, pts_grid, values, R):
        """Trilinear splatting: each element distributes to 8 neighbor voxels."""
        pts = np.clip(pts_grid, 0, R - 1 - 1e-6)
        i0 = np.floor(pts).astype(np.int32)
        frac = pts - i0

        acc = np.zeros((R, R, R), dtype=np.float32)
        wsum = np.zeros((R, R, R), dtype=np.float32)

        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    iz = np.clip(i0[:, 0] + dz, 0, R - 1)
                    iy = np.clip(i0[:, 1] + dy, 0, R - 1)
                    ix = np.clip(i0[:, 2] + dx, 0, R - 1)
                    wz = (frac[:, 0] if dz else 1 - frac[:, 0])
                    wy = (frac[:, 1] if dy else 1 - frac[:, 1])
                    wx = (frac[:, 2] if dx else 1 - frac[:, 2])
                    w = wz * wy * wx
                    np.add.at(acc, (iz, iy, ix), (values * w).astype(np.float32))
                    np.add.at(wsum, (iz, iy, ix), w.astype(np.float32))

        grid = np.zeros_like(acc)
        valid = wsum > 1e-8
        grid[valid] = acc[valid] / wsum[valid]
        return grid, valid.astype(np.float32)

    def _splat_min(self, pts_grid, values, R):
        """Min splatting: each voxel gets the minimum value of its elements."""
        gi = np.round(pts_grid).astype(np.int32)
        gi = np.clip(gi, 0, R - 1)

        grid = np.full((R, R, R), np.inf, dtype=np.float32)
        np.minimum.at(grid, (gi[:, 0], gi[:, 1], gi[:, 2]), values.astype(np.float32))

        occ = np.zeros((R, R, R), dtype=np.float32)
        occ[gi[:, 0], gi[:, 1], gi[:, 2]] = 1.0
        grid[occ < 0.5] = 0.0
        return grid, occ

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        centroids = data['centroids'].astype(np.float32)
        E_raw = data['E'].astype(np.float32)
        S3_raw = data['S3'].astype(np.float32)
        is_bc = data['is_bc'].astype(np.float32)
        is_load = data['is_load'].astype(np.float32)
        u_disp = float(data['u_disp'][0])

        R = self.grid_res

        # ---- 体素化 ----
        E_log = np.log1p(E_raw)
        S3_log = signed_sqrt(S3_raw * 10.0)

        bbox_min, bbox_max = self._sample_bbox(centroids, self.pad)
        pts_grid = self._to_grid_coords(centroids, bbox_min, bbox_max)

        E_grid_raw, occ = self._splat_trilinear(pts_grid, E_log, R)
        if self.s3_splat == 'min':
            S3_grid_raw, _ = self._splat_min(pts_grid, S3_log, R)
        else:
            S3_grid_raw, _ = self._splat_trilinear(pts_grid, S3_log, R)

        mask = _clipped_dilation(occ, max_dist=2, iterations=self.dilate_iter)
        E_grid = _inmask_nearest_fill(E_grid_raw, mask) * mask
        S3_grid = _inmask_nearest_fill(S3_grid_raw, mask) * mask

        # ---- BC / Load mask ----
        BC_grid = np.zeros((R, R, R), dtype=np.float32)
        Load_grid = np.zeros((R, R, R), dtype=np.float32)

        bc_elem_idx = np.where(is_bc > 0.5)[0]
        load_elem_idx = np.where(is_load > 0.5)[0]
        if len(bc_elem_idx) > 0:
            bc_g = np.round(pts_grid[bc_elem_idx]).astype(int)
            bc_g = np.clip(bc_g, 0, R - 1)
            BC_grid[bc_g[:, 0], bc_g[:, 1], bc_g[:, 2]] = 1.0
        if len(load_elem_idx) > 0:
            ld_g = np.round(pts_grid[load_elem_idx]).astype(int)
            ld_g = np.clip(ld_g, 0, R - 1)
            Load_grid[ld_g[:, 0], ld_g[:, 1], ld_g[:, 2]] = 1.0

        u_sqrt = float(np.sign(u_disp) * np.sqrt(abs(u_disp)))
        Load_grid_u = Load_grid * u_sqrt

        # ---- 距离场 ----
        # dist_BC: 每个体素到最近BC体素的距离 (归一化到[0,1])
        if BC_grid.max() > 0:
            dist_BC = distance_transform_edt(BC_grid < 0.5).astype(np.float32)
            dist_BC = dist_BC / (dist_BC.max() + 1e-8)  # [0, 1]
        else:
            dist_BC = np.ones((R, R, R), dtype=np.float32)

        # dist_Load: 每个体素到最近Load体素的距离
        if Load_grid.max() > 0:
            dist_Load = distance_transform_edt(Load_grid < 0.5).astype(np.float32)
            dist_Load = dist_Load / (dist_Load.max() + 1e-8)
        else:
            dist_Load = np.ones((R, R, R), dtype=np.float32)

        # mask内的距离场
        dist_BC = dist_BC * mask
        dist_Load = dist_Load * mask

        # ---- 输出 ----
        # geo:  [E_grid] (1ch)
        # phys: [BC, Load*u, dist_BC, dist_Load] (4ch)
        # 注: 之前的 u_load 常数广播通道已删除。其全局幅值信息由
        # Load*u (sparse path 第 2 通道) 通过 sparse diffusion + encoder
        # 下采样的多层感受野自然传遍整个 patch。
        geo_input = torch.from_numpy(E_grid).unsqueeze(0)  # (1, R, R, R)

        phys_input = torch.stack([
            torch.from_numpy(BC_grid),
            torch.from_numpy(Load_grid_u),
            torch.from_numpy(dist_BC),
            torch.from_numpy(dist_Load),
        ], dim=0)  # (4, R, R, R)

        return {
            'name': self.names[idx],
            'geo_input': geo_input,           # (1, R, R, R)
            'phys_input': phys_input,         # (4, R, R, R)
            'S3_grid': torch.from_numpy(S3_grid).unsqueeze(0),  # (1, R, R, R)
            'mask': torch.from_numpy(mask).unsqueeze(0),        # (1, R, R, R)
        }


# ================================================================
# Patch采样
# ================================================================
def sample_patch(sample, patch_size=64, high_stress_prob=0.5):
    """
    从全图中切一个 patch_size³ 的patch。
    50% 概率以高应力区域为中心, 50% 随机。
    """
    R = sample['geo_input'].shape[-1]
    P = patch_size

    mask_np = sample['mask'].squeeze().numpy()
    S3_np = sample['S3_grid'].squeeze().numpy()

    # 选patch中心
    if np.random.rand() < high_stress_prob:
        abs_s3 = np.abs(S3_np) * mask_np
        flat_idx = np.argpartition(abs_s3.ravel(), -100)[-100:]
        chosen = np.random.choice(flat_idx)
        cz, cy, cx = np.unravel_index(chosen, abs_s3.shape)
    else:
        valid_idx = np.where(mask_np > 0.5)
        if len(valid_idx[0]) == 0:
            cx, cy, cz = R // 2, R // 2, R // 2
        else:
            ri = np.random.randint(len(valid_idx[0]))
            cz, cy, cx = valid_idx[0][ri], valid_idx[1][ri], valid_idx[2][ri]

    def clamp_start(c, P, R):
        return max(0, min(c - P // 2, R - P))

    sx = clamp_start(cx, P, R)
    sy = clamp_start(cy, P, R)
    sz = clamp_start(cz, P, R)

    sl = (slice(sz, sz + P), slice(sy, sy + P), slice(sx, sx + P))

    return {
        'geo_input': sample['geo_input'][:, sl[0], sl[1], sl[2]],      # (1, P³)
        'phys_input': sample['phys_input'][:, sl[0], sl[1], sl[2]],    # (4, P³)
        'S3_grid': sample['S3_grid'][:, sl[0], sl[1], sl[2]],          # (1, P³)
        'mask': sample['mask'][:, sl[0], sl[1], sl[2]],                # (1, P³)
    }


# ================================================================
# PatchTrainDataset (固定grid切patch, 全覆盖)
# ================================================================
class PatchTrainDataset(Dataset):
    """
    训练用: 每个样本固定切 grid 覆盖的 patches (无overlap).
    128³ / 64³ = 2³ = 8 patches per sample.
    """
    def __init__(self, data_dir, grid_res=128, patch_size=64,
                 max_cache=50, sample_indices=None, **kwargs):
        self.full_dataset = HybridDataset(data_dir, grid_res=grid_res)
        self.patch_size = patch_size

        if sample_indices is not None:
            self.indices = list(sample_indices)
        else:
            self.indices = list(range(len(self.full_dataset)))

        # Pre-compute patch positions (30% overlap for training)
        R, P = grid_res, patch_size
        stride = int(P * 0.7)  # 30% overlap → stride=44
        starts = list(range(0, R - P + 1, stride))
        if starts[-1] + P < R:
            starts.append(R - P)
        self.patch_starts = [(sz, sy, sx)
                              for sz in starts for sy in starts for sx in starts]
        self.ppv = len(self.patch_starts)
        self.total = len(self.indices) * self.ppv

        from collections import OrderedDict
        self.max_cache = max_cache
        self._cache = OrderedDict()

        print(f"[PatchTrainDataset] {len(self.indices)} samples x {self.ppv} patches "
              f"= {self.total} total")

    def _get_sample(self, idx):
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        sample = self.full_dataset[idx]
        self._cache[idx] = sample
        if len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)
        return sample

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        sample_idx = self.indices[idx // self.ppv]
        patch_idx = idx % self.ppv
        sample = self._get_sample(sample_idx)

        sz, sy, sx = self.patch_starts[patch_idx]
        P = self.patch_size
        sl = (slice(sz, sz+P), slice(sy, sy+P), slice(sx, sx+P))

        return {
            'geo_input': sample['geo_input'][:, sl[0], sl[1], sl[2]],
            'phys_input': sample['phys_input'][:, sl[0], sl[1], sl[2]],
            'S3_grid': sample['S3_grid'][:, sl[0], sl[1], sl[2]],
            'mask': sample['mask'][:, sl[0], sl[1], sl[2]],
        }


# ================================================================
# PreCutPatchDataset (读离线预切的.pt文件, 最快)
# ================================================================
class PreCutPatchDataset(Dataset):
    """
    直接读预切好的.pt patch文件, 零体素化开销。

    Filtering priority (first non-None wins):
      1. case_filter  — set/list of case names to keep (cleanest)
      2. sample_indices + sample_names — legacy index-based filter
      3. none         — use all patches
    """
    def __init__(self, patch_dir, case_filter=None,
                 sample_indices=None, sample_names=None):
        self.patch_dir = Path(patch_dir)
        all_files = sorted(self.patch_dir.glob("*.pt"))

        if case_filter is not None:
            valid_names = set(case_filter)
            self.files = [f for f in all_files
                          if f.stem.rsplit('_p', 1)[0] in valid_names]
        elif sample_indices is not None and sample_names is not None:
            valid_names = set(sample_names[i] for i in sample_indices)
            self.files = [f for f in all_files
                          if f.stem.rsplit('_p', 1)[0] in valid_names]
        else:
            self.files = all_files

        print(f"[PreCutPatchDataset] {len(self.files)} patches from {self.patch_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        # Convert fp16 back to fp32 for training
        return {k: v.float() if v.dtype == torch.float16 else v for k, v in data.items()}


# ================================================================
# Test
# ================================================================
if __name__ == "__main__":
    ds = HybridDataset("../Data/ElementGraph", grid_res=128)
    print(f"Samples: {len(ds)}")

    if len(ds) > 0:
        s = ds[0]
        print(f"geo_input:  {s['geo_input'].shape}")
        print(f"phys_input: {s['phys_input'].shape}")
        print(f"S3_grid:    {s['S3_grid'].shape}")
        print(f"mask:       {s['mask'].shape}")

        patch = sample_patch(s, patch_size=64)
        print(f"\nPatch:")
        print(f"  geo:  {patch['geo_input'].shape}")
        print(f"  phys: {patch['phys_input'].shape}")
        print(f"  S3:   {patch['S3_grid'].shape}")
        print(f"  mask: {patch['mask'].shape}")
