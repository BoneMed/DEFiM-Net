#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-Encoder DataLoader for SSM-Hybrid.

Source: Data/ElementGraph_*/*.npz (per-element fields)
  -> on-the-fly voxelization to grid_res^3
  -> two output streams:
       geo_input:  [E_grid] (1ch)         -- geometry / material
       phys_input: [BC, Load*u, dist_BC, dist_Load] (4ch) -- BCs / loads
       S3_grid:    [target] (1ch)
       mask:       [bone occupancy] (1ch)

PreCutPatchDataset reads the offline pre-cut .pt patches and attaches
the global Hilbert-curve scan_key for the SSM-Hilbert model. The scan
key is loaded once from dataUtils/hilbert_R{P}.npy where P is the
patch size; air voxels are pushed to BIG_VAL inside the model via the
per-patch bone mask, so the dataset just hands over the raw Hilbert
positions.
"""

import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import distance_transform_edt, binary_fill_holes, binary_dilation


# ================================================================
# Helpers
# ================================================================
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def signed_log(x, scale=100.0):
    """sign(x) * log(1 + |x * scale|) — current target encoding for
    strain. Preserves the extreme tail better than signed_sqrt because
    log has gentler slope at small |x| AND saturates more gracefully
    for large |x|. With scale=100, percent-strain magnitudes get pushed
    into a clean O(1)-O(10) numerical range.
    """
    return np.sign(x) * np.log1p(np.abs(x) * scale)


def inv_signed_log(y, scale=100.0):
    """Inverse: sign(y) * (exp(|y|) - 1) / scale  →  physical strain."""
    return np.sign(y) * np.expm1(np.abs(y)) / scale


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
# HybridDataset: voxelization + distance fields
# ================================================================
class HybridDataset(Dataset):
    """
    Loads ElementGraph npz, voxelizes per-sample.
    Returns:
        geo_input:  (1, R, R, R)  E_grid
        phys_input: (4, R, R, R)  [BC, Load*u, dist_BC, dist_Load]
        S3_grid:    (1, R, R, R)  target stress/strain
        mask:       (1, R, R, R)  bone occupancy
    """
    def __init__(self, data_dir, grid_res=128, dilate_iter=2, pad=0.01,
                 s3_splat='mean'):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npz"))
        self.names = [f.stem for f in self.files]
        self.grid_res = grid_res
        self.dilate_iter = dilate_iter
        self.pad = pad
        self.s3_splat = s3_splat  # 'mean' or 'min'

        print(f"[HybridDataset] {len(self.files)} samples, res={grid_res}, "
              f"s3_splat={s3_splat}")
        print(f"  bbox: per-sample (each sample uses its own bbox)")

    @staticmethod
    def _sample_bbox(centroids, pad=0.01):
        mins = centroids.min(axis=0)
        maxs = centroids.max(axis=0)
        size = maxs - mins
        return ((mins - pad * size).astype(np.float32),
                (maxs + pad * size).astype(np.float32))

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
        """Min splatting: each voxel gets the minimum value among its elements."""
        gi = np.round(pts_grid).astype(np.int32)
        gi = np.clip(gi, 0, R - 1)

        grid = np.full((R, R, R), np.inf, dtype=np.float32)
        np.minimum.at(grid, (gi[:, 0], gi[:, 1], gi[:, 2]),
                      values.astype(np.float32))

        occ = np.zeros((R, R, R), dtype=np.float32)
        occ[gi[:, 0], gi[:, 1], gi[:, 2]] = 1.0
        grid[occ < 0.5] = 0.0
        return grid, occ

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        centroids = data['centroids'].astype(np.float32)
        E_raw     = data['E'].astype(np.float32)
        S3_raw    = data['S3'].astype(np.float32)
        is_bc     = data['is_bc'].astype(np.float32)
        is_load   = data['is_load'].astype(np.float32)
        u_disp    = float(data['u_disp'][0])

        R = self.grid_res

        # ---- Voxelization ----
        E_log = np.log1p(E_raw)
        S3_log = signed_log(S3_raw, scale=100.0)

        bbox_min, bbox_max = self._sample_bbox(centroids, self.pad)
        pts_grid = self._to_grid_coords(centroids, bbox_min, bbox_max)

        E_grid_raw, occ = self._splat_trilinear(pts_grid, E_log, R)
        if self.s3_splat == 'min':
            S3_grid_raw, _ = self._splat_min(pts_grid, S3_log, R)
        else:
            S3_grid_raw, _ = self._splat_trilinear(pts_grid, S3_log, R)

        mask = _clipped_dilation(occ, max_dist=2, iterations=self.dilate_iter)
        E_grid  = _inmask_nearest_fill(E_grid_raw, mask) * mask
        S3_grid = _inmask_nearest_fill(S3_grid_raw, mask) * mask

        # ---- BC / Load voxel masks ----
        BC_grid   = np.zeros((R, R, R), dtype=np.float32)
        Load_grid = np.zeros((R, R, R), dtype=np.float32)

        bc_elem_idx   = np.where(is_bc   > 0.5)[0]
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

        # ---- Distance fields (each normalized to [0,1] inside bone) ----
        if BC_grid.max() > 0:
            dist_BC = distance_transform_edt(BC_grid < 0.5).astype(np.float32)
            dist_BC = dist_BC / (dist_BC.max() + 1e-8)
        else:
            dist_BC = np.ones((R, R, R), dtype=np.float32)

        if Load_grid.max() > 0:
            dist_Load = distance_transform_edt(Load_grid < 0.5).astype(np.float32)
            dist_Load = dist_Load / (dist_Load.max() + 1e-8)
        else:
            dist_Load = np.ones((R, R, R), dtype=np.float32)

        dist_BC   = dist_BC   * mask
        dist_Load = dist_Load * mask

        # ---- Output tensors ----
        # geo:  [E_grid] (1ch)
        # phys: [BC, Load*u, dist_BC, dist_Load] (4ch). Sparse path
        #   ([BC, Load*u]) goes through SparsePhysicsEncoder; dense path
        #   ([dist_BC, dist_Load]) is concatenated inside that encoder.
        geo_input = torch.from_numpy(E_grid).unsqueeze(0)

        phys_input = torch.stack([
            torch.from_numpy(BC_grid),
            torch.from_numpy(Load_grid_u),
            torch.from_numpy(dist_BC),
            torch.from_numpy(dist_Load),
        ], dim=0)

        return {
            'name': self.names[idx],
            'geo_input': geo_input,
            'phys_input': phys_input,
            'S3_grid': torch.from_numpy(S3_grid).unsqueeze(0),
            'mask':    torch.from_numpy(mask).unsqueeze(0),
        }


# ================================================================
# PreCutPatchDataset
# ================================================================
class PreCutPatchDataset(Dataset):
    """
    Reads pre-cut .pt patches; minimal voxelisation work at training time.

    Filtering priority (first non-None wins):
      1. case_filter
      2. sample_indices + sample_names
      3. none

    Hilbert scan_key (required by SSM-Hilbert model):
      Loaded once at __init__ from dataUtils/hilbert_R{patch_size}.npy.
      Each patch shares the same local Hilbert table (curve starts at
      the patch corner), so no per-case slicing is needed. Only the
      patch_size argument is required to pick the right table.
    """
    _hilbert_cache = {}  # patch_size -> shared (P,P,P) torch.float32 tensor

    def __init__(self, patch_dir, case_filter=None,
                 sample_indices=None, sample_names=None,
                 npz_dir=None, grid_res=None, patch_size=None, overlap=None):
        # npz_dir is unused now (no per-case .geodist.npy); grid_res and
        # overlap are stored as bookkeeping so the training script can
        # reconstruct full-volume patch positions during validation.
        del npz_dir

        self.patch_dir = Path(patch_dir)
        # Exclude *_nested.pt sidecar files generated by SSM-Hilbert-Nested —
        # they live alongside originals but only carry tumor-conditioning
        # fields, no 'mask' / 'geo_input' / ..., so loading them blows up here.
        all_files = sorted(f for f in self.patch_dir.glob("*.pt")
                           if not f.name.endswith("_nested.pt"))

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

        self.patch_size = patch_size
        self.grid_res = grid_res
        self.overlap = overlap
        self._scan_key = self._load_hilbert_table(patch_size) \
            if patch_size is not None else None

        if self._scan_key is not None:
            print(f"[PreCutPatchDataset] {len(self.files)} patches from "
                  f"{self.patch_dir}  (scan_key=Hilbert R={patch_size})")
        else:
            print(f"[PreCutPatchDataset] {len(self.files)} patches from "
                  f"{self.patch_dir}  (scan_key OFF — model will fail "
                  f"unless patch_size is provided)")

    @classmethod
    def _load_hilbert_table(cls, patch_size):
        """Load and cache hilbert_R{patch_size}.npy as (P, P, P) float32."""
        if patch_size in cls._hilbert_cache:
            return cls._hilbert_cache[patch_size]
        path = Path(__file__).resolve().parent / f"hilbert_R{patch_size}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Hilbert table not found: {path}\n"
                f"Generate it with:\n"
                f"  python dataUtils/compute_hilbert_order.py --R {patch_size}")
        arr = np.load(path).astype(np.float32)         # (P, P, P) int32 -> f32
        if arr.shape != (patch_size, patch_size, patch_size):
            raise ValueError(
                f"hilbert_R{patch_size}.npy shape mismatch: got {arr.shape}, "
                f"expected ({patch_size},)*3")
        t = torch.from_numpy(arr)                       # CPU float32, shared
        cls._hilbert_cache[patch_size] = t
        return t

    def __len__(self):
        return len(self.files)

    # Keep only the keys the training loop actually uses; bookkeeping
    # fields (`patch_origin`, `bbox_min`, `patch_size`, ...) exist in
    # some patches but are missing in older ones, which crashes
    # DataLoader's default_collate when both versions get batched
    # together. Validation `_reconstruct_case` reads the .pt files
    # directly via `torch.load` and still sees the full dict.
    _TRAIN_KEYS = ('geo_input', 'phys_input', 'S3_grid', 'mask')

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        out = {}
        for k in self._TRAIN_KEYS:
            if k not in data:
                raise KeyError(
                    f"Patch {self.files[idx].name} missing required key "
                    f"'{k}'. Re-cut patches with the current precut script.")
            v = data[k]
            out[k] = v.float() if v.dtype == torch.float16 else v
        # Stored S3_grid is in old signed_sqrt(strain × 10) space; convert
        # on the fly to the current signed_log(strain × 100) target encoding
        # using the algebraic identity
        #     y_old = sign(strain) * sqrt(|strain| * 10)
        #     |strain| * 100  =  y_old**2 * 10
        #     y_new = sign(y_old) * log1p(y_old**2 * 10)
        # Saves having to re-cut every patch.
        y_old = out['S3_grid'].float()
        out['S3_grid'] = torch.sign(y_old) * torch.log1p(y_old.pow(2) * 10.0)
        if self._scan_key is not None:
            # Same global Hilbert table for every patch; add a channel dim.
            out['scan_key'] = self._scan_key.unsqueeze(0)   # (1, P, P, P)
        return out


# ================================================================
# RandomCropDataset  (nnUNet-style on-the-fly sampling)
# ================================================================
class RandomCropDataset(Dataset):
    """On-the-fly random P^3 crops from full-volume .pt files, with
    foreground (low-E / tumor) oversampling — nnUNet-style.

    Why: the offline fixed-grid patches show the tumor at the same few
    positions every epoch and never oversample it, so the loss is
    dominated by near-zero healthy bone and the model regresses to the
    mean. Here each __getitem__ cuts a fresh random crop; with prob
    ``fg_oversample`` the crop is centred on a randomly chosen low-E
    (tumor) voxel, otherwise on a random bone voxel. Fresh randomness
    every epoch also augments tumor position -> better generalisation to
    unseen positions.

    Expects full volumes produced by precut_patches_body.py
    --save_full_volume: one ``{name}.pt`` per case holding half-precision
    ``geo_input (1,R,R,R) / phys_input (4,R,R,R) / S3_grid (1,R,R,R) /
    mask (1,R,R,R)``. S3_grid is in the old signed_sqrt(strain*10) space,
    converted on the fly to signed_log(strain*100) exactly like
    PreCutPatchDataset.

    Foreground = the lowest ``low_e_percentile`` % of E within bone
    (tumor tissue: E ~0.01 vs healthy ~700, so the low tail is the
    tumor). geo_input stores log1p(E); lower value = softer = tumor.

    Output dict matches PreCutPatchDataset:
        geo_input / phys_input / S3_grid / mask  (+ scan_key if enabled).
    """
    _hilbert_cache = PreCutPatchDataset._hilbert_cache  # share table cache

    def __init__(self, full_dir, case_filter=None, patch_size=128,
                 grid_res=256, samples_per_volume=27, fg_oversample=0.33,
                 tumor_e_max=1.0, cache_volumes=8,
                 epoch_length=None, batch_size=None):
        self.full_dir = Path(full_dir)
        all_files = sorted(f for f in self.full_dir.glob("*.pt")
                           if not f.name.endswith("_nested.pt"))
        if case_filter is not None:
            valid = set(case_filter)
            self.files = [f for f in all_files if f.stem in valid]
        else:
            self.files = all_files
        if not self.files:
            raise SystemExit(
                f"RandomCropDataset: no full-volume .pt in {self.full_dir} "
                f"(after case_filter). Generate them with "
                f"precut_patches_body.py --save_full_volume.")

        self.patch_size = patch_size
        self.grid_res = grid_res
        self.samples_per_volume = max(1, int(samples_per_volume))
        self.fg_oversample = float(fg_oversample)
        # Tumor tissue has a FIXED sentinel modulus E=0.01 (next-lowest healthy
        # value is ~17), so an absolute threshold cleanly captures the WHOLE
        # tumor regardless of its size — unlike a percentile, which under-selects
        # big tumors (a VOL_30 tumor is ~29% of bone, not 5%). geo_input stores
        # log1p(E), so compare in that space.
        self.tumor_e_max = float(tumor_e_max)
        self.log_tumor_e_max = float(np.log1p(self.tumor_e_max))
        self.cache_volumes = max(1, int(cache_volumes))
        self._cache = OrderedDict()   # path -> volume dict (half tensors + coords)
        self._scan_key = PreCutPatchDataset._load_hilbert_table(patch_size) \
            if patch_size is not None else None

        # nnUNet-style fixed epoch length + per-batch foreground forcing.
        # epoch_length: fixed #patches drawn per epoch (case chosen at random
        #   per item) instead of len(files)*spv. batch_size: when set, the last
        #   round(batch_size*fg_oversample) items of EVERY batch are forced to be
        #   foreground (tumor-centred) and the rest random — exactly nnUNet's
        #   oversample_foreground_percent. For batch_size=2, fg=0.33 -> 1 tumor +
        #   1 random per batch. Requires the loader to use shuffle=False so item
        #   index order maps to batch position (idx % batch_size).
        self.epoch_length = int(epoch_length) if epoch_length else None
        self.batch_size = int(batch_size) if batch_size else None
        self.n_fg_per_batch = (round(self.batch_size * self.fg_oversample)
                               if self.batch_size else None)
        print(f"[RandomCropDataset] {len(self.files)} full volumes from "
              f"{self.full_dir}  (P={patch_size}, "
              f"epoch_length={self.epoch_length or len(self.files)*self.samples_per_volume}, "
              f"fg_oversample={self.fg_oversample}"
              f"{f' -> {self.n_fg_per_batch}/{self.batch_size} per batch' if self.batch_size else ''}, "
              f"tumor=E<={self.tumor_e_max}, cache={self.cache_volumes})")

    def __len__(self):
        if self.epoch_length is not None:
            return self.epoch_length
        return len(self.files) * self.samples_per_volume

    def _load_volume(self, path):
        """Load (LRU-cached) a full volume; precompute fg / bone voxel coords."""
        c = self._cache
        if path in c:
            c.move_to_end(path)
            return c[path]
        data = torch.load(path, weights_only=False)
        vol = {k: data[k] for k in ('geo_input', 'phys_input', 'S3_grid', 'mask')}
        # bone + tumor (low-E) foreground voxel coordinates (once per volume).
        # Tumor = sentinel modulus E=0.01; absolute threshold on log1p(E)
        # captures the whole tumor at any size.
        mask = vol['mask'][0].float() > 0.5
        e = vol['geo_input'][0].float()                 # log1p(E)
        bone_coords = torch.nonzero(mask, as_tuple=False)   # (Nb, 3)
        fg_mask = mask & (e <= self.log_tumor_e_max)
        fg_coords = torch.nonzero(fg_mask, as_tuple=False)
        vol['fg'] = fg_coords
        vol['bone'] = bone_coords
        c[path] = vol
        if len(c) > self.cache_volumes:
            c.popitem(last=False)
        return vol

    def __getitem__(self, idx):
        if self.epoch_length is not None:
            # Fixed-epoch mode: pick a random case for every drawn patch.
            file_idx = int(torch.randint(len(self.files), ()).item())
        else:
            file_idx = idx // self.samples_per_volume   # locality when shuffle=False
        vol = self._load_volume(self.files[file_idx])
        P, R = self.patch_size, self.grid_res

        fg, bone = vol['fg'], vol['bone']
        if self.batch_size is not None:
            # nnUNet per-batch forcing: last n_fg_per_batch items of each batch
            # are foreground (needs loader shuffle=False so idx -> batch position).
            pos = idx % self.batch_size
            want_fg = pos >= (self.batch_size - self.n_fg_per_batch)
            use_fg = want_fg and (len(fg) > 0)
        else:
            use_fg = (len(fg) > 0) and (float(torch.rand(()).item()) < self.fg_oversample)
        pool = fg if use_fg else bone
        if len(pool) == 0:
            pool = bone
        center = pool[int(torch.randint(len(pool), ()).item())]   # (3,)

        s = [int(min(max(int(center[d]) - P // 2, 0), R - P)) for d in range(3)]
        sl = (slice(s[0], s[0] + P), slice(s[1], s[1] + P), slice(s[2], s[2] + P))

        out = {
            'geo_input':  vol['geo_input'][:, sl[0], sl[1], sl[2]].float(),
            'phys_input': vol['phys_input'][:, sl[0], sl[1], sl[2]].float(),
            'mask':       vol['mask'][:, sl[0], sl[1], sl[2]].float(),
        }
        # Same old signed_sqrt(strain*10) -> signed_log(strain*100) conversion
        # as PreCutPatchDataset, applied to the crop.
        y_old = vol['S3_grid'][:, sl[0], sl[1], sl[2]].float()
        out['S3_grid'] = torch.sign(y_old) * torch.log1p(y_old.pow(2) * 10.0)
        if self._scan_key is not None:
            out['scan_key'] = self._scan_key.unsqueeze(0)
        return out
