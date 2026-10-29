#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-cut tumor patches with BODY-ONLY bbox.

For each tumor case:
  1. Split elements into body / posterior (pure-Y KMeans + graph dilation
     along mesh edges, per-MV dilation override for outliers).
  2. Keep only body elements.
  3. Compute the bbox from BODY centroids (smaller than full vertebra).
  4. Voxelise (geo / phys / S3 / mask) into a 256³ grid covering only the
     body. This naturally raises Y-axis (anterior–posterior) resolution
     ~2.4× since body Y-extent is ~42 mm vs ~99 mm for the full vertebra.
  5. Cut 128³ patches with the requested overlap.

Output:
    Patches_Tumor_Body_Train_256_128/   (overlap 0.5)
    Patches_Tumor_Body_Test_256_128/    (overlap 0.0)

Usage:
    python precut_patches_body.py --all
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import (
    binary_dilation, binary_fill_holes, distance_transform_edt,
)
from sklearn.cluster import KMeans


# ================================================================
# Per-MV dilation overrides — tuned empirically per vertebra group
# ================================================================
DILATE_DEFAULT = 2
DILATE_OVERRIDES = {
    "MV08_L4": 8,
}


# ================================================================
# Helpers (replicated from SSM-Hilbert/dataUtils/DataLoader.py so this
# script is self-contained)
# ================================================================
def signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


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
    mask_bool = mask > 0.5
    dist = distance_transform_edt(~mask_bool)
    dilated = binary_dilation(mask_bool, iterations=iterations)
    result = dilated & (dist <= max_dist)
    result = binary_fill_holes(result)
    return result.astype(np.float32)


def _sample_bbox(centroids, pad=0.01):
    mins = centroids.min(axis=0)
    maxs = centroids.max(axis=0)
    size = maxs - mins
    return ((mins - pad * size).astype(np.float32),
            (maxs + pad * size).astype(np.float32))


def _to_grid_coords(pts, bbox_min, bbox_max, R):
    scale = bbox_max - bbox_min
    scale[scale == 0] = 1.0
    q = (pts - bbox_min) / scale
    q = np.clip(q, 0, 1 - 1e-6)
    return q * (R - 1)


def _splat_trilinear(pts_grid, values, R):
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


def _splat_min(pts_grid, values, R):
    gi = np.round(pts_grid).astype(np.int32)
    gi = np.clip(gi, 0, R - 1)
    grid = np.full((R, R, R), np.inf, dtype=np.float32)
    np.minimum.at(grid, (gi[:, 0], gi[:, 1], gi[:, 2]),
                  values.astype(np.float32))
    occ = np.zeros((R, R, R), dtype=np.float32)
    occ[gi[:, 0], gi[:, 1], gi[:, 2]] = 1.0
    grid[occ < 0.5] = 0.0
    return grid, occ


# ================================================================
# Body / posterior segmentation
# ================================================================
def split_body_posterior(centroids, edge_index=None, dilate_iter=2):
    """KMeans on Y projection + graph dilation. Body = larger cluster."""
    proj = centroids[:, 1].copy()
    km = KMeans(n_clusters=2, n_init=10, random_state=0)
    labels = km.fit_predict(proj.reshape(-1, 1))
    body_label = int(np.argmax(np.bincount(labels)))
    body_mask = labels == body_label

    if edge_index is not None and edge_index.shape[1] > 0 and dilate_iter > 0:
        src = edge_index[0]
        dst = edge_index[1]
        for _ in range(dilate_iter):
            grow = body_mask.copy()
            grow[dst[body_mask[src]]] = True
            if grow.sum() == body_mask.sum():
                break
            body_mask = grow

    return body_mask


def _mv_l_key(name):
    parts = name.split("_")
    mv = parts[0]
    L = next((p for p in parts if len(p) == 2 and p.startswith("L")), "?")
    return f"{mv}_{L}"


# ================================================================
# Per-case extraction
# ================================================================
def extract_body_case(npz_path, R, s3_splat="min", pad=0.01,
                     dilate_iter=2, dilate_iterations_mask=2):
    """Load one tumor npz, filter to body elements, voxelise.

    Returns dict with the same fields HybridDataset.__getitem__ produces:
        geo_input  (1, R, R, R) — log1p(E)
        phys_input (4, R, R, R) — [BC, Load*u, dist_BC, dist_Load]
        S3_grid    (1, R, R, R) — signed_sqrt(S3 * 10)
        mask       (1, R, R, R) — bone occupancy of the BODY region only
        bbox_min, bbox_max      — saved as metadata in the .pt
    """
    data = np.load(npz_path)
    centroids = data["centroids"].astype(np.float32)
    E_raw = data["E"].astype(np.float32)
    S3_raw = data["S3"].astype(np.float32)
    is_bc = data["is_bc"].astype(np.float32)
    is_load = data["is_load"].astype(np.float32)
    u_disp = float(data["u_disp"][0])
    edge_index = data["edge_index"] if "edge_index" in data.files else None

    # ---- Body / posterior split ----
    body_mask = split_body_posterior(centroids, edge_index, dilate_iter)
    centroids = centroids[body_mask]
    E_raw = E_raw[body_mask]
    S3_raw = S3_raw[body_mask]
    is_bc = is_bc[body_mask]
    is_load = is_load[body_mask]

    # ---- bbox from BODY centroids ----
    bbox_min, bbox_max = _sample_bbox(centroids, pad)
    pts_grid = _to_grid_coords(centroids, bbox_min, bbox_max, R)

    # ---- Voxelise ----
    E_log = np.log1p(E_raw)
    S3_log = signed_sqrt(S3_raw * 10.0)

    E_grid_raw, occ = _splat_trilinear(pts_grid, E_log, R)
    if s3_splat == "min":
        S3_grid_raw, _ = _splat_min(pts_grid, S3_log, R)
    else:
        S3_grid_raw, _ = _splat_trilinear(pts_grid, S3_log, R)

    mask = _clipped_dilation(occ, max_dist=2, iterations=dilate_iterations_mask)
    E_grid = _inmask_nearest_fill(E_grid_raw, mask) * mask
    S3_grid = _inmask_nearest_fill(S3_grid_raw, mask) * mask

    # ---- BC / Load voxel masks ----
    BC_grid = np.zeros((R, R, R), dtype=np.float32)
    Load_grid = np.zeros((R, R, R), dtype=np.float32)
    bc_idx = np.where(is_bc > 0.5)[0]
    load_idx = np.where(is_load > 0.5)[0]
    if len(bc_idx) > 0:
        bg = np.round(pts_grid[bc_idx]).astype(int)
        bg = np.clip(bg, 0, R - 1)
        BC_grid[bg[:, 0], bg[:, 1], bg[:, 2]] = 1.0
    if len(load_idx) > 0:
        lg = np.round(pts_grid[load_idx]).astype(int)
        lg = np.clip(lg, 0, R - 1)
        Load_grid[lg[:, 0], lg[:, 1], lg[:, 2]] = 1.0
    u_sqrt = float(np.sign(u_disp) * np.sqrt(abs(u_disp)))
    Load_grid_u = Load_grid * u_sqrt

    # ---- Distance fields ----
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
    dist_BC = dist_BC * mask
    dist_Load = dist_Load * mask

    return {
        "geo_input": torch.from_numpy(E_grid).unsqueeze(0),
        "phys_input": torch.stack([
            torch.from_numpy(BC_grid),
            torch.from_numpy(Load_grid_u),
            torch.from_numpy(dist_BC),
            torch.from_numpy(dist_Load),
        ], dim=0),
        "S3_grid": torch.from_numpy(S3_grid).unsqueeze(0),
        "mask": torch.from_numpy(mask).unsqueeze(0),
        "bbox_min": torch.from_numpy(bbox_min),
        "bbox_max": torch.from_numpy(bbox_max),
        "name": npz_path.stem,
        "n_body_elems": int(body_mask.sum()),
        "n_total_elems": int(len(body_mask)),
    }


# ================================================================
# Patch utilities (replicated from precut_patches.py)
# ================================================================
def get_patch_positions(R, P, overlap):
    stride = max(1, int(P * (1 - overlap)))
    starts = list(range(0, R - P + 1, stride))
    if starts[-1] + P < R:
        starts.append(R - P)
    return [(sz, sy, sx) for sz in starts for sy in starts for sx in starts]


def save_full_volume(sample, name, out_dir, R):
    """Save the whole 256^3 volume as one {name}.pt (half precision) for
    nnUNet-style on-the-fly random cropping (RandomCropDataset).

    Stores the same fields as a patch, just uncut. S3_grid stays in the
    signed_sqrt(strain*10) space — RandomCropDataset converts on the fly,
    identical to PreCutPatchDataset.
    """
    saved = {k: sample[k].half() for k in
             ("geo_input", "phys_input", "S3_grid", "mask")}
    saved["bbox_min"] = sample["bbox_min"]
    saved["bbox_max"] = sample["bbox_max"]
    saved["grid_res"] = torch.tensor(R, dtype=torch.int16)
    torch.save(saved, out_dir / f"{name}.pt")
    return 1


def save_patches(sample, name, patch_positions, P, out_dir, R, overlap):
    for pi, (sz, sy, sx) in enumerate(patch_positions):
        sl = (slice(sz, sz + P), slice(sy, sy + P), slice(sx, sx + P))
        patch = {k: sample[k][:, sl[0], sl[1], sl[2]] for k in
                 ("geo_input", "phys_input", "S3_grid", "mask")}
        saved = {k: v.half() for k, v in patch.items()}
        saved["patch_origin"] = torch.tensor([sz, sy, sx], dtype=torch.int16)
        saved["patch_index"] = torch.tensor(pi, dtype=torch.int16)
        saved["patch_size"] = torch.tensor(P, dtype=torch.int16)
        saved["grid_res"] = torch.tensor(R, dtype=torch.int16)
        saved["overlap"] = torch.tensor(float(overlap), dtype=torch.float32)
        saved["stride"] = torch.tensor(max(1, int(P * (1 - overlap))),
                                        dtype=torch.int16)
        saved["bbox_min"] = sample["bbox_min"]
        saved["bbox_max"] = sample["bbox_max"]
        torch.save(saved, out_dir / f"{name}_p{pi}.pt")
    return len(patch_positions)


# ================================================================
# Driver
# ================================================================
def process_split(npz_dir, out_dir, R, P, overlap, s3_splat, force,
                  save_full=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(npz_dir).glob("*.npz"))
    mode = "FULL VOLUME" if save_full else f"patches overlap={overlap}"
    print(f"\n=== {out_dir.name} (npz: {npz_dir.name}, {mode}) ===")
    print(f"  {len(files)} cases")

    patch_positions = get_patch_positions(R, P, overlap)
    n_per = len(patch_positions)
    if not save_full:
        print(f"  patches/case: {n_per}")

    n_saved = n_skipped = 0
    t_start = time.time()

    for ci, npz in enumerate(files):
        name = npz.stem
        last = (out_dir / f"{name}.pt") if save_full \
            else (out_dir / f"{name}_p{n_per - 1}.pt")
        if last.exists() and not force:
            n_skipped += 1
            continue

        key = _mv_l_key(name)
        dilate_iter = DILATE_OVERRIDES.get(key, DILATE_DEFAULT)

        t0 = time.time()
        try:
            sample = extract_body_case(
                npz, R, s3_splat=s3_splat, dilate_iter=dilate_iter)
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            continue

        if save_full:
            save_full_volume(sample, name, out_dir, R)
        else:
            save_patches(sample, name, patch_positions, P, out_dir, R, overlap)
        n_saved += 1

        elapsed = time.time() - t_start
        rate = n_saved / max(elapsed, 1e-3)
        remain = (len(files) - ci - 1) / max(rate, 1e-3)
        body_pct = 100 * sample["n_body_elems"] / sample["n_total_elems"]
        print(f"  [{ci+1}/{len(files)}] {name}: {time.time()-t0:.1f}s "
              f"body={body_pct:.0f}% (dilate={dilate_iter}) "
              f"ETA {remain/60:.0f}min")

    print(f"  saved={n_saved}, skipped={n_skipped}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_train", type=Path,
                    default=Path("/mnt/training_Data/SSM-Data/ElementGraph_Tumor_Train"))
    ap.add_argument("--npz_test", type=Path,
                    default=Path("/mnt/training_Data/SSM-Data/ElementGraph_Tumor_Test"))
    ap.add_argument("--out_train", type=Path,
                    default=Path("/mnt/training_Data/SSM-Data/Patches_Tumor_Body_Train_256_128"))
    ap.add_argument("--out_test", type=Path,
                    default=Path("/mnt/training_Data/SSM-Data/Patches_Tumor_Body_Test_256_128"))
    ap.add_argument("--save_full_volume", action="store_true",
                    help="Save one uncut {name}.pt per case (256^3) instead of "
                         "fixed-grid patches, for nnUNet-style RandomCropDataset. "
                         "Writes to --out_full_train / --out_full_test.")
    ap.add_argument("--out_full_train", type=Path,
                    default=Path("/mnt/training_Data/SSM-Data/FullVol_Tumor_Body_Train_256"))
    ap.add_argument("--out_full_test", type=Path,
                    default=Path("/mnt/training_Data/SSM-Data/FullVol_Tumor_Body_Test_256"))
    ap.add_argument("--grid_res", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--train_overlap", type=float, default=0.5)
    ap.add_argument("--test_overlap", type=float, default=0.0)
    ap.add_argument("--s3_splat", default="min", choices=["mean", "min"])
    ap.add_argument("--all", action="store_true",
                    help="Process both train and test.")
    ap.add_argument("--only_train", action="store_true")
    ap.add_argument("--only_test", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-process even if patches already exist.")
    args = ap.parse_args()

    do_train = args.all or args.only_train or not (args.only_test)
    do_test = args.all or args.only_test or not (args.only_train)

    out_train = args.out_full_train if args.save_full_volume else args.out_train
    out_test = args.out_full_test if args.save_full_volume else args.out_test

    if do_train:
        process_split(args.npz_train, out_train,
                      args.grid_res, args.patch_size,
                      args.train_overlap, args.s3_splat, args.force,
                      save_full=args.save_full_volume)
    if do_test:
        process_split(args.npz_test, out_test,
                      args.grid_res, args.patch_size,
                      args.test_overlap, args.s3_splat, args.force,
                      save_full=args.save_full_volume)

    print("\nAll done.")


if __name__ == "__main__":
    main()
