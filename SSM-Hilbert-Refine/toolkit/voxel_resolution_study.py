#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantitative study of voxelization resolution (64 / 128 / 256 / 512).

For each R, measure on representative cases:
  - Physical voxel size (mm)
  - Bone voxel count (mask.sum)
  - Unique-voxel coverage: fraction of elements mapped to distinct voxels
  - Collision rate: average elements per occupied voxel
  - Cortical shell layers: mean thickness of mask shell (erosion iterations to empty)
  - Memory footprint per sample (7ch fp16 × R³)
  - Round-trip S3 accuracy: voxelize -> sample back -> compare vs original element S3

Output: CSV table + 2 plots.
"""
import argparse
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, binary_fill_holes, binary_dilation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def inv_signed_sqrt(y):
    return np.sign(y) * (y ** 2)


def _sample_bbox(centroids, pad=0.01):
    mins = centroids.min(axis=0)
    maxs = centroids.max(axis=0)
    size = maxs - mins
    return (mins - pad * size).astype(np.float32), (maxs + pad * size).astype(np.float32)


def _to_grid_coords(pts, bbox_min, bbox_max, R):
    scale = bbox_max - bbox_min
    scale[scale == 0] = 1.0
    q = np.clip((pts - bbox_min) / scale, 0, 1 - 1e-6)
    return q * (R - 1)


def splat_trilinear(pts_grid, values, R):
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
                np.add.at(acc,  (iz, iy, ix), (values * w).astype(np.float32))
                np.add.at(wsum, (iz, iy, ix), w.astype(np.float32))
    grid = np.zeros_like(acc)
    valid = wsum > 1e-8
    grid[valid] = acc[valid] / wsum[valid]
    return grid, valid.astype(np.float32)


def _clipped_dilation(mask, max_dist=2, iterations=2):
    mask = mask > 0.5
    dist = distance_transform_edt(~mask)
    dilated = binary_dilation(mask, iterations=iterations)
    result = dilated & (dist <= max_dist)
    result = binary_fill_holes(result)
    return result.astype(np.float32)


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


def shell_thickness(mask):
    """Return number of erosion iterations to fully empty the mask (rough thickness)."""
    m = mask > 0.5
    if not m.any():
        return 0
    n = 0
    while m.any() and n < 64:
        m = binary_erosion(m)
        n += 1
    return n


def analyze_one(npz_path, R):
    """Full analysis for one case at given resolution."""
    t0 = time.time()
    d = np.load(npz_path)
    centroids = d['centroids'].astype(np.float32)
    S3_raw    = d['S3'].astype(np.float32)
    E_raw     = d['E'].astype(np.float32)

    # Physical bbox
    bbox_min, bbox_max = _sample_bbox(centroids)
    bbox_size_mm = (bbox_max - bbox_min)  # (3,)
    voxel_size_mm = bbox_size_mm / R       # (3,)

    # Grid coords
    pts_grid = _to_grid_coords(centroids, bbox_min, bbox_max, R)

    # Splat (use E as a proxy to build occupancy)
    E_log = np.log1p(E_raw)
    S3_log = signed_sqrt(S3_raw * 10.0)
    _, occ = splat_trilinear(pts_grid, E_log, R)
    S3_grid, _ = splat_trilinear(pts_grid, S3_log, R)

    # Mask (dilated)
    mask = _clipped_dilation(occ, max_dist=2, iterations=2)
    S3_grid_filled = _inmask_nearest_fill(S3_grid, mask)

    # --- Metrics ---
    n_elem = len(centroids)
    bone_voxels = int(mask.sum())

    # Unique voxels hit by element centroids (nearest)
    gi = np.clip(np.round(pts_grid).astype(np.int32), 0, R - 1)
    keys = gi[:, 0].astype(np.int64) * R * R + gi[:, 1] * R + gi[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    unique_voxels_hit = len(uniq)
    collision_rate = 1.0 - unique_voxels_hit / n_elem  # fraction of elements that share a voxel
    avg_elems_per_voxel = n_elem / max(unique_voxels_hit, 1)
    max_elems_per_voxel = int(counts.max())

    # Coverage: hit-voxels / bone-voxels
    coverage = unique_voxels_hit / max(bone_voxels, 1)

    # Shell thickness (number of erosions to vanish)
    shell = shell_thickness(mask)

    # --- Round-trip S3 accuracy ---
    # Sample S3_grid_filled back at element centroid voxels
    pred_elem_log = S3_grid_filled[gi[:, 0], gi[:, 1], gi[:, 2]]
    # Compare in log domain to match what model actually learns
    mae_log = float(np.mean(np.abs(pred_elem_log - S3_log)))
    ss_res = float(np.sum((pred_elem_log - S3_log) ** 2))
    ss_tot = float(np.sum((S3_log - S3_log.mean()) ** 2))
    r2_log = 1 - ss_res / max(ss_tot, 1e-12)
    # Also decode to MPa for interpretability
    pred_mpa = inv_signed_sqrt(pred_elem_log) / 10.0
    gt_mpa   = S3_raw
    mae_mpa  = float(np.mean(np.abs(pred_mpa - gt_mpa)))
    rmse_mpa = float(np.sqrt(np.mean((pred_mpa - gt_mpa) ** 2)))

    # --- Memory ---
    n_channels = 7  # 1 geo + 4 phys + 1 S3 + 1 mask
    bytes_fp16 = n_channels * (R ** 3) * 2
    mem_mb = bytes_fp16 / (1024 ** 2)

    elapsed = time.time() - t0
    return {
        'R': R,
        'voxel_mm_x': voxel_size_mm[0],
        'voxel_mm_y': voxel_size_mm[1],
        'voxel_mm_z': voxel_size_mm[2],
        'voxel_mm_mean': float(voxel_size_mm.mean()),
        'bbox_mm_mean': float(bbox_size_mm.mean()),
        'n_elements': n_elem,
        'bone_voxels': bone_voxels,
        'unique_voxels_hit': unique_voxels_hit,
        'coverage_hit_over_bone': coverage,
        'collision_rate': collision_rate,
        'avg_elems_per_voxel': avg_elems_per_voxel,
        'max_elems_per_voxel': max_elems_per_voxel,
        'shell_erosions': shell,
        'roundtrip_r2_log': r2_log,
        'roundtrip_mae_log': mae_log,
        'roundtrip_mae_mpa': mae_mpa,
        'roundtrip_rmse_mpa': rmse_mpa,
        'mem_mb_fp16_7ch': mem_mb,
        'time_s': elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default=r"H:\pythonproject\PHD_PROGRAM\GCN_Stress\Data\ElementGraph_Tumor_Train")
    ap.add_argument("--out_dir",
                    default=r"H:\pythonproject\PHD_PROGRAM\GCN_Stress\Results\VoxelResolutionStudy")
    ap.add_argument("--n_cases", type=int, default=3,
                    help="Number of representative cases to analyze")
    ap.add_argument("--resolutions", type=int, nargs='+', default=[64, 128, 256, 512])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob('*.npz'))
    print(f"[INFO] Found {len(files)} cases in {data_dir}")
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(files), size=min(args.n_cases, len(files)), replace=False)
    selected = [files[i] for i in sorted(idx)]
    print(f"[INFO] Selected {len(selected)} cases:")
    for f in selected:
        print(f"  {f.name}")

    all_rows = []
    for f in selected:
        for R in args.resolutions:
            print(f"\n[R={R}] {f.name}")
            row = analyze_one(f, R)
            row['case'] = f.stem
            all_rows.append(row)
            print(f"  voxel={row['voxel_mm_mean']:.3f}mm  bone_vox={row['bone_voxels']:,}  "
                  f"coverage={row['coverage_hit_over_bone']:.3f}  "
                  f"collision={row['collision_rate']:.3f}  "
                  f"shell_erosions={row['shell_erosions']}  "
                  f"r2_log={row['roundtrip_r2_log']:.4f}  "
                  f"mae_mpa={row['roundtrip_mae_mpa']:.4f}  "
                  f"mem={row['mem_mb_fp16_7ch']:.1f}MB  "
                  f"({row['time_s']:.1f}s)")

    # ---- Save CSV ----
    import csv
    csv_path = out_dir / 'voxel_resolution_metrics.csv'
    with open(csv_path, 'w', newline='') as fout:
        w = csv.DictWriter(fout, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[saved] {csv_path}")

    # ---- Aggregate per R across cases ----
    Rs = sorted(set(r['R'] for r in all_rows))
    agg = {}
    for R in Rs:
        subset = [r for r in all_rows if r['R'] == R]
        agg[R] = {k: float(np.mean([r[k] for r in subset]))
                  for k in subset[0] if k not in ('case', 'R')}

    # ---- Summary table ----
    print("\n" + "=" * 100)
    print(f"{'R':>5}  {'voxel(mm)':>10}  {'bone_vox':>10}  "
          f"{'coverage':>9}  {'collision':>10}  {'shell_layer':>12}  "
          f"{'r2_log':>8}  {'mae_MPa':>9}  {'mem_MB':>8}")
    print("=" * 100)
    for R in Rs:
        a = agg[R]
        print(f"{R:>5}  {a['voxel_mm_mean']:>10.4f}  {a['bone_voxels']:>10,.0f}  "
              f"{a['coverage_hit_over_bone']:>9.4f}  {a['collision_rate']:>10.4f}  "
              f"{a['shell_erosions']:>12.1f}  {a['roundtrip_r2_log']:>8.4f}  "
              f"{a['roundtrip_mae_mpa']:>9.5f}  {a['mem_mb_fp16_7ch']:>8.1f}")
    print("=" * 100)

    # ---- Plot 1: geometry fidelity vs R ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    Rs_arr = np.array(Rs)

    ax = axes[0, 0]
    ax.plot(Rs_arr, [agg[R]['voxel_mm_mean'] for R in Rs], 'o-', lw=2)
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xlabel('grid resolution R'); ax.set_ylabel('voxel size (mm)')
    ax.set_title('Physical voxel size'); ax.grid(alpha=0.3)
    ax.set_xticks(Rs); ax.set_xticklabels(Rs)

    ax = axes[0, 1]
    ax.plot(Rs_arr, [agg[R]['coverage_hit_over_bone'] for R in Rs], 'o-', lw=2, color='orange')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('R'); ax.set_ylabel('hit-voxels / bone-voxels')
    ax.set_title('Coverage ratio'); ax.grid(alpha=0.3)
    ax.set_xticks(Rs); ax.set_xticklabels(Rs)

    ax = axes[0, 2]
    ax.plot(Rs_arr, [agg[R]['collision_rate'] for R in Rs], 'o-', lw=2, color='red')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('R'); ax.set_ylabel('collision rate')
    ax.set_title('Element collisions (lower is better)'); ax.grid(alpha=0.3)
    ax.set_xticks(Rs); ax.set_xticklabels(Rs)

    ax = axes[1, 0]
    ax.plot(Rs_arr, [agg[R]['shell_erosions'] for R in Rs], 'o-', lw=2, color='purple')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('R'); ax.set_ylabel('erosion steps')
    ax.set_title('Cortical shell layers'); ax.grid(alpha=0.3)
    ax.set_xticks(Rs); ax.set_xticklabels(Rs)

    ax = axes[1, 1]
    ax.plot(Rs_arr, [agg[R]['roundtrip_r2_log'] for R in Rs], 'o-', lw=2, color='green')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('R'); ax.set_ylabel('round-trip R² (log)')
    ax.set_title('Voxelization fidelity (S3)'); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
    ax.set_xticks(Rs); ax.set_xticklabels(Rs)

    ax = axes[1, 2]
    ax.plot(Rs_arr, [agg[R]['mem_mb_fp16_7ch'] for R in Rs], 'o-', lw=2, color='brown')
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xlabel('R'); ax.set_ylabel('per-sample memory (MB)')
    ax.set_title('Memory: 7-ch fp16 × R³'); ax.grid(alpha=0.3)
    ax.set_xticks(Rs); ax.set_xticklabels(Rs)

    plt.tight_layout()
    plt.savefig(out_dir / 'voxel_resolution_analysis.png', dpi=180)
    print(f"[saved] {out_dir / 'voxel_resolution_analysis.png'}")

    # ---- Plot 2: per-case round-trip R² (to show stability) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    cases = sorted(set(r['case'] for r in all_rows))
    for c in cases:
        rs = [r['R']                 for r in all_rows if r['case'] == c]
        ys = [r['roundtrip_r2_log']  for r in all_rows if r['case'] == c]
        ax.plot(rs, ys, 'o-', lw=1.5, alpha=0.7, label=c.split('_')[-1])
    ax.set_xscale('log', base=2)
    ax.set_xlabel('R'); ax.set_ylabel('round-trip R² (log-domain)')
    ax.set_title('Voxelization fidelity per case')
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower right'); ax.set_ylim(0, 1.02)
    ax.set_xticks(Rs); ax.set_xticklabels(Rs)
    plt.tight_layout()
    plt.savefig(out_dir / 'per_case_roundtrip_r2.png', dpi=180)
    print(f"[saved] {out_dir / 'per_case_roundtrip_r2.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
