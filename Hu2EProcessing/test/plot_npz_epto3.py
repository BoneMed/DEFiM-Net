"""
Voxelize EPTO3 from npz and render clipped by X-plane.
"""
import numpy as np
import pyvista as pv
from pathlib import Path

npz_path = Path(r"H:\pythonproject\PHD_PROGRAM\GCN_Stress\Data\ElementGraph_Tumor_Test\MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_10_sp1.npz")
print(f"Loading {npz_path.name}")
d = np.load(npz_path)
centroids = d['centroids'].astype(np.float32)
S3 = d['S3'].astype(np.float32)
print(f"Elements: {len(S3)}, S3 range: [{S3.min():.4f}, {S3.max():.4f}] MPa")

# ---- Voxelize ----
R = 256
pad = 0.02
mins = centroids.min(0); maxs = centroids.max(0)
size = maxs - mins; size[size == 0] = 1.0
bbox_min = (mins - pad * size).astype(np.float32)
bbox_max = (maxs + pad * size).astype(np.float32)
spacing = (bbox_max - bbox_min) / R

# Grid coords for centroids
q = np.clip((centroids - bbox_min) / (bbox_max - bbox_min), 0, 1 - 1e-6)
gi = np.floor(q * R).astype(np.int32)

# Min-splat: take the most negative S3 per voxel
S3_grid = np.zeros((R, R, R), dtype=np.float32)
count_grid = np.zeros((R, R, R), dtype=np.int32)
for (ix, iy, iz), v in zip(gi, S3):
    if count_grid[ix, iy, iz] == 0 or v < S3_grid[ix, iy, iz]:
        S3_grid[ix, iy, iz] = v
    count_grid[ix, iy, iz] += 1

mask = count_grid > 0
print(f"Voxels occupied: {mask.sum()} / {R**3}")

# ---- Build pyvista ImageData ----
grid = pv.ImageData(dimensions=(R+1, R+1, R+1), spacing=spacing, origin=bbox_min)
grid.cell_data['S3'] = S3_grid.ravel(order='F')
grid.cell_data['mask'] = mask.astype(np.float32).ravel(order='F')
solid = grid.threshold(0.5, scalars='mask')

# Clip by X-plane at center, keep right half (invert=True means keep +x side)
center = np.array(solid.center)
clipped = solid.clip(normal='x', origin=center, invert=False)

# Color range
clim = [float(np.percentile(S3[S3 < 0], 2)),
        float(np.percentile(S3[S3 < 0], 98))]
print(f"Color range (P2-P98): {clim}")

# ---- Render ----
pv.global_theme.background = 'white'
pv.global_theme.font.color = 'black'

pl = pv.Plotter(window_size=(1600, 1200), title=f'EPTO3: {npz_path.stem}')
pl.add_mesh(clipped, scalars='S3', cmap='jet', clim=clim, show_edges=False,
            scalar_bar_args={'title': 'S3 (MPa)', 'fmt': '%.4f'})

pl.view_isometric()

save_path = Path(__file__).parent / "npz_epto3.png"
pl.show(screenshot=str(save_path))
print(f"Saved to {save_path}")
