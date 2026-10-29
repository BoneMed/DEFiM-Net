"""
Interactive 3D patch overlap visualization using real vertebra.
Shows a few adjacent patches with highlighted overlap region.

Interactive: use mouse to set view, then close the window. The current
view is saved to patch_overlap_real.png (screenshot on close).
"""
import numpy as np
import pyvista as pv
from pathlib import Path

# ---- Parameters ----
R = 256
P = 128
stride = P // 2   # 64, gives 50% overlap, uniform tiling
overlap = (P - stride) / P

# ---- Load case ----
sample_dir = Path(r"H:\pythonproject\PHD_PROGRAM\GCN_Stress\Results\Tumor_directly_ep162\MV04_L2_VOL_10_MV04_L2_VOL_10_sp18")
def inv_signed_sqrt(y):
    return np.sign(y) * (y ** 2) / 10.0

d = np.load(sample_dir / 'eval_data.npz')
mask = d['mask'] > 0.5
bbox_min = d['bbox_min']
bbox_max = d['bbox_max']
# Target field: minimum principal strain (proxy via same inv transform as GT here)
gt_strain = np.where(mask, inv_signed_sqrt(d['gt_raw']), 0).astype(np.float32)

spacing = (bbox_max - bbox_min) / R
print(f"Grid: {R}^3, Patch: {P}^3, Stride: {stride}, Overlap: {overlap*100:.0f}%")

# ---- Scene ----
pv.global_theme.background = 'white'
pv.global_theme.font.color = 'black'

grid = pv.ImageData(dimensions=(R+1, R+1, R+1), spacing=spacing, origin=bbox_min)
grid.cell_data['mask'] = mask.astype(np.float32).ravel(order='F')
grid.cell_data['strain'] = gt_strain.ravel(order='F')
solid = grid.threshold(0.5, scalars='mask')

# Color range (p2..p98 to avoid extreme saturation)
valid = mask
clim = [float(np.percentile(gt_strain[valid], 2)),
        float(np.percentile(gt_strain[valid], 98))]

pl = pv.Plotter(window_size=(1600, 1200), title='Patch Overlap Demo')

# Helper: convert grid coord -> world coord
def grid_to_world(gx, gy, gz):
    return (bbox_min[0] + gx * spacing[0],
            bbox_min[1] + gy * spacing[1],
            bbox_min[2] + gz * spacing[2])

# 6 patches: 2x3 grid on XY plane (at mid-Z)
# X starts: 0, 90, 128   Y starts: 0, 128
# We pick patches along X at 2 Y rows to show overlap in both directions
patches = [
    {'start': (0,   0,   64), 'color': '#3498DB', 'name': 'Patch 1'},
    {'start': (64,  0,   64), 'color': '#E74C3C', 'name': 'Patch 2'},
    {'start': (128, 0,   64), 'color': '#2ECC71', 'name': 'Patch 3'},
]

# Full vertebra (faint) to give spatial context
pl.add_mesh(solid, scalars='strain', cmap='jet', clim=clim,
            opacity=0.15, show_edges=False, show_scalar_bar=False)

# For each patch: clip vertebra to patch bounds + draw wireframe
for patch in patches:
    sx, sy, sz = patch['start']
    x0, y0, z0 = grid_to_world(sx, sy, sz)
    x1, y1, z1 = grid_to_world(sx + P, sy + P, sz + P)
    bounds = (x0, x1, y0, y1, z0, z1)

    patch_vert = solid.clip_box(bounds, invert=False)
    if patch_vert.n_cells > 0:
        pl.add_mesh(patch_vert, scalars='strain', cmap='jet', clim=clim,
                    opacity=1.0, show_edges=False, show_scalar_bar=False)

    box = pv.Box(bounds=bounds)
    pl.add_mesh(box, color=patch['color'], style='wireframe',
                line_width=5, opacity=1.0)

# Single scalar bar
pl.add_scalar_bar(title='Min principal strain', fmt='%.3e',
                  n_labels=5, vertical=True)

# Annotations
info_text = (
    f'Grid: {R}$^3$\n'
    f'Patch: {P}$^3$\n'
    f'Stride: {stride} voxels\n'
    f'Overlap: {overlap*100:.0f}%'
)
pl.add_text(info_text, position='upper_left', font_size=16, color='black')

pl.add_axes()
pl.view_isometric()

# Interactive: pose the camera, close the window when done, PNG saved on close.
save_path = Path(__file__).parent / "patch_overlap_real.png"
pl.show(screenshot=str(save_path))
print(f"Saved to {save_path}")
