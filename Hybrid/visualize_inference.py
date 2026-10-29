"""
Interactive pyvista 3-panel visualization for inference result.
GT / Pred / Error with clip plane.
"""
import numpy as np
import pyvista as pv
from pathlib import Path
import sys

def inv_signed_sqrt(y):
    return np.sign(y) * (y ** 2) / 10.0

# Allow override from argv
if len(sys.argv) > 1:
    sample_dir = Path(sys.argv[1])
else:
    sample_dir = Path(r"H:\pythonproject\PHD_PROGRAM\GCN_Stress\Results\Tumor_directly_ep125\MV04_L2_VOL_10_MV04_L2_VOL_10_sp18")

name = sample_dir.name
print(f"Loading {name}")

d = np.load(sample_dir / 'eval_data.npz')
mask = d['mask'] > 0.5
bbox_min = d['bbox_min']
bbox_max = d['bbox_max']
pred_mpa = np.where(mask, inv_signed_sqrt(d['pred_raw']), 0).astype(np.float32)
gt_mpa = np.where(mask, inv_signed_sqrt(d['gt_raw']), 0).astype(np.float32)
R = mask.shape[0]

valid = mask
clim = [float(np.percentile(gt_mpa[valid], 2)),
        float(np.percentile(gt_mpa[valid], 98))]
eclim = [0, float(np.percentile(np.abs(pred_mpa - gt_mpa)[valid], 95))]
print(f"  GT range: [{clim[0]:.4f}, {clim[1]:.4f}]")
print(f"  Error range: [{eclim[0]:.4f}, {eclim[1]:.4f}]")

spacing = (bbox_max - bbox_min) / R
grid = pv.ImageData(dimensions=(R+1, R+1, R+1), spacing=spacing, origin=bbox_min)
grid.cell_data['GT'] = gt_mpa.ravel(order='F')
grid.cell_data['Pred'] = pred_mpa.ravel(order='F')
grid.cell_data['Error'] = np.abs(pred_mpa - gt_mpa).ravel(order='F')
grid.cell_data['mask'] = mask.astype(np.float32).ravel(order='F')

solid = grid.threshold(0.5, scalars='mask')

pl = pv.Plotter(shape=(1, 3), window_size=(2400, 900), title=name)

pl.subplot(0, 0)
pl.add_mesh_clip_plane(solid, scalars='GT', cmap='jet', clim=clim,
                       normal='y', show_edges=False,
                       scalar_bar_args={'title': 'GT (MPa)', 'fmt': '%.4f'})
pl.add_text('GT', font_size=16)

pl.subplot(0, 1)
pl.add_mesh_clip_plane(solid, scalars='Pred', cmap='jet', clim=clim,
                       normal='y', show_edges=False,
                       scalar_bar_args={'title': 'Pred (MPa)', 'fmt': '%.4f'})
pl.add_text('Pred', font_size=16)

pl.subplot(0, 2)
pl.add_mesh_clip_plane(solid, scalars='Error', cmap='hot', clim=eclim,
                       normal='y', show_edges=False,
                       scalar_bar_args={'title': '|Error| (MPa)', 'fmt': '%.4f'})
pl.add_text('|Error|', font_size=16)

pl.link_views()
pl.show()
