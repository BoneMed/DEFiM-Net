"""
Interactive pyvista visualisation of min principal strain for L1_504_tp_22.

- Loads 3 CSVs (internal + surface 1st/2nd order nodes) as a point cloud
- Colours points by Min_Principal_Strain
- Provides interactive clip plane (plane widget)
- Colorbar annotates -0.019 (= 1.9% ultimate compressive strain, Bayraktar)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

BASE = Path(r"I:\Second_Year_Data\Morphed_Vertebrae_TrainingSet\WholeVTK2STL\L1_504_tp_22")
CSVS = [
    BASE / "L1_504_tp_22_internal_nodes.csv",
    BASE / "L1_504_tp_22_surface_1st_order_nodes.csv",
    BASE / "L1_504_tp_22_surface_2nd_order_nodes.csv",
]

STRAIN_THRESHOLD = -0.019   # 1.9% compressive = fracture threshold (Bayraktar 2004)

# ---------- Load ----------
dfs = []
for f in CSVS:
    df = pd.read_csv(f, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

pts = df[['X_coord', 'Y_coord', 'Z_coord']].to_numpy(dtype=np.float32)
strain = df['Min_Principal_Strain'].to_numpy(dtype=np.float32)
print(f"Loaded {len(pts)} nodes")
print(f"  strain range: {strain.min():.3e} .. {strain.max():.3e}")
print(f"  mean: {strain.mean():.3e},  median: {np.median(strain):.3e}")

# ---------- Build PolyData ----------
cloud = pv.PolyData(pts)
cloud['strain'] = strain

# Clip colour range to p1..p99 so extreme outliers don't saturate
vmin = float(np.percentile(strain, 1))     # ~-2.6e-2
vmax = float(np.percentile(strain, 99))    # ~-2.6e-7

# Custom discrete annotations on the colorbar
annotations = {
    STRAIN_THRESHOLD: "ε = -0.019\n(1.9% fracture threshold)",
    vmin:              f"p1 = {vmin:.3f}",
    0.0:               "0",
}

# Use jet-like colormap with better low-end visibility
cmap = "jet_r"   # reversed jet: red = most compressive, blue = least

# ---------- Interactive plot ----------
pv.set_plot_theme("document")
pl = pv.Plotter(window_size=(1400, 950))
pl.set_background("white")

# Interactive clip plane
pl.add_mesh_clip_plane(
    cloud,
    normal='z',                  # initial clip plane
    scalars='strain',
    cmap=cmap,
    clim=(vmin, vmax),
    point_size=3,
    render_points_as_spheres=True,
    show_scalar_bar=True,
    annotations=annotations,
    scalar_bar_args={
        'title':         'Min principal strain',
        'n_labels':      6,
        'label_font_size': 14,
        'title_font_size': 16,
        'fmt':           '%.2e',
        'vertical':      True,
        'position_x':    0.88,
        'position_y':    0.1,
        'width':         0.05,
        'height':        0.75,
        'color':         'black',
    },
)

# Axes and orientation
pl.add_axes(line_width=3, color='black',
            xlabel='X', ylabel='Y', zlabel='Z',
            labels_off=False)
pl.add_text(
    f"L1_504_tp_22  |  {len(pts):,} nodes  |  drag plane to clip",
    font_size=11, color='black', position='upper_left')

pl.view_isometric()
pl.show()
