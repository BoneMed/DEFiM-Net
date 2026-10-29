"""
Side-by-side comparison for the SAME case (MV04_L4_VOL_20, sphere_3 / sp3):

  Left:  tet mesh from ANSYS RST, coloured by node-level EPTO3
         (minimum principal strain)
  Right: voxelized volume from the corresponding NPZ, coloured by the
         same quantity after trilinear splatting onto a 256^3 grid.

Interactive: adjust view, close window, screenshot saved.
"""
import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_fill_holes

import sys
# ---- Sample selection via CLI ----
# Usage:  python plot_element_vs_voxel.py [subject] [sphere_id]
# defaults:
SUBJ   = sys.argv[1] if len(sys.argv) > 1 else "MV04_L3_VOL_10"
SPHERE = int(sys.argv[2]) if len(sys.argv) > 2 else 1

rst_dir = Path(rf"I:\Metastasis_Rob\{SUBJ}\{SUBJ}\sphere_{SPHERE}")
# RST filename looks like MV04_L3_VOL10_SS25.rst  (VOL_N becomes VOLN)
subj_short = SUBJ.replace('_VOL_', '_VOL')
rst_files = list(rst_dir.glob(f"{subj_short}_SS*.rst"))
if not rst_files:
    raise FileNotFoundError(f"No .rst file under {rst_dir}")
RST = rst_files[0]
NPZ = Path(rf"H:\pythonproject\PHD_PROGRAM\GCN_Stress\Data\ElementGraph_Tumor_Train\{SUBJ}_{SUBJ}_sp{SPHERE}.npz")
if not NPZ.exists():
    raise FileNotFoundError(f"Matching NPZ not found: {NPZ}")
print(f"Subject: {SUBJ}, sphere {SPHERE}")
print(f"  RST: {RST}")
print(f"  NPZ: {NPZ.name}")
RNUM  = -1   # substep index: -1 = last (fully loaded), 0 = first


def compute_epto3_from_rst(rst, rnum=-1):
    """Compute nodal minimum principal total strain (EPTO3 = EEL + EPL).

    Replicates the extraction logic from dataUtils/extract_rst_to_npz.py.
    Returns (node_ids, node_epto3) in the order exposed by rst.mesh.nnum.
    """
    from scipy.spatial import cKDTree
    node_coords = rst.mesh.nodes.astype(np.float32)
    node_ids    = rst.mesh.nnum
    n_nodes     = len(node_ids)

    if rnum == -1:
        rnum = rst.nsets - 1

    eel, epl = None, None
    for r in range(rnum, -1, -1):
        try:
            _, eel = rst.nodal_elastic_strain(rnum=r)
            _, epl = rst.nodal_plastic_strain(rnum=r)
            if r != rnum:
                print(f"    [WARN] rnum {rnum} no EEL, fell back to {r}")
            rnum = r
            break
        except Exception as ex:
            print(f"    [WARN] rnum={r} read failed: {type(ex).__name__}: {ex}")
            continue
    if eel is None:
        raise RuntimeError("No valid result set with EEL data")

    both_valid = ~np.isnan(eel[:, 0]) & ~np.isnan(epl[:, 0])
    epto = eel[:, :6] + epl[:, :6]

    node_EPTO3 = np.full(n_nodes, np.nan, dtype=np.float32)
    valid_idx  = np.where(both_valid)[0]
    N = len(valid_idx)
    e = epto[valid_idx]

    # Build 3x3 strain tensors (halve engineering shear components)
    T = np.zeros((N, 3, 3), dtype=np.float64)
    T[:, 0, 0] = e[:, 0]
    T[:, 1, 1] = e[:, 1]
    T[:, 2, 2] = e[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = e[:, 3] / 2
    T[:, 1, 2] = T[:, 2, 1] = e[:, 4] / 2
    T[:, 0, 2] = T[:, 2, 0] = e[:, 5] / 2

    batch = 50000
    for i in range(0, N, batch):
        eigs = np.linalg.eigvalsh(T[i:i + batch])
        node_EPTO3[valid_idx[i:i + batch]] = eigs[:, 0].astype(np.float32)

    # Fill NaN (contact/target nodes) via nearest neighbour
    valid_mask = ~np.isnan(node_EPTO3)
    if not valid_mask.all() and valid_mask.any():
        tree = cKDTree(node_coords[valid_mask])
        _, idx = tree.query(node_coords[~valid_mask])
        node_EPTO3[~valid_mask] = node_EPTO3[valid_mask][idx]
    print(f"    EPTO3: {valid_mask.sum()}/{n_nodes} valid nodes (rnum={rnum})")
    return node_ids, node_EPTO3

R_GRID = 256
PAD = 0.01

# ---- Load RST mesh ----
from ansys.mapdl.reader import read_binary
print(f"Reading RST: {RST.name}")
rst = read_binary(str(RST))
ugrid = rst.grid  # pyvista UnstructuredGrid (on the rst object, not rst.mesh)
print(f"  pv UnstructuredGrid: {ugrid.n_points} nodes, {ugrid.n_cells} cells")

# ---- Compute EPTO3 = EEL + EPL, then principal-min eigvals ----
print("Computing EPTO3 from RST (EEL + EPL, eigvalsh of strain tensor)")
node_ids_rst, epto3 = compute_epto3_from_rst(rst, rnum=RNUM)

# Align to the pyvista mesh node order (ugrid exposes ansys_node_num)
ids_in_vtk = ugrid.point_data.get('ansys_node_num', None)
if ids_in_vtk is None:
    strain = epto3
else:
    ids_in_vtk = np.asarray(ids_in_vtk, dtype=np.int64)
    id_to_val = {int(n): v for n, v in zip(node_ids_rst, epto3)}
    strain = np.array([id_to_val.get(int(n), 0.0) for n in ids_in_vtk],
                      dtype=np.float32)
ugrid.point_data['strain'] = strain
print(f"  strain on mesh: range [{strain.min():.3e}, {strain.max():.3e}]")

# ---- Load NPZ, voxelize for RIGHT panel ----
print(f"Reading NPZ: {NPZ.name}")
npz = np.load(NPZ)
centroids = npz['centroids'].astype(np.float32)
# Prefer EPTO3 if present; else use S3 field
if 'EPTO3' in npz.files:
    S = npz['EPTO3'].astype(np.float32)
elif 'S3' in npz.files:
    S = npz['S3'].astype(np.float32)
else:
    raise KeyError("NPZ missing EPTO3 / S3")
E_elem = npz['E'].astype(np.float32) if 'E' in npz.files else None
print(f"  strain range: [{S.min():.3e}, {S.max():.3e}]")
if E_elem is not None:
    print(f"  E (material) range: [{E_elem.min():.1f}, {E_elem.max():.1f}] MPa")

mins = centroids.min(axis=0); maxs = centroids.max(axis=0)
size = maxs - mins
bbox_min = (mins - PAD * size).astype(np.float32)
bbox_max = (maxs + PAD * size).astype(np.float32)
scale = bbox_max - bbox_min; scale[scale == 0] = 1.0
q = np.clip((centroids - bbox_min) / scale, 0, 1 - 1e-6)
pts_grid = q * (R_GRID - 1)

i0 = np.floor(pts_grid).astype(np.int32); frac = pts_grid - i0

def trilinear_splat(values):
    acc  = np.zeros((R_GRID, R_GRID, R_GRID), dtype=np.float32)
    wsum = np.zeros_like(acc)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                iz = np.clip(i0[:, 0] + dz, 0, R_GRID - 1)
                iy = np.clip(i0[:, 1] + dy, 0, R_GRID - 1)
                ix = np.clip(i0[:, 2] + dx, 0, R_GRID - 1)
                wz = (frac[:, 0] if dz else 1 - frac[:, 0])
                wy = (frac[:, 1] if dy else 1 - frac[:, 1])
                wx = (frac[:, 2] if dx else 1 - frac[:, 2])
                w = wz * wy * wx
                np.add.at(acc,  (iz, iy, ix), values * w)
                np.add.at(wsum, (iz, iy, ix), w)
    out = np.zeros_like(acc)
    valid = wsum > 1e-8
    out[valid] = acc[valid] / wsum[valid]
    return out, valid

voxel_S, valid = trilinear_splat(S)
voxel_E = trilinear_splat(E_elem)[0] if E_elem is not None else None

occ = valid.astype(np.float32)
dist = distance_transform_edt(~(occ > 0.5))
dilated = binary_dilation(occ > 0.5, iterations=2)
mask = (dilated & (dist <= 2))
mask = binary_fill_holes(mask).astype(np.float32)
_, inds = distance_transform_edt(~valid, return_indices=True)

def mask_fill(vol):
    filled = vol.copy()
    need = (mask > 0.5) & (~valid)
    filled[need] = vol[tuple(inds)][need]
    return filled * mask
voxel_S = mask_fill(voxel_S)
if voxel_E is not None:
    voxel_E = mask_fill(voxel_E)
print(f"  voxel grid {R_GRID}^3, bone voxels: {int(mask.sum())}")

# ---- Assign E to tet mesh: nearest centroid match ----
if E_elem is not None:
    from scipy.spatial import cKDTree
    tree_c = cKDTree(centroids)
    # use first point of each cell as proxy — better: cell centroid
    try:
        cell_centers = ugrid.cell_centers().points.astype(np.float32)
    except Exception:
        cell_centers = ugrid.points[ugrid.cells.reshape(-1, 5)[:, 1:5]].mean(axis=1).astype(np.float32)
    _, nn = tree_c.query(cell_centers)
    ugrid.cell_data['E'] = E_elem[nn].astype(np.float32)
    print(f"  cell E on mesh: [{ugrid.cell_data['E'].min():.1f}, {ugrid.cell_data['E'].max():.1f}] MPa")

# ---- Shared colour range from the NPZ strain (consistent across views) ----
clim = [float(np.percentile(S, 2)), float(np.percentile(S, 98))]

# ---- Plot: 3 panels ----
pv.global_theme.background = 'white'
pv.global_theme.font.color = 'black'
pl = pv.Plotter(shape=(1, 3), window_size=(2700, 1000),
                title='FE mesh | Voxelized | Patches  (MV04_L4_VOL_20_sp3)')

spacing = (bbox_max - bbox_min) / R_GRID
grid = pv.ImageData(dimensions=(R_GRID+1, R_GRID+1, R_GRID+1),
                    spacing=spacing, origin=bbox_min)
grid.cell_data['strain'] = voxel_S.ravel(order='F')
grid.cell_data['mask']   = mask.ravel(order='F')
if voxel_E is not None:
    grid.cell_data['E'] = voxel_E.ravel(order='F')
solid = grid.threshold(0.5, scalars='mask')

# ---- Shared clip box for all 3 panels (union of 3 patch boxes) ----
P = 128
stride = 64
overlap_pct = (P - stride) / P * 100

def grid_to_world(gx, gy, gz):
    return (bbox_min[0] + gx * spacing[0],
            bbox_min[1] + gy * spacing[1],
            bbox_min[2] + gz * spacing[2])

patches = [
    {'start': (0,   0,   64), 'color': '#3498DB', 'name': 'Patch 1'},
    {'start': (64,  0,   64), 'color': '#E74C3C', 'name': 'Patch 2'},
    {'start': (128, 0,   64), 'color': '#2ECC71', 'name': 'Patch 3'},
]
patch_union_bounds = (
    grid_to_world(patches[0]['start'][0], 0, 0)[0],                 # x_min
    grid_to_world(patches[-1]['start'][0] + P, 0, 0)[0],            # x_max
    grid_to_world(0, patches[0]['start'][1], 0)[1],                 # y_min
    grid_to_world(0, patches[0]['start'][1] + P, 0)[1],             # y_max
    grid_to_world(0, 0, patches[0]['start'][2])[2],                 # z_min
    grid_to_world(0, 0, patches[0]['start'][2] + P)[2],             # z_max
)
print(f"  Shared clip bounds: {[f'{b:.1f}' for b in patch_union_bounds]}")

# ---- Pre-compute clips we reuse across fields ----
ugrid_clip = ugrid.clip_box(patch_union_bounds, invert=False)
solid_clip = solid.clip_box(patch_union_bounds, invert=False)
surface = ugrid.extract_surface()

patch_clips = []
for patch in patches:
    sx, sy, sz = patch['start']
    bounds = (*grid_to_world(sx, sy, sz),
              *grid_to_world(sx + P, sy + P, sz + P))
    bounds = (bounds[0], bounds[3], bounds[1], bounds[4], bounds[2], bounds[5])
    patch_clips.append({'bounds': bounds,
                        'clip': solid.clip_box(bounds, invert=False),
                        'color': patch['color']})

# Scalar field settings
FIELDS = {
    'strain':   dict(cmap='jet',    clim=clim,
                      scalar_bar_title='Min principal strain',
                      fmt='%.3e'),
    'E':        dict(cmap='viridis', clim=[float(np.percentile(E_elem, 2)),
                                           float(np.percentile(E_elem, 98))]
                                    if E_elem is not None else [0, 1],
                      scalar_bar_title="Young's modulus E (MPa)",
                      fmt='%.0f'),
}

# Track actors per field so we can toggle visibility
field_actors = {'strain': [], 'E': []}

def add_solid(subplot_idx, mesh_obj, field):
    """Add a solid mesh with the given scalar field, hidden unless active."""
    cfg = FIELDS[field]
    pl.subplot(0, subplot_idx)
    kwargs = dict(scalars=field, cmap=cfg['cmap'], clim=cfg['clim'],
                  show_scalar_bar=False, show_edges=False)
    if subplot_idx == 0:
        kwargs.update(ambient=0.4, diffuse=0.9, specular=0.1, smooth_shading=True)
    return pl.add_mesh(mesh_obj, **kwargs)

# =========== Panel 1: FE tetrahedral mesh ===========
pl.subplot(0, 0)
pl.add_mesh(surface, color='indianred', opacity=0.15,
            style='wireframe', line_width=0.5, show_scalar_bar=False)
for field in FIELDS:
    if ugrid_clip.n_cells > 0 and field in ugrid.cell_data.dtype.names if False else True:
        pass
    actor = add_solid(0, ugrid_clip, field)
    field_actors[field].append(actor)
pl.add_text(f'FE tetrahedral mesh\n({ugrid.n_cells:,} elements, {ugrid.n_points:,} nodes)',
            font_size=14, color='black', position='upper_left')
pl.add_axes()

# =========== Panel 2: Voxelized volume ===========
pl.subplot(0, 1)
pl.add_mesh(surface, color='indianred', opacity=0.15,
            style='wireframe', line_width=0.5, show_scalar_bar=False)
for field in FIELDS:
    actor = add_solid(1, solid_clip, field)
    field_actors[field].append(actor)
pl.add_text(f'Voxelized ({R_GRID}$^3$)\n({int(mask.sum()):,} bone voxels)',
            font_size=14, color='black', position='upper_left')
pl.add_axes()

# =========== Panel 3: Patch tiling ===========
pl.subplot(0, 2)
pl.add_mesh(surface, color='indianred', opacity=0.15,
            style='wireframe', line_width=0.5, show_scalar_bar=False)
for pc in patch_clips:
    if pc['clip'].n_cells > 0:
        for field in FIELDS:
            actor = add_solid(2, pc['clip'], field)
            field_actors[field].append(actor)
    box = pv.Box(bounds=pc['bounds'])
    pl.add_mesh(box, color=pc['color'], style='wireframe',
                line_width=5, opacity=1.0)

# Initially show strain; hide material
for a in field_actors['E']:
    a.visibility = False

# Shared scalar bar (redrawn on toggle)
scalar_bar_actor = [None]

def _refresh_scalar_bar(field):
    if scalar_bar_actor[0] is not None:
        pl.remove_scalar_bar()
    cfg = FIELDS[field]
    # Use one of the visible actors as the mapper source
    visible = field_actors[field][0]
    pl.add_scalar_bar(title=cfg['scalar_bar_title'], fmt=cfg['fmt'],
                      n_labels=5, vertical=True, mapper=visible.mapper)
    scalar_bar_actor[0] = 'set'

_refresh_scalar_bar('strain')

# ---- Toggle button: strain <-> material ----
current = ['strain']
def toggle_field(flag_is_material):
    new = 'E' if flag_is_material else 'strain'
    for a in field_actors[current[0]]:
        a.visibility = False
    for a in field_actors[new]:
        a.visibility = True
    current[0] = new
    _refresh_scalar_bar(new)
    pl.render()

pl.add_checkbox_button_widget(toggle_field, value=False,
                              position=(10, 10), size=30,
                              color_on='#2ECC71', color_off='#3498DB',
                              background_color='white')
pl.add_text("strain / material (click)", position=(50, 15),
            font_size=11, color='black')

pl.add_text(f'Patch tiling\n$P={P}^3$, $s={stride}$, overlap {overlap_pct:.0f}\\%',
            font_size=14, color='black', position='upper_left')
pl.add_axes()

pl.link_views()
pl.view_isometric()

save_path = Path(__file__).parent / "element_vs_voxel.png"
pl.show(screenshot=str(save_path))
print(f"Saved: {save_path}")
