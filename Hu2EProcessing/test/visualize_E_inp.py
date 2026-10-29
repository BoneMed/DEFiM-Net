"""
Visualize INP mesh with element E values using pyvista.
Supports clipping plane for cross-section view.
"""
import numpy as np
import pandas as pd
import pyvista as pv
import meshio

# ---- Paths ----
subj_id = "504"
inp_path = rf"N:\SecondYear\Morphed_Vertebrae_TrainingSet\WholeVTK2STL\INPFiles\L1_{subj_id}_tp_119.inp"
csv_path = rf"N:\SecondYear\CT_MASK_E_PHANTOMLESS\Train_FEM\{subj_id}\{subj_id}_element_E_tp_119.csv"

# ---- Read mesh ----
print(f"Reading INP: {inp_path}")
mesh = meshio.read(inp_path)
points = mesh.points

# ---- Build pyvista mesh ----
cells, cell_types = [], []
for block in mesh.cells:
    for cell in block.data:
        n = len(cell)
        cells.extend([n] + list(cell))
        if block.type in ("tetra", "tetra4"):
            cell_types.append(pv.CellType.TETRA)
        elif block.type in ("tetra10",):
            cell_types.append(pv.CellType.QUADRATIC_TETRA)
        elif block.type in ("hexahedron", "hex8"):
            cell_types.append(pv.CellType.HEXAHEDRON)
        else:
            cell_types.append(pv.CellType.TETRA)

grid = pv.UnstructuredGrid(np.array(cells), np.array(cell_types), points)
print(f"Mesh: {grid.n_points} points, {grid.n_cells} cells")

# ---- Read E CSV and map to mesh ----
df = pd.read_csv(csv_path)
print(f"CSV: {len(df)} elements, E: [{df['E_MPa'].min():.1f}, {df['E_MPa'].max():.1f}]")

# INP element IDs: parse to get ordering
# meshio cells are in file order, INP element IDs may differ
# We need to match by element ID
# For simplicity: assume CSV ElementID corresponds to 1-based INP element ordering
# Build a lookup
eid_to_E = dict(zip(df['ElementID'], df['E_MPa']))

# Parse INP element IDs in order
elem_ids_ordered = []
in_elem = False
with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
    cur_eid = None
    expected_nodes = 10
    node_count = 0
    for line in f:
        line = line.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            low = line.lower()
            if low.startswith("*element") and "output" not in low:
                in_elem = True
                cur_eid = None
                node_count = 0
            else:
                in_elem = False
            continue
        if in_elem:
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if cur_eid is not None and node_count < expected_nodes:
                # continuation line
                node_count += len(parts)
            else:
                # new element
                if cur_eid is not None:
                    elem_ids_ordered.append(cur_eid)
                try:
                    cur_eid = int(float(parts[0]))
                    node_count = len(parts) - 1
                except Exception:
                    cur_eid = None
    if cur_eid is not None:
        elem_ids_ordered.append(cur_eid)

print(f"Parsed {len(elem_ids_ordered)} element IDs from INP")

# Map E to cell order
E_array = np.zeros(grid.n_cells, dtype=np.float32)
for i, eid in enumerate(elem_ids_ordered):
    if i < grid.n_cells and eid in eid_to_E:
        E_array[i] = eid_to_E[eid]

grid.cell_data['E_MPa'] = E_array

# ---- Visualize ----
pv.global_theme.font.size = 14

p = pv.Plotter(title=f"L1 {subj_id} - Element E (MPa) [drag clip plane]")

p.add_mesh_clip_plane(grid, scalars='E_MPa', cmap='jet', clim=[0, 15000],
                      show_edges=False, scalar_bar_args={'title': 'E (MPa)'},
                      normal='y', interaction_event='always',
                      widget_color='white')

p.show()
