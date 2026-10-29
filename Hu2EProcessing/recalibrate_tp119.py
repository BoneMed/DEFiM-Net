"""
Phantomless calibration for L1 vertebra.

Step 1: Find L1 centroid slice, define ROI (+-20 slices),
        extract air/fat/muscle reference HU, plot for verification.
Step 2: 3-point HU correction -> rho_QCT -> rho_ash -> rho_app -> E (Morgan)
Step 3: IDW interpolate E onto FEM mesh, output element CSV for abaqus_add_material
"""

import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import nibabel as nib
import meshio
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import (
    binary_closing,
    binary_erosion,
    binary_opening,
    generate_binary_structure,
    label,
)


# ==========================================
# 配置
# ==========================================
CONFIG = {
    'ct_dir':    r"N:\SecondYear\CT_MASK_E\Train_CT",
    'mask_dir':  r"N:\SecondYear\CT_MASK_E\Train_Mask",
    'inp_dir':   r"N:\SecondYear\Morphed_Vertebrae_TrainingSet\WholeVTK2STL\INPFiles",
    'out_root':  r"N:\SecondYear\CT_MASK_E_PHANTOMLESS\Train_FEM",

    'ct_suffix':   '_ct.nii.gz',
    'mask_suffix': '_seg-vert_msk.nii.gz',
    'inp_pattern': 'L1_{id}_tp_{tp}.inp',
    'ref_id': '510',             # reference template subject
    'max_tp': 119,

    'target_label': 20,          # L1 label in mask
    'roi_half_width': 20,        # +-20 slices around L1 centroid
    'idw_k': 4,
    'idw_p': 2.0,

    # 3-point HU correction targets (canonical HU values)
    'hu_targets': {
        "air":    -1000.0,
        "fat":     -100.0,
        "muscle":    40.0,
    },

    # Ash fraction: rho_app = rho_ash / ash_fraction
    # Schileo et al. (2008), typical range 0.55-0.65
    'ash_fraction': 0.6,

    # Keller (1994): E = 10200 * rho_ash^2.01
    'morgan_a': 10200,
    'morgan_b': 2.01,

    'vol_types': {"tetra", "tetra4", "tetra10",
                  "hexahedron", "hex8", "hex20", "wedge", "pyramid"},
}


# ==========================================
# Utils
# ==========================================
def extract_subject_id(filename):
    match = re.search(r'verse(\d+)', filename)
    return match.group(1) if match else None


def robust_mean(x, trim=0.1):
    x = np.asarray(x, dtype=np.float32).ravel()
    n = x.size
    if n == 0:
        return np.nan
    if n < 50:
        return float(np.median(x))
    lo, hi = int(trim * n), int((1 - trim) * n)
    if hi <= lo:
        return float(np.median(x))
    x_part = np.partition(x, (lo, hi - 1))
    return float(np.mean(x_part[lo:hi]))


def get_largest_component(mask):
    labeled, num = label(mask)
    if num <= 1:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == np.argmax(counts)


# ==========================================
# Step 1: L1 ROI -> extract air/fat/muscle
# ==========================================
def get_l1_roi(mask_data, target_label, half_width):
    """Find L1 centroid axial slice, return (z_min, z_max, z_center)."""
    l1_mask = (mask_data == target_label)
    if l1_mask.sum() == 0:
        return None
    coords = np.argwhere(l1_mask)
    z_center = int(np.mean(coords[:, 2]))
    z_min = max(0, z_center - half_width)
    z_max = min(mask_data.shape[2] - 1, z_center + half_width)
    return z_min, z_max, z_center


def extract_refs_in_roi(ct_data, mask_data, z_min, z_max, target_label, spacing):
    """
    Within ROI slab [z_min:z_max+1], extract air/fat/muscle HU.
    Muscle uses erector spinae targeting (posterior to L1, 10-50mm, ±40mm lateral).
    Returns refs dict and 3D masks (within ROI) for plotting.
    """
    roi = ct_data[:, :, z_min:z_max+1].copy()

    # Body mask
    body = roi > -500
    body = binary_closing(body, structure=generate_binary_structure(3, 1))
    body = get_largest_component(body)

    shell = body & (~binary_erosion(body, iterations=8))
    deep = binary_erosion(body, iterations=5)

    # Exclude L1 bone from soft tissue search
    l1_roi = (mask_data[:, :, z_min:z_max+1] == target_label)

    # L1 centroid for erector spinae targeting
    l1_coords = np.argwhere(l1_roi)
    l1_cx = int(np.mean(l1_coords[:, 0]))
    l1_cy = int(np.mean(l1_coords[:, 1]))

    # Distance from L1 centroid in mm
    yy, xx = np.meshgrid(np.arange(roi.shape[1]), np.arange(roi.shape[0]))
    dist_mm = np.sqrt(((xx - l1_cx) * spacing[0])**2 + ((yy - l1_cy) * spacing[1])**2)
    dist_3d = np.broadcast_to(dist_mm[:, :, np.newaxis], roi.shape)

    # Erector spinae zone: posterior to vertebra, 10-50mm, lateral < 40mm
    posterior_3d = np.broadcast_to((yy < l1_cy - 5)[:, :, np.newaxis], roi.shape)
    lateral_mm = np.abs(xx - l1_cx) * spacing[0]
    lateral_3d = np.broadcast_to((lateral_mm < 40)[:, :, np.newaxis], roi.shape)
    erector_zone = (dist_3d > 10) & (dist_3d < 50) & posterior_3d & lateral_3d

    refs = {}
    masks_3d = {}

    # Air: outside body
    air_m = (roi < -800) & (~body)
    if air_m.sum() > 500:
        refs["air"] = robust_mean(roi[air_m])
        masks_3d["air"] = air_m

    # Fat: subcutaneous (shell region)
    fat_m = (-200 < roi) & (roi < -50) & shell & (~l1_roi)
    fat_m = binary_opening(fat_m)
    if fat_m.sum() > 500:
        refs["fat"] = robust_mean(roi[fat_m])
        masks_3d["fat"] = fat_m

    # Muscle: erector spinae (posterior to L1)
    mus_m = (0 < roi) & (roi < 100) & deep & (~l1_roi) & erector_zone
    if mus_m.sum() > 500:
        refs["muscle"] = robust_mean(roi[mus_m])
        masks_3d["muscle"] = mus_m

    return refs, masks_3d, roi


def plot_step1(ct_data, mask_data, z_center, z_min, z_max,
               refs, masks_3d, roi, subj_id):
    """3-panel plot: CT slice | tissue masks | HU histogram."""
    z_local = z_center - z_min
    ct_slice = roi[:, :, z_local]
    l1_slice = (mask_data[:, :, z_center] == CONFIG['target_label'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: CT + L1 contour
    ax = axes[0]
    ax.imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-400, vmax=1500)
    ax.contour(l1_slice.T, levels=[0.5], colors='cyan', linewidths=1.5)
    ax.set_title(f'CT z={z_center} (L1=cyan)')
    ax.axis('off')

    # Panel 2: tissue masks overlay
    ax = axes[1]
    ax.imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-400, vmax=1500)
    ax.contour(l1_slice.T, levels=[0.5], colors='cyan', linewidths=1)
    colors = {"air": "blue", "fat": "yellow", "muscle": "red"}
    handles = []
    for tissue, color in colors.items():
        if tissue in masks_3d:
            m = masks_3d[tissue][:, :, z_local]
            if m.sum() > 0:
                ax.contourf(m.T, levels=[0.5, 1.5], colors=[color],
                            alpha=0.35, origin='lower')
        if tissue in refs:
            handles.append(plt.Line2D([0], [0], color=color, lw=6, alpha=0.5,
                                      label=f'{tissue}: {refs[tissue]:.1f} HU'))
    ax.legend(handles=handles, loc='upper right', fontsize=10)
    ax.set_title('Air / Fat / Muscle masks')
    ax.axis('off')

    # Panel 3: HU histogram
    ax = axes[2]
    flat = roi.flatten()
    flat = flat[(flat > -1200) & (flat < 2000)]
    ax.hist(flat, bins=200, color='gray', alpha=0.7, density=True, label='ROI')
    for tissue, color in colors.items():
        if tissue in refs:
            ax.axvline(refs[tissue], color=color, lw=2, ls='--',
                       label=f'{tissue}={refs[tissue]:.1f}')
    ax.set_xlabel('HU')
    ax.set_ylabel('Density')
    ax.set_title('HU histogram (ROI)')
    ax.legend()
    ax.set_xlim(-1200, 2000)

    plt.suptitle(f'[{subj_id}] Phantomless Step 1 - L1 ROI '
                 f'(z={z_center} +/- {CONFIG["roi_half_width"]})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==========================================
# Step 2: HU correction -> rho_QCT -> rho_ash -> rho_app -> E
# ==========================================
def fit_hu_correction(refs):
    """
    3-point weighted linear fit: HU_std = a * HU_raw + b

    Maps measured HU to standardized HU so that:
      air -> -1000, fat -> -100, muscle -> 40

    Returns (a, b, used_refs) or (None, None, []) if < 2 refs.
    """
    targets = CONFIG['hu_targets']
    # Air anchors the HU scale most strongly (widest lever arm)
    weights = {"air": 3.0, "fat": 1.0, "muscle": 1.5}

    xs, ys, ws, used = [], [], [], []
    for k in ("air", "fat", "muscle"):
        if k in refs and k in targets:
            xs.append(refs[k])
            ys.append(targets[k])
            ws.append(weights[k])
            used.append(k)

    if len(xs) < 2:
        return None, None, used

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    w = np.asarray(ws, dtype=np.float64)

    A = np.column_stack([x, np.ones_like(x)])
    Aw = A * np.sqrt(w[:, None])
    yw = y * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    a, b = float(coef[0]), float(coef[1])

    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None, None, used
    return a, b, used


def hu_std_to_rho_qct(hu_std):
    """
    HU_std -> rho_QCT (g/cm3)

    By definition of HU: water (HU=0) has rho_QCT ~ 0,
    and the scale is approximately linear.
    rho_QCT (g/cm3) = HU_std / 1000
    """
    return np.asarray(hu_std, dtype=np.float64) / 1000.0


def rho_qct_to_ash(rho_qct):
    """
    rho_QCT (g/cm3) -> rho_ash (g/cm3)

    rho_ash = 0.8772 * rho_QCT + 0.07895

    Ref: Kopperdahl, Morgan & Keaveny (2002) J Orthop Res 20(4):801-805
         (Mindways QCT Pro phantom calibration)
    """
    return 0.8772 * np.asarray(rho_qct, dtype=np.float64) + 0.07895


def rho_ash_to_app(rho_ash):
    """
    rho_ash -> rho_app (apparent density)

    rho_app = rho_ash * ash_fraction
    (ash is ~60% of total bone mass, so rho_ash > rho_app)

    Ref: Schileo et al. (2008), ash_fraction ~ 0.6
    """
    return np.asarray(rho_ash, dtype=np.float64) * CONFIG['ash_fraction']


def rho_to_E(rho):
    """
    rho -> E (MPa)

    E = 10200 * rho_ash^2.01

    Ref: Keller (1994), using rho_ash directly
    """
    rho = np.maximum(np.asarray(rho, dtype=np.float64), 0)
    return CONFIG['morgan_a'] * np.power(np.maximum(rho, 1e-10), CONFIG['morgan_b'])


def full_pipeline(hu_raw, cal_a, cal_b):
    """
    Full chain: HU_raw -> HU_std -> rho_QCT -> rho_ash -> rho_app -> E

    Returns (E, rho_app, rho_ash, rho_qct, hu_std) for diagnostics.
    """
    hu_std = cal_a * np.asarray(hu_raw, dtype=np.float64) + cal_b
    rho_qct = hu_std_to_rho_qct(hu_std)
    rho_ash = rho_qct_to_ash(rho_qct)
    rho_ash = np.maximum(rho_ash, 0)  # clamp negative
    rho_app = rho_ash_to_app(rho_ash)
    E = rho_to_E(rho_ash)             # Keller: E = f(rho_ash) directly
    return E, rho_app, rho_ash, rho_qct, hu_std


def plot_step2(refs, cal_a, cal_b, bone_hu, bone_E, subj_id):
    """
    4-panel plot:
      1) HU correction verification (raw vs std)
      2) Calibration chain curve: HU_std -> E
      3) Bone voxel E histogram
      4) Bone voxel E vs HU scatter
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    targets = CONFIG['hu_targets']

    # Panel 1: HU correction verification
    ax = axes[0, 0]
    for k in ("air", "fat", "muscle"):
        if k in refs:
            raw = refs[k]
            std = cal_a * raw + cal_b
            tgt = targets[k]
            ax.plot(raw, std, 'o', markersize=10, label=f'{k}: {raw:.1f} -> {std:.1f}')
            ax.plot(raw, tgt, 'x', markersize=10, color='red')
    hu_range = np.array([-1200, 200])
    ax.plot(hu_range, cal_a * hu_range + cal_b, 'k-', lw=1)
    ax.plot(hu_range, hu_range, '--', color='gray', lw=0.5, label='identity')
    ax.set_xlabel('HU_raw')
    ax.set_ylabel('HU_std')
    ax.set_title(f'HU correction: std = {cal_a:.4f}*raw + {cal_b:.2f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: HU_std -> E curve
    ax = axes[0, 1]
    hu_curve = np.linspace(-200, 2000, 500)
    rho_qct_c = hu_std_to_rho_qct(hu_curve)
    rho_ash_c = np.maximum(rho_qct_to_ash(rho_qct_c), 0)
    E_c = rho_to_E(rho_ash_c)
    ax.plot(hu_curve, E_c, 'b-', lw=2)
    ax.set_xlabel('HU_std')
    ax.set_ylabel('E (MPa)')
    ax.set_title('HU_std -> E mapping')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-200, 2000)

    # Panel 3: Bone E histogram
    ax = axes[1, 0]
    ax.hist(bone_E, bins=100, color='steelblue', alpha=0.8, edgecolor='none')
    ax.axvline(np.median(bone_E), color='red', ls='--', lw=1.5,
               label=f'median={np.median(bone_E):.0f}')
    ax.axvline(np.mean(bone_E), color='orange', ls='--', lw=1.5,
               label=f'mean={np.mean(bone_E):.0f}')
    ax.set_xlabel('E (MPa)')
    ax.set_ylabel('Count')
    ax.set_title(f'L1 bone E distribution [{bone_E.min():.0f}, {bone_E.max():.0f}]')
    ax.legend()

    # Panel 4: E vs HU scatter
    ax = axes[1, 1]
    n = len(bone_hu)
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
        ax.scatter(bone_hu[idx], bone_E[idx], s=1, alpha=0.3, c='steelblue')
    else:
        ax.scatter(bone_hu, bone_E, s=1, alpha=0.3, c='steelblue')
    ax.set_xlabel('HU_raw (bone voxels)')
    ax.set_ylabel('E (MPa)')
    ax.set_title('E vs HU (bone voxels)')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'[{subj_id}] Step 2: Phantomless Calibration Pipeline',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==========================================
# Step 3: Map E onto FEM mesh -> element CSV
# ==========================================
def parse_inp_nodes_and_elements(inp_path):
    """Parse INP file, return (node_ids, elements).
    elements: list of (elem_type, elem_id, [node_ids])
    Handles C3D10 continuation lines (10 nodes split across 2 lines).
    """
    node_ids = []
    elements = []
    cur_section = None

    # Pass 1: Nodes
    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                cur_section = "node" if line.lower().startswith("*node") else None
                continue
            if cur_section == "node":
                parts = line.split(',')
                try:
                    node_ids.append(int(parts[0]))
                except Exception:
                    pass

    # Pass 2: Elements (C3D10 continuation)
    in_elem = False
    cur_type = None
    cur_eid = None
    cur_nodes = []
    EXPECTED = 10

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                if cur_eid is not None and len(cur_nodes) >= EXPECTED:
                    elements.append((cur_type or "tetra10", cur_eid, cur_nodes[:EXPECTED]))
                cur_eid, cur_nodes = None, []
                low = line.lower()
                if low.startswith("*element") and "output" not in low:
                    in_elem = True
                    cur_type = "tetra10"
                    if "type=" in low:
                        t = low.split("type=")[1].split(",")[0].strip()
                        if t.startswith("c3d4"):
                            cur_type = "tetra"
                        elif t.startswith("c3d10"):
                            cur_type = "tetra10"
                        elif t.startswith("c3d8"):
                            cur_type = "hexahedron"
                else:
                    in_elem = False
                continue
            if in_elem:
                parts = [p.strip() for p in line.split(',') if p.strip()]
                if cur_eid is not None and len(cur_nodes) < EXPECTED:
                    for tok in parts:
                        try:
                            cur_nodes.append(int(float(tok)))
                        except Exception:
                            pass
                else:
                    if cur_eid is not None and len(cur_nodes) >= EXPECTED:
                        elements.append((cur_type or "tetra10", cur_eid, cur_nodes[:EXPECTED]))
                    try:
                        cur_eid = int(float(parts[0]))
                        cur_nodes = [int(float(t)) for t in parts[1:]]
                    except Exception:
                        cur_eid, cur_nodes = None, []

    if cur_eid is not None and len(cur_nodes) >= EXPECTED:
        elements.append((cur_type or "tetra10", cur_eid, cur_nodes[:EXPECTED]))

    return node_ids, elements


def idw_interp(src_xyz, src_val, query_xyz, k=4, p=2.0):
    """Inverse distance weighted interpolation."""
    tree = cKDTree(src_xyz)
    d, idx = tree.query(query_xyz, k=k, workers=-1)
    if k == 1:
        return src_val[idx]
    w = 1.0 / np.maximum(d, 1e-12) ** p
    w_sum = np.sum(w, axis=1, keepdims=True)
    return np.sum((w / w_sum) * src_val[idx], axis=1)


def process_step3(subj_id, ct_data, mask_data, affine, cal_a, cal_b):
    """
    Map E onto FEM mesh and output element CSV.

    Output: {out_root}/{subj_id}/{subj_id}_element_E_tp_119.csv
            columns: ElementID, E_MPa
    """
    inp_name = CONFIG['inp_pattern'].format(id=subj_id, tp=119)
    inp_path = os.path.join(CONFIG['inp_dir'], inp_name)
    if not os.path.exists(inp_path):
        print(f"  [SKIP] INP not found: {inp_path}")
        return None

    # 1. Extract L1 bone voxels -> E
    l1_indices = np.where(mask_data == CONFIG['target_label'])
    l1_hu = ct_data[l1_indices]
    voxels = np.vstack(l1_indices).T
    ct_xyz = nib.affines.apply_affine(affine, voxels).astype(np.float32)

    bone_E, _, _, _, _ = full_pipeline(l1_hu, cal_a, cal_b)
    bone_E = bone_E.astype(np.float32)

    print(f"  [Step3] L1 voxels: {len(l1_hu)}, E: [{bone_E.min():.1f}, {bone_E.max():.1f}]")

    # 2. Parse INP
    real_node_ids, elements_raw = parse_inp_nodes_and_elements(inp_path)
    mesh = meshio.read(inp_path)
    points = mesh.points.astype(np.float32)

    # Build element ID map by corner connectivity
    node_id_to_idx = {nid: i for i, nid in enumerate(real_node_ids)}
    tet_id_map = {}
    for etype, eid, nlist in elements_raw:
        corners = nlist[:4]
        try:
            key = tuple(sorted(node_id_to_idx[n] for n in corners))
            tet_id_map[key] = eid
        except KeyError:
            pass

    print(f"  [Step3] INP: {len(real_node_ids)} nodes, {len(elements_raw)} elements")

    # 3. Geometric alignment (flip X/Y + translate centroid)
    points[:, 0] = -points[:, 0]
    points[:, 1] = -points[:, 1]
    trans = np.mean(ct_xyz, axis=0) - np.mean(points, axis=0)
    aligned_points = points + trans

    # 4. IDW interpolation -> node E
    node_E = idw_interp(ct_xyz, bone_E, aligned_points,
                        k=CONFIG['idw_k'], p=CONFIG['idw_p'])

    print(f"  [Step3] Node E: [{node_E.min():.1f}, {node_E.max():.1f}], "
          f"mean={node_E.mean():.1f}, median={np.median(node_E):.1f}")

    # 5. Element E = mean of corner node E -> CSV
    final_ids, final_E = [], []
    for block in mesh.cells:
        if block.type not in CONFIG['vol_types']:
            continue
        conn = block.data
        block_E = np.mean(node_E[conn], axis=1)
        for i, cell_nodes in enumerate(conn):
            key = tuple(sorted(cell_nodes[:4]))
            eid = tet_id_map.get(key, -1)
            if eid != -1:
                final_ids.append(eid)
                final_E.append(block_E[i])

    out_dir = os.path.join(CONFIG['out_root'], subj_id)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{subj_id}_element_E_tp_119.csv")

    df = pd.DataFrame({'ElementID': final_ids, 'E_MPa': final_E})
    df = df.sort_values('ElementID')
    df.to_csv(csv_path, index=False)

    print(f"  [Step3] Saved {len(df)} elements -> {csv_path}")
    print(f"          Element E: [{df['E_MPa'].min():.1f}, {df['E_MPa'].max():.1f}], "
          f"mean={df['E_MPa'].mean():.1f}, median={df['E_MPa'].median():.1f}")

    return {
        'csv_path': csv_path,
        'aligned_points': aligned_points,
        'node_E': node_E,
        'tet_id_map': tet_id_map,
    }


# ==========================================
# Step 4: Interpolate intermediate tp 1~118
# ==========================================
def get_rho_ash_from_E(E):
    """Inverse Keller: E = a * rho^b  ->  rho = (E/a)^(1/b)"""
    E = np.maximum(np.asarray(E, dtype=np.float64), 0)
    return np.power(E / CONFIG['morgan_a'], 1.0 / CONFIG['morgan_b'])


def process_step4(subj_id, ref_data, tgt_data):
    """
    Interpolate E for tp 1~118.

    For each intermediate tp:
      1. Read INP mesh geometry
      2. IDW map ref rho_ash and target rho_ash onto intermediate mesh
      3. Blend: rho = (1-alpha)*rho_ref + alpha*rho_tgt,  alpha = tp/119
      4. E = Keller(rho)
      5. Output element CSV
    """
    out_dir = os.path.join(CONFIG['out_root'], subj_id)
    os.makedirs(out_dir, exist_ok=True)

    # Prepare ref and target KDTrees (centered)
    ref_pts = ref_data['aligned_points']
    tgt_pts = tgt_data['aligned_points']

    ref_rho = get_rho_ash_from_E(ref_data['node_E'])
    tgt_rho = get_rho_ash_from_E(tgt_data['node_E'])

    c_ref = np.mean(ref_pts, axis=0)
    c_tgt = np.mean(tgt_pts, axis=0)

    tree_ref = cKDTree(ref_pts - c_ref)
    tree_tgt = cKDTree(tgt_pts - c_tgt)

    max_tp = CONFIG['max_tp']
    k, p = 8, 2.0
    count = 0

    for tp in range(1, max_tp):
        inp_name = CONFIG['inp_pattern'].format(id=subj_id, tp=tp)
        inp_path = os.path.join(CONFIG['inp_dir'], inp_name)
        if not os.path.exists(inp_path):
            continue

        # Read intermediate mesh
        real_node_ids, elements_raw = parse_inp_nodes_and_elements(inp_path)
        mesh = meshio.read(inp_path)
        pts = mesh.points.astype(np.float32)

        # Flip + center
        pts[:, 0] = -pts[:, 0]
        pts[:, 1] = -pts[:, 1]
        c_tp = np.mean(pts, axis=0)
        query = pts - c_tp

        # IDW from ref
        d_r, idx_r = tree_ref.query(query, k=k, workers=-1)
        w_r = 1.0 / np.maximum(d_r, 1e-12) ** p
        w_r /= np.sum(w_r, axis=1, keepdims=True)
        rho_from_ref = np.sum(w_r * ref_rho[idx_r], axis=1)

        # IDW from target
        d_t, idx_t = tree_tgt.query(query, k=k, workers=-1)
        w_t = 1.0 / np.maximum(d_t, 1e-12) ** p
        w_t /= np.sum(w_t, axis=1, keepdims=True)
        rho_from_tgt = np.sum(w_t * tgt_rho[idx_t], axis=1)

        # Blend
        alpha = tp / max_tp
        rho_blend = (1.0 - alpha) * rho_from_ref + alpha * rho_from_tgt
        node_E = rho_to_E(rho_blend)

        # Element E -> CSV
        node_id_to_idx = {nid: i for i, nid in enumerate(real_node_ids)}
        tet_map = {}
        for etype, eid, nlist in elements_raw:
            corners = nlist[:4]
            try:
                key = tuple(sorted(node_id_to_idx[n] for n in corners))
                tet_map[key] = eid
            except KeyError:
                pass

        eids, evals = [], []
        for block in mesh.cells:
            if block.type not in CONFIG['vol_types']:
                continue
            for cell_nodes in block.data:
                key = tuple(sorted(cell_nodes[:4]))
                eid = tet_map.get(key, -1)
                if eid != -1:
                    eids.append(eid)
                    evals.append(float(np.mean(node_E[cell_nodes])))

        csv_path = os.path.join(out_dir, f"{subj_id}_element_E_tp_{tp}.csv")
        pd.DataFrame({'ElementID': eids, 'E_MPa': evals}) \
          .sort_values('ElementID').to_csv(csv_path, index=False)

        count += 1
        if tp % 20 == 0 or tp == 1:
            E_arr = np.array(evals)
            print(f"  [Step4] tp={tp:3d} alpha={alpha:.3f} | "
                  f"E: [{E_arr.min():.0f}, {E_arr.max():.0f}], "
                  f"mean={E_arr.mean():.0f}, median={np.median(E_arr):.0f}")

    print(f"  [Step4] Done: {count} intermediate CSVs -> {out_dir}")
    return count


# ==========================================
# Run Steps 1-3 for one subject
# ==========================================
def run_steps_123(subj_id, plot=False):
    """Run Steps 1-3 for a single subject. Returns step3 result dict or None."""

    # Find CT + mask
    ct_files = sorted(glob.glob(
        os.path.join(CONFIG['ct_dir'], f"*{CONFIG['ct_suffix']}")))
    ct_path, mask_path = None, None
    for f in ct_files:
        sid = extract_subject_id(os.path.basename(f))
        if sid == subj_id:
            mn = os.path.basename(f).replace(
                CONFIG['ct_suffix'], CONFIG['mask_suffix'])
            mp = os.path.join(CONFIG['mask_dir'], mn)
            if os.path.exists(mp):
                ct_path, mask_path = f, mp
            break

    if ct_path is None:
        print(f"  [SKIP] No CT+mask for {subj_id}")
        return None

    print(f"\n{'='*60}")
    print(f"  Subject: {subj_id}")
    print(f"{'='*60}")
    print(f"  CT:   {ct_path}")
    print(f"  Mask: {mask_path}")

    ct_img = nib.as_closest_canonical(nib.load(ct_path))
    ct_data = ct_img.get_fdata(dtype=np.float32)
    mask_data = np.asarray(nib.as_closest_canonical(nib.load(mask_path)).dataobj)

    # Step 1
    roi_info = get_l1_roi(mask_data, CONFIG['target_label'],
                          CONFIG['roi_half_width'])
    if roi_info is None:
        print(f"  [SKIP] L1 not found in mask")
        return None

    z_min, z_max, z_center = roi_info
    spacing = np.abs(np.diag(ct_img.affine)[:3])
    refs, masks_3d, roi = extract_refs_in_roi(
        ct_data, mask_data, z_min, z_max, CONFIG['target_label'], spacing)

    print(f"  [Step1] ROI z={z_center}+-{CONFIG['roi_half_width']}")
    for k, v in refs.items():
        print(f"    {k:8s}: {v:8.2f} HU")

    if plot:
        plot_step1(ct_data, mask_data, z_center, z_min, z_max,
                   refs, masks_3d, roi, subj_id)

    # Step 2
    cal_a, cal_b, used = fit_hu_correction(refs)
    if cal_a is None:
        print(f"  [SKIP] HU correction failed")
        return None
    print(f"  [Step2] HU_std = {cal_a:.6f} * HU_raw + {cal_b:.3f}  (refs: {used})")

    l1_hu = ct_data[np.where(mask_data == CONFIG['target_label'])]
    bone_E, _, bone_rho_ash, _, _ = full_pipeline(l1_hu, cal_a, cal_b)
    print(f"  [Step2] E: [{bone_E.min():.0f}, {bone_E.max():.0f}], "
          f"mean={bone_E.mean():.0f}, median={np.median(bone_E):.0f}")

    if plot:
        plot_step2(refs, cal_a, cal_b, l1_hu, bone_E, subj_id)

    # Step 3
    result = process_step3(subj_id, ct_data, mask_data,
                           ct_img.affine, cal_a, cal_b)
    return result


# ==========================================
# Discover subjects from INP directory
# ==========================================
def get_all_subject_ids():
    """Scan INP dir for L1_*_tp_119.inp, return sorted list of subject IDs."""
    pattern = os.path.join(CONFIG['inp_dir'], 'L1_*_tp_119.inp')
    ids = []
    for f in glob.glob(pattern):
        # L1_504_tp_119.inp -> 504
        base = os.path.basename(f)
        parts = base.replace('.inp', '').split('_')
        # parts: ['L1', '504', 'tp', '119']
        if len(parts) >= 2:
            ids.append(parts[1])
    return sorted(ids)


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", default=None,
                    help="Single subject ID (default: batch all from INP dir)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--skip-interp", action="store_true",
                    help="Only run Steps 1-3, skip intermediate tp interpolation")
    args = ap.parse_args()

    ref_id = CONFIG['ref_id']

    # Determine subject list
    if args.subj:
        target_ids = [args.subj]
    else:
        all_ids = get_all_subject_ids()
        target_ids = [s for s in all_ids if s != ref_id]
        print(f"Found {len(all_ids)} subjects in INP dir, {len(target_ids)} targets (ref={ref_id})")

    # 1. Run ref once
    print(f"\n>>> Processing REFERENCE: {ref_id}")
    ref_data = run_steps_123(ref_id, plot=False)
    if ref_data is None:
        print(f"Reference {ref_id} failed!")
        sys.exit(1)

    # 2. Loop targets
    t_start = time.time()
    ok, fail, skip = 0, 0, 0

    for i, tgt_id in enumerate(target_ids):
        print(f"\n>>> [{i+1}/{len(target_ids)}] TARGET: {tgt_id}")

        # Check if already done
        out_dir = os.path.join(CONFIG['out_root'], tgt_id)
        last_csv = os.path.join(out_dir, f"{tgt_id}_element_E_tp_118.csv")
        tp119_csv = os.path.join(out_dir, f"{tgt_id}_element_E_tp_119.csv")
        if os.path.exists(last_csv) and os.path.exists(tp119_csv):
            print(f"  [SKIP] Already complete (tp_118 + tp_119 exist)")
            skip += 1
            continue

        try:
            tgt_data = run_steps_123(tgt_id, plot=args.plot)
            if tgt_data is None:
                fail += 1
                continue

            if not args.skip_interp:
                process_step4(tgt_id, ref_data, tgt_data)

            ok += 1
        except Exception as e:
            print(f"  [ERROR] {tgt_id}: {e}")
            import traceback; traceback.print_exc()
            fail += 1

        elapsed = time.time() - t_start
        done = ok + fail
        if done > 0:
            eta = elapsed / done * (len(target_ids) - i - 1)
            print(f"  Progress: {i+1}/{len(target_ids)} | "
                  f"ok={ok} fail={fail} skip={skip} | "
                  f"ETA {eta/60:.0f}min")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Done. ok={ok}, fail={fail}, skip={skip}")
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"{'='*60}")
