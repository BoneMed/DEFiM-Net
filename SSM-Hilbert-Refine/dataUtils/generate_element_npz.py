#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate ElementGraph .npz from INP + node CSV files (no abaqus needed).

Sources:
  1. INP file →node coords + C3D10 connectivity
  2. *_internal_nodes.csv + *_surface_1st_order_nodes.csv →S11,S22,S33, E per node
  3. *_BC.csv, *_FORCE.csv →BC/Load node coordinates

Usage:
  python generate_element_npz.py --mode test
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree


# ================================================================
# Parse INP file
# ================================================================
def parse_inp(inp_path):
    """
    Parse Abaqus INP: extract node coords and C3D10 element connectivity.
    Returns:
      node_labels: (N_nodes,) int
      node_coords: (N_nodes, 3) float
      elem_labels: (N_elems,) int
      connectivity: (N_elems, 10) int (node indices, 0-based)
    """
    node_labels, node_coords = [], []
    elem_labels, elem_conns = [], []

    in_nodes = False
    in_elements = False
    elem_line_buf = []

    with open(inp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('**'):
                continue

            if line.startswith('*NODE'):
                in_nodes = True
                in_elements = False
                continue
            elif line.startswith('*ELEMENT') and 'C3D10' in line:
                in_nodes = False
                in_elements = True
                elem_line_buf = []
                continue
            elif line.startswith('*'):
                # Any other keyword ends current section
                if in_elements and elem_line_buf:
                    # Flush last element
                    nums = [int(x) for x in ','.join(elem_line_buf).split(',') if x.strip()]
                    elem_labels.append(nums[0])
                    elem_conns.append(nums[1:11])
                    elem_line_buf = []
                in_nodes = False
                in_elements = False
                continue

            if in_nodes:
                parts = line.split(',')
                try:
                    nid = int(parts[0])
                except ValueError:
                    in_nodes = False
                    continue
                node_labels.append(nid)
                node_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif in_elements:
                # Check if line is still numeric data
                first_token = line.split(',')[0].strip()
                if not first_token.lstrip('-').isdigit():
                    in_elements = False
                    continue
                elem_line_buf.append(line)
                # Count numbers so far
                all_nums = [x for x in ','.join(elem_line_buf).split(',') if x.strip()]
                if len(all_nums) >= 11:  # elem_id + 10 nodes
                    nums = [int(x) for x in all_nums[:11]]
                    elem_labels.append(nums[0])
                    elem_conns.append(nums[1:11])
                    elem_line_buf = []

    node_labels = np.array(node_labels, dtype=np.int64)
    node_coords = np.array(node_coords, dtype=np.float64)
    elem_labels = np.array(elem_labels, dtype=np.int64)

    # Convert node labels to 0-based indices
    label_to_idx = {int(lbl): i for i, lbl in enumerate(node_labels)}
    connectivity = np.array(
        [[label_to_idx[n] for n in conn] for conn in elem_conns],
        dtype=np.int64)

    print(f"  INP: {len(node_labels)} nodes, {len(elem_labels)} elements (C3D10)")
    return node_labels, node_coords, elem_labels, connectivity


# ================================================================
# Load node CSVs (stress + E)
# ================================================================
def load_node_csv(*csv_paths):
    """Load node CSVs →dict: NodeID →(S11, S22, S33, Mises, E)."""
    node_data = {}
    for p in csv_paths:
        p = Path(p)
        if not p.exists():
            continue
        data = np.loadtxt(p, delimiter=',', skiprows=1)
        for row in data:
            nid = int(row[0])
            # X,Y,Z,S11,S22,S33,Mises,E
            node_data[nid] = row[3:].astype(np.float32)  # S11,S22,S33,Mises,E
    return node_data


# ================================================================
# Compute element quantities
# ================================================================
def compute_element_S3(node_coords, connectivity, node_data, node_labels):
    """
    Element S3 = min principal stress, computed from node stress tensors.
    Use diagonal components only (S11, S22, S33 from CSV).
    S3 = min of averaged corner-node stress eigenvalues.
    """
    N_elem = len(connectivity)
    n_corner = 4  # C3D10: first 4 nodes are corners

    label_to_idx = {int(lbl): i for i, lbl in enumerate(node_labels)}

    # Build node-level stress tensor (diagonal only)
    node_S = np.zeros((len(node_labels), 6), dtype=np.float32)
    node_E_arr = np.zeros(len(node_labels), dtype=np.float32)
    node_has_data = np.zeros(len(node_labels), dtype=bool)

    for nid, vals in node_data.items():
        idx = label_to_idx.get(nid)
        if idx is not None and len(vals) >= 5:
            node_S[idx, 0] = vals[0]  # S11
            node_S[idx, 1] = vals[1]  # S22
            node_S[idx, 2] = vals[2]  # S33
            node_E_arr[idx] = vals[4]  # E
            node_has_data[idx] = True

    # Element stress: average corner nodes, then compute eigenvalues
    corner_conn = connectivity[:, :n_corner]

    # Average stress tensor per element
    elem_S = np.zeros((N_elem, 3, 3), dtype=np.float32)
    for i in range(n_corner):
        nidx = corner_conn[:, i]
        elem_S[:, 0, 0] += node_S[nidx, 0]
        elem_S[:, 1, 1] += node_S[nidx, 1]
        elem_S[:, 2, 2] += node_S[nidx, 2]
    elem_S /= n_corner

    # Min eigenvalue = S3
    eigs = np.linalg.eigvalsh(elem_S)
    S3 = eigs[:, 0].astype(np.float32)  # smallest eigenvalue

    # Element E: average corner node E
    elem_E = node_E_arr[corner_conn].mean(axis=1).astype(np.float32)

    return S3, elem_E


def compute_centroids(node_coords, connectivity):
    n_corner = 4
    return node_coords[connectivity[:, :n_corner]].mean(axis=1).astype(np.float32)


def compute_volume(node_coords, connectivity):
    verts = node_coords[connectivity[:, :4]]
    a = verts[:, 1] - verts[:, 0]
    b = verts[:, 2] - verts[:, 0]
    c = verts[:, 3] - verts[:, 0]
    return (np.abs(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0).astype(np.float32)


def find_bc_load_elements(centroids, bc_coords, load_coords, radius=2.0):
    is_bc = np.zeros(len(centroids), dtype=np.float32)
    is_load = np.zeros(len(centroids), dtype=np.float32)
    tree = cKDTree(centroids)
    if len(bc_coords) > 0:
        for idx_list in tree.query_ball_point(bc_coords, r=radius):
            for idx in idx_list:
                is_bc[idx] = 1.0
    if len(load_coords) > 0:
        for idx_list in tree.query_ball_point(load_coords, r=radius):
            for idx in idx_list:
                is_load[idx] = 1.0
    return is_bc, is_load


def build_topo_edges(connectivity):
    from collections import defaultdict
    n_corner = 4
    conn = connectivity[:, :n_corner]
    face_to_elem = defaultdict(list)
    for ei in range(len(conn)):
        ns = sorted(conn[ei])
        for face in [(ns[0],ns[1],ns[2]), (ns[0],ns[1],ns[3]),
                     (ns[0],ns[2],ns[3]), (ns[1],ns[2],ns[3])]:
            face_to_elem[face].append(ei)
    src, dst = [], []
    for elems in face_to_elem.values():
        if len(elems) == 2:
            src.append(elems[0]); dst.append(elems[1])
            src.append(elems[1]); dst.append(elems[0])
    return np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)


# ================================================================
# Process one sample
# ================================================================
def process_one_sample(inp_path, sample_dir, e_csv_path, out_path):
    inp_path = Path(inp_path)
    sample_dir = Path(sample_dir)

    # 1. Parse INP
    node_labels, node_coords, elem_labels, connectivity = parse_inp(inp_path)

    # 2. Load node stress/E from CSVs
    internal_csv = list(sample_dir.glob("*_internal_nodes.csv"))
    surface_csv = list(sample_dir.glob("*_surface_1st_order_nodes.csv"))
    csv_paths = []
    if internal_csv: csv_paths.append(internal_csv[0])
    if surface_csv: csv_paths.append(surface_csv[0])
    node_data = load_node_csv(*csv_paths)
    print(f"  Node CSVs: {len(node_data)} nodes with stress/E")

    # 3. Compute element quantities
    S3, E_elem = compute_element_S3(node_coords, connectivity, node_data, node_labels)
    centroids = compute_centroids(node_coords, connectivity)
    volume = compute_volume(node_coords, connectivity)
    N = len(elem_labels)
    print(f"  Elements: {N}, S3: [{S3.min():.2f}, {S3.max():.2f}] MPa")

    # 4. Override E from element csv if available
    if e_csv_path and Path(e_csv_path).exists():
        e_data = np.loadtxt(e_csv_path, delimiter=',', skiprows=1)
        eid_to_E = dict(zip(e_data[:, 0].astype(int), e_data[:, 1]))
        for i, eid in enumerate(elem_labels):
            if int(eid) in eid_to_E:
                E_elem[i] = eid_to_E[int(eid)]
        print(f"  E from element CSV: {len(eid_to_E)} entries")

    # 5. BC / Load
    bc_csv = list(sample_dir.glob("*_BC.csv"))
    force_csv = list(sample_dir.glob("*_FORCE.csv"))
    bc_coords = np.empty((0, 3), dtype=np.float32)
    load_coords = np.empty((0, 3), dtype=np.float32)
    if bc_csv:
        bc_data = np.loadtxt(bc_csv[0], delimiter=',', skiprows=1)
        bc_coords = bc_data[:, 1:4].astype(np.float32)
    if force_csv:
        force_data = np.loadtxt(force_csv[0], delimiter=',', skiprows=1)
        load_coords = force_data[:, 1:4].astype(np.float32)

    is_bc, is_load = find_bc_load_elements(centroids, bc_coords, load_coords)
    bc_centroid = bc_coords.mean(axis=0).astype(np.float32) if len(bc_coords) > 0 else centroids.mean(axis=0)
    load_centroid = load_coords.mean(axis=0).astype(np.float32) if len(load_coords) > 0 else centroids.mean(axis=0)
    print(f"  BC elems: {int(is_bc.sum())}, Load elems: {int(is_load.sum())}")

    # 6. u_disp
    u_disp = np.array([-0.5], dtype=np.float32)

    # 7. Topo edges
    edge_index = build_topo_edges(connectivity)
    E_log = np.log1p(E_elem)
    if edge_index.shape[1] > 0:
        s, d = edge_index[0], edge_index[1]
        delta = centroids[d] - centroids[s]
        dist = np.linalg.norm(delta, axis=1, keepdims=True).clip(min=1e-6)
        edge_physics = np.hstack([delta/dist, dist, ((E_log[s]+E_log[d])/2).reshape(-1,1)]).astype(np.float32)
    else:
        edge_physics = np.zeros((0, 5), dtype=np.float32)
    print(f"  Topo edges: {edge_index.shape[1]}")

    # 8. Save
    np.savez_compressed(
        out_path,
        centroids=centroids, E=E_elem, S3=S3, elem_ids=elem_labels,
        is_bc=is_bc, is_load=is_load, u_disp=u_disp, volume=volume,
        bc_centroid=bc_centroid, load_centroid=load_centroid,
        edge_index=edge_index, edge_physics=edge_physics,
        near_edge_index=np.zeros((2, 0), dtype=np.int64),
        near_edge_physics=np.zeros((0, 5), dtype=np.float32),
        far_edge_index=np.zeros((2, 0), dtype=np.int64),
        far_edge_physics=np.zeros((0, 5), dtype=np.float32),
    )
    print(f"  Saved: {out_path}")
    return True


# ================================================================
# Main
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="test", choices=["test"])
    ap.add_argument("--inp_dir", default=r"N:\SecondYear\Morphed_Vertebrae_TestSet\WholeVTK2STL\INPFiles")
    ap.add_argument("--odb_root", default=r"N:\SecondYear\Aba_ODB_Set_TEST")
    ap.add_argument("--e_root", default=r"N:\SecondYear\CT_MASK_E_PHANTOMLESS\Test_FEM")
    ap.add_argument("--out_dir", default="../Data/ElementGraph_Test")
    args = ap.parse_args()

    inp_dir = Path(args.inp_dir)
    odb_root = Path(args.odb_root)
    e_root = Path(args.e_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find INP files
    inp_files = sorted(inp_dir.glob("*.inp"))
    print(f"Found {len(inp_files)} INP files")
    print(f"Output: {out_dir}\n")

    n_ok, n_fail = 0, 0
    for inp_path in inp_files:
        name = inp_path.stem  # e.g. L1_502_TP_119
        out_path = out_dir / f"{name}.npz"

        if out_path.exists():
            print(f"[SKIP] {name}")
            n_ok += 1
            continue

        print(f"[{n_ok+n_fail+1}] {name}...")

        # Find corresponding ODB dir (for BC/FORCE csv)
        subj = name.split('_')[1]  # e.g. "502"
        sample_dir = odb_root / subj / name

        # E csv
        tp_num = name.split('_')[-1]
        e_csv = e_root / subj / f"{subj}_element_E_tp_{tp_num}.csv"

        try:
            if process_one_sample(inp_path, sample_dir, e_csv, out_path):
                n_ok += 1
            else:
                n_fail += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
            n_fail += 1

    print(f"\nDone. Success: {n_ok}, Failed: {n_fail}")


if __name__ == "__main__":
    main()
