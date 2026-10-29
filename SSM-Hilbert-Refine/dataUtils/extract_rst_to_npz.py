#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract ElementGraph .npz from ANSYS RST files (metastasis data).

From each RST:
  - Node coords + element connectivity (from rst.grid)
  - EPTO3 (3rd principal total strain) from EPTO.csv in same directory
  - E Young's modulus (rst.materials per element)
  - BC/Load from displacement boundary conditions
  - u_disp from Failure_Load.csv

Auto split: 90% train, 10% test (random by RST file).

Output:
  Data/ElementGraph_Tumor_Train/*.npz
  Data/ElementGraph_Tumor_Test/*.npz

Usage:
  C:/Users/Administrator/anaconda3/envs/pyansys/python.exe extract_rst_to_npz.py
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import json
import time


def extract_one_rst(rst_path, out_path, rnum=-1):
    """
    Extract one RST → ElementGraph npz.

    Args:
        rst_path: path to .rst file
        out_path: output .npz path
        rnum: result set number (-1 = last)
    """
    from ansys.mapdl.reader import read_binary

    rst = read_binary(str(rst_path))
    grid = rst.grid

    n_nodes = grid.n_points
    n_cells = grid.n_cells
    print(f"    Nodes: {n_nodes}, Elements: {n_cells}")

    # ---- Node coordinates ----
    node_coords = np.array(grid.points, dtype=np.float64)

    # ---- Element connectivity (from VTK cells) ----
    # Extract cell connectivity from pyvista grid
    cells = grid.cells
    cell_types = grid.celltypes
    offsets = grid.offset

    # Build element connectivity array
    # For C3D10 (VTK_QUADRATIC_TETRA=24): 10 nodes per element
    connectivity_list = []
    elem_labels = []
    pos = 0
    for ei in range(n_cells):
        n_per = cells[pos]
        conn = cells[pos + 1: pos + 1 + n_per]
        connectivity_list.append(conn)
        elem_labels.append(ei)
        pos += n_per + 1

    # Pad to uniform width (max nodes per element)
    max_npe = max(len(c) for c in connectivity_list)
    connectivity = np.zeros((n_cells, max_npe), dtype=np.int64)
    for i, c in enumerate(connectivity_list):
        connectivity[i, :len(c)] = c
    elem_labels = np.array(elem_labels, dtype=np.int64)

    # ---- Centroids (first 4 corner nodes for tet) ----
    n_corner = min(4, max_npe)
    centroids = node_coords[connectivity[:, :n_corner]].mean(axis=1).astype(np.float32)

    # ---- Volume ----
    if max_npe >= 4:
        verts = node_coords[connectivity[:, :4]]
        a = verts[:, 1] - verts[:, 0]
        b = verts[:, 2] - verts[:, 0]
        c = verts[:, 3] - verts[:, 0]
        volume = (np.abs(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0).astype(np.float32)
    else:
        volume = np.ones(n_cells, dtype=np.float32)

    # ---- EPTO3 (3rd principal total strain) from RST: EEL + EPL ----
    if rnum == -1:
        rnum = rst.nsets - 1

    # 有些 result set 没有 element solution 或 header 损坏，往前找
    eel, epl = None, None
    for r in range(rnum, -1, -1):
        try:
            _, eel = rst.nodal_elastic_strain(rnum=r)
            _, epl = rst.nodal_plastic_strain(rnum=r)
            if r != rnum:
                print(f"    [WARN] rnum {rnum} no EEL, using rnum={r}")
            rnum = r
            break
        except Exception as ex:
            print(f"    [WARN] rnum={r} EEL read failed: {type(ex).__name__}: {ex}")
            continue
    if eel is None:
        print(f"    [SKIP] No valid result set with EEL data")
        return False

    both_valid = ~np.isnan(eel[:, 0]) & ~np.isnan(epl[:, 0])
    epto = eel[:, :6] + epl[:, :6]

    node_EPTO3 = np.full(n_nodes, np.nan, dtype=np.float32)
    valid_idx = np.where(both_valid)[0]
    N = len(valid_idx)
    e = epto[valid_idx]
    tensors = np.zeros((N, 3, 3), dtype=np.float64)
    tensors[:, 0, 0] = e[:, 0]
    tensors[:, 1, 1] = e[:, 1]
    tensors[:, 2, 2] = e[:, 2]
    tensors[:, 0, 1] = tensors[:, 1, 0] = e[:, 3] / 2
    tensors[:, 1, 2] = tensors[:, 2, 1] = e[:, 4] / 2
    tensors[:, 0, 2] = tensors[:, 2, 0] = e[:, 5] / 2

    batch = 50000
    for i in range(0, N, batch):
        eigs = np.linalg.eigvalsh(tensors[i:i+batch])
        node_EPTO3[valid_idx[i:i+batch]] = eigs[:, 0].astype(np.float32)

    # Interpolate NaN (contact/target nodes)
    valid_mask = ~np.isnan(node_EPTO3)
    if not valid_mask.all() and valid_mask.any():
        tree = cKDTree(node_coords[valid_mask])
        _, idx = tree.query(node_coords[~valid_mask])
        node_EPTO3[~valid_mask] = node_EPTO3[valid_mask][idx]
    print(f"    EPTO3: {valid_mask.sum()}/{n_nodes} valid nodes")

    # Element EPTO3: average corner nodes
    elem_EPTO3 = node_EPTO3[connectivity[:, :n_corner]].mean(axis=1).astype(np.float32)

    # ---- E (Young's modulus per element) ----
    try:
        materials = rst.materials
        mat_ids = rst.mesh.material_type
        E_elem = np.array([materials[m]['EX'] for m in mat_ids], dtype=np.float32)
    except Exception as e:
        print(f"    [WARN] Cannot read E: {e}, using default")
        E_elem = np.full(n_cells, 1000.0, dtype=np.float32)

    # ---- BC / Load (从 RST 直接读 D 命令记录) ----
    # APDL 的 D 命令同时记录固定约束和位移加载:
    #   D, node, UX, 0     -> 固定 (BC)
    #   D, node, UZ, -0.5  -> 指定位移 (Load)
    # nodal_boundary_conditions 返回 (nnum, dof, value)，
    #   value == 0  -> BC
    #   value != 0  -> Load (= u_disp)
    bc_coords = np.empty((0, 3), dtype=np.float32)
    load_coords = np.empty((0, 3), dtype=np.float32)
    u_disp_val = -0.5  # fallback
    try:
        # rnum=0 即可，BC 在所有 substep 里都一样定义
        bc_data = rst.nodal_boundary_conditions(0)
        bc_nnum, bc_val = bc_data[0], bc_data[2]
        nnum_all = rst.mesh.nnum
        nnum_to_idx = {int(n): i for i, n in enumerate(nnum_all)}

        fixed_mask = np.abs(bc_val) < 1e-9
        loaded_mask = ~fixed_mask

        fixed_nodes = np.unique(bc_nnum[fixed_mask])
        loaded_nodes = np.unique(bc_nnum[loaded_mask])

        bc_idx = np.array(
            [nnum_to_idx[int(n)] for n in fixed_nodes if int(n) in nnum_to_idx],
            dtype=np.int64)
        load_idx = np.array(
            [nnum_to_idx[int(n)] for n in loaded_nodes if int(n) in nnum_to_idx],
            dtype=np.int64)

        if len(bc_idx) > 0:
            bc_coords = node_coords[bc_idx].astype(np.float32)
        if len(load_idx) > 0:
            load_coords = node_coords[load_idx].astype(np.float32)

        # u_disp = 出现次数最多的非零 D 值
        if loaded_mask.any():
            uniq_vals, counts = np.unique(np.round(bc_val[loaded_mask], 6),
                                           return_counts=True)
            u_disp_val = float(uniq_vals[counts.argmax()])

        print(f"    BC nodes: {len(bc_idx)}, Load nodes: {len(load_idx)}, "
              f"u_disp={u_disp_val:.4f}")
    except Exception as e:
        print(f"    [WARN] Cannot read nodal_boundary_conditions: "
              f"{type(e).__name__}: {e}")

    # 把节点级 BC/Load 映射到 element 级 (one-hot)：
    # 一个 element 的任意一个角节点是 BC/Load -> 该 element 标记为 BC/Load
    is_bc = np.zeros(n_cells, dtype=np.float32)
    is_load = np.zeros(n_cells, dtype=np.float32)

    if len(bc_coords) > 0 or len(load_coords) > 0:
        bc_node_set = set(bc_idx.tolist()) if len(bc_coords) > 0 else set()
        load_node_set = set(load_idx.tolist()) if len(load_coords) > 0 else set()
        for ei in range(n_cells):
            corners = connectivity[ei, :n_corner]
            for cn in corners:
                if cn in bc_node_set:
                    is_bc[ei] = 1.0
                    break
            for cn in corners:
                if cn in load_node_set:
                    is_load[ei] = 1.0
                    break

    bc_centroid = bc_coords.mean(axis=0).astype(np.float32) if len(bc_coords) > 0 else centroids.mean(axis=0)
    load_centroid = load_coords.mean(axis=0).astype(np.float32) if len(load_coords) > 0 else centroids.mean(axis=0)

    print(f"    BC elems: {int(is_bc.sum())}, Load elems: {int(is_load.sum())}")

    u_disp = np.array([u_disp_val], dtype=np.float32)

    # ---- Topological edges ----
    from collections import defaultdict
    face_to_elem = defaultdict(list)
    for ei in range(n_cells):
        ns = sorted(connectivity[ei, :n_corner].tolist())
        for face in [(ns[0],ns[1],ns[2]), (ns[0],ns[1],ns[3]),
                     (ns[0],ns[2],ns[3]), (ns[1],ns[2],ns[3])]:
            face_to_elem[face].append(ei)
    src, dst = [], []
    for elems in face_to_elem.values():
        if len(elems) == 2:
            src.append(elems[0]); dst.append(elems[1])
            src.append(elems[1]); dst.append(elems[0])
    edge_index = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)

    if edge_index.shape[1] > 0:
        s, d = edge_index[0], edge_index[1]
        delta = centroids[d] - centroids[s]
        dist = np.linalg.norm(delta, axis=1, keepdims=True).clip(min=1e-6)
        edge_physics = np.hstack([delta/dist, dist,
                                   ((np.log1p(E_elem[s])+np.log1p(E_elem[d]))/2).reshape(-1,1)
                                   ]).astype(np.float32)
    else:
        edge_physics = np.zeros((0, 5), dtype=np.float32)

    print(f"    Topo edges: {edge_index.shape[1]}")
    print(f"    EPTO3: [{elem_EPTO3.min():.6e}, {elem_EPTO3.max():.6e}]")
    print(f"    E: [{E_elem.min():.1f}, {E_elem.max():.1f}] MPa")

    # ---- Save ----
    np.savez_compressed(
        out_path,
        centroids=centroids, E=E_elem, S3=elem_EPTO3, elem_ids=elem_labels,
        is_bc=is_bc, is_load=is_load,
        u_disp=u_disp, volume=volume,
        bc_centroid=bc_centroid, load_centroid=load_centroid,
        edge_index=edge_index, edge_physics=edge_physics,
        near_edge_index=np.zeros((2, 0), dtype=np.int64),
        near_edge_physics=np.zeros((0, 5), dtype=np.float32),
        far_edge_index=np.zeros((2, 0), dtype=np.int64),
        far_edge_physics=np.zeros((0, 5), dtype=np.float32),
    )
    print(f"    Saved: {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rst_root", default=r"I:\Metastasis_Rob")
    ap.add_argument("--out_train", default=r"H:\pythonproject\PHD_PROGRAM\GCN_Stress\Data\ElementGraph_Tumor_Train")
    ap.add_argument("--out_test", default=r"H:\pythonproject\PHD_PROGRAM\GCN_Stress\Data\ElementGraph_Tumor_Test")
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rst_root = Path(args.rst_root)
    out_train = Path(args.out_train)
    out_test = Path(args.out_test)
    out_train.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    # Find all RST files
    rst_files = sorted(rst_root.rglob("*.rst"))
    print(f"Found {len(rst_files)} RST files")

    # Generate unique names
    rst_info = []
    for f in rst_files:
        # e.g. J:/Metastasis_Rob/MV00/L3/VOL_10/sphere_1/MV00_L3_VOL10_SS25.rst
        parts = f.relative_to(rst_root).parts
        # Build name like: MV00_L3_VOL10_sphere1
        name = '_'.join(parts[:-1]).replace('sphere_', 'sp')
        rst_info.append({'path': f, 'name': name})

    # Random train/test split
    rng = np.random.RandomState(args.seed)
    indices = list(range(len(rst_info)))
    rng.shuffle(indices)
    n_test = max(1, int(len(indices) * args.test_ratio))
    test_indices = set(indices[:n_test])
    train_indices = set(indices[n_test:])

    print(f"Split: {len(train_indices)} train / {len(test_indices)} test")

    # Save split info
    split_info = {
        'train': [rst_info[i]['name'] for i in sorted(train_indices)],
        'test': [rst_info[i]['name'] for i in sorted(test_indices)],
    }
    with open(out_train.parent / 'tumor_split.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    n_ok, n_fail = 0, 0
    t_start = time.time()
    failed_log = out_train.parent / 'failed_samples.txt'

    # Hard skip list: samples whose RST files crash pyansys C extension
    # (SIGSEGV on corrupted result-set headers, cannot be caught from Python).
    HARD_SKIP = {
        'MV00_ALL_VOL_MV00_ALL_VOL_L3_VOL_20_sp28',
    }

    def _record_failure(name, split_tag, rst_path, reason):
        with open(failed_log, 'a', encoding='utf-8') as fh:
            fh.write(f"{name}\t{split_tag}\t{rst_path}\t{reason}\n")

    for i, info in enumerate(rst_info):
        name = info['name']
        is_test = i in test_indices
        out_dir = out_test if is_test else out_train
        out_path = out_dir / f"{name}.npz"
        split_tag = "TEST" if is_test else "TRAIN"

        if name in HARD_SKIP:
            print(f"[HARD-SKIP] {name}  (known segfault sample)", flush=True)
            n_fail += 1
            _record_failure(name, split_tag, info['path'], "hard-skip: segfault")
            continue

        if out_path.exists():
            print(f"[SKIP] {name}", flush=True)
            n_ok += 1
            continue

        print(f"\n[{i+1}/{len(rst_info)}] {name} ({split_tag})", flush=True)
        try:
            ok = extract_one_rst(info['path'], out_path)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                _record_failure(name, split_tag, info['path'], "extract_one_rst returned False")
        except Exception as e:
            print(f"    [ERROR] {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            n_fail += 1
            _record_failure(name, split_tag, info['path'],
                            f"{type(e).__name__}: {e}")

        if (n_ok + n_fail) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (n_ok + n_fail) / elapsed
            remaining = (len(rst_info) - n_ok - n_fail) / max(rate, 0.01)
            print(f"    Progress: {n_ok+n_fail}/{len(rst_info)}, "
                  f"{rate:.1f} samples/min, ETA {remaining:.0f}min", flush=True)

    print(f"\nDone. Success: {n_ok}, Failed: {n_fail}")
    if n_fail > 0:
        print(f"Failed samples logged to: {failed_log}")
    print(f"Train: {out_train} ({len(train_indices)} samples)")
    print(f"Test:  {out_test} ({len(test_indices)} samples)")


if __name__ == "__main__":
    main()
