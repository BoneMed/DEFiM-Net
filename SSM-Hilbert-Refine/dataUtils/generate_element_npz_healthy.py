#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate ElementGraph .npz for HEALTHY vertebra dataset (no Abaqus needed).

Each newly-discovered case (non-empty ODB dir, not yet present in either
output dir) is processed and routed directly into either the train pool
or the test set per a 9:1 deterministic split.  Subsequent runs only
process NEW cases —already-partitioned npz files stay where they are.
This makes incremental data ingestion safe: download a batch, run this,
ratios accumulate toward 90/10.

Differences vs the tumor / test variant:
  1. Node CSVs contain  NodeID,X,Y,Z,Min_Principal_Strain  (single scalar
     per node —pre-computed minimum principal STRAIN, not stress).
  2. BC / FORCE node sets are parsed directly from the INP file
     (NSET=BC, NSET=FORCE) —no external *_BC.csv / *_FORCE.csv.
  3. Element label written to npz `S3` is min principal STRAIN per element
     (mean of 4 corner-node strains for each C3D10).  Field name kept as
     `S3` so downstream DataLoader / training code is unchanged.

Inputs:
  --inp_dir    N:/SecondYear/Morphed_Vertebrae_TrainingSet/WholeVTK2STL/INPFiles
  --odb_root   N:/SecondYear/Aba_ODB_Set       (non-empty case dirs only)
  --e_root     N:/SecondYear/CT_MASK_E_PHANTOMLESS/Train_FEM
  --train_dir  ../../Data/ElementGraph_Healthy_Train
  --test_dir   ../../Data/ElementGraph_Healthy_Test
  --test_ratio 0.10
  --seed       42

Usage:
  python generate_element_npz_healthy.py
"""

import argparse
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# ================================================================
# INP parsing  (nodes + C3D10 connectivity + NSET BC / NSET FORCE)
# ================================================================
def parse_inp(inp_path):
    """
    Returns:
      node_labels   (N_nodes,) int
      node_coords   (N_nodes, 3) float
      elem_labels   (N_elems,) int
      connectivity  (N_elems, 10) int (node array indices, 0-based)
      nset_bc       set of node labels in NSET=BC
      nset_force    set of node labels in NSET=FORCE
    """
    node_labels, node_coords = [], []
    elem_labels, elem_conns = [], []
    nset_bc, nset_force = [], []

    state = None  # 'nodes' | 'elements' | 'nset_bc' | 'nset_force' | None
    elem_buf = []

    nset_pat = re.compile(r'NSET\s*=\s*([A-Za-z0-9_]+)', re.I)

    with open(inp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('**'):
                continue

            if line.startswith('*'):
                # Flush pending element line buffer when leaving element block.
                if state == 'elements' and elem_buf:
                    nums = [int(x) for x in ','.join(elem_buf).split(',') if x.strip()]
                    if len(nums) >= 11:
                        elem_labels.append(nums[0])
                        elem_conns.append(nums[1:11])
                    elem_buf = []

                up = line.upper()
                if up.startswith('*NODE'):
                    state = 'nodes'; continue
                if up.startswith('*ELEMENT') and 'C3D10' in up:
                    state = 'elements'; continue
                if up.startswith('*NSET'):
                    m = nset_pat.search(up)
                    if m:
                        name = m.group(1).upper()
                        if name == 'BC':       state = 'nset_bc';    continue
                        if name == 'FORCE':    state = 'nset_force'; continue
                state = None
                continue

            if state == 'nodes':
                parts = line.split(',')
                try:
                    nid = int(parts[0])
                except ValueError:
                    state = None; continue
                node_labels.append(nid)
                node_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif state == 'elements':
                first = line.split(',')[0].strip()
                if not first.lstrip('-').isdigit():
                    state = None; continue
                elem_buf.append(line)
                tokens = [x for x in ','.join(elem_buf).split(',') if x.strip()]
                if len(tokens) >= 11:
                    nums = [int(x) for x in tokens[:11]]
                    elem_labels.append(nums[0])
                    elem_conns.append(nums[1:11])
                    elem_buf = []

            elif state in ('nset_bc', 'nset_force'):
                # NSET lines list node labels separated by commas, possibly
                # with trailing commas or generated triples (start,end,step).
                tokens = [t.strip() for t in line.split(',') if t.strip()]
                if not tokens or not tokens[0].lstrip('-').isdigit():
                    state = None; continue
                ints = [int(t) for t in tokens]
                target = nset_bc if state == 'nset_bc' else nset_force
                # Abaqus generated form has 3 items: start,end,step (when
                # the NSET keyword had GENERATE flag —rare here, but
                # safe to handle).
                if len(ints) == 3 and any('GENERATE' in line.upper() for _ in [0]):
                    target.extend(range(ints[0], ints[1] + 1, ints[2]))
                else:
                    target.extend(ints)

    node_labels = np.array(node_labels, dtype=np.int64)
    node_coords = np.array(node_coords, dtype=np.float64)
    elem_labels = np.array(elem_labels, dtype=np.int64)

    label_to_idx = {int(lbl): i for i, lbl in enumerate(node_labels)}
    connectivity = np.array(
        [[label_to_idx[n] for n in conn] for conn in elem_conns],
        dtype=np.int64)

    print(f"  INP: {len(node_labels)} nodes, {len(elem_labels)} C3D10 elements, "
          f"NSET BC={len(set(nset_bc))}, NSET FORCE={len(set(nset_force))}")

    return (node_labels, node_coords, elem_labels, connectivity,
            set(nset_bc), set(nset_force))


# ================================================================
# Strain CSVs
# ================================================================
def load_strain_csv(*csv_paths):
    """
    Load CSVs with columns:  NodeID, X, Y, Z, Min_Principal_Strain.
    Returns dict NodeID -> (strain, X, Y, Z).
    """
    out = {}
    for p in csv_paths:
        p = Path(p)
        if not p.exists():
            continue
        # skiprows=1 to skip header.  Columns delimited by ',' (with spaces).
        try:
            data = np.loadtxt(p, delimiter=',', skiprows=1, dtype=np.float64)
        except Exception as e:
            print(f"  [WARN] cannot read {p.name}: {e}")
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 5:
            print(f"  [WARN] {p.name} has {data.shape[1]} cols (<5), skipping")
            continue
        for row in data:
            nid = int(row[0])
            out[nid] = (float(row[4]), float(row[1]), float(row[2]), float(row[3]))
    return out


def compute_element_strain(node_labels, connectivity, node_strain):
    """
    Element min principal strain = mean of 4 corner-node strains.
    Nodes without strain data are skipped in the mean.
    """
    label_to_idx = {int(lbl): i for i, lbl in enumerate(node_labels)}
    node_str_arr = np.full(len(node_labels), np.nan, dtype=np.float32)
    for nid, (strain, *_) in node_strain.items():
        idx = label_to_idx.get(nid)
        if idx is not None:
            node_str_arr[idx] = strain

    n_corner = 4
    corner = connectivity[:, :n_corner]                # (N, 4)
    corner_str = node_str_arr[corner]                  # (N, 4)
    mask = ~np.isnan(corner_str)
    counts = mask.sum(axis=1).clip(min=1)
    summed = np.where(mask, corner_str, 0.0).sum(axis=1)
    elem_strain = (summed / counts).astype(np.float32)

    # Fill any element with all NaN corners using nearest non-NaN element value
    bad = ~mask.any(axis=1)
    if bad.any() and (~bad).any():
        elem_strain[bad] = float(np.median(elem_strain[~bad]))

    return elem_strain


# ================================================================
# Element geometry
# ================================================================
def compute_centroids(node_coords, connectivity):
    return node_coords[connectivity[:, :4]].mean(axis=1).astype(np.float32)


def compute_volume(node_coords, connectivity):
    verts = node_coords[connectivity[:, :4]]
    a = verts[:, 1] - verts[:, 0]
    b = verts[:, 2] - verts[:, 0]
    c = verts[:, 3] - verts[:, 0]
    return (np.abs(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0).astype(np.float32)


def nset_to_element_mask(nset_labels, label_to_idx, connectivity):
    """
    is_X[e] = 1 if any of the element's 10 connectivity nodes is in nset.
    """
    mask = np.zeros(len(connectivity), dtype=np.float32)
    if not nset_labels:
        return mask
    nset_idx = {label_to_idx[n] for n in nset_labels if n in label_to_idx}
    if not nset_idx:
        return mask
    for ei, conn in enumerate(connectivity):
        if any(int(n) in nset_idx for n in conn):
            mask[ei] = 1.0
    return mask


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
# Process one healthy sample
# ================================================================
def process_one_sample(inp_path, sample_dir, e_csv_path, out_path):
    inp_path = Path(inp_path)
    sample_dir = Path(sample_dir)

    # ---- 1. INP ----
    (node_labels, node_coords, elem_labels, connectivity,
     nset_bc, nset_force) = parse_inp(inp_path)
    label_to_idx = {int(lbl): i for i, lbl in enumerate(node_labels)}

    # ---- 2. Strain CSVs ----
    csvs = []
    for pat in ('*_internal_nodes.csv', '*_surface_1st_order_nodes.csv'):
        csvs.extend(sample_dir.glob(pat))
    if not csvs:
        raise FileNotFoundError(f"No node strain CSVs in {sample_dir}")
    node_strain = load_strain_csv(*csvs)
    if not node_strain:
        raise RuntimeError(f"Empty strain data after loading {len(csvs)} CSVs")
    print(f"  Strain CSVs: {len(node_strain)} nodes")

    # ---- 3. Element strain ----
    elem_strain = compute_element_strain(node_labels, connectivity, node_strain)
    print(f"  Elements: {len(elem_labels)}, "
          f"strain range [{elem_strain.min():.4e}, {elem_strain.max():.4e}]")

    # ---- 4. Centroids / volume ----
    centroids = compute_centroids(node_coords, connectivity)
    volume    = compute_volume(node_coords, connectivity)

    # ---- 5. Element E (override from element CSV) ----
    elem_E = np.ones(len(elem_labels), dtype=np.float32)
    if e_csv_path and Path(e_csv_path).exists():
        e_data = np.loadtxt(e_csv_path, delimiter=',', skiprows=1, dtype=np.float64)
        if e_data.ndim == 1:
            e_data = e_data.reshape(1, -1)
        eid_to_E = dict(zip(e_data[:, 0].astype(int), e_data[:, 1].astype(np.float32)))
        for i, eid in enumerate(elem_labels):
            v = eid_to_E.get(int(eid))
            if v is not None:
                elem_E[i] = v
        print(f"  Element E: {len(eid_to_E)} entries")
    else:
        print(f"  [WARN] no element_E CSV at {e_csv_path}, using uniform E=1")

    # ---- 6. BC / Load element masks from NSETs ----
    is_bc   = nset_to_element_mask(nset_bc,    label_to_idx, connectivity)
    is_load = nset_to_element_mask(nset_force, label_to_idx, connectivity)
    print(f"  BC elements: {int(is_bc.sum())}, Load elements: {int(is_load.sum())}")

    # ---- 7. Centroids of BC / Load ----
    bc_centroid   = centroids[is_bc   > 0.5].mean(axis=0).astype(np.float32) if is_bc.sum()   else centroids.mean(axis=0).astype(np.float32)
    load_centroid = centroids[is_load > 0.5].mean(axis=0).astype(np.float32) if is_load.sum() else centroids.mean(axis=0).astype(np.float32)

    # ---- 8. u_disp (healthy = uniform -0.5 mm) ----
    u_disp = np.array([-0.5], dtype=np.float32)

    # ---- 9. Topology edges ----
    edge_index = build_topo_edges(connectivity)
    E_log = np.log1p(elem_E)
    if edge_index.shape[1] > 0:
        s, d = edge_index[0], edge_index[1]
        delta = centroids[d] - centroids[s]
        dist = np.linalg.norm(delta, axis=1, keepdims=True).clip(min=1e-6)
        edge_physics = np.hstack(
            [delta / dist, dist, ((E_log[s] + E_log[d]) / 2).reshape(-1, 1)]
        ).astype(np.float32)
    else:
        edge_physics = np.zeros((0, 5), dtype=np.float32)
    print(f"  Topology edges: {edge_index.shape[1]}")

    # ---- 10. Save ----
    np.savez_compressed(
        out_path,
        centroids=centroids, E=elem_E,
        S3=elem_strain,            # field name kept as `S3` for pipeline compat
        elem_ids=elem_labels,
        is_bc=is_bc, is_load=is_load,
        u_disp=u_disp, volume=volume,
        bc_centroid=bc_centroid, load_centroid=load_centroid,
        edge_index=edge_index, edge_physics=edge_physics,
        near_edge_index=np.zeros((2, 0), dtype=np.int64),
        near_edge_physics=np.zeros((0, 5), dtype=np.float32),
        far_edge_index=np.zeros((2, 0), dtype=np.int64),
        far_edge_physics=np.zeros((0, 5), dtype=np.float32),
    )
    print(f"  Saved: {out_path}")


# ================================================================
# Main
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp_dir",   default=r"N:\SecondYear\Morphed_Vertebrae_TrainingSet\WholeVTK2STL\INPFiles")
    ap.add_argument("--odb_root",  default=r"N:\SecondYear\Aba_ODB_Set")
    ap.add_argument("--e_root",    default=r"N:\SecondYear\CT_MASK_E_PHANTOMLESS\Train_FEM")
    ap.add_argument("--train_dir", default=r"../../Data/ElementGraph_Healthy_Train")
    ap.add_argument("--test_dir",  default=r"../../Data/ElementGraph_Healthy_Test")
    ap.add_argument("--test_ratio", type=float, default=0.10,
                    help="Fraction of NEW cases routed to test (default 10%, "
                         "matches tumor's ~90:10 split).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inp_dir   = Path(args.inp_dir)
    odb_root  = Path(args.odb_root)
    e_root    = Path(args.e_root)
    train_dir = Path(args.train_dir); train_dir.mkdir(parents=True, exist_ok=True)
    test_dir  = Path(args.test_dir);  test_dir.mkdir(parents=True, exist_ok=True)

    # Cases already partitioned in earlier runs are immutable.
    existing_train = {f.stem for f in train_dir.glob("*.npz")}
    existing_test  = {f.stem for f in test_dir.glob("*.npz")}
    existing       = existing_train | existing_test
    print(f"[INFO] Existing  train={len(existing_train)}  test={len(existing_test)}")

    # Discover non-empty source case dirs.
    src_cases = sorted([d for d in odb_root.iterdir()
                        if d.is_dir() and any(d.iterdir())])
    print(f"[INFO] Non-empty case dirs in {odb_root}: {len(src_cases)}")

    # Filter out cases that are already partitioned
    new_cases = [d for d in src_cases if d.name not in existing]
    print(f"[INFO] New (unprocessed) cases this run: {len(new_cases)}")
    if not new_cases:
        print("[DONE] Nothing new to process.")
        return

    # Deterministic 9:1 split of new cases. Shuffling by NAME (not by dir
    # iteration order) keeps the split reproducible across machines.
    new_names = sorted([d.name for d in new_cases])
    rng = random.Random(args.seed)
    rng.shuffle(new_names)
    if args.test_ratio <= 0.0:
        n_test = 0
    else:
        n_test = max(1, round(len(new_names) * args.test_ratio))
    test_set = set(new_names[:n_test])
    print(f"[INFO] Split this batch: train={len(new_names) - n_test}  test={n_test}")

    case_by_name = {d.name: d for d in new_cases}
    n_ok = n_fail = n_skip = 0
    t_start = time.time()

    for k, name in enumerate(new_names, 1):
        case_dir = case_by_name[name]
        out_dir  = test_dir if name in test_set else train_dir
        tag      = 'TEST' if name in test_set else 'TRAIN'
        out_path = out_dir / f"{name}.npz"

        if out_path.exists():
            n_skip += 1
            continue

        inp_path = inp_dir / f"{name}.inp"
        if not inp_path.exists():
            print(f"[SKIP-NO-INP] [{tag}] {name}: {inp_path} not found")
            n_skip += 1
            continue

        # element_E CSV path:  <e_root>/<subj>/<subj>_element_E_tp_<tp>.csv
        # name format: L1_527_tp_1  -> subj=527  tp=1
        parts = name.split('_')
        subj  = parts[1] if len(parts) >= 2 else ''
        tp_num = parts[-1]
        e_csv = e_root / subj / f"{subj}_element_E_tp_{tp_num}.csv"

        print(f"\n[{k}/{len(new_names)}] [{tag}] {name}")
        try:
            process_one_sample(inp_path, case_dir, e_csv, out_path)
            n_ok += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
            n_fail += 1

    final_train = len(existing_train) + sum(1 for n in new_names if n not in test_set)
    final_test  = len(existing_test)  + len(test_set)
    final_total = final_train + final_test
    print(f"\n[DONE] ok={n_ok}  skipped={n_skip}  failed={n_fail}  "
          f"time={(time.time()-t_start)/60:.1f} min")
    print(f"[INFO] Cumulative: train={final_train} ({final_train/final_total*100:.1f}%)  "
          f"test={final_test} ({final_test/final_total*100:.1f}%)")


if __name__ == "__main__":
    main()
