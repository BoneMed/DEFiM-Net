import os
import re
import csv

# ===== ??????? INP ??? =====
inp_root = r"N:\SecondYear\Morphed_Vertebrae_TestSet\WholeVTK2STL\INPFiles"

# ===== ??????? =====
out_root = r"N:\SecondYear\test_ODB_Set"

# ======== ????????? ========
def parse_nodes(lines):
    node_coords = {}
    node_section = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("*node"):
            node_section = True
            continue
        if node_section:
            if line.startswith("*"):
                node_section = False
                continue
            parts = line.split(",")
            if len(parts) >= 4:
                try:
                    nid = int(parts[0].strip())
                    x = float(parts[1].strip())
                    y = float(parts[2].strip())
                    z = float(parts[3].strip())
                    node_coords[nid] = (x, y, z)
                except:
                    pass
    return node_coords

# ======== ????? NSET ========
def extract_nset(lines, name):
    ids = []
    capture = False
    name_lower = name.lower()
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.lower().startswith("*nset") and ("nset=" + name_lower) in line_stripped.lower():
            capture = True
            continue
        if capture:
            if line_stripped.startswith("*"):
                break
            parts = [p.strip() for p in line_stripped.split(",") if p.strip()]
            for p in parts:
                try:
                    ids.append(int(p))
                except:
                    pass
    return ids

# ======== ???? CSV ========
def write_csv(path, node_ids, node_coords):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["NodeID", "X_coord", "Y_coord", "Z_coord"])
        for nid in node_ids:
            if nid in node_coords:
                x, y, z = node_coords[nid]
                writer.writerow([nid, x, y, z])

# ======== ??? ========
for fname in os.listdir(inp_root):
    if not fname.lower().endswith(".inp"):
        continue

    inp_path = os.path.join(inp_root, fname)
    base = os.path.splitext(fname)[0].split('T_')[-1]  # e.g. L1_504_tp_1
    print("Processing:", base)

    # ?? subject id??? L1_504_tp_1 ? subj = 504
    try:
        subj_id = re.findall(r"L1_(\d+)_", base)[0]
    except:
        print("  ?? Cannot parse subject id from name:", base)
        continue

    # ??????
    out_dir = os.path.join(out_root, subj_id, base)

    # ? ???????????
    if not os.path.exists(out_dir):
        print("  ?? Skip: output folder not found ->", out_dir)
        continue

    # ? ????????????????????????
    bc_csv = os.path.join(out_dir, "BC_Nodes.csv")
    force_csv = os.path.join(out_dir, "FORCE_Nodes.csv")
    if os.path.exists(bc_csv) and os.path.exists(force_csv):
        print("  ?? Skip: both CSV files already exist.\n")
        continue

    # ?? inp ??
    with open(inp_path, "r") as f:
        lines = f.readlines()

    # ??????
    node_coords = parse_nodes(lines)

    # ?? BC ? FORCE ????
    bc_nodes = extract_nset(lines, "BC")
    force_nodes = extract_nset(lines, "FORCE")

    print("  Found %d BC nodes, %d FORCE nodes" % (len(bc_nodes), len(force_nodes)))

    # ?? CSV
    if bc_nodes:
        write_csv(bc_csv, bc_nodes, node_coords)
    if force_nodes:
        write_csv(force_csv, force_nodes, node_coords)

    print("  Saved to", out_dir, "\n")

print("Batch extraction completed.")
