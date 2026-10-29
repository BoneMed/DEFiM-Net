import os
import glob
import numpy as np
import pandas as pd
import meshio
import pyvista as pv
from scipy.spatial import cKDTree
import traceback

# ==========================================
# [配置区域]
# ==========================================
CONFIG = {
    # 1. 根目录 (包含所有 Subj 文件夹的目录)
    'fem_root_dir': r"N:\SecondYear/CT_MASK_E_PHANTOMLESS/Train_FEM",

    # 2. INP 源目录 (存放 L1_xxx_tp_xx.inp 的地方)
    'inp_root_dir': r"N:\SecondYear/Morphed_Vertebrae_TrainingSet\WholeVTK2STL/INPFiles",

    # 3. 命名规则
    'ref_id': '510',  # 参考模版 ID
    'max_tp': 119,  # 最大 TP
    'tp_range': range(1, 119),  # 生成范围: 1 到 118
    'inp_template': 'L1_{id}_tp_{tp}.inp',  # INP 文件名模板

    # 4. 材料参数
    'morgan_a': 4730,
    'morgan_b': 1.56,
}

# ==========================================
# 全局缓存（不改你原函数返回值，用全局把解析信息带出来）
# ==========================================
_INP_NODE_IDS_LIST = None      # list[int] 文件顺序 NodeID
_INP_NODE_COORDS = None        # (N,3) 文件顺序 Node坐标（未flip）
_INP_ELEM_CONN_NIDS = None     # dict[eid] -> [node_ids...]
_INP_ELEM_TYPE = None          # dict[eid] -> 'c3d10' / ...


# ==========================================
# 工具函数
# ==========================================
def get_rho(E):
    E = np.maximum(E, 0)
    return (E / CONFIG['morgan_a']) ** (1.0 / CONFIG['morgan_b'])


def get_E(rho):
    rho = np.maximum(rho, 0)
    return CONFIG['morgan_a'] * (rho ** CONFIG['morgan_b'])


def load_vtu_data(vtu_path):
    """ 读取 VTU 的坐标和密度 """
    if not os.path.exists(vtu_path):
        raise FileNotFoundError(f"VTU not found: {vtu_path}")

    mesh = pv.read(vtu_path)
    points = mesh.points

    # 优先找 Rho, 其次找 E 反算
    if 'Rho' in mesh.point_data:
        rho = mesh.point_data['Rho']
    elif 'Rho_g_cm3' in mesh.point_data:
        rho = mesh.point_data['Rho_g_cm3']
    elif 'E_MPa' in mesh.point_data:
        rho = get_rho(mesh.point_data['E_MPa'])
    elif 'E' in mesh.point_data:
        rho = get_rho(mesh.point_data['E'])
    else:
        raise ValueError(f"VTU无有效数据字段: {vtu_path}")

    return points, rho


# ==========================================
# 核心：严格 ID 解析 (支持 C3D10 续行 + 缓存 element conn)
# ==========================================
def parse_inp_strict_ids(inp_path):
    """
    手动解析 INP 文件（支持 *Element 续行）。
    返回:
      1) real_node_ids: 真实节点 ID 列表 (对应文件顺序)
      2) fingerprint_map: {Sorted_Corner_Indices -> Element_ID}  （保持你原接口）
    同时全局缓存：
      _INP_NODE_COORDS, _INP_ELEM_CONN_NIDS, _INP_ELEM_TYPE
    """
    global _INP_NODE_IDS_LIST, _INP_NODE_COORDS, _INP_ELEM_CONN_NIDS, _INP_ELEM_TYPE

    node_ids_list = []
    node_coords_list = []
    node_id_to_idx = {}  # Map NodeID -> 0-based index

    # --- Pass 1: Parse Nodes (ID + xyz) ---
    with open(inp_path, 'r', encoding='utf-8', errors='ignore') as f:
        mode = None
        idx_counter = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith('**'):
                continue
            if line.startswith('*'):
                mode = 'node' if line.lower().startswith('*node') else None
                continue

            if mode == 'node':
                parts = [p.strip() for p in line.split(',') if p.strip() != ""]
                if len(parts) >= 4:
                    try:
                        nid = int(float(parts[0]))
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        node_ids_list.append(nid)
                        node_coords_list.append([x, y, z])
                        node_id_to_idx[nid] = idx_counter
                        idx_counter += 1
                    except:
                        pass

    _INP_NODE_IDS_LIST = node_ids_list
    _INP_NODE_COORDS = np.asarray(node_coords_list, dtype=np.float32)

    # --- Pass 2: Parse Elements (C3D10 only, fixed version) ---
    # 修复：正确处理所有唯一元素 ID，包括重复定义的情况
    fingerprint_map = {}   # {Sorted_Corner_Indices -> Element_ID}
    elem_conn = {}         # eid -> [node_ids...]
    elem_type = {}         # eid -> type string

    in_elem_section = False
    current_type = None
    current_eid = None
    current_nodes = []
    EXPECTED_NODES = 10  # C3D10 固定 10 个节点

    with open(inp_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('**'):
                continue

            # 关键字行
            if line.startswith('*'):
                # 先 flush 当前元素
                if current_eid is not None and len(current_nodes) >= EXPECTED_NODES:
                    nids = current_nodes[:EXPECTED_NODES]
                    elem_conn[current_eid] = nids
                    elem_type[current_eid] = current_type if current_type else 'c3d10'
                    if len(nids) >= 4:
                        corners = [node_id_to_idx[n] for n in nids[:4] if n in node_id_to_idx]
                        if len(corners) == 4:
                            fingerprint_map[tuple(sorted(corners))] = current_eid

                current_eid = None
                current_nodes = []

                # 检查是否进入元素段
                lower = line.lower()
                if lower.startswith('*element'):
                    if 'output' not in lower:  # 排除 *ELEMENT OUTPUT
                        in_elem_section = True
                        current_type = 'c3d10'  # 默认 C3D10
                        if 'type=' in lower:
                            t_str = lower.split('type=')[1].split(',')[0].strip()
                            current_type = t_str
                    else:
                        in_elem_section = False
                else:
                    in_elem_section = False
                continue

            # 元素数据行
            if in_elem_section:
                parts = [p.strip() for p in line.split(',') if p.strip() != ""]
                if not parts:
                    continue

                # 判断是新元素还是续行
                # 关键：C3D10 需要 10 个节点，如果当前节点数 < 10，这是续行
                if current_eid is not None and len(current_nodes) < EXPECTED_NODES:
                    # 续行：所有列都是节点 ID
                    for tok in parts:
                        try:
                            current_nodes.append(int(float(tok)))
                        except:
                            pass
                else:
                    # 新元素（current_eid is None 或 len(current_nodes) >= 10）
                    # Flush 上一个元素
                    if current_eid is not None and len(current_nodes) >= EXPECTED_NODES:
                        nids = current_nodes[:EXPECTED_NODES]
                        elem_conn[current_eid] = nids
                        elem_type[current_eid] = current_type if current_type else 'c3d10'
                        if len(nids) >= 4:
                            corners = [node_id_to_idx[n] for n in nids[:4] if n in node_id_to_idx]
                            if len(corners) == 4:
                                fingerprint_map[tuple(sorted(corners))] = current_eid

                    # 开始新元素
                    try:
                        current_eid = int(float(parts[0]))
                        current_nodes = []
                        # 添加剩余列的节点 ID
                        for tok in parts[1:]:
                            try:
                                current_nodes.append(int(float(tok)))
                            except:
                                pass
                    except:
                        # 第一列不是整数，可能是格式错误，跳过
                        current_eid = None
                        current_nodes = []

    # Flush 最后一个元素
    if current_eid is not None and len(current_nodes) >= EXPECTED_NODES:
        nids = current_nodes[:EXPECTED_NODES]
        elem_conn[current_eid] = nids
        elem_type[current_eid] = current_type if current_type else 'c3d10'
        if len(nids) >= 4:
            corners = [node_id_to_idx[n] for n in nids[:4] if n in node_id_to_idx]
            if len(corners) == 4:
                fingerprint_map[tuple(sorted(corners))] = current_eid

    _INP_ELEM_CONN_NIDS = elem_conn
    _INP_ELEM_TYPE = elem_type

    return np.array(node_ids_list), fingerprint_map


# ==========================================
# 单样本处理逻辑
# ==========================================
def process_subject(subj_id, ref_data):
    print(f"\n>>> Processing Subject: {subj_id}")

    subj_dir = os.path.join(CONFIG['fem_root_dir'], subj_id)

    # 1. 加载 Target (TP119) 数据（Phase 1 生成的 Ground Truth）
    tgt_vtu_path = os.path.join(subj_dir, f"{subj_id}_E_tp_119.vtu")
    if not os.path.exists(tgt_vtu_path):
        print(f"   [Skip] Target VTU not found: {tgt_vtu_path}")
        return

    try:
        xyz_tgt, rho_tgt = load_vtu_data(tgt_vtu_path)
    except Exception as e:
        print(f"   [Error] Loading Target: {e}")
        return

    # 2. 准备三方对齐 (Ref 和 Target 的 KDTree)
    c_ref = np.mean(ref_data['xyz'], axis=0)
    c_tgt = np.mean(xyz_tgt, axis=0)

    tree_ref = cKDTree(ref_data['xyz'] - c_ref)
    tree_tgt = cKDTree(xyz_tgt - c_tgt)

    # 3. 循环处理中间 TP
    for tp in CONFIG['tp_range']:
        try:
            # A. 定位中间态 INP
            inp_name = CONFIG['inp_template'].format(id=subj_id, tp=tp)
            inp_path = os.path.join(CONFIG['inp_root_dir'], inp_name)

            if not os.path.exists(inp_path):
                continue

            # B. 读取几何 (meshio)
            mesh = meshio.read(inp_path)
            xyz_tp = mesh.points.astype(np.float32)

            # --- 几何变换 (TP) ---
            # 1) 旋转 (Z-flip)
            xyz_tp[:, 0] = -xyz_tp[:, 0]
            xyz_tp[:, 1] = -xyz_tp[:, 1]

            # 2) 归零 (用于查询)
            c_tp = np.mean(xyz_tp, axis=0)
            xyz_query = xyz_tp - c_tp

            # C. 插值映射
            # Ref -> TP
            d_ref, idx_ref = tree_ref.query(xyz_query, k=8, workers=-1)
            w_ref = 1.0 / np.maximum(d_ref, 1e-12) ** 2
            rho_mapped_ref = np.sum((w_ref / np.sum(w_ref, axis=1, keepdims=True)) * ref_data['rho'][idx_ref], axis=1)

            # Target -> TP
            d_tgt, idx_tgt = tree_tgt.query(xyz_query, k=8, workers=-1)
            w_tgt = 1.0 / np.maximum(d_tgt, 1e-12) ** 2
            rho_mapped_tgt = np.sum((w_tgt / np.sum(w_tgt, axis=1, keepdims=True)) * rho_tgt[idx_tgt], axis=1)

            # D. 混合
            alpha = tp / CONFIG['max_tp']
            rho_final = (1.0 - alpha) * rho_mapped_ref + alpha * rho_mapped_tgt
            E_final = get_E(rho_final)

            # E. 保存文件
            # 解析真实 ID（并缓存 element conn）
            real_nids, _ = parse_inp_strict_ids(inp_path)  # fingerprint 不再使用

            # --- 关键修复：直接使用 INP 解析的节点 ID 和坐标，不再依赖 meshio 的点顺序 ---
            # 问题：KDTree 匹配可能导致多个 meshio 点匹配同一 INP 节点，造成某些节点 ID 丢失
            # 解决：直接使用 INP 解析的节点顺序，同时用 INP 坐标计算 E 值
            max_node_match_dist = np.nan

            if (_INP_NODE_COORDS is not None) and (len(_INP_NODE_COORDS) == len(real_nids)):
                # 直接使用 INP 解析的坐标（做同样的 flip）
                xyz_inp = _INP_NODE_COORDS.astype(np.float32).copy()
                xyz_inp[:, 0] = -xyz_inp[:, 0]
                xyz_inp[:, 1] = -xyz_inp[:, 1]

                # 用 INP 坐标重新计算 E_final（而不是用 meshio 的 xyz_tp）
                c_inp = np.mean(xyz_inp, axis=0)
                xyz_query_inp = xyz_inp - c_inp

                # Ref -> INP nodes
                d_ref, idx_ref = tree_ref.query(xyz_query_inp, k=8, workers=-1)
                w_ref = 1.0 / np.maximum(d_ref, 1e-12) ** 2
                rho_mapped_ref = np.sum((w_ref / np.sum(w_ref, axis=1, keepdims=True)) * ref_data['rho'][idx_ref], axis=1)

                # Target -> INP nodes
                d_tgt, idx_tgt = tree_tgt.query(xyz_query_inp, k=8, workers=-1)
                w_tgt = 1.0 / np.maximum(d_tgt, 1e-12) ** 2
                rho_mapped_tgt = np.sum((w_tgt / np.sum(w_tgt, axis=1, keepdims=True)) * rho_tgt[idx_tgt], axis=1)

                # 混合
                rho_final = (1.0 - alpha) * rho_mapped_ref + alpha * rho_mapped_tgt
                E_final = get_E(rho_final)

                # xyz_tp 更新为 INP 坐标（用于后续 VTU 保存）
                xyz_tp = xyz_inp

                # real_nids 保持不变（直接是 INP 文件顺序的节点 ID）
            else:
                # 兜底：保持原逻辑
                if len(real_nids) != len(xyz_tp):
                    real_nids = np.arange(1, len(xyz_tp) + 1)

            # Node CSV
            csv_node = os.path.join(subj_dir, f"{subj_id}_node_E_tp_{tp}.csv")
            pd.DataFrame({'NodeIndex_in_FileOrder': real_nids, 'E_MPa': E_final}).to_csv(csv_node, index=False)

            # --- Element CSV（关键改：不再依赖 mesh.cells，改为遍历 INP 手动解析的全量 element conn） ---
            eids, evals = [], []
            missing_elem_nid = 0

            # NodeID -> mesh点索引（修复后: real_nids 是 INP 文件顺序，不再是 KDTree 重排后的）
            nid2midx = {}
            nid_list = real_nids.tolist() if hasattr(real_nids, 'tolist') else list(real_nids)
            for midx, nid in enumerate(nid_list):
                if nid not in nid2midx:
                    nid2midx[nid] = midx

            # 诊断：检查节点 ID 覆盖率
            unique_nids_in_map = len(nid2midx)
            total_nids_parsed = len(_INP_NODE_IDS_LIST) if _INP_NODE_IDS_LIST else 0

            if _INP_ELEM_CONN_NIDS is None or len(_INP_ELEM_CONN_NIDS) == 0:
                # 理论上不会发生：parse_inp_strict_ids 已经解析
                print(f"   [Error] No element connectivity parsed for TP {tp}!")
                print(f"   Please check the INP file format.")
                continue

            # 按 INP 的 element 列表写（处理所有 C3D10 元素）
            for eid, nids in _INP_ELEM_CONN_NIDS.items():
                # 所有元素都是 C3D10，不需要类型检查
                midxs = []
                ok = True
                for nid in nids:
                    mi = nid2midx.get(nid, None)
                    if mi is None:
                        ok = False
                        break
                    midxs.append(mi)

                if not ok:
                    missing_elem_nid += 1
                    continue

                eids.append(eid)
                evals.append(float(np.mean(E_final[midxs])))

            csv_elem = os.path.join(subj_dir, f"{subj_id}_element_E_tp_{tp}.csv")
            pd.DataFrame({'ElementID': eids, 'E_MPa': evals}).sort_values('ElementID').to_csv(csv_elem, index=False)

            # VTU (保存旋转后的几何)
            vtu_out = os.path.join(subj_dir, f"{subj_id}_E_tp_{tp}.vtu")
            grid = pv.PolyData(xyz_tp)
            grid.point_data['E_MPa'] = E_final
            grid.cast_to_unstructured_grid().save(vtu_out)

            print(f"   -> Generated TP {tp} | nodes={len(real_nids)} nid_coverage={unique_nids_in_map}/{total_nids_parsed} elems_written={len(eids)} bad_elem={missing_elem_nid}")

        except Exception as e:
            print(f"   [Error] TP {tp}: {e}")
            # traceback.print_exc()


# ==========================================
# 批处理主控
# ==========================================
def run_batch_interpolation():
    print("=== 开始批量中间态插值 (Aligned + Strict IDs) ===")

    # 1. 加载 Reference 数据 (只加载一次)
    ref_id = CONFIG['ref_id']
    ref_vtu = os.path.join(CONFIG['fem_root_dir'], ref_id, f"{ref_id}_E_tp_119.vtu")

    print(f"Loading Reference ({ref_id})...")
    try:
        xyz_ref, rho_ref = load_vtu_data(ref_vtu)
        ref_data = {'xyz': xyz_ref, 'rho': rho_ref}
    except Exception as e:
        print(f"[Fatal Error] Failed to load Reference: {e}")
        return

    # 2. 遍历所有 Subject 文件夹
    subfolders = [f.path for f in os.scandir(CONFIG['fem_root_dir']) if f.is_dir()]

    for folder in subfolders:
        subj_id = os.path.basename(folder)

        # 跳过 Ref 本身
        if subj_id == ref_id:
            continue

        # 执行处理
        try:
            process_subject(subj_id, ref_data)
        except Exception as e:
            print(f"   [Failed] Subject {subj_id}: {e}")

    print("\n所有批处理任务完成。")


if __name__ == "__main__":
    run_batch_interpolation()
