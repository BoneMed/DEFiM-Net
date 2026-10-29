# -*- coding: utf-8 -*-
"""
用 abaqus python 运行（由 run_extract_and_visualize.py 自动调用）
从 ODB 提取单元级应力, 模拟 Abaqus Average 75% 过滤

Average 75% 逻辑:
  1. 收集每个单元所有积分点的应力张量
  2. 计算平均值 avg 和范围 range = max - min
  3. 剔除偏离 avg 超过 75% × range 的积分点
  4. 用剩余积分点重新平均

输出 .npz
"""
from odbAccess import openOdb
from abaqusConstants import INTEGRATION_POINT
import numpy as np
import sys

odb_path = sys.argv[1]
out_path = sys.argv[2]

print("Opening: %s" % odb_path)
odb = openOdb(odb_path, readOnly=True)

# -------- Instance --------
inst_name = list(odb.rootAssembly.instances.keys())[0]
inst = odb.rootAssembly.instances[inst_name]
print("Instance: %s" % inst_name)

# -------- 节点坐标 --------
nodes = inst.nodes
node_labels = np.array([n.label for n in nodes])
node_coords = np.array([n.coordinates for n in nodes], dtype=np.float64)
label_to_idx = {}
for i, lbl in enumerate(node_labels):
    label_to_idx[int(lbl)] = i
print("Nodes: %d" % len(nodes))

# -------- 单元 --------
elements = inst.elements
n_per = len(elements[0].connectivity)
elem_type = str(elements[0].type)

elem_labels = np.array([int(e.label) for e in elements])
connectivity = np.array(
    [[label_to_idx[int(n)] for n in e.connectivity] for e in elements],
    dtype=np.int64
)
print("Elements: %d, type: %s, nodes/elem: %d" % (len(elements), elem_type, n_per))

# -------- 单元质心 --------
n_corner = 4 if n_per == 10 else n_per
centroids = node_coords[connectivity[:, :n_corner]].mean(axis=1)

# -------- 应力 (INTEGRATION_POINT) --------
step = list(odb.steps.values())[-1]
frame = step.frames[-1]
print("Step: %s, Time: %.4f" % (step.name, frame.frameValue))

S_field = frame.fieldOutputs['S']
S_sub = S_field.getSubset(position=INTEGRATION_POINT)

# 建立 element label -> index 映射
eid_to_idx = {}
for i, lbl in enumerate(elem_labels):
    eid_to_idx[int(lbl)] = i

n_elem = len(elements)

# -------- 收集每个单元的所有积分点数据 --------
# C3D10 有 4 个积分点
elem_ip_stress = {}  # eid_idx -> list of [S11,S22,S33,S12,S13,S23]
elem_ip_mises = {}   # eid_idx -> list of mises

for v in S_sub.values:
    eid = int(v.elementLabel)
    idx = eid_to_idx.get(eid)
    if idx is not None:
        s6 = [float(v.data[k]) for k in range(6)]
        mi = float(v.mises)
        if idx not in elem_ip_stress:
            elem_ip_stress[idx] = []
            elem_ip_mises[idx] = []
        elem_ip_stress[idx].append(s6)
        elem_ip_mises[idx].append(mi)

print("Collected integration point data for %d elements" % len(elem_ip_stress))

# -------- Average 75% 过滤 --------
AVG_THRESHOLD = 0.75

S_tensor = np.full((n_elem, 6), np.nan)
mises = np.full(n_elem, np.nan)

for idx in range(n_elem):
    if idx not in elem_ip_stress:
        continue

    ip_data = np.array(elem_ip_stress[idx])  # (n_ip, 6)
    ip_mi = np.array(elem_ip_mises[idx])     # (n_ip,)
    n_ip = len(ip_data)

    if n_ip == 1:
        S_tensor[idx] = ip_data[0]
        mises[idx] = ip_mi[0]
        continue

    # 对每个分量计算 average 75% 过滤
    # 用 Mises 作为标量指标来判断哪些积分点是 outlier
    avg_mi = ip_mi.mean()
    mi_range = ip_mi.max() - ip_mi.min()

    if mi_range < 1e-10:
        # 所有积分点几乎相同
        S_tensor[idx] = ip_data.mean(axis=0)
        mises[idx] = ip_mi.mean()
        continue

    # 保留偏离平均不超过 75% × range 的积分点
    deviation = np.abs(ip_mi - avg_mi)
    keep = deviation <= AVG_THRESHOLD * mi_range

    if keep.sum() == 0:
        # 全部被剔除(不应该发生), fallback到全平均
        S_tensor[idx] = ip_data.mean(axis=0)
        mises[idx] = ip_mi.mean()
    else:
        S_tensor[idx] = ip_data[keep].mean(axis=0)
        mises[idx] = ip_mi[keep].mean()

n_valid = int(np.sum(~np.isnan(S_tensor[:, 0])))
print("Stress averaged (75%% filter): %d / %d" % (n_valid, n_elem))

# -------- 主应力 --------
print("Computing principal stresses...")
valid = ~np.isnan(S_tensor).any(axis=1)
T = np.zeros((n_elem, 3, 3))
T[:, 0, 0] = S_tensor[:, 0]
T[:, 1, 1] = S_tensor[:, 1]
T[:, 2, 2] = S_tensor[:, 2]
T[:, 0, 1] = T[:, 1, 0] = S_tensor[:, 3]
T[:, 0, 2] = T[:, 2, 0] = S_tensor[:, 4]
T[:, 1, 2] = T[:, 2, 1] = S_tensor[:, 5]

eigs = np.full((n_elem, 3), np.nan)
eigs[valid] = np.linalg.eigvalsh(T[valid])
S1 = eigs[:, 2]
S2 = eigs[:, 1]
S3 = eigs[:, 0]

# -------- 保存 --------
np.savez_compressed(
    out_path,
    elem_labels=elem_labels,
    centroids=centroids,
    connectivity=connectivity,
    S_tensor=S_tensor,
    S1=S1,
    S2=S2,
    S3=S3,
    mises=mises,
)
odb.close()
print("Saved: %s" % out_path)
print("S3 (MinPrincipal) range: %.4f ~ %.4f MPa" % (np.nanmin(S3), np.nanmax(S3)))
print("S1 (MaxPrincipal) range: %.4f ~ %.4f MPa" % (np.nanmin(S1), np.nanmax(S1)))
print("Mises range: %.4f ~ %.4f MPa" % (np.nanmin(mises), np.nanmax(mises)))
