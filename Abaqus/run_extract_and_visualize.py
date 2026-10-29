#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主脚本：
  1. subprocess 调用 abaqus python extract_element_stress.py → .npz
  2. 读取 .npz + element E CSV
  3. pyvista 3D 可视化 Max Principal Stress
"""

import subprocess
import sys
import numpy as np
import os

# ================================================================
# 路径配置
# ================================================================
ODB_PATH   = r"N:\SecondYear\Aba_ODB_Set\504\L1_504_tp_1\L1_504_tp_1.odb"
NPZ_PATH   = r"N:\SecondYear\Aba_ODB_Set\504\L1_504_tp_1\L1_504_tp_1_element_stress.npz"
E_CSV_PATH = r"N:\SecondYear\CT_MASK_E_PHANTOMLESS\Train_FEM\504\504_element_E_tp_1.csv"
ABAQUS_EXE = r"E:\SIMULIA\Commands\abaqus.bat"
EXTRACT_SCRIPT = os.path.join(os.path.dirname(__file__), "extract_element_stress.py")


# ================================================================
# Step 1: 调用 abaqus python 提取
# ================================================================
def run_abaqus_extract(odb_path, npz_path):
    print(f"[1/2] 调用 abaqus python 提取单元应力...")
    cmd = [ABAQUS_EXE, "python", EXTRACT_SCRIPT, odb_path, npz_path]
    print(f"  CMD: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(result.stdout)
    if result.stderr:
        print(f"  STDERR: {result.stderr}")
    if result.returncode != 0:
        print(f"  [ERROR] 返回码: {result.returncode}")
        sys.exit(1)


# ================================================================
# Step 2: 可视化
# ================================================================
def visualize(npz_path, e_csv_path=None):
    import pyvista as pv

    print(f"[2/2] 可视化...")
    data = np.load(npz_path)

    centroids   = data['centroids']       # (N, 3)
    S1          = data['S1']              # Max Principal
    S3          = data['S3']              # Min Principal
    mises       = data['mises']
    elem_labels = data['elem_labels']

    valid = ~np.isnan(S1)
    print(f"  单元数: {len(elem_labels)}, 有效: {valid.sum()}")
    print(f"  Max Principal: [{np.nanmin(S1):.4f}, {np.nanmax(S1):.4f}] MPa")
    print(f"  Min Principal: [{np.nanmin(S3):.4f}, {np.nanmax(S3):.4f}] MPa")
    print(f"  Mises:         [{np.nanmin(mises):.4f}, {np.nanmax(mises):.4f}] MPa")

    # 只用有效单元
    c = centroids[valid]
    s1 = S1[valid]
    s3 = S3[valid]
    mi = mises[valid]
    eids = elem_labels[valid]

    # E
    E_vals = None
    if e_csv_path and os.path.exists(e_csv_path):
        e_data = np.loadtxt(e_csv_path, delimiter=',', skiprows=1)
        eid_to_E = dict(zip(e_data[:, 0].astype(int), e_data[:, 1]))
        E_vals = np.array([eid_to_E.get(int(eid), np.nan) for eid in eids])
        n_matched = np.sum(~np.isnan(E_vals))
        print(f"  E匹配: {n_matched}/{len(eids)}")

    # pyvista
    points = pv.PolyData(c)
    points['Max Principal (MPa)'] = s1
    points['Min Principal (MPa)'] = s3
    points['Mises (MPa)'] = mi
    if E_vals is not None:
        points['E (MPa)'] = E_vals

    # 可视化 S1 (最大主应力) 和 S3 (最小主应力)
    pl = pv.Plotter(shape=(1, 2), window_size=(2000, 900))

    # S1 Max Principal
    pl.subplot(0, 0)
    pl.add_text("S1 - Max Principal (Avg 75) MPa", font_size=12)
    cl = [np.percentile(s1, 2), np.percentile(s1, 98)]
    pl.add_mesh(points, scalars='Max Principal (MPa)', point_size=4,
                cmap='jet', render_points_as_spheres=True,
                clim=cl)

    # S3 Min Principal
    pl.subplot(0, 1)
    pl.add_text("S3 - Min Principal (Avg 75) MPa", font_size=12)
    cl = [np.percentile(s3, 2), np.percentile(s3, 98)]
    pl.add_mesh(points, scalars='Min Principal (MPa)', point_size=4,
                cmap='jet', render_points_as_spheres=True,
                clim=cl)

    pl.link_views()
    pl.show()


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    # Step 1: 提取（如果npz不存在）
    if not os.path.exists(NPZ_PATH):
        run_abaqus_extract(ODB_PATH, NPZ_PATH)
    else:
        print(f"[跳过提取] NPZ已存在: {NPZ_PATH}")

    # Step 2: 可视化
    visualize(NPZ_PATH, E_CSV_PATH)
