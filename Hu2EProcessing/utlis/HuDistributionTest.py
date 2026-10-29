#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize HU quantile curves from two CT volumes and their masks.
Author: Ryan & GPT-5 (2025)
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ========================= 用户输入 =========================
ref_ct_path = r"N:\SecondYear\CT_MASK_E\Train_CT\sub-verse510_dir-ax_ct.nii.gz"
ref_mask_path = r"N:\SecondYear\CT_MASK_E\Train_Mask\sub-verse510_dir-ax_seg-vert_msk.nii.gz"

subj_ct_path = r"N:\SecondYear\CT_MASK_E\Train_CT\sub-verse527_dir-ax_ct.nii.gz"
subj_mask_path = r"N:\SecondYear\CT_MASK_E\Train_Mask\sub-verse527_dir-ax_seg-vert_msk.nii.gz"



# 当前目标帧，例如 tp = 60
tp_index = 60
tp_total = 120
t = tp_index / tp_total  # 插值权重 [0~1]

num_quantiles = 5001  # 曲线采样精度

# ========================= 函数定义 =========================
def load_hu_values(ct_path, mask_path):
    """读取CT与mask，返回mask区域内的HU值"""
    ct = nib.load(ct_path).get_fdata().astype(np.float32)
    mask = nib.load(mask_path).get_fdata().astype(np.float32)
    hu = ct[mask > 0].ravel()
    return hu[np.isfinite(hu)]


def compute_quantile_curve(hu_values, num_q=5001):
    """计算分位数曲线"""
    qs = np.linspace(0.0, 1.0, num_q)
    q_vals = np.quantile(hu_values, qs)
    return qs, q_vals


# ========================= 主流程 =========================
print(f"[INFO] Loading CT data...")
ref_hu = load_hu_values(ref_ct_path, ref_mask_path)
subj_hu = load_hu_values(subj_ct_path, subj_mask_path)

print(f"[INFO] Computing quantile curves...")
qs, q_ref = compute_quantile_curve(ref_hu, num_quantiles)
_, q_subj = compute_quantile_curve(subj_hu, num_quantiles)

# 分位插值生成 tp=60 的目标分布
q_target = (1 - t) * q_ref + t * q_subj

plt.figure(figsize=(7, 5))
plt.plot(qs, q_ref, 'r-', label='Ref (tp119)')
plt.plot(qs, q_subj, 'g-', label='Subj (tp0)')
plt.plot(qs, q_target, 'b--', linewidth=2.0, label=f'Interpolated (tp={tp_index})')

plt.xlabel('Quantile')
plt.ylabel('HU value')
plt.title(f'HU Quantile Curves with Interpolation (tp={tp_index})')
plt.legend()
plt.grid(alpha=0.3)

# ✅ 放宽 y 轴范围
ymin = min(q_ref.min(), q_subj.min()) - 200
ymax = max(q_ref.max(), q_subj.max()) + 200
plt.ylim(ymin, ymax)

plt.tight_layout()
plt.show()


print(f"[OK] Interpolated quantile for tp={tp_index} generated (t={t:.3f})")
