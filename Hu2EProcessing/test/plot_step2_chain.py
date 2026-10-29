"""
Step 2 PPT figure: HU correction + density conversion chain.
Left: formula chain diagram
Right: conversion curves with example subject
"""
import numpy as np
import nibabel as nib
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ---- Load subject 504 for real data ----
subj = "504"
ct_files = glob.glob(rf"N:\SecondYear\CT_MASK_E\Train_CT\*{subj}*ct.nii.gz")
ct_img = nib.as_closest_canonical(nib.load(ct_files[0]))
ct = ct_img.get_fdata(dtype=np.float32)
mask_path = ct_files[0].replace("Train_CT", "Train_Mask").replace("_ct.nii.gz", "_seg-vert_msk.nii.gz")
mask = np.asarray(nib.as_closest_canonical(nib.load(mask_path)).dataobj)

# Refs (from previous run)
refs = {"air": -1001.6, "fat": -117.0, "muscle": 43.4}
targets = {"air": -1000.0, "fat": -100.0, "muscle": 40.0}

# Fit
weights = {"air": 3.0, "fat": 1.0, "muscle": 1.5}
xs = [refs[k] for k in ("air", "fat", "muscle")]
ys = [targets[k] for k in ("air", "fat", "muscle")]
ws = [weights[k] for k in ("air", "fat", "muscle")]
x, y, w = np.array(xs), np.array(ys), np.array(ws)
A = np.column_stack([x, np.ones_like(x)])
Aw = A * np.sqrt(w[:, None])
yw = y * np.sqrt(w)
coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
cal_a, cal_b = coef

# Pipeline functions
def pipeline(hu_raw):
    hu_std = cal_a * hu_raw + cal_b
    rho_qct = hu_std / 1000.0
    rho_ash = 0.8772 * rho_qct + 0.07895
    rho_ash = np.maximum(rho_ash, 0)
    E = 10200 * np.power(np.maximum(rho_ash, 1e-10), 2.01)
    return hu_std, rho_qct, rho_ash, E

# L1 bone data
l1_hu = ct[mask == 20]
hu_std, rho_qct, rho_ash, bone_E = pipeline(l1_hu)

# ============================================================
# FIGURE: 2x2
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
plt.rcParams.update({'font.size': 14})

tissue_colors = {"air": "#3498DB", "fat": "#F1C40F", "muscle": "#E74C3C"}

# ===== Panel 1: HU correction (3-point fit) =====
ax = axes[0, 0]
hu_range = np.linspace(-1100, 100, 200)
hu_std_line = cal_a * hu_range + cal_b
ax.plot(hu_range, hu_std_line, 'k-', lw=2, label=f'Fit: HU_std = {cal_a:.4f} * HU + {cal_b:.1f}')
ax.plot(hu_range, hu_range, '--', color='gray', lw=1, alpha=0.5, label='Identity (no correction)')

for k in ("air", "fat", "muscle"):
    raw = refs[k]
    std = cal_a * raw + cal_b
    tgt = targets[k]
    color = tissue_colors[k]
    ax.plot(raw, std, 'o', color=color, markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax.plot(raw, tgt, 'x', color=color, markersize=12, markeredgewidth=2.5, zorder=5)
    ax.annotate('', xy=(raw, tgt), xytext=(raw, std),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, ls='--'))
    offset = 20 if k != "muscle" else -60
    ax.text(raw + offset, (std+tgt)/2, k.capitalize(), color=color, fontsize=12, fontweight='bold')

ax.set_xlabel('HU_raw (measured)', fontsize=14)
ax.set_ylabel('HU_std (corrected)', fontsize=14)
ax.set_title('3-Point HU Standardization', fontsize=16, fontweight='bold')
handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', markersize=10,
           markeredgecolor='black', label='Measured → Corrected'),
    Line2D([0],[0], marker='x', color='gray', markersize=10, markeredgewidth=2, label='Target (canonical)'),
]
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)

# ===== Panel 2: HU_std -> rho_QCT -> rho_ash =====
ax = axes[0, 1]
hu_curve = np.linspace(-200, 1800, 500)
hu_std_c = cal_a * hu_curve + cal_b
rho_qct_c = hu_std_c / 1000.0
rho_ash_c = np.maximum(0.8772 * rho_qct_c + 0.07895, 0)

ax.plot(hu_curve, rho_qct_c, '-', color='#3498DB', lw=2.5, label=r'$\rho_{QCT}$ = HU_std / 1000')
ax.plot(hu_curve, rho_ash_c, '-', color='#E67E22', lw=2.5,
        label=r'$\rho_{ash}$ = 0.877·$\rho_{QCT}$ + 0.079')

# Mark key points
for hu_mark, label_txt in [(0, 'Water'), (300, 'Trabecular'), (1000, 'Cortical')]:
    rq = (cal_a * hu_mark + cal_b) / 1000.0
    ra = max(0.8772 * rq + 0.07895, 0)
    ax.plot(hu_mark, rq, 'o', color='#3498DB', markersize=8)
    ax.plot(hu_mark, ra, 's', color='#E67E22', markersize=8)
    ax.annotate(f'{label_txt}\nHU={hu_mark}', xy=(hu_mark, ra), fontsize=11,
                xytext=(hu_mark+50, ra+0.1), arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

ax.set_xlabel('HU_raw', fontsize=14)
ax.set_ylabel('Density (g/cm³)', fontsize=14)
ax.set_title('HU → Density Conversion', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.2, 1.8)

# HU definition + Kopperdahl reference
ax.text(0.98, 0.38, (r'$HU = 1000 \times \frac{\mu - \mu_{water}}{\mu_{water} - \mu_{air}}$'
                      '\n\n'
                      r'$\therefore\ \rho_{QCT} = HU_{std}\, /\, 1000$  (g/cm³)'
                      '\n\n'
                      r'$\rho_{ash} = 0.877 \cdot \rho_{QCT} + 0.079$'
                      '\n      Kopperdahl et al. (2002)'),
        transform=ax.transAxes, fontsize=12, ha='right', va='top',
        bbox=dict(boxstyle='round', fc='wheat', alpha=0.9))

# ===== Panel 3: rho_ash -> E =====
ax = axes[1, 0]
rho_range = np.linspace(0, 1.6, 300)
E_range = 10200 * np.power(np.maximum(rho_range, 1e-10), 2.01)

ax.plot(rho_range, E_range, '-', color='#E74C3C', lw=2.5)

# Mark regions
ax.axvspan(0, 0.3, alpha=0.1, color='#3498DB', label='Trabecular (ρ < 0.3)')
ax.axvspan(0.3, 1.6, alpha=0.1, color='#E74C3C', label='Cortical (ρ > 0.3)')
ax.axvline(0.3, color='gray', lw=1, ls='--', alpha=0.5)

# Key points
for rho_mark, label_txt in [(0.1, ''), (0.3, 'Transition'), (0.5, ''), (1.0, ''), (1.5, '')]:
    E_mark = 10200 * rho_mark**2.01
    ax.plot(rho_mark, E_mark, 'ko', markersize=5)
    ax.text(rho_mark, E_mark + 800, f'ρ={rho_mark}\nE={E_mark:.0f}', fontsize=11, ha='center')

ax.set_xlabel(r'$\rho_{ash}$ (g/cm³)', fontsize=14)
ax.set_ylabel('E (MPa)', fontsize=14)
ax.set_title(r'$E = 10200 \cdot \rho_{ash}^{2.01}$ (Keller 1994)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1.6)
ax.set_ylim(0, 25000)

# ===== Panel 4: Full chain example on bone voxels =====
ax = axes[1, 1]

# Scatter: HU_raw vs E
n = len(l1_hu)
idx = np.random.choice(n, min(5000, n), replace=False)
sc = ax.scatter(l1_hu[idx], bone_E[idx], c=rho_ash[idx], cmap='jet', s=2, alpha=0.4,
                vmin=0, vmax=1.5)
plt.colorbar(sc, ax=ax, label=r'$\rho_{ash}$ (g/cm³)', shrink=0.8)

ax.set_xlabel('HU_raw (bone voxels)', fontsize=14)
ax.set_ylabel('E (MPa)', fontsize=14)
ax.set_title(f'Result: L1 Bone E (Subject {subj}, n={n})', fontsize=16, fontweight='bold')

# Stats text
stats = (f'Median E = {np.median(bone_E):.0f} MPa\n'
         f'Mean E = {np.mean(bone_E):.0f} MPa\n'
         f'Range = [{bone_E.min():.0f}, {bone_E.max():.0f}]')
ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=12, va='top',
        bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))
ax.grid(True, alpha=0.3)

plt.suptitle('Phantomless Calibration - HU to Elastic Modulus',
             fontsize=20, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('H:/pythonproject/PHD_PROGRAM/GCN_Stress/Hu2EProcessing/test/step2_chain.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print("Saved step2_chain.png")
plt.show()
