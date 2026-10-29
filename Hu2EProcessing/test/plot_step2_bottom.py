"""
Generate two separate figures:
  1) Keller power law curve (rho_ash -> E)
  2) L1 bone E vs HU scatter (colored by rho_ash)
"""
import numpy as np
import nibabel as nib
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

# ---- Subject 504 data ----
subj = "504"
ct_files = glob.glob(rf"N:\SecondYear\CT_MASK_E\Train_CT\*{subj}*ct.nii.gz")
ct_img = nib.as_closest_canonical(nib.load(ct_files[0]))
ct = ct_img.get_fdata(dtype=np.float32)
mask_path = ct_files[0].replace("Train_CT", "Train_Mask").replace("_ct.nii.gz", "_seg-vert_msk.nii.gz")
mask = np.asarray(nib.as_closest_canonical(nib.load(mask_path)).dataobj)

cal_a, cal_b = 1.0017, 4.0

l1_hu = ct[mask == 20]
hu_std = cal_a * l1_hu + cal_b
rho_qct = hu_std / 1000.0
rho_ash = np.maximum(0.8772 * rho_qct + 0.07895, 0)
bone_E = 10200 * np.power(np.maximum(rho_ash, 1e-10), 2.01)

# ============================================================
# Figure 1: Keller power law
# ============================================================
fig1, ax = plt.subplots(figsize=(12, 10))

rho_range = np.linspace(0, 1.6, 300)
E_range = 10200 * np.power(np.maximum(rho_range, 1e-10), 2.01)

ax.plot(rho_range, E_range, '-', color='#E74C3C', lw=3.5)


# Key points
for rho_mark in [0.1, 0.3, 0.5, 1.0, 1.5]:
    E_mark = 10200 * rho_mark**2.01
    ax.plot(rho_mark, E_mark, 'ko', markersize=8)
    ax.text(rho_mark + 0.03, E_mark + 600, f'ρ={rho_mark}\nE={E_mark:.0f}',
            fontsize=18, ha='left')

ax.set_xlabel(r'$\rho_{ash}$ (g/cm³)', fontsize=26)
ax.set_ylabel('E (MPa)', fontsize=26)
ax.set_title(r'$E = 10200 \cdot \rho_{ash}^{2.01}$  (Keller 1994)', fontsize=28, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=20)
ax.set_xlim(0, 1.6)
ax.set_ylim(0, 25000)

plt.tight_layout()
plt.savefig('H:/pythonproject/PHD_PROGRAM/GCN_Stress/Hu2EProcessing/test/keller_curve.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved keller_curve.png")
plt.close()

# ============================================================
# Figure 2: L1 bone E vs HU scatter
# ============================================================
fig2, ax = plt.subplots(figsize=(12, 10))

n = len(l1_hu)
idx = np.random.choice(n, min(8000, n), replace=False)
sc = ax.scatter(l1_hu[idx], bone_E[idx], c=rho_ash[idx], cmap='jet', s=4, alpha=0.5,
                vmin=0, vmax=1.5)
cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label(r'$\rho_{ash}$ (g/cm³)', fontsize=22)
cbar.ax.tick_params(labelsize=18)

ax.set_xlabel('HU_raw (bone voxels)', fontsize=26)
ax.set_ylabel('E (MPa)', fontsize=26)
ax.set_title(f'L1 Bone E Distribution (Subject {subj}, n={n})', fontsize=28, fontweight='bold')
ax.tick_params(labelsize=20)
ax.grid(True, alpha=0.3)

stats = (f'Median E = {np.median(bone_E):.0f} MPa\n'
         f'Mean E = {np.mean(bone_E):.0f} MPa\n'
         f'Range = [{bone_E.min():.0f}, {bone_E.max():.0f}]')
ax.text(0.03, 0.97, stats, transform=ax.transAxes, fontsize=20, va='top',
        bbox=dict(boxstyle='round', fc='wheat', alpha=0.9))

plt.tight_layout()
plt.savefig('H:/pythonproject/PHD_PROGRAM/GCN_Stress/Hu2EProcessing/test/bone_E_scatter.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved bone_E_scatter.png")
plt.close()

print("Done.")
