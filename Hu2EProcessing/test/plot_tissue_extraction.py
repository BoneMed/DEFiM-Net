"""
Visualize the tissue extraction process for PPT.
Shows how body/shell/deep masks are built, then how air/fat/muscle are extracted.
"""
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import (binary_erosion, binary_opening, binary_closing,
                           generate_binary_structure, label)

# ---- Load ----
subj = "532"
import glob as _glob
_ct_files = _glob.glob(rf"N:\SecondYear\CT_MASK_E\Train_CT\*{subj}*ct.nii.gz")
ct_path = _ct_files[0]
mask_path = ct_path.replace("Train_CT", "Train_Mask").replace("_ct.nii.gz", "_seg-vert_msk.nii.gz")

print("Loading...")
ct = nib.load(ct_path).get_fdata(dtype=np.float32)
mask = np.asarray(nib.load(mask_path).dataobj)

# ---- L1 ROI ----
l1_mask = (mask == 20)
coords = np.argwhere(l1_mask)
z_center = int(np.mean(coords[:, 2]))
half_w = 20
z_min = max(0, z_center - half_w)
z_max = min(ct.shape[2] - 1, z_center + half_w)

# ---- Build masks in ROI ----
roi = ct[:, :, z_min:z_max+1].copy()
z_local = z_center - z_min

# Body
body = roi > -500
body = binary_closing(body, structure=generate_binary_structure(3, 1))
labeled, num = label(body)
if num > 1:
    counts = np.bincount(labeled.ravel()); counts[0] = 0
    body = labeled == np.argmax(counts)

# Shell and Deep
shell = body & (~binary_erosion(body, iterations=8))
deep = binary_erosion(body, iterations=5)
l1_roi = (mask[:, :, z_min:z_max+1] == 20)

# Tissue masks
air_m = (roi < -800) & (~body)
fat_m = (-200 < roi) & (roi < -50) & shell & (~l1_roi)
fat_m = binary_opening(fat_m)
mus_m = (0 < roi) & (roi < 100) & deep & (~l1_roi)

# Get 2D slices at centroid
ct_slice = roi[:, :, z_local]
body_2d = body[:, :, z_local]
shell_2d = shell[:, :, z_local]
deep_2d = deep[:, :, z_local]
l1_2d = l1_roi[:, :, z_local]
air_2d = air_m[:, :, z_local]
fat_2d = fat_m[:, :, z_local]
mus_2d = mus_m[:, :, z_local]

# Robust mean
def robust_mean(x, trim=0.1):
    x = np.sort(x.ravel())
    n = len(x)
    if n < 50: return float(np.median(x))
    lo, hi = int(trim*n), int((1-trim)*n)
    return float(np.mean(x[lo:hi]))

refs = {}
if air_m.sum() > 500: refs['air'] = robust_mean(roi[air_m])
if fat_m.sum() > 500: refs['fat'] = robust_mean(roi[fat_m])
if mus_m.sum() > 500: refs['muscle'] = robust_mean(roi[mus_m])

print("Refs:", {k: f"{v:.1f}" for k, v in refs.items()})

# ============================================================
# FIGURE: 2 rows x 3 cols
# Row 1: Body mask building (CT -> Body -> Shell/Deep)
# Row 2: Tissue extraction (Air -> Fat -> Muscle)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

vmin, vmax = -400, 1500

# ---------- Row 1: Mask construction ----------

# Panel 1a: Raw CT
ax = axes[0, 0]
ax.imshow(ct_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax.contour(l1_2d.T, levels=[0.5], colors='cyan', linewidths=1.5)
ax.set_title('(a) CT Slice (z={})'.format(z_center), fontsize=13, fontweight='bold')
ax.text(10, ct_slice.shape[1]-20, 'HU range:\n-1000 ~ +1600', color='white',
        fontsize=9, va='top', bbox=dict(boxstyle='round', fc='black', alpha=0.6))
ax.axis('off')

# Panel 1b: Body mask
ax = axes[0, 1]
ax.imshow(ct_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
# Body contour
ax.contourf(body_2d.T, levels=[0.5, 1.5], colors=['#2ECC71'], alpha=0.25, origin='lower')
ax.contour(body_2d.T, levels=[0.5], colors='#2ECC71', linewidths=2)
ax.contour(l1_2d.T, levels=[0.5], colors='cyan', linewidths=1.5)
ax.set_title('(b) Body Mask\n(HU > -500, closing, largest CC)', fontsize=13, fontweight='bold')
ax.axis('off')

# Panel 1c: Shell + Deep
ax = axes[0, 2]
ax.imshow(ct_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
# Shell in orange
ax.contourf(shell_2d.T, levels=[0.5, 1.5], colors=['#E67E22'], alpha=0.3, origin='lower')
# Deep in purple
ax.contourf(deep_2d.T, levels=[0.5, 1.5], colors=['#8E44AD'], alpha=0.3, origin='lower')
ax.contour(l1_2d.T, levels=[0.5], colors='cyan', linewidths=1.5)

handles_c = [
    plt.Rectangle((0,0),1,1, fc='#E67E22', alpha=0.5, label='Shell (subcutaneous)\n= body - erode(8)'),
    plt.Rectangle((0,0),1,1, fc='#8E44AD', alpha=0.5, label='Deep (interior)\n= erode(5)'),
    Line2D([0],[0], color='cyan', lw=2, label='L1 vertebra'),
]
ax.legend(handles=handles_c, loc='center left', fontsize=8, facecolor='white', framealpha=0.9,
          bbox_to_anchor=(1.02, 0.5))
ax.set_title('(c) Shell vs Deep Regions', fontsize=13, fontweight='bold')
ax.axis('off')

# ---------- Row 2: Tissue extraction ----------

# Panel 2a: Air
ax = axes[1, 0]
ax.imshow(ct_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax.contourf(air_2d.T, levels=[0.5, 1.5], colors=['#3498DB'], alpha=0.5, origin='lower')
ax.contour(body_2d.T, levels=[0.5], colors='#2ECC71', linewidths=1, linestyles='--')
ax.contour(l1_2d.T, levels=[0.5], colors='cyan', linewidths=1)

hu_val = refs.get('air', 0)
ax.set_title(f'(d) Air: HU < -800, outside body\nMeasured: {hu_val:.0f} HU',
             fontsize=13, fontweight='bold', color='#3498DB')
ax.text(10, 20, 'Canonical: -1000 HU', color='#3498DB', fontsize=10,
        bbox=dict(boxstyle='round', fc='white', alpha=0.8))
ax.axis('off')

# Panel 2b: Fat
ax = axes[1, 1]
ax.imshow(ct_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax.contourf(fat_2d.T, levels=[0.5, 1.5], colors=['#F1C40F'], alpha=0.5, origin='lower')
ax.contour(shell_2d.T, levels=[0.5], colors='#E67E22', linewidths=1, linestyles='--')
ax.contour(l1_2d.T, levels=[0.5], colors='cyan', linewidths=1)

hu_val = refs.get('fat', 0)
ax.set_title(f'(e) Fat: -200 < HU < -50, in Shell\nMeasured: {hu_val:.0f} HU',
             fontsize=13, fontweight='bold', color='#C8A415')
ax.text(10, 20, 'Canonical: -100 HU', color='#C8A415', fontsize=10,
        bbox=dict(boxstyle='round', fc='white', alpha=0.8))
ax.axis('off')

# Panel 2c: Muscle
ax = axes[1, 2]
ax.imshow(ct_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax.contourf(mus_2d.T, levels=[0.5, 1.5], colors=['#E74C3C'], alpha=0.5, origin='lower')
ax.contour(deep_2d.T, levels=[0.5], colors='#8E44AD', linewidths=1, linestyles='--')
ax.contour(l1_2d.T, levels=[0.5], colors='cyan', linewidths=1)

hu_val = refs.get('muscle', 0)
ax.set_title(f'(f) Muscle: 0 < HU < 100, in Deep\n(exclude L1)  Measured: {hu_val:.0f} HU',
             fontsize=13, fontweight='bold', color='#E74C3C')
ax.text(10, 20, 'Canonical: +40 HU', color='#E74C3C', fontsize=10,
        bbox=dict(boxstyle='round', fc='white', alpha=0.8))
ax.axis('off')

# ---- Arrows between rows ----
# Add arrow annotations between row 1 and row 2
fig.text(0.17, 0.48, r'$\downarrow$', fontsize=30, ha='center', color='#3498DB', fontweight='bold')
fig.text(0.50, 0.48, r'$\downarrow$', fontsize=30, ha='center', color='#F1C40F', fontweight='bold')
fig.text(0.83, 0.48, r'$\downarrow$', fontsize=30, ha='center', color='#E74C3C', fontweight='bold')

plt.suptitle(f'Tissue Reference Extraction Process (Subject {subj}, L1 ROI)',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'H:/pythonproject/PHD_PROGRAM/GCN_Stress/Hu2EProcessing/test/tissue_extraction_{subj}.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print("Saved tissue_extraction.png")
plt.show()
