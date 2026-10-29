"""
Phantomless calibration visualization for PPT.
4 panels:
  1) Centroid slice with air/fat/muscle masks
  2) 3-point HU correction fit
  3) Density conversion chain curves
  4) Bone voxel E distribution
"""
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import binary_erosion, binary_opening, generate_binary_structure, label, binary_closing

# ---- Load ----
subj = "514"
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

# ---- Extract tissue masks in ROI ----
roi = ct[:, :, z_min:z_max+1].copy()
body = roi > -500
body = binary_closing(body, structure=generate_binary_structure(3, 1))
labeled, num = label(body)
if num > 1:
    counts = np.bincount(labeled.ravel()); counts[0] = 0
    body = labeled == np.argmax(counts)

shell = body & (~binary_erosion(body, iterations=8))
deep = binary_erosion(body, iterations=5)
l1_roi = (mask[:, :, z_min:z_max+1] == 20)

# L1 centroid for psoas targeting
l1_coords_roi = np.argwhere(l1_roi)
l1_cx = int(np.mean(l1_coords_roi[:, 0]))
l1_cy = int(np.mean(l1_coords_roi[:, 1]))
spacing = np.abs(np.diag(nib.load(ct_path).affine)[:3])

# Distance and psoas mask
yy, xx = np.meshgrid(np.arange(roi.shape[1]), np.arange(roi.shape[0]))
dist_from_l1 = np.sqrt(((xx - l1_cx)*spacing[0])**2 + ((yy - l1_cy)*spacing[1])**2)
dist_3d = np.broadcast_to(dist_from_l1[:,:,np.newaxis], roi.shape)
lateral_mm = np.abs(xx - l1_cx) * spacing[0]
lateral_3d = np.broadcast_to(lateral_mm[:,:,np.newaxis], roi.shape)
anterior_3d = np.broadcast_to((yy >= l1_cy - 15)[:,:,np.newaxis], roi.shape)
psoas_zone = (dist_3d > 15) & (dist_3d < 50) & (lateral_3d > 8) & anterior_3d

z_local = z_center - z_min

air_m = (roi < -800) & (~body)
fat_m = (-200 < roi) & (roi < -50) & shell & (~l1_roi)
fat_m = binary_opening(fat_m)
mus_m = (0 < roi) & (roi < 100) & deep & (~l1_roi) & psoas_zone

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

print("Refs:", refs)

# ---- Step 2: HU correction ----
targets = {"air": -1000.0, "fat": -100.0, "muscle": 40.0}
weights = {"air": 3.0, "fat": 1.0, "muscle": 1.5}

xs, ys, ws = [], [], []
for k in ("air", "fat", "muscle"):
    if k in refs:
        xs.append(refs[k])
        ys.append(targets[k])
        ws.append(weights[k])

x = np.array(xs); y = np.array(ys); w = np.array(ws)
A = np.column_stack([x, np.ones_like(x)])
Aw = A * np.sqrt(w[:, None])
yw = y * np.sqrt(w)
coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
cal_a, cal_b = coef

print(f"HU_std = {cal_a:.6f} * HU_raw + {cal_b:.3f}")

# ---- Full pipeline ----
def pipeline(hu_raw):
    hu_std = cal_a * hu_raw + cal_b
    rho_qct = hu_std / 1000.0
    rho_ash = 0.8772 * rho_qct + 0.07895
    rho_ash = np.maximum(rho_ash, 0)
    E = 10200 * np.power(np.maximum(rho_ash, 1e-10), 2.01)
    return hu_std, rho_qct, rho_ash, E

l1_hu = ct[l1_mask]
hu_std, rho_qct, rho_ash, bone_E = pipeline(l1_hu)

# ============================================================
# FIGURE
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

tissue_colors = {"air": "#3498DB", "fat": "#F1C40F", "muscle": "#E74C3C"}
tissue_names = {"air": "Air", "fat": "Fat", "muscle": "Muscle"}

# ===== Panel 1: Centroid slice with tissue masks =====
ax = axes[0, 0]
ct_slice = roi[:, :, z_local]
l1_slice = l1_roi[:, :, z_local]

ax.imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-400, vmax=1500)

# L1 contour
if l1_slice.sum() > 0:
    ax.contour(l1_slice.T, levels=[0.5], colors='cyan', linewidths=2)

# Tissue masks
masks_2d = {"air": air_m[:,:,z_local], "fat": fat_m[:,:,z_local], "muscle": mus_m[:,:,z_local]}
for tissue, color in tissue_colors.items():
    m = masks_2d[tissue]
    if m.sum() > 10:
        ax.contourf(m.T, levels=[0.5, 1.5], colors=[color], alpha=0.35, origin='lower')

# Legend
handles = [
    Line2D([0], [0], color='cyan', lw=3, label='L1 vertebra'),
]
for tissue, color in tissue_colors.items():
    if tissue in refs:
        handles.append(plt.Rectangle((0,0),1,1, fc=color, alpha=0.5,
                       label=f'{tissue_names[tissue]}: {refs[tissue]:.0f} HU'))

ax.legend(handles=handles, loc='upper right', fontsize=9)
ax.set_title(f'Centroid Slice (z={z_center}) - Reference Tissues', fontsize=12, fontweight='bold')
ax.axis('off')

# ===== Panel 2: 3-point HU correction =====
ax = axes[0, 1]

# Plot fit line
hu_range = np.linspace(-1100, 100, 200)
hu_std_line = cal_a * hu_range + cal_b
ax.plot(hu_range, hu_std_line, 'k-', lw=2, label=f'Fit: HU_std = {cal_a:.4f}*HU + {cal_b:.1f}')

# Identity line
ax.plot(hu_range, hu_range, '--', color='gray', lw=1, alpha=0.5, label='Identity')

# Plot reference points (measured -> corrected)
for k in ("air", "fat", "muscle"):
    if k in refs:
        raw = refs[k]
        std = cal_a * raw + cal_b
        tgt = targets[k]
        color = tissue_colors[k]
        ax.plot(raw, std, 'o', color=color, markersize=12, markeredgecolor='black',
                markeredgewidth=1.5, zorder=5)
        ax.plot(raw, tgt, 'x', color=color, markersize=12, markeredgewidth=2.5, zorder=5)
        # Arrow from corrected to target
        ax.annotate('', xy=(raw, tgt), xytext=(raw, std),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5, ls='--'))
        ax.text(raw + 15, (std+tgt)/2, f'{tissue_names[k]}', color=color,
                fontsize=9, fontweight='bold')

# Labels
ax.set_xlabel('HU_raw (measured)', fontsize=11)
ax.set_ylabel('HU_std (corrected)', fontsize=11)
ax.set_title('3-Point HU Correction', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Custom legend for markers
handles2 = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', markersize=10,
           markeredgecolor='black', label='Measured → Corrected'),
    Line2D([0],[0], marker='x', color='gray', markersize=10,
           markeredgewidth=2, label='Target (canonical)'),
]
ax.legend(handles=[ax.get_lines()[0], ax.get_lines()[1]] + handles2, fontsize=8, loc='lower right')

# ===== Panel 3: Density conversion chain =====
ax = axes[1, 0]

hu_curve = np.linspace(-200, 1800, 500)
hu_std_c, rho_qct_c, rho_ash_c, E_c = pipeline(hu_curve)

# Twin axes: left = density, right = E
ax.plot(hu_curve, rho_qct_c, '-', color='#3498DB', lw=2, label='rho_QCT (g/cm3)')
ax.plot(hu_curve, rho_ash_c, '-', color='#E67E22', lw=2, label='rho_ash (g/cm3)')
ax.set_xlabel('HU_raw', fontsize=11)
ax.set_ylabel('Density (g/cm3)', fontsize=11, color='#2C3E50')
ax.set_ylim(-0.1, 2.0)
ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ax2.plot(hu_curve, E_c, '-', color='#E74C3C', lw=2, label='E (MPa)')
ax2.set_ylabel('E (MPa)', fontsize=11, color='#E74C3C')
ax2.set_ylim(-500, 25000)

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

# Annotate formulas
ax.text(0.98, 0.35, 'rho_QCT = HU_std / 1000\n'
        'rho_ash = 0.877*rho_QCT + 0.079\n'
        '               (Kopperdahl 2002)\n'
        'E = 10200 * rho_ash^2.01\n'
        '               (Keller 1994)',
        transform=ax.transAxes, fontsize=8, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_title('Density Conversion Chain', fontsize=12, fontweight='bold')

# ===== Panel 4: Bone voxel E distribution =====
ax = axes[1, 1]

# Histogram
ax.hist(bone_E, bins=150, color='steelblue', alpha=0.8, edgecolor='none', density=True)
ax.axvline(np.median(bone_E), color='red', lw=2, ls='--',
           label=f'Median = {np.median(bone_E):.0f} MPa')
ax.axvline(np.mean(bone_E), color='orange', lw=2, ls='--',
           label=f'Mean = {np.mean(bone_E):.0f} MPa')

# Percentile annotations
for p, c in [(5, '#95A5A6'), (95, '#95A5A6')]:
    val = np.percentile(bone_E, p)
    ax.axvline(val, color=c, lw=1, ls=':')
    ax.text(val, ax.get_ylim()[1]*0.9, f'P{p}\n{val:.0f}', color=c,
            fontsize=8, ha='center', va='top')

ax.set_xlabel('E (MPa)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'L1 Bone E Distribution (n={len(bone_E)} voxels)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, 25000)

# ---- Global ----
plt.suptitle(f'Phantomless Calibration Pipeline (Subject {subj})',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'H:/pythonproject/PHD_PROGRAM/GCN_Stress/Hu2EProcessing/test/calibration_results_{subj}.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print("Saved calibration_results.png")
plt.show()
