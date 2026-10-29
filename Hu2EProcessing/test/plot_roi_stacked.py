"""
L1 ROI extraction - 3D orthogonal slices visualization.
Center: axial slice (flat), Left wall: sagittal, Back wall: coronal.
Like the femur reference image with 3 orthogonal planes in 3D space.
"""
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import binary_erosion, binary_opening, binary_closing, generate_binary_structure, label

# ---- Load ----
ct_path = r"N:\SecondYear\CT_MASK_E\Train_CT\sub-verse504_dir-iso_ct.nii.gz"
mask_path = r"N:\SecondYear\CT_MASK_E\Train_Mask\sub-verse504_dir-iso_seg-vert_msk.nii.gz"

print("Loading...")
ct = nib.load(ct_path).get_fdata(dtype=np.float32)
mask = np.asarray(nib.load(mask_path).dataobj)

# ---- L1 ROI ----
l1_mask = (mask == 20)
coords = np.argwhere(l1_mask)
z_center = int(np.mean(coords[:, 2]))
x_center = int(np.mean(coords[:, 0]))
y_center = int(np.mean(coords[:, 1]))
half_w = 20
z_min = max(0, z_center - half_w)
z_max = min(ct.shape[2] - 1, z_center + half_w)

print(f"L1 centroid: ({x_center}, {y_center}, {z_center})")

# ---- Crop to body region ----
body = ct[:, :, z_center] > -500
rows = np.any(body, axis=1)
cols = np.any(body, axis=0)
r0, r1 = np.where(rows)[0][[0, -1]]
c0, c1 = np.where(cols)[0][[0, -1]]
pad = 15
r0, r1 = max(0, r0-pad), min(ct.shape[0]-1, r1+pad)
c0, c1 = max(0, c0-pad), min(ct.shape[1]-1, c1+pad)

# Z crop: centroid +- 30 for sagittal/coronal
z_crop0 = max(0, z_center - 30)
z_crop1 = min(ct.shape[2]-1, z_center + 30)

# ---- Prepare slices ----
def normalize(img, vmin=-400, vmax=1500):
    return np.clip((img - vmin) / (vmax - vmin), 0, 1)

# Axial (z = z_center), cropped
axial = normalize(ct[r0:r1, c0:c1, z_center])
l1_axial = l1_mask[r0:r1, c0:c1, z_center]
ax_h, ax_w = axial.shape

# Sagittal (x = x_center), cropped
sag = normalize(ct[x_center, c0:c1, z_crop0:z_crop1])
l1_sag = l1_mask[x_center, c0:c1, z_crop0:z_crop1]
sag_h, sag_w = sag.shape  # (ny_crop, nz_crop)

# Coronal (y = y_center), cropped
cor = normalize(ct[r0:r1, y_center, z_crop0:z_crop1])
l1_cor = l1_mask[r0:r1, y_center, z_crop0:z_crop1]
cor_h, cor_w = cor.shape  # (nx_crop, nz_crop)

# ---- Make RGBA with L1 cyan highlight ----
def to_rgba(gray, l1, highlight_color=(0, 0.9, 0.9)):
    h, w = gray.shape
    rgba = np.zeros((h, w, 4))
    rgba[:,:,0] = gray
    rgba[:,:,1] = gray
    rgba[:,:,2] = gray
    rgba[:,:,3] = 1.0
    if l1.sum() > 0:
        from scipy.ndimage import binary_dilation
        contour = binary_dilation(l1, iterations=1) & (~l1)
        rgba[contour, 0] = highlight_color[0]
        rgba[contour, 1] = highlight_color[1]
        rgba[contour, 2] = highlight_color[2]
        # Also tint interior slightly
        rgba[l1, 0] = rgba[l1, 0] * 0.5
        rgba[l1, 1] = np.clip(rgba[l1, 1] * 0.5 + 0.4, 0, 1)
        rgba[l1, 2] = np.clip(rgba[l1, 2] * 0.5 + 0.4, 0, 1)
    return rgba

axial_rgba = to_rgba(axial, l1_axial)
sag_rgba = to_rgba(sag, l1_sag)
cor_rgba = to_rgba(cor, l1_cor)

# ---- 3D Plot ----
fig = plt.figure(figsize=(12, 10), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')

# Scale: map everything to similar coordinate range
# Axial sits on z=0 plane, x=[0, ax_w], y=[0, ax_h]
# Sagittal on x=0 wall (YZ plane)
# Coronal on y=ax_h wall (XZ plane)

z_scale = ax_h / (z_crop1 - z_crop0)  # scale z to match

# --- Axial plane at centroid height (middle of z range) ---
centroid_z = (z_center - z_crop0) * z_scale
xx_ax, yy_ax = np.meshgrid(np.arange(ax_w), np.arange(ax_h))
zz_ax = np.full_like(xx_ax, centroid_z, dtype=float)
ax.plot_surface(xx_ax, yy_ax, zz_ax, facecolors=axial_rgba,
                rstride=2, cstride=2, shade=False, antialiased=False)
# Red border for centroid slice
bx = [0, ax_w, ax_w, 0, 0]
by = [0, 0, ax_h, ax_h, 0]
bz = [centroid_z]*5
ax.plot(bx, by, bz, color='red', lw=3)

# --- Sagittal plane (x = 0, vertical wall on left) ---
# Maps to: x=0, y=[0..sag_h], z=[0..sag_w*z_scale]
yy_s, zz_s = np.meshgrid(np.arange(sag_h), np.arange(sag_w))
xx_s = np.zeros_like(yy_s, dtype=float)
# sag_rgba shape is (sag_h, sag_w, 4), we need (sag_w, sag_h, 4) for meshgrid
sag_colors = np.transpose(sag_rgba, (1, 0, 2))  # (sag_w, sag_h, 4)
ax.plot_surface(xx_s, yy_s, zz_s * z_scale, facecolors=sag_colors,
                rstride=2, cstride=2, shade=False, antialiased=False)
# Border
bx_s = [0, 0, 0, 0, 0]
by_s = [0, sag_h, sag_h, 0, 0]
bz_s = [0, 0, sag_w*z_scale, sag_w*z_scale, 0]
ax.plot(bx_s, by_s, bz_s, color='#45B7D1', lw=2)

# --- Coronal plane (y = ax_h, vertical wall on back) ---
# Maps to: y=ax_h, x=[0..cor_h], z=[0..cor_w*z_scale]
xx_c, zz_c = np.meshgrid(np.arange(cor_h), np.arange(cor_w))
yy_c = np.full_like(xx_c, ax_h, dtype=float)
cor_colors = np.transpose(cor_rgba, (1, 0, 2))  # (cor_w, cor_h, 4)
ax.plot_surface(xx_c, yy_c, zz_c * z_scale, facecolors=cor_colors,
                rstride=2, cstride=2, shade=False, antialiased=False)
# Border
bx_c = [0, cor_h, cor_h, 0, 0]
by_c = [ax_h, ax_h, ax_h, ax_h, ax_h]
bz_c = [0, 0, cor_w*z_scale, cor_w*z_scale, 0]
ax.plot(bx_c, by_c, bz_c, color='#45B7D1', lw=2)

# --- ROI band on sagittal and coronal ---
# ROI z range in local coords
roi_z_lo = (z_min - z_crop0) * z_scale
roi_z_hi = (z_max - z_crop0) * z_scale

# Sagittal ROI band
ax.plot([0, 0], [0, sag_h], [roi_z_lo, roi_z_lo], color='#45B7D1', lw=1.5, ls='--')
ax.plot([0, 0], [0, sag_h], [roi_z_hi, roi_z_hi], color='#45B7D1', lw=1.5, ls='--')
# Centroid line on sagittal
ax.plot([0, 0], [0, sag_h], [centroid_z, centroid_z], color='red', lw=2, ls='--')

# Coronal ROI band
ax.plot([0, cor_h], [ax_h, ax_h], [roi_z_lo, roi_z_lo], color='#45B7D1', lw=1.5, ls='--')
ax.plot([0, cor_h], [ax_h, ax_h], [roi_z_hi, roi_z_hi], color='#45B7D1', lw=1.5, ls='--')
ax.plot([0, cor_h], [ax_h, ax_h], [centroid_z, centroid_z], color='red', lw=2, ls='--')

# --- Annotations ---
ax.text(ax_w/2, -15, centroid_z, 'Axial (Centroid)', color='red',
        fontsize=11, fontweight='bold', ha='center')
ax.text(-10, sag_h/2, sag_w*z_scale/2, 'Sagittal', color='#45B7D1',
        fontsize=11, fontweight='bold', ha='center', rotation=90)
ax.text(cor_h/2, ax_h+15, cor_w*z_scale/2, 'Coronal', color='#45B7D1',
        fontsize=11, fontweight='bold', ha='center')

# ROI label
ax.text(-8, sag_h+5, (roi_z_lo+roi_z_hi)/2, f'ROI\n±{half_w}',
        color='#45B7D1', fontsize=10, fontweight='bold', ha='center')

# --- Styling ---
ax.set_xlim(-20, max(ax_w, cor_h) + 10)
ax.set_ylim(-10, ax_h + 20)
ax.set_zlim(-5, max(sag_w, cor_w) * z_scale + 5)
ax.set_axis_off()
ax.view_init(elev=25, azim=-55)

ax.set_title('L1 ROI Extraction - Orthogonal Views',
             fontsize=15, fontweight='bold', color='white', pad=15)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    plt.Rectangle((0,0),1,1, fc='red', alpha=0.5, ec='red', lw=2, label='Centroid slice'),
    Line2D([0],[0], color='#45B7D1', lw=2, ls='--', label='ROI boundary'),
    Line2D([0],[0], color='cyan', lw=3, label='L1 vertebra'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
          facecolor='#333333', edgecolor='white', labelcolor='white')

plt.tight_layout()
plt.savefig('H:/pythonproject/PHD_PROGRAM/GCN_Stress/Hu2EProcessing/test/roi_stacked_3d.png',
            dpi=200, bbox_inches='tight', facecolor='black')
print("Saved roi_stacked_3d.png")
plt.show()
