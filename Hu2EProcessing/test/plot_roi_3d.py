"""
3D visualization of L1 ROI extraction for PPT.
Shows CT volume with ROI slab highlighted, centroid slice in red.
"""
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---- Load data ----
ct_path = r"N:\SecondYear\CT_MASK_E\Train_CT\sub-verse504_dir-iso_ct.nii.gz"
mask_path = r"N:\SecondYear\CT_MASK_E\Train_Mask\sub-verse504_dir-iso_seg-vert_msk.nii.gz"

print("Loading CT...")
ct_img = nib.load(ct_path)
ct = ct_img.get_fdata(dtype=np.float32)
affine = ct_img.affine
mask = np.asarray(nib.load(mask_path).dataobj)

print(f"CT shape: {ct.shape}")

# ---- Find L1 ----
l1_mask = (mask == 20)
coords = np.argwhere(l1_mask)
z_center = int(np.mean(coords[:, 2]))
half_w = 20
z_min = max(0, z_center - half_w)
z_max = min(ct.shape[2] - 1, z_center + half_w)

print(f"L1 centroid: z={z_center}, ROI: [{z_min}, {z_max}]")

# ---- Get voxel spacing ----
spacing = np.abs(np.diag(affine)[:3])
print(f"Spacing: {spacing}")

# ---- Build figure ----
fig = plt.figure(figsize=(14, 8))

# === Left: 3D volume with ROI slab ===
ax1 = fig.add_subplot(121, projection='3d')

nx, ny, nz = ct.shape
sx, sy, sz = spacing

# Draw volume outline (wireframe box)
x_len, y_len, z_len = nx * sx, ny * sy, nz * sz

# Volume edges
def draw_box(ax, x0, x1, y0, y1, z0, z1, color='gray', alpha=0.3, lw=0.8):
    verts = [
        [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],  # bottom
        [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],  # top
    ]
    for v in verts:
        poly = Poly3DCollection([v], alpha=alpha*0.3, facecolor=color, edgecolor=color, lw=lw)
        ax.add_collection3d(poly)
    # vertical edges
    for x, y in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]:
        ax.plot([x,x],[y,y],[z0,z1], color=color, lw=lw, alpha=alpha)

# Full volume box (light gray)
draw_box(ax1, 0, x_len, 0, y_len, 0, z_len, color='gray', alpha=0.2)

# ROI slab (blue, semi-transparent)
z_roi_min = z_min * sz
z_roi_max = (z_max + 1) * sz
roi_verts_bottom = [[0,0,z_roi_min],[x_len,0,z_roi_min],[x_len,y_len,z_roi_min],[0,y_len,z_roi_min]]
roi_verts_top = [[0,0,z_roi_max],[x_len,0,z_roi_max],[x_len,y_len,z_roi_max],[0,y_len,z_roi_max]]
roi_verts_front = [[0,0,z_roi_min],[x_len,0,z_roi_min],[x_len,0,z_roi_max],[0,0,z_roi_max]]
roi_verts_back = [[0,y_len,z_roi_min],[x_len,y_len,z_roi_min],[x_len,y_len,z_roi_max],[0,y_len,z_roi_max]]
roi_verts_left = [[0,0,z_roi_min],[0,y_len,z_roi_min],[0,y_len,z_roi_max],[0,0,z_roi_max]]
roi_verts_right = [[x_len,0,z_roi_min],[x_len,y_len,z_roi_min],[x_len,y_len,z_roi_max],[x_len,0,z_roi_max]]

for v in [roi_verts_bottom, roi_verts_top, roi_verts_front, roi_verts_back, roi_verts_left, roi_verts_right]:
    poly = Poly3DCollection([v], alpha=0.12, facecolor='#45B7D1', edgecolor='#45B7D1', lw=1)
    ax1.add_collection3d(poly)

# Centroid slice (red plane)
z_c = z_center * sz
center_verts = [[0,0,z_c],[x_len,0,z_c],[x_len,y_len,z_c],[0,y_len,z_c]]
poly_center = Poly3DCollection([center_verts], alpha=0.35, facecolor='red', edgecolor='red', lw=2)
ax1.add_collection3d(poly_center)

# L1 vertebra body as point cloud (subsample)
l1_coords = coords[::20]  # subsample
l1_xyz = l1_coords * spacing
ax1.scatter(l1_xyz[:, 0], l1_xyz[:, 1], l1_xyz[:, 2],
            c='cyan', s=0.5, alpha=0.4, label='L1 vertebra')

# Labels
ax1.set_xlabel('X (mm)', fontsize=9)
ax1.set_ylabel('Y (mm)', fontsize=9)
ax1.set_zlabel('Z (mm)', fontsize=9)
ax1.set_title('CT Volume + L1 ROI Slab', fontsize=12, fontweight='bold')

# Annotations
ax1.text(x_len/2, -20, z_c, f'Centroid\nz={z_center}', color='red',
         fontsize=9, fontweight='bold', ha='center')
ax1.text(x_len + 10, y_len/2, (z_roi_min+z_roi_max)/2,
         f'ROI\n+/-{half_w}\nslices', color='#45B7D1',
         fontsize=9, fontweight='bold', ha='center')

ax1.set_xlim(0, x_len)
ax1.set_ylim(0, y_len)
ax1.set_zlim(0, z_len)
ax1.view_init(elev=20, azim=-60)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='gray', lw=2, label='CT volume'),
    plt.Rectangle((0,0), 1, 1, fc='#45B7D1', alpha=0.3, label='ROI slab'),
    plt.Rectangle((0,0), 1, 1, fc='red', alpha=0.4, label='Centroid slice'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan',
           markersize=6, label='L1 voxels'),
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=8)

# === Right: 3 representative slices ===
slices_to_show = [z_min, z_center, z_max]
slice_labels = [f'z={z_min}\n(ROI bottom)', f'z={z_center}\n(Centroid)', f'z={z_max}\n(ROI top)']
border_colors = ['#45B7D1', 'red', '#45B7D1']

for i, (zs, label, bc) in enumerate(zip(slices_to_show, slice_labels, border_colors)):
    ax = fig.add_subplot(3, 2, 2*(i+1), aspect='equal')
    ct_slice = ct[:, :, zs]
    l1_slice = (mask[:, :, zs] == 20)

    ax.imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-400, vmax=1500)
    if l1_slice.sum() > 0:
        ax.contour(l1_slice.T, levels=[0.5], colors='cyan', linewidths=1.5)

    # Border
    for spine in ax.spines.values():
        spine.set_edgecolor(bc)
        spine.set_linewidth(3)

    ax.set_title(label, fontsize=10, color=bc, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Step 1: L1 ROI Extraction (Subject 504)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('H:/pythonproject/PHD_PROGRAM/GCN_Stress/Hu2EProcessing/test/roi_extraction_3d.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print("Saved roi_extraction_3d.png")
plt.show()
