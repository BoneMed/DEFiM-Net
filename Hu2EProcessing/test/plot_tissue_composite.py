"""
Composite single-slice visualization: overlay Air / Fat / Muscle masks
on the same CT slice. No text, no titles, no legend.
"""
import glob as _glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import (binary_erosion, binary_opening, binary_closing,
                           binary_dilation, binary_fill_holes,
                           generate_binary_structure, label)

SUBJ = "514"

# ---- Load ----
ct_files = _glob.glob(rf"N:\SecondYear\CT_MASK_E\Train_CT\*{SUBJ}*ct.nii.gz")
ct_path = ct_files[0]
mask_path = ct_path.replace("Train_CT", "Train_Mask").replace("_ct.nii.gz", "_seg-vert_msk.nii.gz")

print(f"Loading subject {SUBJ}...")
ct_nii   = nib.as_closest_canonical(nib.load(ct_path))
ct       = ct_nii.get_fdata(dtype=np.float32)
mask     = np.asarray(nib.as_closest_canonical(nib.load(mask_path)).dataobj)
spacing  = np.abs(np.diag(ct_nii.affine)[:3])

# ---- L1 ROI (vertebra label 20) ----
l1_mask = (mask == 20)
coords  = np.argwhere(l1_mask)
z_center = int(np.mean(coords[:, 2]))
half_w = 20
z_min = max(0, z_center - half_w)
z_max = min(ct.shape[2] - 1, z_center + half_w)

roi   = ct[:, :, z_min:z_max + 1].copy()
z_local = z_center - z_min

# ---- Body / Shell / Deep ----
body = roi > -500
body = binary_closing(body, structure=generate_binary_structure(3, 1))
labeled, num = label(body)
if num > 1:
    counts = np.bincount(labeled.ravel()); counts[0] = 0
    body = labeled == np.argmax(counts)

# Fill internal air cavities (bowel / stomach gas) so that the "shell"
# below refers ONLY to the subcutaneous outer rim, never the boundary
# around internal gas.
body_solid = np.stack([binary_fill_holes(body[:, :, k]) for k in range(body.shape[2])],
                      axis=-1)

shell   = body_solid & (~binary_erosion(body_solid, iterations=8))
deep    = binary_erosion(body_solid, iterations=5)
l1_roi  = (mask[:, :, z_min:z_max + 1] == 20)

# ---- Paravertebral zone (posterior only, 10-50 mm of L1 centroid) ----
l1_coords_roi = np.argwhere(l1_roi)
l1_cx = int(np.mean(l1_coords_roi[:, 0]))
l1_cy = int(np.mean(l1_coords_roi[:, 1]))

yy, xx = np.meshgrid(np.arange(roi.shape[1]), np.arange(roi.shape[0]))
dist_from_l1 = np.sqrt(((xx - l1_cx) * spacing[0]) ** 2 +
                      ((yy - l1_cy) * spacing[1]) ** 2)
dist_3d = np.repeat(dist_from_l1[:, :, np.newaxis], roi.shape[2], axis=2)

# Tighter posterior zone: strictly behind vertebra, away from bony processes
posterior_3d = np.broadcast_to((yy < l1_cy - 15)[:, :, np.newaxis], roi.shape)
lateral_mm = np.abs(xx - l1_cx) * spacing[0]
lateral_3d = np.broadcast_to((lateral_mm < 30)[:, :, np.newaxis], roi.shape)

# Keep only the strictly-posterior lumbodorsal muscle band (15-45 mm from L1 centroid)
para_mask = ((dist_3d > 15) & (dist_3d < 45) &
             posterior_3d & lateral_3d)

# Expand L1 vertebra mask by a safety buffer to also exclude pedicles / laminae /
# spinous processes that are anatomically part of the vertebra but may not be
# fully contained in the segmented L1 label.
l1_buffer = binary_dilation(l1_roi, iterations=4)

# ---- Tissue masks ----
# AIR: keep only EXTERNAL air (connected to image boundary), not internal cavities
air_raw = (roi < -800) & (~body)
lab, nlab = label(air_raw)
# Find labels that touch any image border -> that's the external-air component
border_labels = set()
# X-min / X-max faces
for s in (lab[0, :, :], lab[-1, :, :], lab[:, 0, :], lab[:, -1, :],
          lab[:, :, 0], lab[:, :, -1]):
    border_labels.update(np.unique(s))
border_labels.discard(0)
air_m = np.isin(lab, list(border_labels)) if border_labels else np.zeros_like(air_raw)

fat_m = (-200 < roi) & (roi < -50) & shell & (~l1_buffer)
fat_m = binary_opening(fat_m)

mus_m = (0 < roi) & (roi < 100) & deep & (~l1_buffer) & para_mask
# Keep only the largest connected component (drops small scattered voxels)
lab_m, nlab_m = label(mus_m)
if nlab_m > 0:
    counts = np.bincount(lab_m.ravel()); counts[0] = 0
    mus_m = lab_m == np.argmax(counts)

# ---- 2D slices at centroid ----
ct_slice = roi[:, :, z_local]
air_2d = air_m[:, :, z_local]
fat_2d = fat_m[:, :, z_local]
mus_2d = mus_m[:, :, z_local]

# ============================================================
# 3-panel side-by-side: Air | Fat | Muscle (same slice, no text)
# ============================================================
vmin, vmax = -400, 1500
AIR_COLOR    = '#3498DB'   # blue
FAT_COLOR    = '#F1C40F'   # yellow
MUSCLE_COLOR = '#E74C3C'   # red

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for ax, mask_2d, color in [
    (axes[0], air_2d, AIR_COLOR),
    (axes[1], fat_2d, FAT_COLOR),
    (axes[2], mus_2d, MUSCLE_COLOR),
]:
    ax.imshow(ct_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    if mask_2d.any():
        ax.contourf(mask_2d.T, levels=[0.5, 1.5], colors=[color],
                    alpha=0.6, origin='lower')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(''); ax.axis('off')

out = rf"H:\pythonproject\PHD_PROGRAM\GCN_Stress\Hu2EProcessing\test\tissue_composite_{SUBJ}.png"
plt.subplots_adjust(wspace=0.01, left=0, right=1, top=1, bottom=0)
plt.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0, facecolor='black')
print(f"Saved: {out}")
