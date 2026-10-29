"""
Plot the 3 nan cases (561, 588, 605) to see why erector spinae muscle extraction fails.
"""
import numpy as np
import nibabel as nib
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_opening, binary_closing, generate_binary_structure, label

ct_dir = r"N:\SecondYear\CT_MASK_E\Train_CT"
mask_dir = r"N:\SecondYear\CT_MASK_E\Train_Mask"

nan_subjs = ["561", "588", "605"]

fig, axes = plt.subplots(3, 3, figsize=(18, 18))

for row, subj in enumerate(nan_subjs):
    ct_files = glob.glob(f"{ct_dir}/*{subj}*ct.nii.gz")
    ct_path = ct_files[0]
    mask_path = ct_path.replace(ct_dir, mask_dir).replace("_ct.nii.gz", "_seg-vert_msk.nii.gz")

    ct = nib.load(ct_path).get_fdata(dtype=np.float32)
    msk = np.asarray(nib.load(mask_path).dataobj)
    spacing = np.abs(np.diag(nib.load(ct_path).affine)[:3])

    l1 = (msk == 20)
    coords = np.argwhere(l1)
    zc = int(np.mean(coords[:, 2]))
    z0, z1 = max(0, zc - 20), min(ct.shape[2]-1, zc + 20)

    roi = ct[:, :, z0:z1+1]
    z_local = zc - z0

    body = roi > -500
    body = binary_closing(body, structure=generate_binary_structure(3, 1))
    lab, num = label(body)
    if num > 1:
        c = np.bincount(lab.ravel()); c[0] = 0
        body = lab == np.argmax(c)

    deep = binary_erosion(body, iterations=5)
    l1_roi = (msk[:, :, z0:z1+1] == 20)

    l1c = np.argwhere(l1_roi)
    l1_cx, l1_cy = int(np.mean(l1c[:, 0])), int(np.mean(l1c[:, 1]))

    yy, xx = np.meshgrid(np.arange(roi.shape[1]), np.arange(roi.shape[0]))
    dist = np.sqrt(((xx-l1_cx)*spacing[0])**2 + ((yy-l1_cy)*spacing[1])**2)
    dist_3d = np.broadcast_to(dist[:,:,np.newaxis], roi.shape)
    posterior_3d = np.broadcast_to((yy < l1_cy - 5)[:,:,np.newaxis], roi.shape)
    lateral_mm = np.abs(xx - l1_cx) * spacing[0]
    lateral_3d = np.broadcast_to((lateral_mm < 40)[:,:,np.newaxis], roi.shape)
    para = (dist_3d > 10) & (dist_3d < 50) & posterior_3d & lateral_3d

    mus_old = (0 < roi) & (roi < 100) & deep & (~l1_roi)
    mus_new = mus_old & para

    ct_slice = roi[:, :, z_local]
    l1_2d = l1_roi[:, :, z_local]
    deep_2d = deep[:, :, z_local]
    mus_old_2d = mus_old[:, :, z_local]
    mus_new_2d = mus_new[:, :, z_local]
    para_2d = para[:, :, z_local]

    n_old = mus_old.sum()
    n_new = mus_new.sum()

    # Panel 1: CT + L1
    ax = axes[row, 0]
    ax.imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-400, vmax=1500)
    ax.contour(l1_2d.T, levels=[0.5], colors='cyan', linewidths=2)
    ax.plot(l1_cx, l1_cy, 'r+', markersize=15, markeredgewidth=2)
    ax.set_title(f'[{subj}] CT z={zc}\nL1 centroid=({l1_cx},{l1_cy})', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Panel 2: Old muscle (whole deep)
    ax = axes[row, 1]
    ax.imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-400, vmax=1500)
    if mus_old_2d.sum() > 0:
        ax.contourf(mus_old_2d.T, levels=[0.5, 1.5], colors=['#E74C3C'], alpha=0.4, origin='lower')
    ax.contour(l1_2d.T, levels=[0.5], colors='cyan', linewidths=1)
    ax.set_title(f'OLD muscle (whole deep)\n{n_old} voxels', fontsize=11, fontweight='bold', color='#E74C3C')
    ax.axis('off')

    # Panel 3: New muscle (erector spinae) + search zone
    ax = axes[row, 2]
    ax.imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-400, vmax=1500)
    if para_2d.sum() > 0:
        ax.contour(para_2d.T, levels=[0.5], colors='white', linewidths=1, linestyles=':', origin='lower')
    if mus_new_2d.sum() > 0:
        ax.contourf(mus_new_2d.T, levels=[0.5, 1.5], colors=['#E74C3C'], alpha=0.5, origin='lower')
    ax.contour(l1_2d.T, levels=[0.5], colors='cyan', linewidths=1)
    ax.plot(l1_cx, l1_cy, 'r+', markersize=15, markeredgewidth=2)
    ax.set_title(f'NEW muscle (erector spinae)\n{n_new} voxels (min=500)', fontsize=11,
                 fontweight='bold', color='red' if n_new < 500 else '#E74C3C')
    ax.axis('off')

plt.suptitle('NaN Cases: Why Erector Spinae Extraction Fails', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('H:/pythonproject/PHD_PROGRAM/GCN_Stress/Hu2EProcessing/test/nan_cases.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print("Saved nan_cases.png")
plt.show()
