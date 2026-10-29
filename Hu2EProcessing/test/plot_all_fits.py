"""
HU standardization plot:
  - Ref line = average fit of all subjects' tp_119
  - 504's 119 timepoints as individual lines
  - tp_40 highlighted in red
"""
import numpy as np
import pandas as pd
import glob, os
import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_opening, binary_closing, generate_binary_structure, label
plt.rcParams.update({'font.size': 20})

ct_dir = r"N:\SecondYear\CT_MASK_E\Train_CT"
mask_dir = r"N:\SecondYear\CT_MASK_E\Train_Mask"
out_root = r"N:\SecondYear\CT_MASK_E_PHANTOMLESS\Train_FEM"

targets = {"air": -1000.0, "fat": -100.0, "muscle": 40.0}
weights_dict = {"air": 3.0, "fat": 1.0, "muscle": 1.5}

def robust_mean(x, trim=0.1):
    x = np.sort(x.ravel().astype(np.float32))
    n = len(x)
    if n < 50: return float(np.median(x))
    lo, hi = int(trim*n), int((1-trim)*n)
    return float(np.mean(x[lo:hi]))

def get_fit(subj):
    """Get HU fit (a, b) and refs for a subject."""
    ct_files = glob.glob(os.path.join(ct_dir, f'*{subj}*ct.nii.gz'))
    if not ct_files: return None
    ct_path = ct_files[0]
    mask_path = ct_path.replace(ct_dir, mask_dir).replace('_ct.nii.gz', '_seg-vert_msk.nii.gz')
    if not os.path.exists(mask_path): return None

    ct_img = nib.as_closest_canonical(nib.load(ct_path))
    ct = ct_img.get_fdata(dtype=np.float32)
    msk = np.asarray(nib.as_closest_canonical(nib.load(mask_path)).dataobj)
    spacing = np.abs(np.diag(ct_img.affine)[:3])

    l1 = (msk == 20)
    if l1.sum() == 0: return None
    coords = np.argwhere(l1)
    zc = int(np.mean(coords[:, 2]))
    z0, z1 = max(0, zc-20), min(ct.shape[2]-1, zc+20)

    roi = ct[:, :, z0:z1+1]
    body = roi > -500
    body = binary_closing(body, structure=generate_binary_structure(3, 1))
    lab, num = label(body)
    if num > 1:
        c = np.bincount(lab.ravel()); c[0] = 0
        body = lab == np.argmax(c)

    shell = body & (~binary_erosion(body, iterations=8))
    deep = binary_erosion(body, iterations=5)
    l1_roi = (msk[:, :, z0:z1+1] == 20)

    l1c = np.argwhere(l1_roi)
    l1_cx, l1_cy = int(np.mean(l1c[:,0])), int(np.mean(l1c[:,1]))
    yy, xx = np.meshgrid(np.arange(roi.shape[1]), np.arange(roi.shape[0]))
    dist = np.sqrt(((xx-l1_cx)*spacing[0])**2 + ((yy-l1_cy)*spacing[1])**2)
    dist_3d = np.broadcast_to(dist[:,:,np.newaxis], roi.shape)
    posterior_3d = np.broadcast_to((yy < l1_cy - 5)[:,:,np.newaxis], roi.shape)
    lateral_3d = np.broadcast_to((np.abs(xx-l1_cx)*spacing[0] < 40)[:,:,np.newaxis], roi.shape)
    para = (dist_3d > 10) & (dist_3d < 50) & posterior_3d & lateral_3d

    refs = {}
    air_m = (roi < -800) & (~body)
    if air_m.sum() > 500: refs["air"] = robust_mean(roi[air_m])
    fat_m = (-200 < roi) & (roi < -50) & shell & (~l1_roi)
    fat_m = binary_opening(fat_m)
    if fat_m.sum() > 500: refs["fat"] = robust_mean(roi[fat_m])
    mus_m = (0 < roi) & (roi < 100) & deep & (~l1_roi) & para
    if mus_m.sum() > 500: refs["muscle"] = robust_mean(roi[mus_m])

    xs, ys, ws = [], [], []
    for k in ("air", "fat", "muscle"):
        if k in refs:
            xs.append(refs[k]); ys.append(targets[k]); ws.append(weights_dict[k])
    if len(xs) < 2: return None

    x, y, w = np.array(xs), np.array(ys), np.array(ws)
    A = np.column_stack([x, np.ones_like(x)])
    Aw = A * np.sqrt(w[:, None])
    yw = y * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    a, b = coef
    if not np.isfinite(a) or a <= 0: return None
    return a, b, refs

# ---- Get all subjects' fits for ref average ----
all_dirs = sorted([d for d in os.listdir(out_root) if os.path.isdir(os.path.join(out_root, d))])
all_a, all_b = [], []
print("Computing reference average from all subjects' tp_119 fits...")
for subj in all_dirs:
    result = get_fit(subj)
    if result:
        a, b, refs = result
        all_a.append(a)
        all_b.append(b)
        print(f"  {subj}: a={a:.6f}, b={b:.2f}")

ref_a = np.mean(all_a)
ref_b = np.mean(all_b)
print(f"\nRef average: a={ref_a:.6f}, b={ref_b:.2f} (from {len(all_a)} subjects)")

# ---- Get 504's fit ----
result_504 = get_fit("504")
a_504, b_504, refs_504 = result_504

# ---- Figure ----
fig, ax = plt.subplots(figsize=(16, 14))

hu_range = np.linspace(-1100, 100, 200)
highlight_tp = 40

# Ref average line (thick blue)
hu_ref_line = ref_a * hu_range + ref_b
ax.plot(hu_range, hu_ref_line, '-', color='#3498DB', lw=4,
        label=f'Ref mean: a={ref_a:.4f}, b={ref_b:.1f}', zorder=6)

# 504's tp lines: interpolate between ref and 504's fit
# tp_i: a_tp = (1-alpha)*ref_a + alpha*a_504, b_tp = (1-alpha)*ref_b + alpha*b_504
for tp in range(1, 120):
    alpha = tp / 119.0
    a_tp = (1 - alpha) * ref_a + alpha * a_504
    b_tp = (1 - alpha) * ref_b + alpha * b_504
    hu_tp_line = a_tp * hu_range + b_tp

    if tp == highlight_tp:
        ax.plot(hu_range, hu_tp_line, '-', color='#E74C3C', lw=3.5, zorder=7,
                label=f'tp_{tp} (α={alpha:.2f})')
    elif tp == 119:
        ax.plot(hu_range, hu_tp_line, '-', color='#E67E22', lw=3, zorder=5,
                label=f'504 tp_119 (Target): a={a_504:.4f}, b={b_504:.1f}')
    elif tp == 1:
        ax.plot(hu_range, hu_tp_line, '--', color='#85C1E9', lw=1.5, alpha=0.6, zorder=2)
    else:
        gray = 0.55 + 0.3 * alpha
        ax.plot(hu_range, hu_tp_line, '-', color=str(gray), lw=0.4, alpha=0.3, zorder=1)

# Identity line
ax.plot(hu_range, hu_range, '--', color='gray', lw=2, alpha=0.4, label='Identity')

# Canonical target lines
tc = {"air": "#3498DB", "fat": "#F1C40F", "muscle": "#E74C3C"}
for k, tgt in targets.items():
    ax.axhline(tgt, color=tc[k], lw=1.5, ls=':', alpha=0.5)
    ax.text(105, tgt + 15, f'{k.capitalize()} = {tgt:.0f}',
            fontsize=20, color=tc[k], fontweight='bold')

# 504's ref points
for k in ("air", "fat", "muscle"):
    if k in refs_504:
        raw = refs_504[k]
        std = a_504 * raw + b_504
        ax.plot(raw, std, 'o', color=tc[k], markersize=14,
                markeredgecolor='black', markeredgewidth=1.5, zorder=8)

# Arrow to tp_40
alpha_40 = highlight_tp / 119.0
a_40 = (1 - alpha_40) * ref_a + alpha_40 * a_504
b_40 = (1 - alpha_40) * ref_b + alpha_40 * b_504
hu_at = -500
y_at = a_40 * hu_at + b_40
ax.annotate(f'tp_{highlight_tp}\nα = {alpha_40:.2f}',
            xy=(hu_at, y_at), xytext=(hu_at + 150, y_at - 100),
            fontsize=24, fontweight='bold', color='#E74C3C',
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5))

ax.set_xlabel('HU_raw (measured)', fontsize=28)
ax.set_ylabel('HU_std (corrected)', fontsize=28)
ax.set_title('HU Standardization: Ref → 504 (119 Timepoints)', fontsize=30, fontweight='bold')
ax.legend(fontsize=20, loc='lower right')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=22)

plt.tight_layout()
plt.savefig('H:/pythonproject/PHD_PROGRAM/GCN_Stress/Hu2EProcessing/test/all_fits.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved all_fits.png")
plt.show()
