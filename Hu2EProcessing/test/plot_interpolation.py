"""
Temporal interpolation: rho_ash histograms.
Ref mean, 504 target (tp_119), tp_40 interpolated.
"""
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

root = r"N:\SecondYear\CT_MASK_E_PHANTOMLESS\Train_FEM"

def E_to_rho(E):
    return np.power(np.maximum(E, 0) / 10200.0, 1.0 / 2.01)

# ---- Load all subjects' tp_119 for ref ----
all_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
all_rho_flat = []
for subj in all_dirs:
    csv = os.path.join(root, subj, f"{subj}_element_E_tp_119.csv")
    if not os.path.exists(csv): continue
    E = pd.read_csv(csv)['E_MPa'].values
    all_rho_flat.append(E_to_rho(E))
    print(f"  Loaded {subj}: {len(E)} elements")

# Ref: concatenate all, then compute mean histogram
# Use fixed bins for all
bins = np.linspace(0, 1.6, 120)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Mean histogram across subjects
hist_per_subj = []
for rho in all_rho_flat:
    h, _ = np.histogram(rho, bins=bins, density=True)
    hist_per_subj.append(h)
ref_hist_mean = np.mean(hist_per_subj, axis=0)

# ---- 504 target (tp_119) ----
E_tgt = pd.read_csv(os.path.join(root, "504", "504_element_E_tp_119.csv"))['E_MPa'].values
rho_tgt = E_to_rho(E_tgt)
hist_tgt, _ = np.histogram(rho_tgt, bins=bins, density=True)

# ---- 504 tp_40 ----
E_40 = pd.read_csv(os.path.join(root, "504", "504_element_E_tp_40.csv"))['E_MPa'].values
rho_40 = E_to_rho(E_40)
hist_40, _ = np.histogram(rho_40, bins=bins, density=True)

print(f"\nRef: {len(all_rho_flat)} subjects")
print(f"504 tp_119: {len(E_tgt)} elements, rho_ash median={np.median(rho_tgt):.3f}")
print(f"504 tp_40:  {len(E_40)} elements, rho_ash median={np.median(rho_40):.3f}")

# ---- Figure ----
fig, ax = plt.subplots(figsize=(14, 10))

ax.fill_between(bin_centers, ref_hist_mean, alpha=0.3, color='#3498DB')
ax.plot(bin_centers, ref_hist_mean, '-', color='#3498DB', lw=3, label='Ref mean')

ax.fill_between(bin_centers, hist_tgt, alpha=0.2, color='#E67E22')
ax.plot(bin_centers, hist_tgt, '-', color='#E67E22', lw=3, label='504 Target (tp_119)')

ax.fill_between(bin_centers, hist_40, alpha=0.2, color='#E74C3C')
ax.plot(bin_centers, hist_40, '-', color='#E74C3C', lw=3.5, label=r'504 tp_40 ($\alpha$ = 0.34)')

# Median lines
ax.axvline(np.median(np.concatenate(all_rho_flat)), color='#3498DB', lw=2, ls='--', alpha=0.7)
ax.axvline(np.median(rho_tgt), color='#E67E22', lw=2, ls='--', alpha=0.7)
ax.axvline(np.median(rho_40), color='#E74C3C', lw=2, ls='--', alpha=0.7)

ax.set_xlabel(r'$\rho_{ash}$ (g/cm³)', fontsize=26)
ax.set_ylabel('Density', fontsize=26)
ax.set_title(r'$\rho_{ash}$ Distribution: Ref → tp_40 → Target', fontsize=28, fontweight='bold')
ax.legend(fontsize=20, loc='upper right')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=20)
ax.set_xlim(0, 1.6)

plt.tight_layout()
plt.savefig('H:/pythonproject/PHD_PROGRAM/GCN_Stress/Hu2EProcessing/test/interpolation_diagram.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved interpolation_diagram.png")
plt.show()
