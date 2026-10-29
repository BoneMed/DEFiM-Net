"""
Step 2: Two panels + formula chain below. Large fonts for PPT.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

# ---- Data ----
refs = {"air": -1001.6, "fat": -117.0, "muscle": 43.4}
targets = {"air": -1000.0, "fat": -100.0, "muscle": 40.0}

xs = [refs[k] for k in ("air", "fat", "muscle")]
ys = [targets[k] for k in ("air", "fat", "muscle")]
ws = [3.0, 1.0, 1.5]
x, y, w = np.array(xs), np.array(ys), np.array(ws)
A = np.column_stack([x, np.ones_like(x)])
Aw = A * np.sqrt(w[:, None])
yw = y * np.sqrt(w)
coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
cal_a, cal_b = coef

# ---- Figure ----
fig = plt.figure(figsize=(28, 18))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.3], hspace=0.22)
gs_top = gs[0].subgridspec(1, 2, wspace=0.3)

tissue_colors = {"air": "#3498DB", "fat": "#F1C40F", "muscle": "#E74C3C"}
hu_curve = np.linspace(-1100, 1800, 500)
hu_std_c = cal_a * hu_curve + cal_b

# ===== Panel (a): HU -> rho_QCT =====
ax = fig.add_subplot(gs_top[0])
hu_range = np.linspace(-1100, 1500, 500)
rho_qct_line = hu_range / 1000.0  # absorbed: rho_QCT = a*HU + b (a=1/1000, b=0 in canonical)

ax.plot(hu_range, rho_qct_line, '-', color='#3498DB', lw=3.5,
        label=r'$\rho_{QCT} = a \cdot HU + b$')

# Mark reference tissues (HU -> rho_QCT anchors)
tissue_hu = {"air": -1000.0, "fat": -100.0, "muscle": 40.0}
for k in ("air", "fat", "muscle"):
    hu = tissue_hu[k]
    rho = hu / 1000.0
    color = tissue_colors[k]
    ax.plot(hu, rho, 'o', color=color, markersize=18,
            markeredgecolor='black', markeredgewidth=2, zorder=5)
    xt = 60 if k != "air" else 80
    yt = 0.06
    ax.text(hu + xt, rho + yt, k.capitalize(), color=color,
            fontsize=22, fontweight='bold')

ax.set_xlabel(r'$HU$', fontsize=26)
ax.set_ylabel(r'$\rho_{QCT}$ (g/cm³)', fontsize=26)
ax.set_title(r'(a) $HU \rightarrow \rho_{QCT}$', fontsize=28, fontweight='bold')
ax.legend(fontsize=22, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.2, 1.6)
ax.tick_params(labelsize=22)

# ===== Panel (b): rho_QCT -> rho_ash =====
ax = fig.add_subplot(gs_top[1])
rho_qct_range = np.linspace(-1.1, 1.6, 500)
rho_ash_range = np.maximum(0.8772 * rho_qct_range + 0.07895, 0)

ax.plot(rho_qct_range, rho_ash_range, '-', color='#E67E22', lw=3.5,
        label=r'$\rho_{ash} = 0.877 \cdot \rho_{QCT} + 0.079$')

ax.set_xlabel(r'$\rho_{QCT}$ (g/cm³)', fontsize=26)
ax.set_ylabel(r'$\rho_{ash}$ (g/cm³)', fontsize=26)
ax.set_title(r'(b) $\rho_{QCT} \rightarrow \rho_{ash}$', fontsize=28, fontweight='bold')
ax.legend(fontsize=22, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(-1.2, 1.6)
ax.set_ylim(-0.1, 1.5)
ax.tick_params(labelsize=22)

# ===== Bottom: Formula chain =====
ax_bottom = fig.add_subplot(gs[1])
ax_bottom.set_xlim(0, 12)
ax_bottom.set_ylim(0, 2.5)
ax_bottom.axis('off')

formulas = [
    (r'$HU$', '#4ECDC4'),
    (r'$\rho_{QCT} = a \cdot HU + b$' + '\n(3-point weighted fit)', '#3498DB'),
    (r'$\rho_{ash} = 0.877 \cdot \rho_{QCT} + 0.079$' + '\n(Schileo 2008)', '#E67E22'),
    (r'$E = 10200 \cdot \rho_{ash}^{2.01}$' + '\n(Keller 1994)', '#E74C3C'),
]

box_w = 2.65
box_h = 1.6
gap = 0.3
x_start = 0.3

for i, (text, color) in enumerate(formulas):
    x_pos = x_start + i * (box_w + gap)
    rect = plt.Rectangle((x_pos, 0.45), box_w, box_h, facecolor=color, edgecolor='#2C3E50',
                          linewidth=3, alpha=0.85, zorder=3)
    ax_bottom.add_patch(rect)
    fontcolor = 'white' if color in ('#E74C3C', '#3498DB') else '#2C3E50'
    ax_bottom.text(x_pos + box_w/2, 1.25, text, ha='center', va='center',
                   fontsize=20, fontweight='bold', color=fontcolor, zorder=4)

    if i < len(formulas) - 1:
        ax_bottom.annotate('', xy=(x_pos + box_w + gap - 0.05, 1.25),
                           xytext=(x_pos + box_w + 0.05, 1.25),
                           arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=3.5))

plt.savefig('H:/pythonproject/PHD_PROGRAM/GCN_Stress/Hu2EProcessing/test/step2_combined.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved step2_combined.png")
plt.show()
