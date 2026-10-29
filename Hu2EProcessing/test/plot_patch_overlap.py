"""
3D patch overlap visualization for PPT.
Shows grid volume with patches cut at 30% overlap.
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

plt.rcParams.update({'font.size': 14})

# ---- Parameters ----
R = 256       # Grid resolution
P = 128       # Patch size
overlap = 0.30
stride = int(round(P * (1 - overlap)))  # 90
print(f"Grid={R}, Patch={P}, Stride={stride}, Overlap={overlap*100:.0f}%")

# Compute patch starts
starts = list(range(0, R - P + 1, stride))
if starts[-1] + P < R:
    starts.append(R - P)
print(f"Patch starts: {starts}")
print(f"Patches per axis: {len(starts)}, total: {len(starts)**3}")

# ---- Figure ----
fig = plt.figure(figsize=(18, 9))

# ========== Panel 1: Full 3D volume with all patches ==========
ax1 = fig.add_subplot(121, projection='3d')

def draw_cube(ax, origin, size, color, alpha=0.15, edge_alpha=0.6, lw=1.2):
    x0, y0, z0 = origin
    sx, sy, sz = size, size, size
    verts = np.array([
        [x0,     y0,     z0],      [x0+sx, y0,     z0],
        [x0+sx, y0+sy, z0],        [x0,     y0+sy, z0],
        [x0,     y0,     z0+sz],   [x0+sx, y0,     z0+sz],
        [x0+sx, y0+sy, z0+sz],     [x0,     y0+sy, z0+sz],
    ])
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],  # bottom
        [verts[4], verts[5], verts[6], verts[7]],  # top
        [verts[0], verts[1], verts[5], verts[4]],  # front
        [verts[2], verts[3], verts[7], verts[6]],  # back
        [verts[1], verts[2], verts[6], verts[5]],  # right
        [verts[0], verts[3], verts[7], verts[4]],  # left
    ]
    poly = Poly3DCollection(faces, facecolor=color, alpha=alpha,
                             edgecolor=color, linewidth=lw)
    ax.add_collection3d(poly)

# Main volume (wireframe)
draw_cube(ax1, (0, 0, 0), R, 'gray', alpha=0.05, lw=2)

# Draw all patch cubes with different colors per axis grouping
colors = plt.cm.tab20(np.linspace(0, 1, len(starts)**3))
idx = 0
for iz, sz in enumerate(starts):
    for iy, sy in enumerate(starts):
        for ix, sx in enumerate(starts):
            # Color based on corner position for variety
            col = plt.cm.viridis((ix + iy + iz) / (3 * (len(starts) - 1) + 0.01))
            draw_cube(ax1, (sx, sy, sz), P, col, alpha=0.08, lw=0.8)
            idx += 1

ax1.set_xlim(0, R); ax1.set_ylim(0, R); ax1.set_zlim(0, R)
ax1.set_xlabel('X', fontsize=13); ax1.set_ylabel('Y', fontsize=13); ax1.set_zlabel('Z', fontsize=13)
ax1.set_title(f'Full Volume: {len(starts)**3} Patches\n'
              f'Grid {R}³, Patch {P}³, Stride {stride}',
              fontsize=16, fontweight='bold')
ax1.view_init(elev=20, azim=-55)

# ========== Panel 2: Zoomed view showing 2 overlapping patches ==========
ax2 = fig.add_subplot(122, projection='3d')

# Show 2 adjacent patches with overlap
draw_cube(ax2, (0, 0, 0), P, '#3498DB', alpha=0.2, lw=2.5)
draw_cube(ax2, (stride, 0, 0), P, '#E74C3C', alpha=0.2, lw=2.5)

# Highlight overlap region
overlap_size = P - stride
draw_cube(ax2, (stride, 0, 0), overlap_size, '#F39C12', alpha=0.45, lw=2)

# Labels
ax2.text(P/2, -15, -5, 'Patch A', color='#3498DB', fontsize=14, fontweight='bold')
ax2.text(stride + P/2, -15, -5, 'Patch B', color='#E74C3C', fontsize=14, fontweight='bold')
ax2.text(stride + overlap_size/2, P + 10, P + 20, 'Overlap\nRegion\n(30%)',
         color='#D35400', fontsize=13, fontweight='bold', ha='center')

# Stride arrow
ax2.plot([0, stride], [-30, -30], [P+10, P+10], color='black', lw=2)
ax2.plot([0, 0], [-33, -27], [P+10, P+10], color='black', lw=2)
ax2.plot([stride, stride], [-33, -27], [P+10, P+10], color='black', lw=2)
ax2.text(stride/2, -35, P+25, f'stride = {stride}\n(0.7 × P)',
         color='black', fontsize=12, ha='center')

# Patch size arrow
ax2.plot([0, P], [-50, -50], [-10, -10], color='gray', lw=2, ls='--')
ax2.text(P/2, -55, -25, f'P = {P}', color='gray', fontsize=12, ha='center')

ax2.set_xlim(-10, P + stride + 10)
ax2.set_ylim(-60, P + 20)
ax2.set_zlim(-30, P + 40)
ax2.set_xlabel('X', fontsize=13); ax2.set_ylabel('Y', fontsize=13); ax2.set_zlabel('Z', fontsize=13)
ax2.set_title(f'Overlap Detail: Stride = {stride}\n'
              f'Overlap = {overlap_size} ({overlap*100:.0f}% of patch)',
              fontsize=16, fontweight='bold')
ax2.view_init(elev=22, azim=-60)

plt.suptitle('3D Patch Extraction with 30% Overlap',
             fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('H:/pythonproject/PHD_PROGRAM/GCN_Stress/Hu2EProcessing/test/patch_overlap.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved patch_overlap.png")
plt.show()
