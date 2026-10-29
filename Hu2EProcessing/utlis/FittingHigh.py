import numpy as np
import trimesh
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import pyvista as pv


# ============================================================
# 1️⃣ 读取 STL 并 PCA 对齐
# ============================================================
def load_stl_points(stl_path):
    mesh = trimesh.load_mesh(stl_path)
    return mesh.vertices.copy()

def align_pca(points):
    """PCA 对齐，让主方向成为新的 XYZ 轴"""
    pca = PCA(n_components=3)
    pca.fit(points)
    aligned = (points - pca.mean_) @ pca.components_.T
    return aligned, pca


# ============================================================
# 2️⃣ 选择椎体区域：按 Y 轴排序取前 40%
# ============================================================
def select_body_region_by_y(points, ratio=0.4):
    """
    按 Y 从大到小排序，取前 ratio 比例的点。
    一般用于去掉椎弓，只保留椎体区域。
    """
    y_sorted = np.argsort(points[:, 0])  # 从大到小排序
    n_select = int(len(points) * ratio)
    selected = points[y_sorted[:n_select]]
    print(f"[INFO] 已筛选出 {n_select}/{len(points)} 个点 ({ratio*100:.0f}%) 作为椎体区域")
    return selected


# ============================================================
# 3️⃣ 提取上下端板点集
# ============================================================
def extract_endplates(points_aligned, top_ratio=0.1, bottom_ratio=0.1, radius_mm=10.0):
    """
    从点云中提取上下端板，并只保留中心区域（半径 ≤ radius_mm）。
    参数:
        top_ratio, bottom_ratio : 按 z 取端板厚度比例
        radius_mm : 以端板中心为圆心保留的半径 (mm)
    返回:
        top_pts, bot_pts
    """
    z_values = points_aligned[:, 2]
    z_min, z_max = np.min(z_values), np.max(z_values)
    z_range = z_max - z_min

    # Step 1: 按 z 取上下层
    top_thresh = z_max - z_range * top_ratio
    bot_thresh = z_min + z_range * bottom_ratio
    top_pts = points_aligned[z_values >= top_thresh]
    bot_pts = points_aligned[z_values <= bot_thresh]

    # Step 2: 分别计算质心
    top_center = np.mean(top_pts, axis=0)
    bot_center = np.mean(bot_pts, axis=0)

    # Step 3: 按半径筛选（以中心为圆心）
    top_dist = np.linalg.norm(top_pts - top_center, axis=1)
    bot_dist = np.linalg.norm(bot_pts - bot_center, axis=1)
    top_pts = top_pts[top_dist <= radius_mm]
    bot_pts = bot_pts[bot_dist <= radius_mm]

    print(f"[INFO] 上端板中心: {top_center}, 点数: {len(top_pts)}")
    print(f"[INFO] 下端板中心: {bot_center}, 点数: {len(bot_pts)}")
    return top_pts, bot_pts


# ============================================================
# 4️⃣ 平面拟合 + 最小垂直厚度计算
# ============================================================
def fit_plane_svd(X):
    """拟合平面，返回质心 c 和单位法向 n"""
    c = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - c, full_matrices=False)
    n = Vt[-1]
    return c, n / np.linalg.norm(n)


def unify_normal(n_top, n_bot, c_top, c_bot):
    """
    让上下法向一致，并确保最终法向 n 指向从下端板 -> 上端板的方向。
    """
    # 保证法向方向一致
    if np.dot(n_top, n_bot) < 0:
        n_bot = -n_bot

    n = n_top + n_bot
    n /= np.linalg.norm(n)

    # 检查 n 是否朝向从下 -> 上
    vec = c_top - c_bot  # 从下到上的向量
    if np.dot(vec, n) < 0:
        n = -n  # 翻转

    return n


def min_vertical_thickness(top_pts: np.ndarray, bot_pts: np.ndarray):
    """
    几何最小距离版厚度计算：
    直接计算上下端板点之间的欧氏距离矩阵，
    取最小距离作为椎体厚度。
    返回:
        min_d: 最小厚度 (mm)
        p_top: 上端板点坐标 (3,)
        p_bot: 下端板点坐标 (3,)
        n:     伪法向方向 (默认竖直方向 [0,0,1])
    """
    if len(top_pts) == 0 or len(bot_pts) == 0:
        print("[WARN] 上下端板点数量为0，无法计算厚度。")
        return np.inf, None, None, np.array([0, 0, 1])

    # 计算所有点对之间的距离矩阵 (N_top, N_bot)
    diff = top_pts[:, None, :] - bot_pts[None, :, :]   # shape (N_top, N_bot, 3)
    dist = np.linalg.norm(diff, axis=2)                # 欧氏距离

    # 找最小距离对应点对
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    min_d = dist[i, j]
    p_top = top_pts[i]
    p_bot = bot_pts[j]

    # 伪法向（竖直方向，兼容旧结构）
    n = np.array([0, 0, 1])

    print(f"[INFO] 找到最小距离点对: index_top={i}, index_bot={j}, 距离={min_d:.3f} mm")
    return float(min_d), p_top, p_bot, n


# ============================================================
# 5️⃣ 主流程
# ============================================================
def measure_min_height_from_stl(
    stl_path,
    y_ratio=0.35,
    top_ratio=0.15,
    bottom_ratio=0.15,
    visualize=True
):
    print(f"\n[INFO] 读取 STL: {stl_path}")
    points = load_stl_points(stl_path)
    aligned, pca = align_pca(points)
    print(f"[INFO] PCA 对齐完成。点数: {len(points)}")

    # Step1: 只保留椎体区域（按 Y 轴取前 40%）
    body_points = select_body_region_by_y(aligned, ratio=y_ratio)

    # Step2: 提取端板
    top_pts, bot_pts = extract_endplates(body_points, top_ratio, bottom_ratio)
    print(f"[INFO] 上端板: {len(top_pts)}，下端板: {len(bot_pts)}")
    # Step3: 计算最小垂直厚度
    min_h, p_top, p_bot, n_vert = min_vertical_thickness(top_pts, bot_pts)
    print(f"最小垂直厚度: {min_h:.3f} mm")

    # Step4: 可视化
    if visualize:
        pv.set_plot_theme("document")
        p = pv.Plotter()
        p.add_points(aligned, color='lightgray', point_size=2, label="Full")
        p.add_points(body_points, color='orange', point_size=3, label="Body (Y40%)")
        p.add_points(top_pts, color='red', point_size=6, label="Top")
        p.add_points(bot_pts, color='blue', point_size=6, label="Bottom")
        if p_top is not None and p_bot is not None:
            line = np.array([p_top, p_bot])
            p.add_lines(line, color='green', width=4, label=f"min H = {min_h:.2f} mm")
        p.add_legend()
        p.show()

    return min_h, p_top, p_bot, n_vert


# ============================================================
# 6️⃣ 示例
# ============================================================
if __name__ == "__main__":
    stl_file = r"J:\Second_Year_Data\Morphed_Vertebrae_TrainingSet\WholeVTK2STL\DeterministicAtlas__flow__reference__subject_L1_553__tp_64.stl"  # ←改成你的路径
    measure_min_height_from_stl(stl_file)
