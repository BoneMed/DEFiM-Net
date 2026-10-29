import os
import glob
import re
import numpy as np
from ansys.mapdl.reader import read_binary
from joblib import Parallel, delayed
from tensorboard.compat.proto.variable_pb2 import SaveSliceInfoDef


# -------------------------------------
# Helper functions: normalize, rotate, mapping
# -------------------------------------

def normalize(points):
    center = points.mean(axis=0)
    pts = points - center
    m = np.linalg.norm(pts, axis=1).max()
    return pts / m if m > 0 else pts


def rotate(points, angles=(20, -10, 5)):
    """Rotate point cloud around XYZ (deg)."""
    ax, ay, az = np.deg2rad(angles)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
    return points @ Rx.T @ Ry.T @ Rz.T


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

# -------------------------------------
# Mapping functions for Y‑axis reorder
# -------------------------------------

def build_Y_mapping(pts):
    """Return a 2‑column array [original_index (1‑based), rank_by_descending_Y]."""
    N = pts.shape[0]
    idx = np.argsort(pts[:, 1])[::-1]  # Y descending
    ranks = np.empty(N, dtype=int)
    ranks[idx] = np.arange(1, N + 1)
    return np.stack([np.arange(1, N + 1), ranks], axis=1)


def reorder_data(mapping_Ori, mapping_Target, data):
    """Two‑step reorder: original -> Y‑sorted -> target order."""
    temp = np.empty_like(data)
    temp[mapping_Ori[:, 1] - 1] = data[mapping_Ori[:, 0] - 1]
    out = np.empty_like(temp)
    out[mapping_Target[:, 0] - 1] = temp[mapping_Target[:, 1] - 1]
    return out

# -------------------------------------
# Main routine (one sampling pass)
# -------------------------------------

def GetStress(RSTFile,
              Name,
              suffix="",
              stl_norm_dir="../Data/test_Nor_Coordinate_STL",
              n_interior=23528,
              seed=42):
    """Process one RST file and *one* interior‑sampling pass.

    Parameters
    ----------
    RSTFile : str
        Path to .rst file.
    Name : str
        Base model name (without sampling suffix) – used to locate STL normalisation file.
    suffix : str, optional
        Extra string appended to every output file name (e.g. 's1').  Default "" (no suffix).
    stl_norm_dir : str
        Directory containing Nor_{Name}.npy from STL.
    n_interior : int
        Down‑sample target for interior points.
    seed : int
        RNG seed to initialise farthest‑point sampling.  Different seeds → different samples.
    """

    # -------- read RST --------
    rst = read_binary(RSTFile)
    grid = rst.grid
    _, ps = rst.principal_nodal_stress(rnum=0)
    stress_all = ps[:, :5]
    # 去掉第四列
    stress_all = np.delete(stress_all, 3, axis=1)  # Remove the 4th column
    grid.point_data["stress"] = stress_all

    # -------- separate surface / interior --------
    surf = grid.extract_surface()
    surf_idx = surf.point_data.active_scalars
    all_idx = grid.point_data.active_scalars
    int_idx = np.setdiff1d(all_idx, surf_idx)

    # ---- optional FPS down‑sample interior ----
    if len(int_idx) > n_interior:
        rng = np.random.default_rng(seed)
        pts_int = grid.points[int_idx - 1]
        # Farthest‑point sampling
        centroids = [rng.integers(len(pts_int))]
        dist2 = np.full(len(pts_int), np.inf)
        for _ in range(1, n_interior):
            d = np.sum((pts_int - pts_int[centroids[-1]]) ** 2, axis=1)
            dist2 = np.minimum(dist2, d)
            centroids.append(np.argmax(dist2))
        int_idx = int_idx[np.asarray(centroids)]

    # -------- gather, rotate, normalise --------
    chosen = np.concatenate([surf_idx, int_idx])
    pts = grid.points[chosen - 1]
    st = grid.point_data["stress"][chosen - 1]
    pts = rotate(pts)
    pts_norm = normalize(pts)

    # split back
    ns = len(surf_idx)

    pts_s, pts_i = pts[:ns], pts[ns:]
    norm_s, norm_i = pts_norm[:ns], pts_norm[ns:]
    stress_s, stress_i = st[:ns], st[ns:]

    # -------- surface re‑order (STL‑guided) --------
    stl_norm = np.load(os.path.join(stl_norm_dir, f"Nor_{Name}.npy"))
    map_ori_s = build_Y_mapping(norm_s)
    map_stl_s = build_Y_mapping(stl_norm)

    surf_pts_re = reorder_data(map_ori_s, map_stl_s, pts_s)
    surf_norm_re = reorder_data(map_ori_s, map_stl_s, norm_s)
    surf_stress_re = reorder_data(map_ori_s, map_stl_s, stress_s)

    # -------- interior re‑order (same two‑step scheme) --------

    map_int = build_Y_mapping(norm_i)

    int_pts_re = reorder_data(map_int, map_stl_s, pts_i)
    int_norm_re = reorder_data(map_int, map_stl_s, norm_i)
    int_stress_re = reorder_data(map_int, map_stl_s, stress_i)

    # -------- save --------
    Test_out_dirs = {
        "norm": "../Data/TrainingSet_Cor_Stress/Surface_Test_Coordinate_Nor",
        "stress": "../Data/TrainingSet_Cor_Stress/Surface_Test_Stress",
        "int_coord_Nor": "../Data/TrainingSet_Cor_Stress/Inter_Test_Coordinate_Nor",
        "int_stress": "../Data/TrainingSet_Cor_Stress/Inter_Test_Stress",
    }

    Train_out_dirs = {
        "norm": "../Data/TrainingSet_Cor_Stress/Surface_Train_Coordinate_Nor",
        "stress": "../Data/TrainingSet_Cor_Stress/Surface_Train_Stress",
        "int_coord_Nor": "../Data/TrainingSet_Cor_Stress/Inter_Train_Coordinate_Nor",
        "int_stress": "../Data/TrainingSet_Cor_Stress/Inter_Train_Stress",
    }


    if stl_norm_dir == '../Data/train_Nor_Coordinate_STL':
        out_dirs = Train_out_dirs
    else:
        out_dirs = Test_out_dirs


    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    # surface + interior combined (legacy output)
    np.save(
        os.path.join(out_dirs["norm"], f"Nor_{Name}.npy"),
        np.vstack([surf_pts_re]),
    )
    np.save(
        os.path.join(out_dirs["stress"], f"{Name}.npy"),
        np.vstack([surf_stress_re]),
    )

    # interior‑only outputs
    np.save(
        os.path.join(out_dirs["int_coord_Nor"], f"{Name}{suffix}.npy"), int_pts_re
    )
    np.save(
        os.path.join(out_dirs["int_stress"], f"{Name}{suffix}.npy"), int_stress_re
    )

    print(
        f"Reordered {Name}{suffix}: surface {ns} + interior {len(int_idx)} points (seed {seed})"
    )


# ---------------- batch driver ----------------

def process_file(f,stl_norm_dir):
    base = os.path.splitext(os.path.basename(f))[0]
    # five independent interior samplings with different seeds
    for i in range(1, 6):
        GetStress(
            f,
            Name=base,          # base model name (for STL)
            stl_norm_dir=stl_norm_dir,
            suffix=f"s{i}",     # append s1..s5 to output files
            seed=42 + i - 1,    # seed 42,43,44,45,46
        )


if __name__ == "__main__":

    SaveRoot = '../Data/TrainingSet_Cor_Stress'
    os.makedirs(SaveRoot, exist_ok=True)


    TestRSTDir = r"I:\3D_Slicer\L1_VTK\Morphed_Vertebrae_TestSet\WholeVTK2STL\RSTFile"
    TrainRSTDir = r"I:\3D_Slicer\L1_VTK\Morphed_Vertebrae_TrainingSet\WholeVTK\RSTFile"
    #
    Trainfiles = glob.glob(os.path.join(TrainRSTDir, "L1_504__tp_119.rst"))
    Trainfiles.sort(key=natural_sort_key)

    # Testfiles = glob.glob(os.path.join(TestRSTDir, "*.rst"))
    # Testfiles.sort(key=natural_sort_key)


    Parallel(n_jobs=-1)(delayed(process_file)(f, '../Data/train_Nor_Coordinate_STL') for f in Trainfiles)
    # Parallel(n_jobs=-1)(delayed(process_file)(f, '../Data/test_Nor_Coordinate_STL') for f in Testfiles)
