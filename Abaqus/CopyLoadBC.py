import os
import shutil

# ===== 输入根目录 =====
src_root = r"N:\SecondYear\test_ODB_Set"
dst_load = r"H:\python project\PHD_PROGRAM\StressPrediction_With_E\Data\Data_no_FPS\Load_csv"
dst_bc = r"H:\python project\PHD_PROGRAM\StressPrediction_With_E\Data\Data_no_FPS\BC_csv"

# ===== 每个样本复制次数 =====
num_samples = 1

# ===== 遍历 subj 文件夹 =====
for subj in os.listdir(src_root):
    subj_path = os.path.join(src_root, subj)
    if not os.path.isdir(subj_path):
        continue

    # 遍历 tp 文件夹
    for tp_folder in os.listdir(subj_path):
        tp_path = os.path.join(subj_path, tp_folder)
        if not os.path.isdir(tp_path):
            continue

        # 构造 FORCE / BC 文件路径
        force_file = os.path.join(tp_path, "FORCE_Nodes.csv")
        bc_file = os.path.join(tp_path, "BC_Nodes.csv")

        # ===== FORCE =====
        if os.path.exists(force_file):
            for i in range(1, num_samples + 1):
                dst_name = f"{tp_folder}_s{i}.csv"
                dst_path = os.path.join(dst_load, dst_name)
                shutil.copy2(force_file, dst_path)
            print(f"✅ Copied FORCE for {tp_folder}")

        # ===== BC =====
        if os.path.exists(bc_file):
            for i in range(1, num_samples + 1):
                dst_name = f"{tp_folder}_s{i}.csv"
                dst_path = os.path.join(dst_bc, dst_name)
                shutil.copy2(bc_file, dst_path)
            print(f"✅ Copied BC for {tp_folder}")

print("\n🎉 所有文件已成功复制完成！")
