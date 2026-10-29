import os
import pandas as pd

# ===== 输入路径 =====
min_height_csv = r"N:\SecondYear/min_height_results.csv"
base_dir = r"N:\SecondYear\Aba_ODB_Set"

# 读取 min_height_results.csv
df = pd.read_csv(min_height_csv)

for idx, row in df.iterrows():
    filename = row["filename"]
    displacement = row["min_height_mm"]

    # 解析 subject 和 tp
    parts = filename.split("__tp_")
    subj = parts[0].split("_")[-1]  # 例如 L1_504 → subj = 504
    tp = f"L1_{subj}_tp_{parts[1]}"  # 还原成文件夹名

    # 构造 FORCE_Nodes.csv 路径
    force_csv = os.path.join(base_dir, subj, tp, "FORCE_Nodes.csv")

    if os.path.exists(force_csv):
        print(f"Processing: {force_csv}")
        df_force = pd.read_csv(force_csv)

        # 添加 displacement 列（如果不存在）
        df_force["displacement"] = displacement

        # 保存（覆盖原文件）
        df_force.to_csv(force_csv, index=False)
    else:
        print(f"⚠️ 文件不存在: {force_csv}")

print("✅ 所有文件已处理完成！")
