import os
import pandas as pd

# === 配置区 ===
csv_path = r"J:\Second_Year_Data\Morphed_Vertebrae_TrainingSet\WholeVTK2STL\min_height_results.csv" # CSV 路径
inp_dir = r"J:\Second_Year_Data\Morphed_Vertebrae_TrainingSet\WholeVTK2STL\INPFiles"  # INP 文件夹路径
# === 读取CSV ===
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={df.columns[0]: "filename", df.columns[1]: "min_height_mm"})

# === 遍历并修改INP文件 ===
for _, row in df.iterrows():
    name = str(row["filename"])
    name = name.replace('__', '_')
    h = float(row["min_height_mm"])
    new_force = round(0.019 * h, 6)

    inp_path = os.path.join(inp_dir, f"{name}.inp")
    if not os.path.exists(inp_path):
        print(f"[跳过] 未找到 {inp_path}")
        continue

    with open(inp_path, "r") as f:
        lines = f.readlines()

        # 从最后一行向前找 FORCE
    for i in range(len(lines) - 1, -1, -1):
        if "FORCE" in lines[i].upper():
            lines[i] = f"FORCE,3, ,{-new_force:.6f}\n"
            print(f"[已修改] {name}.inp → FORCE = {-new_force:.6f}")
            break
    else:
        print(f"[未找到FORCE行] {name}.inp")

    with open(inp_path, "w") as f:
        f.writelines(lines)

print("\n✅ 所有INP文件已更新完成！")