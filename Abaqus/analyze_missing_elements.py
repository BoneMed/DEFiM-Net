"""
分析缺失的元素 ID：
1. 检查这些元素是否在 Morphed INP 中
2. 检查这些元素是否在 CSV 中
3. 找出问题根源
"""
import pandas as pd
import os

def analyze_missing_elements(missing_ids_file, morphed_inp, csv_file, sim_inp=None):
    """分析缺失元素

    Args:
        missing_ids_file: 缺失元素ID文件
        morphed_inp: 原始Morphed INP文件
        csv_file: CSV材料属性文件
        sim_inp: 仿真用的INP文件（可选）
    """

    print("=" * 80)
    print("分析缺失材料属性的元素")
    print("=" * 80)

    # 1. 读取缺失的元素 ID
    print(f"\n1. 读取缺失元素 ID 列表...")
    missing_ids = []
    with open(missing_ids_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    missing_ids.append(int(line))
                except:
                    pass

    missing_ids = sorted(missing_ids)
    print(f"   缺失元素数量: {len(missing_ids)}")
    if missing_ids:
        print(f"   ID 范围: {missing_ids[0]} - {missing_ids[-1]}")

    # 2. 读取 Morphed INP 的所有元素 ID
    print(f"\n2. 读取 Morphed INP 的元素...")
    morphed_elem_ids = set()
    in_elem = False

    with open(morphed_inp, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('*'):
                if line.lower().startswith('*element') and 'output' not in line.lower():
                    in_elem = True
                else:
                    in_elem = False
                continue

            if in_elem and line and not line.startswith('**'):
                parts = line.split(',')
                if parts and parts[0].strip():
                    try:
                        eid = int(float(parts[0].strip()))
                        morphed_elem_ids.add(eid)
                    except:
                        pass

    print(f"   Morphed INP 元素数: {len(morphed_elem_ids)}")

    # 3. 读取 CSV 的元素 ID
    print(f"\n3. 读取 CSV 的元素...")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        csv_elem_ids = set(df['ElementID'].values)
        print(f"   CSV 元素数: {len(csv_elem_ids)}")
    else:
        print(f"   ✗ CSV 文件不存在: {csv_file}")
        csv_elem_ids = set()

    # 4. 检查仿真 INP 中的材料定义
    sim_material_ids = set()
    if sim_inp and os.path.exists(sim_inp):
        print(f"\n4. 检查仿真 INP 的材料定义...")
        with open(sim_inp, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip().startswith('*SOLID SECTION'):
                    # 格式: *SOLID SECTION, ELSET=Element_12345, MATERIAL=Material_12345
                    if 'Element_' in line:
                        try:
                            eid_str = line.split('Element_')[1].split(',')[0].split()[0]
                            eid = int(eid_str)
                            sim_material_ids.add(eid)
                        except:
                            pass
        print(f"   仿真 INP 中有材料定义的元素数: {len(sim_material_ids)}")
    else:
        if sim_inp:
            print(f"\n4. [跳过] 仿真 INP 文件不存在: {sim_inp}")

    # 5. 分析
    print(f"\n" + "=" * 80)
    print("分析结果:\n")

    missing_set = set(missing_ids)

    # 检查缺失元素是否在 Morphed INP 中
    in_morphed = missing_set & morphed_elem_ids
    not_in_morphed = missing_set - morphed_elem_ids

    print(f"缺失的 {len(missing_ids)} 个元素中:")
    print(f"  - 在 Morphed INP 中:  {len(in_morphed)} 个")
    print(f"  - 不在 Morphed INP 中: {len(not_in_morphed)} 个")

    if csv_elem_ids:
        # 检查缺失元素是否在 CSV 中
        in_csv = missing_set & csv_elem_ids
        not_in_csv = missing_set - csv_elem_ids

        print(f"\n  - 在 CSV 中:  {len(in_csv)} 个")
        print(f"  - 不在 CSV 中: {len(not_in_csv)} 个")

    if sim_material_ids:
        # 检查缺失元素是否有材料定义
        has_material = missing_set & sim_material_ids
        no_material = missing_set - sim_material_ids

        print(f"\n  - 在仿真 INP 中有材料定义: {len(has_material)} 个")
        print(f"  - 在仿真 INP 中无材料定义: {len(no_material)} 个 <-- 这是问题根源")

        if len(no_material) == len(missing_ids):
            print(f"\n    [验证成功] 所有 ODB 报告缺失的元素确实没有材料定义！")
        elif len(has_material) > 0:
            print(f"\n    [异常] 有 {len(has_material)} 个元素 ODB 报错但实际有材料定义")
            print(f"          示例: {sorted(list(has_material))[:10]}")

    # 6. 结论
    print(f"\n" + "=" * 80)
    print("问题诊断:\n")

    if len(not_in_morphed) > 0:
        print(f"[根本原因] 有 {len(not_in_morphed)} 个元素不在 Morphed INP 中!")
        print(f"  → 仿真 INP 被修改过，增加了额外的元素")
        print(f"  → 这些元素 ID: {sorted(list(not_in_morphed))[:20]}")

        if len(not_in_morphed) == len(missing_ids):
            print(f"\n[结论] 所有缺失的元素都是额外添加的，不在原始 Morphed INP 中")
            print(f"       解决方法：使用原始的 Morphed INP 进行仿真")

    elif csv_elem_ids and len(not_in_csv) > 0:
        print(f"[根本原因] 有 {len(not_in_csv)} 个元素在 Morphed INP 中，但不在 CSV 中!")
        print(f"  → 解析器有 bug，没有处理这些元素")
        print(f"  → 需要修复 phantomless_aug.py")
        print(f"  → 这些元素 ID: {sorted(list(not_in_csv))[:20]}")

    elif sim_material_ids and len(no_material) == len(missing_ids):
        print(f"[根本原因] CSV 有所有元素，但仿真 INP 中缺少材料定义!")
        print(f"  → abaqus_add_material_2.py 没有为这些元素添加材料定义")
        print(f"  → 可能的原因:")
        print(f"    1. CSV 文件名不匹配（时间点不对）")
        print(f"    2. 循环写入过程中被中断")
        print(f"    3. 文件写入出错")
        print(f"  → 这些元素 ID: {sorted(list(no_material))[:20]}")

    else:
        print(f"[奇怪] 所有缺失的元素都在 Morphed INP 和 CSV 中")
        print(f"  → 可能 abaqus_add_material_2.py 有问题")
        print(f"  → 或者材料添加过程中出错")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # 配置文件路径
    missing_ids_file = "missing_element_ids.txt"
    morphed_inp = r"N:\SecondYear\Morphed_Vertebrae_TrainingSet\WholeVTK2STL\INPFiles\L1_514_tp_116.inp"  # 原始 Morphed INP
    csv_file = r"N:\SecondYear\CT_MASK_E_PHANTOMLESS\Train_FEM\514\514_element_E_tp_116.csv"
    sim_inp = r"C:\Users\Administrator\Desktop\L1_514_tp_116.inp"  # 仿真用的 INP

    try:
        analyze_missing_elements(missing_ids_file, morphed_inp, csv_file, sim_inp)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
