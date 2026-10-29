"""
详细诊断：对比 CSV、INP 和缺失元素
"""
import pandas as pd

def diagnose():
    print("=" * 80)
    print("详细诊断：CSV vs INP vs 缺失元素")
    print("=" * 80)

    # 1. 读取缺失元素 ID
    print("\n1. 读取缺失的元素 ID (从 ODB)...")
    missing_ids = []
    with open('missing_element_ids.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                missing_ids.append(int(line))

    missing_ids.sort()
    print(f"   缺失元素数: {len(missing_ids)}")
    print(f"   ID 范围: {missing_ids[0]} - {missing_ids[-1]}")

    # 2. 读取 CSV
    print("\n2. 读取 CSV 文件...")
    csv_path = r"N:\SecondYear\CT_MASK_E_PHANTOMLESS\Train_FEM\514\514_element_E_tp_116.csv"
    df = pd.read_csv(csv_path)
    csv_ids = set(df['ElementID'].values)

    print(f"   CSV 总元素数: {len(df)}")
    print(f"   CSV ID 范围: {df['ElementID'].min()} - {df['ElementID'].max()}")

    # 3. 读取仿真 INP 中有材料定义的元素
    print("\n3. 读取仿真 INP 中有材料定义的元素...")
    inp_path = r"C:\Users\Administrator\Desktop\L1_514_tp_116.inp"

    # 从 *SOLID SECTION 行提取已赋予材料的元素
    section_elem_ids = set()
    with open(inp_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip().startswith('*SOLID SECTION'):
                # 格式: *SOLID SECTION, ELSET=Element_12345, MATERIAL=Material_12345
                if 'Element_' in line:
                    try:
                        eid_str = line.split('Element_')[1].split(',')[0].split()[0]
                        eid = int(eid_str)
                        section_elem_ids.add(eid)
                    except:
                        pass

    print(f"   INP 中有材料定义的元素数: {len(section_elem_ids)}")

    # 4. 详细对比
    print("\n" + "=" * 80)
    print("详细对比:\n")

    missing_set = set(missing_ids)

    # CSV 检查
    in_csv = missing_set & csv_ids
    not_in_csv = missing_set - csv_ids

    print(f"缺失的 {len(missing_ids)} 个元素:")
    print(f"  在 CSV 中:     {len(in_csv):4d} 个  {'<-- CSV 有数据' if in_csv else ''}")
    print(f"  不在 CSV 中:   {len(not_in_csv):4d} 个  {'<-- CSV 缺失' if not_in_csv else ''}")

    # INP 检查
    in_inp = missing_set & section_elem_ids
    not_in_inp = missing_set - section_elem_ids

    print(f"\n  在 INP 材料定义中:   {len(in_inp):4d} 个")
    print(f"  不在 INP 材料定义中: {len(not_in_inp):4d} 个  <-- 这些是问题元素")

    # 5. 结论
    print("\n" + "=" * 80)
    print("结论:\n")

    if len(not_in_csv) > 0:
        print(f"[问题 A] CSV 缺少 {len(not_in_csv)} 个元素")
        print(f"  -> phantomless_aug.py 没有生成这些元素的材料属性")
        print(f"  -> 需要重新运行 phantomless_aug.py")
    else:
        print(f"[OK] CSV 完整，包含所有 {len(missing_ids)} 个缺失元素")

    if len(in_csv) > 0 and len(not_in_inp) > 0:
        print(f"\n[问题 B] CSV 有 {len(in_csv)} 个元素，但 INP 中没有材料定义")
        print(f"  -> abaqus_add_material_2.py 读取了 CSV")
        print(f"  -> 但没有为这些元素添加材料定义")
        print(f"  -> 需要检查 abaqus_add_material_2.py 的逻辑")

        # 显示示例
        print(f"\n  示例元素 (CSV 有但 INP 没写入):")
        for eid in sorted(list(in_csv & not_in_inp))[:10]:
            e_val = df[df['ElementID'] == eid]['E_MPa'].values[0]
            print(f"    Element {eid}: E = {e_val:.2f} MPa  <-- CSV 有这个值，但没写入 INP")

    # 6. CSV vs 脚本读取的字典
    print("\n" + "=" * 80)
    print("进一步检查：模拟 abaqus_add_material_2.py 的读取过程\n")

    # 模拟 ReadElementModulus 函数
    elementmodulusdict = {}
    for i in range(len(df['ElementID'])):
        elementmodulusdict[df['ElementID'][i]] = df['E_MPa'][i]

    dict_ids = set(elementmodulusdict.keys())

    print(f"CSV 原始行数: {len(df)}")
    print(f"字典元素数:   {len(elementmodulusdict)}")

    if len(elementmodulusdict) != len(df):
        print(f"\n[警告] 字典元素数 != CSV 行数!")
        print(f"        可能有重复的 ElementID")
    else:
        print(f"\n[OK] 字典读取正常")

    # 检查缺失元素是否在字典中
    in_dict = missing_set & dict_ids
    not_in_dict = missing_set - dict_ids

    print(f"\n缺失元素在字典中: {len(in_dict)} 个")
    print(f"缺失元素不在字典中: {len(not_in_dict)} 个")

    if len(in_dict) == len(missing_ids):
        print(f"\n[结论] 所有缺失元素都在读取的字典中")
        print(f"       问题出在循环写入 INP 时 (第 32-36 行)")
        print(f"       可能的原因:")
        print(f"         1. 循环被提前终止")
        print(f"         2. 文件写入时出错")
        print(f"         3. *STEP 位置查找错误")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        diagnose()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
