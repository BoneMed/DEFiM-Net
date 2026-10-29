"""
测试字典迭代是否完整
重点检查是否所有元素都被处理
"""
import pandas as pd
import os


def test_dictionary_iteration():
    print("=" * 80)
    print("测试字典迭代完整性 - 514 tp119")
    print("=" * 80)

    # 读取 CSV
    csv_path = r"N:\SecondYear\CT_MASK_E_PHANTOMLESS\Train_FEM\514\514_element_E_tp_119.csv"
    print(f"\n读取 CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"CSV 行数: {len(df)}")
    print(f"ElementID 范围: {df['ElementID'].min()} - {df['ElementID'].max()}")

    # 构建字典（和原脚本相同的方式）
    Elementmodulusdict = {}
    for i in range(len(df['ElementID'])):
        Elementmodulusdict[df['ElementID'][i]] = df['E_MPa'][i]

    print(f"\n字典构建:")
    print(f"  字典元素数: {len(Elementmodulusdict)}")

    # 检查是否有重复
    if len(Elementmodulusdict) != len(df):
        print(f"  [WARNING] 字典元素数 != CSV 行数，可能有重复 ElementID")
        duplicates = df[df.duplicated(subset=['ElementID'], keep=False)]
        print(f"  重复的 ElementID 数量: {len(duplicates)}")
    else:
        print(f"  [OK] 无重复 ElementID")

    # 获取字典中的所有 key
    dict_keys = sorted(list(Elementmodulusdict.keys()))
    print(f"\n字典 key 范围: {dict_keys[0]} - {dict_keys[-1]}")

    # 模拟循环构建 newcontent（只计数，不实际构建字符串）
    print(f"\n模拟材料定义循环:")
    count = 0
    processed_ids = []

    for elemid in Elementmodulusdict:
        count += 1
        processed_ids.append(elemid)

    print(f"  循环次数: {count}")
    print(f"  处理的元素数: {len(processed_ids)}")

    # 检查缺失的元素
    csv_ids = set(df['ElementID'].values)
    dict_ids = set(Elementmodulusdict.keys())
    processed_ids_set = set(processed_ids)

    print(f"\n集合对比:")
    print(f"  CSV 中的元素: {len(csv_ids)}")
    print(f"  字典中的元素: {len(dict_ids)}")
    print(f"  循环处理的元素: {len(processed_ids_set)}")

    missing_in_dict = csv_ids - dict_ids
    missing_in_loop = dict_ids - processed_ids_set

    if missing_in_dict:
        print(f"\n  [ERROR] CSV 有但字典没有: {len(missing_in_dict)} 个")
        print(f"          示例: {sorted(list(missing_in_dict))[:10]}")
    else:
        print(f"\n  [OK] 字典包含所有 CSV 元素")

    if missing_in_loop:
        print(f"\n  [ERROR] 字典有但循环没处理: {len(missing_in_loop)} 个")
        print(f"          示例: {sorted(list(missing_in_loop))[:10]}")
    else:
        print(f"\n  [OK] 循环处理了字典中所有元素")

    # 检查 ODB 中缺失的元素
    print(f"\n" + "=" * 80)
    print("检查 ODB 报告缺失的元素是否在字典中:")

    missing_file = "missing_element_ids.txt"
    if os.path.exists(missing_file):
        missing_ids = []
        with open(missing_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        missing_ids.append(int(line))
                    except:
                        pass

        print(f"\nODB 缺失元素数: {len(missing_ids)}")
        if missing_ids:
            print(f"ID 范围: {missing_ids[0]} - {missing_ids[-1]}")

            missing_set = set(missing_ids)
            in_csv = missing_set & csv_ids
            in_dict = missing_set & dict_ids
            in_loop = missing_set & processed_ids_set

            print(f"\n这 {len(missing_ids)} 个元素:")
            print(f"  在 CSV 中:  {len(in_csv):4d} 个")
            print(f"  在字典中:   {len(in_dict):4d} 个")
            print(f"  被循环处理: {len(in_loop):4d} 个")

            if len(in_loop) == len(missing_ids):
                print(f"\n[结论] 所有缺失元素都被循环处理了！")
                print(f"       问题不在循环，而在写入或其他地方")
            else:
                not_in_loop = missing_set - processed_ids_set
                print(f"\n[结论] 有 {len(not_in_loop)} 个缺失元素没被循环处理")
                print(f"       示例: {sorted(list(not_in_loop))[:20]}")
    else:
        print(f"未找到 {missing_file}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        test_dictionary_iteration()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
