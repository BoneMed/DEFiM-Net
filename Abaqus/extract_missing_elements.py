"""
Abaqus Python 脚本：从 .odb 文件提取 ErrElemMissingSection 的元素 ID
使用方法：
  abaqus python extract_missing_elements.py
"""
from odbAccess import openOdb
import sys

# ODB 文件路径
odb_path = r"N:/L1_514_tp_116.odb"

print("=" * 80)
print("提取缺失材料属性的元素 ID")
print("=" * 80)
print(f"\nODB 文件: {odb_path}\n")

try:
    # 打开 ODB
    odb = openOdb(path=odb_path, readOnly=True)

    # 获取 assembly
    assembly = odb.rootAssembly

    # 查找 ErrElemMissingSection element set
    elset_name = 'ErrElemMissingSection'

    # 尝试在不同位置查找 element set
    elset = None

    # 1. 在 rootAssembly 中查找
    if elset_name in assembly.elementSets.keys():
        elset = assembly.elementSets[elset_name]
        print(f"✓ 在 assembly 中找到 element set: {elset_name}")
    else:
        # 2. 在各个 instance 中查找
        for instance_name, instance in assembly.instances.items():
            if elset_name in instance.elementSets.keys():
                elset = instance.elementSets[elset_name]
                print(f"✓ 在 instance '{instance_name}' 中找到 element set: {elset_name}")
                break

    if elset is None:
        print(f"✗ 未找到 element set: {elset_name}")
        print(f"\n可用的 element sets:")
        for name in assembly.elementSets.keys():
            print(f"  - {name}")
        odb.close()
        sys.exit(1)

    # 提取元素 ID
    element_ids = []

    print(f"\n正在提取元素 ID...")

    # elset.elements 是一个 tuple，包含 OdbMeshElementArray 对象
    # 需要两层遍历
    for elem_array in elset.elements:
        # elem_array 是 OdbMeshElementArray，需要遍历其中的元素
        for elem in elem_array:
            element_ids.append(elem.label)

    element_ids.sort()

    print(f"\n缺失材料属性的元素数量: {len(element_ids)}")

    if element_ids:
        print(f"元素 ID 范围: {element_ids[0]} - {element_ids[-1]}")

        # 保存到文件
        output_file = "missing_element_ids.txt"
        with open(output_file, 'w') as f:
            f.write(f"# 缺失材料属性的元素 ID (从 {odb_path})\n")
            f.write(f"# 总数: {len(element_ids)}\n")
            f.write(f"# 范围: {element_ids[0]} - {element_ids[-1]}\n\n")
            for eid in element_ids:
                f.write(f"{eid}\n")

        print(f"\n✓ 元素 ID 已保存到: {output_file}")

        # 显示前 20 个
        print(f"\n前 20 个缺失的元素 ID:")
        for eid in element_ids[:20]:
            print(f"  {eid}")

        if len(element_ids) > 20:
            print(f"  ...")
            print(f"  (共 {len(element_ids)} 个)")

    odb.close()
    print("\n" + "=" * 80)
    print("完成！")

except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
