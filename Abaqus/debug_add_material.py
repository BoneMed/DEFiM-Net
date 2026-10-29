"""
Debug version of abaqus_add_material_2.py
添加详细日志以追踪为什么会丢失 1190 个元素
"""
import pandas as pd
import os


def ReadElementModulus(csvfilepath,inp_filename,csvfile):
    Elementmodulusdict={}
    csvfilefullpath=os.path.join(csvfilepath,inp_filename,csvfile)
    data=pd.read_csv(csvfilefullpath)

    for i in range(len(data['ElementID'])):
        Elementmodulusdict[data['ElementID'][i]]=data['E_MPa'][i]

    print(f"[DEBUG] CSV 读取完成:")
    print(f"        CSV 行数: {len(data)}")
    print(f"        字典元素数: {len(Elementmodulusdict)}")
    print(f"        ElementID 范围: {data['ElementID'].min()} - {data['ElementID'].max()}")

    return Elementmodulusdict


def Addmaterialdata(inpfilepath, inpfile, elementmodulusdict):
    inpfilefullpath=os.path.join(inpfilepath,inpfile)
    newcontent=[]

    with open(inpfilefullpath, 'r') as file1:
        lines=file1.readlines()

    print(f"\n[DEBUG] INP 文件读取:")
    print(f"        总行数: {len(lines)}")

    # 查找 *STEP 位置
    step_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('*STEP'):
            step_index = i
            break

    print(f"        *STEP 位置: 第 {step_index} 行")

    # 构建材料内容
    print(f"\n[DEBUG] 开始构建材料定义...")
    print(f"        字典元素数: {len(elementmodulusdict)}")

    elem_ids_processed = []
    for elemid in elementmodulusdict:
        newcontent.append("*ELSET, ELSET=Element_%s  \n   %s   \n" %(elemid,elemid))
        newcontent.append("*MATERIAL, NAME=Material_%s  \n*ELASTIC, TYPE = ISOTROPIC  \n  %s,     0.3,     0 \n" %(elemid,elementmodulusdict[elemid]))
        newcontent.append("*SOLID SECTION, ELSET=Element_%s, MATERIAL=Material_%s  \n" %(elemid,elemid))
        newcontent.append("**=================================== \n")
        elem_ids_processed.append(elemid)

    print(f"        循环完成，处理了 {len(elem_ids_processed)} 个元素")
    print(f"        newcontent 行数: {len(newcontent)}")
    print(f"        预期 SOLID SECTION 数: {len(elem_ids_processed)}")

    # 检查最后几个元素
    print(f"\n[DEBUG] 最后 10 个处理的元素 ID:")
    for eid in sorted(elem_ids_processed)[-10:]:
        print(f"        {eid}")

    # 插入到 *STEP 之前
    newlines = lines[:step_index] + newcontent + lines[step_index:]
    print(f"\n[DEBUG] 组装新文件:")
    print(f"        原始 INP: {len(lines)} 行")
    print(f"        新增材料: {len(newcontent)} 行")
    print(f"        合并后: {len(newlines)} 行")

    # 检查是否已有材料定义
    material_found=False
    for line in lines:
        if line.strip().startswith('*MATERIAL'):
            print(f"\n[ERROR] INP 文件 {inpfile} 中已经存在材料参数定义，请先删除")
            material_found=True
            break

    if not material_found:
        print(f"\n[DEBUG] 写入文件...")
        with open(inpfilefullpath,'w') as file2:
            file2.writelines(newlines)

        print(f"[OK] 成功写入 {len(elementmodulusdict)} 个元素的材料属性")

        # 验证写入
        print(f"\n[DEBUG] 验证写入结果...")
        section_count = 0
        with open(inpfilefullpath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip().startswith('*SOLID SECTION'):
                    section_count += 1

        print(f"        文件中 SOLID SECTION 数量: {section_count}")

        if section_count == len(elementmodulusdict):
            print(f"        [OK] 验证通过!")
        else:
            print(f"        [ERROR] 验证失败! 预期 {len(elementmodulusdict)}, 实际 {section_count}")
            print(f"        丢失了 {len(elementmodulusdict) - section_count} 个元素")

    return material_found


# 测试单个文件
if __name__ == "__main__":
    print("=" * 80)
    print("调试材料添加过程 - 514 tp116")
    print("=" * 80)

    # 测试 514 tp116
    inp_folder = r"J:\Second_Year_Data\Morphed_Vertebrae_TrainingSet\WholeVTK2STL\INPFiles"
    csv_folder = r"N:\SecondYear\CT_MASK_E_PHANTOMLESS\Train_FEM"

    inp_file = "L1_514_tp_119_2nd_Order.inp"
    inp_filename = "514"
    csv_file = "514_element_E_tp_119.csv"

    print(f"\nINP 文件: {inp_file}")
    print(f"CSV 文件: {csv_file}")

    # 读取 CSV
    Elementmodulusdict = ReadElementModulus(csv_folder, inp_filename, csv_file)

    # 添加材料
    materialadded = Addmaterialdata(inp_folder, inp_file, Elementmodulusdict)

    print("\n" + "=" * 80)
    print("完成!")
