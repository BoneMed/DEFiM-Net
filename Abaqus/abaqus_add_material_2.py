import pandas as pd
import os



def ReadElementModulus(csvfilepath,inp_filename,csvfile):
    Elementmodulusdict={}
    csvfilefullpath=os.path.join(csvfilepath,inp_filename,csvfile)
    data=pd.read_csv(csvfilefullpath)

    print(f"  [检查点1] 读取 CSV: {csvfile}")
    print(f"            CSV 行数: {len(data)}")
    print(f"            ElementID 范围: {data['ElementID'].min()} - {data['ElementID'].max()}")

    for i in range(len(data['ElementID'])):
        Elementmodulusdict[data['ElementID'][i]]=data['E_MPa'][i]

    print(f"  [检查点2] 字典构建完成: {len(Elementmodulusdict)} 个元素")

    return Elementmodulusdict


#print(Elementmodulusdict)


def Addmaterialdata(inpfilepath, inpfile,elementmodulusdict):
    inpfilefullpath=os.path.join(inpfilepath,inpfile)
    newcontent=[]
    with open(inpfilefullpath, 'r') as file1:
        lines=file1.readlines()

    print(f"  [检查点3] 准备写入材料属性到: {inpfile}")

    step_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('*STEP'):
            step_index = i
            break

    print(f"            *STEP 位置: 第 {step_index} 行")

    for elemid in elementmodulusdict:
        newcontent.append("*ELSET, ELSET=Element_%s  \n   %s   \n" %(elemid,elemid))
        newcontent.append("*MATERIAL, NAME=Material_%s  \n*ELASTIC, TYPE = ISOTROPIC  \n  %s,     0.3,     0 \n" %(elemid,elementmodulusdict[elemid]))
        newcontent.append("*SOLID SECTION, ELSET=Element_%s, MATERIAL=Material_%s  \n" %(elemid,elemid))
        newcontent.append("**=================================== \n")

    print(f"  [检查点4] 材料定义构建完成: {len(newcontent)} 行 (预期 SOLID SECTION: {len(elementmodulusdict)} 个)")

    newlines = lines[:step_index] + newcontent + lines[step_index:]

    material_found=False
    for line in lines:
        if line.strip().startswith('*MATERIAL'):
            print("INP 文件 %s 中已经存在材料参数定义，请先删除" %(inpfile))
            material_found=True
            break
    if not material_found:
        with open(inpfilefullpath,'w') as file2:
            file2.writelines(newlines)
        print(f"  [检查点5] 文件写入完成")

        # 验证写入
        section_count = 0
        with open(inpfilefullpath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip().startswith('*SOLID SECTION'):
                    section_count += 1

        print(f"  [检查点6] 验证: INP 中 SOLID SECTION 数量 = {section_count}")
        if section_count == len(elementmodulusdict):
            print(f"            ✓ 成功写入 {len(elementmodulusdict)} 个元素的材料属性")
        else:
            print(f"            ✗ 警告: 预期 {len(elementmodulusdict)}, 实际 {section_count}, 丢失 {len(elementmodulusdict) - section_count} 个")

    return material_found



def GenerateInp(INPFiles_path, CSV_path):
    inp_folder=INPFiles_path
    
    inp_files = [f for f in os.listdir(inp_folder) if f.endswith('.inp')]
    print(inp_files)

    csv_folder=os.path.join(CSV_path)

    csv_files = [f for f in os.walk(csv_folder) if any("element_E_tp" in f and f.endswith(".csv") for f in f[2])]
    csv_files = [f for sublist in csv_files for f in sublist[2] if "element_E_tp" in f and f.endswith('.csv')]
    print(csv_files)

    os.chdir(inp_folder)

    successfulinp=[]

    if not inp_files:
        print("文件夹: %s 中没有找到 .inp 文件!" %(inp_folder))
    else:
        print("文件夹: %s 找到 %d 个.inp 文件!" %(inp_folder,len(inp_files)))
        for inp_file in inp_files:
            # === 新增判断：如果文件太大，跳过 ===
            inp_path = os.path.join(inp_folder, inp_file)
            file_size_kb = os.path.getsize(inp_path) / 1024  # 转为KB
            if file_size_kb > 1000000:  # 100000 KB = 100 MB
                print(f"[跳过] {inp_file} 已存在材料属性 (文件大小 {file_size_kb / 1024:.2f} MB)")
                continue
            inp_filename = os.path.splitext(inp_file)[0].split('_')[1]
            tp_num = inp_file.split('_')[3].split('.')[0]
            csv_filename=inp_filename +"_element_E_tp_" + str(tp_num) + ".csv"
            print(f"\n处理 INP: {inp_file} (Subject {inp_filename}, TP {tp_num})")
            print(f"        匹配 CSV: {csv_filename}")
            if csv_filename in csv_files:
                Elementmodulusdict=ReadElementModulus(csv_folder,inp_filename,csv_filename)
                materialadded=Addmaterialdata(inp_folder,inp_file,Elementmodulusdict)
                if not materialadded:
                    successfulinp.append(inp_file)
            else:
                print(f"        ✗ 未找到匹配的 CSV 文件，跳过")
    
    if successfulinp:
        print("成功添加材料参数的inp文件: %s" %(successfulinp))
    else:
        print("inp文件都已经定义了材料参数，请先重新生成inp文件")




scriptpath=os.path.dirname(__file__)
INPFiles_path = r"J:\Second_Year_Data/Morphed_Vertebrae_TrainingSet\WholeVTK2STL/INPFiles"
CSV_path = r"N:\SecondYear/CT_MASK_E_PHANTOMLESS/Train_FEM"

print(scriptpath)


GenerateInp(INPFiles_path, CSV_path)



#Elementmodulusdict=ReadElementModulus(scriptpath,'510_element_E_New_Mesh.csv')       

#Addmaterialdata(scriptpath,'L1_510_tp_119_2nd_Order.inp',Elementmodulusdict)




