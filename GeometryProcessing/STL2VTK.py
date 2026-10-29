import pyvista as pv
import vtk


stl_file = r"J:\Second_Year_Data\Deformetrica_STL_Process\L1_510_REF_Endplate.stl"
vtk_file = r"J:\Second_Year_Data\Deformetrica_STL_Process\VTK_Model\aligned_L1_510_REF_PRE_ALIGN.vtk"

stl_reader = vtk.vtkSTLReader()
stl_reader.SetFileName(stl_file)
stl_reader.Update()

# 获取读取到的PolyData
polydata = stl_reader.GetOutput()

# 写入VTK文件
vtk_writer = vtk.vtkPolyDataWriter()
vtk_writer.SetFileName(vtk_file)
vtk_writer.SetInputData(polydata)
vtk_writer.Write()