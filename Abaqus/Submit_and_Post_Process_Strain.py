#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
import shutil
from abaqus import *
from abaqusConstants import *
from odbAccess import openOdb


def GetStrain(strain_data):
    strain_data_dict = {}
    count_dict={}
    for sd in strain_data:
        nodeLabel, data= sd
        #print sd
        if nodeLabel not in strain_data_dict: 
            strain_data_dict[nodeLabel]=data
            count_dict[nodeLabel]=1
            #print strain_data_dict
        else:
            count_dict[nodeLabel]+=1
            strain_data_dict[nodeLabel]=strain_data_dict[nodeLabel]+data
    strain_data_dict_New = {key: strain_data_dict[key] / count_dict[key] for key in strain_data_dict}
    return strain_data_dict_New


def ReadOdb(inp_file,Model_folder,input_folder):
    
    job_name = os.path.splitext(inp_file)[0]
    odb_path = os.path.join(input_folder,(job_name+".odb"))

    odb_newpath=os.path.join(Model_folder,job_name)
    os.mkdir(odb_newpath)

    surface_1st_csv_path = os.path.join(odb_newpath,(job_name+"_surface_1st_order_nodes"+".csv"))
    surface_2nd_csv_path = os.path.join(odb_newpath,(job_name+"_surface_2nd_order_nodes"+".csv"))
    internal_csv_path = os.path.join(odb_newpath,(job_name+"_internal_nodes"+".csv"))
    
    
    odb = openOdb(odb_path)

    step_name = 'Step-1'  
    frame_index = -1      
    step = odb.steps[step_name]
    frame = step.frames[frame_index]
    strain_field = frame.fieldOutputs['E']

    ############################################################
    surface_2nd_node_set_name= "SURFACE_NODES_2ND"
    
    surface_2nd_node_set=odb.rootAssembly.instances["PART-1-1"].nodeSets[surface_2nd_node_set_name]
    
    surface_2nd_node_set_nodes=[node.label for node in surface_2nd_node_set.nodes]

    strain_values_surface_2nd = strain_field.getSubset(region=surface_2nd_node_set,position=ELEMENT_NODAL).getScalarField(invariant=MIN_PRINCIPAL).values

    strain_data_surface_2nd = map(lambda x:[x.nodeLabel, x.data], strain_values_surface_2nd)

    strain_data_dict_surface_2nd=GetStrain(strain_data_surface_2nd)

    ############################################################
    surface_1st_node_set_name= "SURFACE_NODES"

    surface_1st_node_set=odb.rootAssembly.instances["PART-1-1"].nodeSets[surface_1st_node_set_name]

    surface_1st_node_set_nodes=[node.label for node in surface_1st_node_set.nodes]

    strain_values_surface_1st = strain_field.getSubset(region=surface_1st_node_set,position=ELEMENT_NODAL).getScalarField(invariant=MIN_PRINCIPAL).values

    strain_data_surface_1st = map(lambda x:[x.nodeLabel, x.data], strain_values_surface_1st)

    stress_data_dict_surface_1st=GetStrain(strain_data_surface_1st)

    ############################################################
    # 所有节点
    node_all=odb.rootAssembly.instances["PART-1-1"].nodes
    node_all_nodes=[node.label for node in node_all]

    coords = map(lambda x:[x.label, x.coordinates], node_all)
    #print coords
    coords_dict= {}
    
    for coord in coords: 
        coords_dict[coord[0]] = coord[1]

    #print coords_dict

    ###########################################################
    internalnodes=[]
    for node in node_all_nodes:
        if node not in surface_2nd_node_set_nodes:
            internalnodes.append(node)

    internalnodeset=odb.rootAssembly.instances["PART-1-1"].NodeSetFromNodeLabels(name='InternalNodesSet',nodeLabels=tuple(internalnodes))

    strain_values_internal = strain_field.getSubset(region=internalnodeset,position=ELEMENT_NODAL).getScalarField(invariant=MIN_PRINCIPAL).values

    strain_data_internal = map(lambda x:[x.nodeLabel, x.data], strain_values_internal)

    strain_data_dict_internal=GetStrain(strain_data_internal)

    ###########################################################
    with open(surface_1st_csv_path,mode='wt') as f:
        f.write('NodeID'+' , '+'X_coord'+' , '+'Y_coord'+' , '+'Z_coord'+' , '+'Min_Principal_Strain'+'\n')
        for nodeid in stress_data_dict_surface_1st:
            if nodeid in coords_dict:
                f.write(str(nodeid)+' , '+ str(coords_dict[nodeid][0])+' , '+ str(coords_dict[nodeid][1])+' , '+str(coords_dict[nodeid][2])+' , '+\
                        str(stress_data_dict_surface_1st[nodeid])+'\n')

    with open(surface_2nd_csv_path,mode='wt') as f:
        f.write('NodeID'+' , '+'X_coord'+' , '+'Y_coord'+' , '+'Z_coord'+' , '+'Min_Principal_Strain'+'\n')
        for nodeid in strain_data_dict_surface_2nd:
            if nodeid in coords_dict:
                f.write(str(nodeid)+' , '+ str(coords_dict[nodeid][0])+' , '+ str(coords_dict[nodeid][1])+' , '+str(coords_dict[nodeid][2])+' , '+\
                        str(strain_data_dict_surface_2nd[nodeid])+'\n')
    
    with open(internal_csv_path,mode='wt') as f:
        f.write('NodeID'+' , '+'X_coord'+' , '+'Y_coord'+' , '+'Z_coord'+' , '+'Min_Principal_Strain'+'\n')
        for nodeid in strain_data_dict_internal:
            if nodeid in coords_dict:
                f.write(str(nodeid)+' , '+ str(coords_dict[nodeid][0])+' , '+ str(coords_dict[nodeid][1])+' , '+str(coords_dict[nodeid][2])+' , '+\
                        str(strain_data_dict_internal[nodeid])+'\n')
    
    
    odb.close()

    shutil.move(odb_path, odb_newpath)








def SubmitJob(Model_folder):
    input_folder=os.path.join(Model_folder,'INPFiles')

    inp_files = [f for f in os.listdir(input_folder) if f.endswith('.inp')]
    print (inp_files)
    os.chdir(input_folder)

    if not inp_files:
        print ("folder: %s not .inp " %(input_folder))
    else:
        print ("folder: %s find %d .inp " %(input_folder,len(inp_files)))
        for inp_file in inp_files:
            inp_path = os.path.join(input_folder,inp_file) 
            job_name = os.path.splitext(inp_file)[0]
            mdb.JobFromInputFile(name=job_name,inputFileName=inp_path)
            mdb.jobs[job_name].submit()
            mdb.jobs[job_name].waitForCompletion()

            ReadOdb(inp_file,Model_folder,input_folder)

            print ("Model: %s Calculation Done!" %(inp_file))

#请输入stl文件的目录
Model_folder = r"I:\Second_Year_Data\Morphed_Vertebrae_TrainingSet\WholeVTK2STL"


SubmitJob(Model_folder)
