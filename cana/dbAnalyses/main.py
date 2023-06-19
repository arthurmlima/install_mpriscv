# This Python file uses the following encoding: utf-8
import sys
from glob import glob
import os
import numpy as np
import csv
import pandas as pd
from natsort import natsorted


# **************** START APPLICATION *******************************************************
# ******************************************************************************************

load_labelings = glob(os.path.join('label_*.csv'))  # Ants Path

n_files = len(load_labelings)
list_gt = []
i = 0
for fn in load_labelings:
    base = os.path.basename(fn)
    file_name = (os.path.splitext(base)[0])
    print("========================================")
    print(file_name)

    #==================================================================
    # Reads raw csv, inserts the header, address the index

    df = pd.read_csv(fn, sep=',', header=None, names = list(range(0,106)))
    row, col = df.shape

    nameCol = ['setName','object','user','date','figure','nLines']
    for header in range (0, (col-len(nameCol))):
        nameCol.append('data%i'% header)

    df.columns = nameCol

    index_nLines = df.columns.get_loc('nLines')
    col_nLines = df['nLines']
    max_lines = np.max(col_nLines)

    for cc in range(0,max_lines):
        # creation new column
        df['Ang_%i' % cc] = None

    # creation new column
    df['Average'] = None
    df['Std Deviation'] = None
    df['Warning List'] = None


    index_average = df.columns.get_loc('Average')
    index_std = df.columns.get_loc('Std Deviation')
    index_wl = df.columns.get_loc('Warning List')

    #==================================================================
    # Calculates each angle of the nLines
    #==================================================================

    # - - - - Choose the coordinates system - - - - - - - -
    #reference = "0to180Right"
    reference = "-90to90Front"


    list_angle = []
    for n in range(0, row):
        nLines = df.iat[n,index_nLines]
        angle_elem = []
        for line in range(1,nLines+1):
            x1 = df.iat[n, 4 * line + 2]
            y1 = df.iat[n, 4 * line + 3]
            x2 = df.iat[n, 4 * line + 4]
            y2 = df.iat[n, 4 * line + 5]

            alfa1 = (y2 - y1)/(x2 - x1)
            #print("y2", x2)
            line_len = np.sqrt(np.square(x2 - x1)+ np.square(y2 - y1))
            #print(line_len)

            if line_len > 200:
                #--------------------------------------
                if reference == "0to180Right":
                    angle = 90 + np.arctan(alfa1)*(180/np.pi)

                #--------------------------------------
                elif reference == "-90to90Front":
                    angle = np.arctan(alfa1)*(180/np.pi)
                    if angle < 0:
                        angle = + (angle + 90)
                    else:
                        angle = (angle - 90)

                #--------------------------------------
                angle_elem.append(angle)
                df.iat[n,(index_nLines + (4 * max_lines) + line)] = round(angle,2)



        df.iat[n,index_average] = round(np.mean(angle_elem),2)
        df.iat[n,index_std] = round(np.std(angle_elem),2)
        if df.iat[n,index_std] > 2.5:
            df.iat[n,index_wl] =  "yes"


    figures = df["figure"].str.split(".", n = 1, expand = True)
    angle_avg = df['Average']
    angle_std = df['Std Deviation']
    warn_list = df['Warning List']
    data_local = list(zip(figures[0].T, angle_avg.T,angle_std.T, warn_list.T))
    data_Local_ordered = natsorted(data_local)

    list_gt.append(data_Local_ordered)

    print("========================================")
    print(file_name)
    print("Total of Images",len(data_Local_ordered))
    print("========================================")
    print(data_Local_ordered)

    df.to_csv('full_analyse_%s.csv' % file_name)
    i += 1

print("")
print("**********************************************************************************")
print("%i files were read" %i, ":", load_labelings)
print("**********************************************************************************")


#==================================================================
#==================================================================
# WORKS ON ALL GROUND TRUTH CSV FILE
#==================================================================
#==================================================================
print("")

array_figures = []
for af in range(0,len(list_gt[0])):
    array_figures.append(list_gt[0][af][0])
print(array_figures)

array_data_0 = []
array_data_1 = []
array_data_2 = []
array_data_3 = []
array_data_4 = []
array_std_0 = []
array_std_1 = []
array_std_2 = []
array_std_3 = []
array_std_4 = []
array_warn_0 = []
array_warn_1 = []
array_warn_2 = []
array_warn_3 = []
array_warn_4 = []
for a in range(0,len(list_gt)):
    for ag in range(0,len(list_gt[0])):
        if a == 0:
            array_data_0.append(list_gt[a][ag][1])
            array_std_0.append(list_gt[a][ag][2])
            array_warn_0.append(list_gt[a][ag][3])
        elif a == 1:
            array_data_1.append(list_gt[a][ag][1])
            array_std_1.append(list_gt[a][ag][2])
            array_warn_1.append(list_gt[a][ag][3])
        elif a == 2:
            array_data_2.append(list_gt[a][ag][1])
            array_std_2.append(list_gt[a][ag][2])
            array_warn_2.append(list_gt[a][ag][3])
        elif a == 3:
            array_data_3.append(list_gt[a][ag][1])
            array_std_3.append(list_gt[a][ag][2])
            array_warn_3.append(list_gt[a][ag][3])
        elif a == 4:
            array_data_4.append(list_gt[a][ag][1])
            array_std_4.append(list_gt[a][ag][2])
            array_warn_4.append(list_gt[a][ag][3])


#print(array_warn_0)
#print(array_warn_1)
#print(array_warn_2)

trylist = list(zip(array_figures,array_data_0,array_data_1,array_data_2,array_data_3,array_data_4,array_std_0,array_std_1,array_std_2,array_std_3,array_std_4, array_warn_0,array_warn_1,array_warn_2,array_warn_3,array_warn_4))
#print(trylist)
np.savetxt('all_ground_truth.csv',trylist,fmt='%s',delimiter=',')

#========      WORK ON GROUND TRUTH GLOBAL     ================================

gt_global_df = pd.read_csv('all_ground_truth.csv', sep=',', header=None)
row1, col1 = gt_global_df.shape

colnames = ['figures']
for hd in range (0, n_files):
    colnames.append('specialist_%i'% hd)

for hd in range (0, n_files):
    colnames.append('std_%i'% hd)

for hd in range (0, n_files):
    colnames.append('warn_%i'% hd)

gt_global_df.columns = colnames

gt_global_df['Mean'] = None
gt_global_df['Std Deviation'] = None
gt_global_df['ShouldBeUsed?'] = None

indexAvg_0 = gt_global_df.columns.get_loc('specialist_0')
indexAvg_1 = gt_global_df.columns.get_loc('specialist_1')
indexAvg_2 = gt_global_df.columns.get_loc('specialist_2')
indexAvg_3 = gt_global_df.columns.get_loc('specialist_3')
indexAvg_4 = gt_global_df.columns.get_loc('specialist_4')

indexStd_0 = gt_global_df.columns.get_loc('std_0')
indexStd_1 = gt_global_df.columns.get_loc('std_1')
indexStd_2 = gt_global_df.columns.get_loc('std_2')
indexStd_3 = gt_global_df.columns.get_loc('std_3')
indexStd_4 = gt_global_df.columns.get_loc('std_4')

indexWarn_0 = gt_global_df.columns.get_loc('warn_0')
indexWarn_1 = gt_global_df.columns.get_loc('warn_1')
indexWarn_2 = gt_global_df.columns.get_loc('warn_2')
indexWarn_3 = gt_global_df.columns.get_loc('warn_3')
indexWarn_4 = gt_global_df.columns.get_loc('warn_4')

indexStdGlobal  = gt_global_df.columns.get_loc('Std Deviation')
indexMeanGlobal = gt_global_df.columns.get_loc('Mean')
indexSBU        = gt_global_df.columns.get_loc('ShouldBeUsed?')

for nn in range(0,row1):

    array_gt_global = []

    u_0 = gt_global_df.iat[nn,indexAvg_0] # Get the average value from spec_0
    u_1 = gt_global_df.iat[nn,indexAvg_1] # Get the average value from spec_1
    u_2 = gt_global_df.iat[nn,indexAvg_2] # Get the average value from spec_2
    u_3 = gt_global_df.iat[nn,indexAvg_3] # Get the average value from spec_3
    u_4 = gt_global_df.iat[nn,indexAvg_4] # Get the average value from spec_4

    var_0 = np.square(gt_global_df.iat[nn,indexStd_0]) # Get the variance value from spec_0
    var_1 = np.square(gt_global_df.iat[nn,indexStd_1]) # Get the variance value from spec_1
    var_2 = np.square(gt_global_df.iat[nn,indexStd_2]) # Get the variance value from spec_2
    var_3 = np.square(gt_global_df.iat[nn,indexStd_3]) # Get the variance value from spec_2
    var_4 = np.square(gt_global_df.iat[nn,indexStd_4]) # Get the variance value from spec_2

    u_01 = ((var_1/(var_0 + var_1)) * u_0) + ((var_0/(var_0 + var_1)) * u_1)
    var_01 = 1/((1/var_0)+(1/var_1))

    u_01_2 = ((var_2/(var_01 + var_2)) * u_01) + ((var_01/(var_01 + var_2)) * u_2)
    var_01_2 = 1/((1/var_01)+(1/var_2))

    u_012_3 = ((var_3/(var_01_2 + var_3)) * u_01_2) + ((var_01_2/(var_01_2 + var_3)) * u_3)
    var_012_3 = 1/((1/var_01_2)+(1/var_3))

    u_0123_4 = ((var_4/(var_012_3 + var_4)) * u_012_3) + ((var_012_3/(var_012_3 + var_4)) * u_4)
    var_0123_4 = 1/((1/var_012_3)+(1/var_4))


    gt_global_df.iat[nn,indexMeanGlobal] = round(u_0123_4,3)
    gt_global_df.iat[nn,indexStdGlobal]  = round(np.sqrt(var_0123_4),3)

    sumSBU = 0
    if (gt_global_df.iat[nn,indexWarn_0] == 'yes'):
        sumSBU += 1
    if (gt_global_df.iat[nn,indexWarn_1] == 'yes'):
        sumSBU += 1
    if (gt_global_df.iat[nn,indexWarn_2] == 'yes'):
        sumSBU += 1
    if (gt_global_df.iat[nn,indexWarn_3] == 'yes'):
        sumSBU += 1
    if (gt_global_df.iat[nn,indexWarn_4] == 'yes'):
        sumSBU += 1


    if sumSBU == 1:     # If only one specialist std dev > 2.5°
        if (gt_global_df.iat[nn,indexStdGlobal] > 1): # if global std dev > 1° (will be discarded)
            gt_global_df.iat[nn,indexSBU] = 'no'
    if sumSBU >= 2:     # If two specialist std dev > 2.5° (will be discarded)
        gt_global_df.iat[nn,indexSBU] = 'no'


print(gt_global_df)
gt_global_df.to_csv('all_ground_truth.csv')
