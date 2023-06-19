# Criado por: Raphael P Ferreira
# Data: 05/03/2022
# Modificado: 15/11/2022

# This Python file uses the following encoding: utf-8
import sys
from glob import glob
import os
import numpy as np
import csv
import pandas as pd
from natsort import natsorted
#import matplotlib.pyplot as plt

print(sys.argv, len(sys.argv))

# Acess core folder to gets algoritm orientation angle results
os.chdir("../core")
# Creates a initial data frame based on 2 columns from final results file
df_fr = pd.read_csv('final_results.csv', header=None)
row,col = df_fr.shape
# Since the initial dataframe has no column name, let's give them
colm_names = ['Figure','Angle_algorithm']
df_fr.columns = colm_names
# Creates another 3 columns to be filled ahead
df_fr['Angle_Specialist'] = None
df_fr['ABS ERROR'] = None
df_fr['SBU'] = None
#print(df_fr)

# Acess dbAnalyses folder to gets ground truth orientation angle results
os.chdir("../dbAnalyses")
df_ground_th = pd.read_csv('all_ground_truth.csv')
#print(df_ground_th)
angle_gt = df_ground_th['Mean']
# Agregates the analyses from specialists in our dataframe
df_fr['Angle_Specialist'] = angle_gt
warn_col = df_ground_th['ShouldBeUsed?']
df_fr['SBU'] = warn_col

# Generates and incorates on dataframe a simple comparison betwenn both results
df_fr['ABS ERROR'] = round(abs(df_fr['Angle_algorithm'] - df_fr['Angle_Specialist']),2)
#print(df_fr)
# This delets images results with SBU marks (no)
df_fr.drop(df_fr.index[(df_fr["SBU"] == "no")],axis=0,inplace=True)
df_fr.drop("SBU",axis=1,inplace=True)
#print(df_fr)
index_absError = df_fr.columns.get_loc('ABS ERROR')

# Creates a array from all dataframe rows (only at one index_columns) to calculate stats
array_sae = []
for sae in range(0,df_fr.shape[0]):
    array_sae.append(df_fr.iat[sae,index_absError])
avg_abs_err = round(np.mean(array_sae),3)  # Mean
std_avg_abs_err = round(np.std(array_sae),3) # Standard deviation

# Append the stats results at the end of row data
df_fr.loc[(row+1)] = ['Average Absolute Error','-----','-----',avg_abs_err]
df_fr.loc[(row+2)] = ['Std dev','-----','-----', std_avg_abs_err]
# Acess result_analyses folder to save the new dataframe as a single file
os.chdir("../result_analyses")
os.chdir(sys.argv[1])
df_fr.to_csv('data_performance.csv')

# Write on the terminal the summary results
print(" ")
print("======== Results ==========================")
print(df_fr)

# Generates a box plot of absolute error
if(0):
    green_diamond = dict(markerfacecolor='g', marker='D')
    fig1, ax1 = plt.subplots()
    ax1.set_title('Absolute angle error')
    ax1.boxplot(df_fr['ABS ERROR'], flierprops=green_diamond)
    plt.show()
