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
import matplotlib.pyplot as plt
import scipy.stats as stats

#===========================================================================================
# ================ USEFUL FUNCTIONS ========================================================


"""--------------------------------------------------------------
Median Absolute Deviation (MAD) based outlier detection
Removes outliers
---------------------------------------------------------------"""
def remove_outliers(data, fill=False, threshold=3.5):

    med = np.median(data)
    mad = np.abs(stats.median_abs_deviation(data))
    outlier = []

    for i, v in enumerate(data):
        t = (v-med)/mad
        if t > threshold:
            outlier.append(i)
        else:
            continue

    print(outlier)
    new_data = np.delete(data, outlier)

    return new_data
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Acess all orientation angle results files
os.chdir("PROJECTION")

#1.data_performance_01_H_7
#2.data_performance_1_H_7
#3.data_performance_01_H_15
#4.data_performance_1_H_15
#5.data_performance_01_H_23
#6.data_performance_1_H_23

#7.data_performance_01_Q_7
#8.data_performance_1_Q_7
#9.data_performance_01_Q_15
#10.data_performance_1_Q_15
#11.data_performance_01_Q_23
#12.data_performance_1_Q_23

# Creates dataframes based final results files
df_fr_01_H_7  = pd.read_csv('data_performance_01_H_7.csv')
df_fr_1_H_7   = pd.read_csv('data_performance_1_H_7.csv')
df_fr_01_H_15 = pd.read_csv('data_performance_01_H_15.csv')
df_fr_1_H_15  = pd.read_csv('data_performance_1_H_15.csv')
df_fr_01_H_23 = pd.read_csv('data_performance_01_H_23.csv')
df_fr_1_H_23  = pd.read_csv('data_performance_1_H_23.csv')

# Creates dataframes based final results files
df_fr_01_Q_7   = pd.read_csv('data_performance_01_Q_7.csv')
df_fr_1_Q_7    = pd.read_csv('data_performance_1_Q_7.csv')
df_fr_01_Q_15  = pd.read_csv('data_performance_01_Q_15.csv')
df_fr_1_Q_15   = pd.read_csv('data_performance_1_Q_15.csv')
df_fr_01_Q_23  = pd.read_csv('data_performance_01_Q_23.csv')
df_fr_1_Q_23   = pd.read_csv('data_performance_1_Q_23.csv')

# Since our dataframe has no column name, let's give them
colm_names_H = ['0.1H7','1H7','0.1H15','1H15','0.1H23','1H23']
colm_names_Q = ['0.1Q7','1Q7','0.1Q15','1Q15','0.1Q23','1Q23']

# Creates our dataframes to be based on final results files
all_df_fr_H = pd.DataFrame(columns=colm_names_H)
all_df_fr_Q = pd.DataFrame(columns=colm_names_Q)

# Incorporates the data to our dataframe
all_df_fr_H['0.1H7']  = df_fr_01_H_7['ABS ERROR']
all_df_fr_H['1H7']    = df_fr_1_H_7['ABS ERROR']
all_df_fr_H['0.1H15'] = df_fr_01_H_15['ABS ERROR']
all_df_fr_H['1H15']   = df_fr_1_H_15['ABS ERROR']
all_df_fr_H['0.1H23'] =  df_fr_01_H_23['ABS ERROR']
all_df_fr_H['1H23']   =  df_fr_1_H_23['ABS ERROR']

# Incorporates the data to our dataframe
all_df_fr_Q['0.1Q7']  = df_fr_01_Q_7['ABS ERROR']
all_df_fr_Q['1Q7']    = df_fr_1_Q_7['ABS ERROR']
all_df_fr_Q['0.1Q15'] = df_fr_01_Q_15['ABS ERROR']
all_df_fr_Q['1Q15']   = df_fr_1_Q_15['ABS ERROR']
all_df_fr_Q['0.1Q23'] =  df_fr_01_Q_23['ABS ERROR']
all_df_fr_Q['1Q23']   =  df_fr_1_Q_23['ABS ERROR']

print(all_df_fr_H)
print(all_df_fr_Q)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#newData = remove_outliers(all_df_fr_H['0.1H7'])
median = np.median(all_df_fr_H['0.1H7'])
Q1, Q3 = np.percentile(all_df_fr_H['0.1H7'], [25,75], method='midpoint')


IQR = Q3 - Q1

print('median: ',median)
print('iqr: ',IQR)
print('q1: ',Q1)
print('q3: ',Q3)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Generates a box plot of absolute error
my_dict_H = {'0.1H7': all_df_fr_H['0.1H7'], '1H7': all_df_fr_H['1H7'], '0.1H15': all_df_fr_H['0.1H15'] , '1H15': all_df_fr_H['1H15'], '0.1H23': all_df_fr_H['0.1H23'], '1H23': all_df_fr_H['1H23']}
fig_H, ax_H = plt.subplots()
#ax.set_title('Box plot of all results')
bp1_H = ax_H.boxplot(my_dict_H.values())
ax_H.set_xticklabels(my_dict_H.keys())
ax_H.set_ylabel('Absolute error [°]')
ax_H.set_xlabel('Configurations')
#ax.legend([bp1["boxes"][0], bp1["boxes"][1]], ['A', 'B'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Generates a box plot of absolute error
my_dict_Q = {'0.1Q7': all_df_fr_Q['0.1Q7'], '1Q7': all_df_fr_Q['1Q7'], '0.1Q15': all_df_fr_Q['0.1Q15'] , '1Q15': all_df_fr_Q['1Q15'], '0.1Q23': all_df_fr_Q['0.1Q23'], '1Q23': all_df_fr_Q['1Q23']}
fig_Q, ax_Q = plt.subplots()
#ax.set_title('Box plot of all results')
bp1_Q = ax_Q.boxplot(my_dict_Q.values())
ax_Q.set_xticklabels(my_dict_Q.keys())
ax_Q.set_ylabel('Absolute error [°]')
ax_Q.set_xlabel('Configurations')
#ax.legend([bp1["boxes"][0], bp1["boxes"][1]], ['A', 'B'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()


