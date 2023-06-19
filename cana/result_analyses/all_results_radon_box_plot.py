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

# Acess all orientation angle results files
os.chdir("RADON")

# data_performance_0.1_half.csv
# data_performance_0.1_quarter.csv
# data_performance_0.5_half.csv
# data_performance_0.5_quarter.csv
# data_performance_1_half.csv
# data_performance_1_quarter.csv

# Creates dataframes based final results files
df_fr_01H = pd.read_csv('data_performance_0.1_half.csv')
df_fr_01Q = pd.read_csv('data_performance_0.1_quarter.csv')
df_fr_05H = pd.read_csv('data_performance_0.5_half.csv')
df_fr_05Q = pd.read_csv('data_performance_0.5_quarter.csv')
df_fr_1H  = pd.read_csv('data_performance_1_half.csv')
df_fr_1Q  = pd.read_csv('data_performance_1_quarter.csv')

# Since our dataframe has no column name, let's give them
colm_names = ['0.1H','0.1Q','0.5H','0.5Q' ,'1H','1Q']
# Creates our dataframes to be based on final results files
all_df_fr = pd.DataFrame(columns=colm_names)

# Incorporates the data to our dataframe
all_df_fr['0.1H'] = df_fr_01H['ABS ERROR']
all_df_fr['0.1Q'] = df_fr_01Q['ABS ERROR']
all_df_fr['0.5H'] = df_fr_05H['ABS ERROR']
all_df_fr['0.5Q'] = df_fr_05Q['ABS ERROR']
all_df_fr['1H']  =  df_fr_1H['ABS ERROR']
all_df_fr['1Q']  =  df_fr_1Q['ABS ERROR']

print(all_df_fr)

# Generates a box plot of absolute error
my_dict = {'0.1H': all_df_fr['0.1H'], '0.1Q': all_df_fr['0.1Q'], '0.5H': all_df_fr['0.5H'] , '0.5Q': all_df_fr['0.5Q'], '1H': all_df_fr['1H'], '1Q': all_df_fr['1Q']}
fig, ax = plt.subplots()
#ax.set_title('Box plot of all results')
bp1 = ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())
ax.set_ylabel('Absolute error [°]')
ax.set_xlabel('Configurations')
#ax.legend([bp1["boxes"][0], bp1["boxes"][1]], ['A', 'B'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


