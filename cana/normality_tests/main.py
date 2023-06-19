# This Python file uses the following encoding: utf-8
import sys
import matplotlib.pyplot as plt
from glob import glob
import os
import pandas as pd
import numpy as np
import pylab
import scipy.stats as stats
import scipy.stats as shapiro
import scipy.stats as normaltest
import seaborn as sns
from statsmodels.stats.diagnostic import lilliefors


# **************** USEFUL FUNCTIONS  *******************************************************
# ******************************************************************************************
def remove_outliers(data, fill=False, threshold=3.5):
    """
    Median Absolute Deviation (MAD) based outlier detection
    Removes outliers
    """
    med = np.median(data)
    mad = np.abs(stats.median_abs_deviation(data))
    outlier = []

    for i, v in enumerate(data):
        t = (v-med)/mad
        if t > threshold:
            outlier.append(i)
        else:
            continue

    #print(outlier)
    new_data = np.delete(data, outlier)

    return new_data

# **************** START APPLICATION *******************************************************
# ******************************************************************************************


os.chdir("../dbAnalyses")
load_labelings = glob(os.path.join('full_analyse_*.csv'))  # Ants Path

n_files = len(load_labelings)
list_values = []
ii = 0

# Calling DataFrame constructor
frame_global = pd.DataFrame()
## creation new column
#frame_global['0'] = None
#frame_global['1'] = None
#frame_global['2'] = None
#print(frame_global)

# Reads raw csv, address the index
for fn in load_labelings:

    base = os.path.basename(fn)
    file_name = (os.path.splitext(base)[0])

    df = pd.read_csv(fn)
    row, col = df.shape
    index_Avg = df.columns.get_loc('Average')
    index_Ang_0 = df.columns.get_loc('Ang_0')
    print("index_Ang_0", index_Ang_0)

    error_spec = []

    for j in range(0,row):
        for i in range(0,12):
            diff = round((df.iat[j,index_Ang_0 + i] - df.iat[j,index_Avg]),3)
            error_spec.append(diff)


    mylist = [str(x) for x in error_spec]   # To eliminates nan values
    diff_list = [x for x in mylist if x != 'nan'] # Eliminates nan values

    arr = np.array(diff_list) # converting python string list to numpy string array
    num_arr = arr.astype(float) # converting numpy string array to numpy float array
    #print(np.sort(num_arr))
    len_before = len(num_arr)
    #num_arr = remove_outliers(num_arr) #  MAD outlier remove


    #------------------------------------------

    frame_local = pd.DataFrame()
    frame_local["Specialist_%i" %ii] = num_arr
    frame_global = pd.concat([frame_global,frame_local], axis=1)
    ii += 1
    #------------------------------------------

    print("")
    print("")
    print("====================================================")
    print("%s" % file_name)
    print("====================================================")
    print("Outliers removed:",len(num_arr) - len_before)
    print(num_arr)

    # =================     Normality tests      =================================================

    #      Shapiro-Wilk      #
    print("........ Shapiro-Wilk Test .........")
    stat, p = stats.shapiro(num_arr)
    print('stat=%.3f, p=%.3f' % (stat,p))
    if p > 0.05:
        print('Probably Gaussian')
    else:
        print('Probably not Gaussian')


    #         D’Agostino’s K-squared       #
    print("........ D’Agostino’s K-squared test .........")
    stat, p = stats.normaltest(num_arr)
    print('stat=%.3f, p=%.3f' % (stat,p))
    if p > 0.05:
        print('Probably Gaussian')
    else:
        print('Probably not Gaussian')


    #         Anderson-Darling       #
    print("........ Anderson-Darling Test .........")
    result = stats.anderson(num_arr)
    print('stat=%.3f' % (result.statistic))
    for o in range(len(result.critical_values)):
        sig_lev, crit_val =  result.significance_level[o], result.critical_values[o]
        if result.statistic < crit_val:
            print(f'Probably Gaussian: {crit_val} critical value at {sig_lev} level of significance')
        else:
            print(f'Probably not Gaussian: {crit_val} critical value at {sig_lev} level of significance')


    #         Chi-Square       #
    print("........ Chi-Square Test .........")
    statistic, pvalue = stats.chisquare(num_arr)
    print('stat=%.3f, p=%.3f' % ( statistic, pvalue))
    if pvalue > 0.05:
        print('Probably Gaussian')
    else:
        print('Probably not Gaussian')


    #         Lilliefors        #
    print("........ Lilliefors test .........")
    statistic, pvalue = lilliefors(num_arr)
    print('stat=%.3f, p=%.3f' % ( statistic, pvalue))
    if pvalue > 0.05:
        print('Probably Gaussian')
    else:
        print('Probably not Gaussian')


    #         Kolmogorov-Smirnov        #
    print("........ Kolmogorov-Smirnov test .........")
    statistic, pvalue = stats.kstest(num_arr,'norm')
    print('stat=%.3f, p=%.3f' % (statistic, pvalue))
    if pvalue > 0.05:
        print('Probably Gaussian')
    else:
        print('Probably not Gaussian')

    # =================     Normality plots    =================================================

    plt.figure(1)
    stats.probplot(num_arr, dist="norm", plot=pylab)
    plt.grid()

    plt.figure(2)
    ax =  sns.displot(num_arr,kde=True)
    plt.grid()
    #pylab.title("%s" % file_name)

    #plt.show()


print(frame_global)
os.chdir("../normality_tests")
frame_global.to_csv('data_distribuition.csv')
