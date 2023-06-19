"""
Criado por: Ivan Perissini
Data: 15/11/2020
Função: Blur detector
Última alteração:
"""
import cv2
from codes.core.demeter import descriptors as info
from codes.core.demeter import sptools as spt
from codes.core.demeter import dbtools as dbt
from codes.core.demeter import parameters
from codes.core.demeter import mltools as mlt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import time
from scipy.special import expit
from codes.core.demeter import colorindex
import math
import os
from sklearn.ensemble import RandomForestClassifier

# # #---------------------BLUR DETECTOR----------------------
# img_directory = "D:/Demeter Banco de imagens/.Selecionadas/Clima RUN1/"
# img_list = ['A_PANICUM_2D_PRE_I_3_R2', 'A_PANICUM_2D_PRE_III_ 3_R3', 'A_PANICUM_3D_PRE_I_2_R3',
#             'A_PANICUM_3D_PRE_II_3_R1', 'A_PANICUM_3D_PRE_IV_2_R1', 'A_PANICUM_3D_PRE_V_3_R1', 'S_CAYANA_PRE_I_3',
#             'S_PANICUM_1B_PRE_III_2', 'S_PANICUM_1B_PRE_III_3', 'S_PANICUM_1B_PRE_IV_4', 'S_PANICUM_1B_PRE_V_2',
#             'S_PANICUM_1B_PRE_V_3', 'S_PANICUM_3D_PRE_V_3', 'S_PANICUM_6B_PRE_IV_1_R3', 'S_PANICUM_6B_PRE_V_2_R1',
#             'T_PIQUETE153_PRE_I_3_R3', 'T_PIQUETE153_PRE_II_1_R1', 'T_PIQUETE153_PRE_II_4_R3',
#             'T_PIQUETE153_PRE_III_3_R1', 'T_PIQUETE153_PRE_IV_3_R2', 'T_PIQUETE153_PRE_V_3_R2']


full_dir = "D:/Demeter Banco de imagens/.Selecionadas/Blur RUN1/all/"


def blur_detect(image):
    blur_dictionary = {}
    low_limit = 1250
    high_limit = 2000

    focus_index = cv2.Laplacian(image, cv2.CV_64F).var()
    blur_dictionary['focus index'] = focus_index

    if focus_index < low_limit:
        blur_dictionary['quality'] = 'low'
    elif focus_index < high_limit:
        blur_dictionary['quality'] = 'medium'
    else:
        blur_dictionary['quality'] = 'high'

    return blur_dictionary


for root, directories, files in os.walk(full_dir):
    for file in files:
        path = root + file
        img = spt.img_read(path)
        out = blur_detect(img)
        print(file, out)


print('Finalizado')
