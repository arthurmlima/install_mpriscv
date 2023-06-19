"""
Criado por: Ivan Perissini
Data: 09/06/2020
Função: Codigo para integração dos módulos e desenvolvimento base do programa
Última alteração:
"""

from core.demeter import sptools as spt
from core.demeter import mltools as mlt
from core.demeter import metadata
from core.demeter import dbtools as dbt
from core.demeter import descriptors as info
from core.demeter import parameters
from core.demeter import colorindex
from core.demeter import results
import matplotlib.pyplot as plt
import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import os
import math
from joblib import dump, load
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# #================================ MAIN CODE ======================================

# #----------------------CODE TO PRE PROCESS THE IMAGENS---------------------
# img_directory = "D:/Drive/PROFISSIONAL/Projeto Drone/Banco de dados/Imagens/Imagens Drone/Selecionadas3/"
# img_directory = r"D:\\Drive\\PROFISSIONAL\\Projeto Drone\\Banco de dados\\Imagens\\Imagens Drone\\Selecionadas3\\"
# out_dir = img_directory + r'output\\'
# # img_list = ['PRE_III_ 3_R1.JPG', 'teste.JPG', 'POS_I_1_R1.JPG']
# img_list = ['T_POS_IV_1_R1.JPG', 'T_PRE_II_ 1_R3.JPG']
# # img_list = ['D_4_A_100M.JPG', 'D_6_B_120M_PRE.JPG']

# for img_name in img_list:
#     img_path = img_directory + img_name
#     dbt.image_db(img_path, out_dir, debug=True)

# img_db_dir = out_dir

# img = spt.img_read(img_directory + img_list[1])
# spt.img_show_share(img4=img, img3=img, img2=img, img1=img, share=True)

# for img_name in img_list:
#     img_path = img_directory + img_name
#     db_path = img_db_dir + img_name + '.csv'
#     print(db_path)
#     img_label, img_out, img_class = dbt.img_test(image_path=img_path,
#                                                  database_path=db_path,
#                                                  model_name='RF_1596561713',
#                                                  debug=True,
#                                                  save=False)


# # # # # # # # # # # #
# # # PARKING LOT # # #
# # # # # # # # # # # #

# # -------------------CODE TO EVALUATE FEATURES -------------------

# db_name = 'bloco'
# db_name = 'bloco_cat'
# db_dir = r'D:\\Demeter Banco de imagens\\Testes\\Cor\\'
# db_path = db_dir + db_name + '.csv'
# db_data = pd.read_csv(db_path, delimiter=';')
#
# mlt.descriptor_evaluation(db_data, n_features=5)

# # # -------------------CODE TO TRAIN THE MODELS -------------------
# model = RandomForestClassifier(n_estimators=10, random_state=20, max_depth=6)
# model = mlt.train_model(model, X_train, y_train, model_name='RF', save_model=False)
# mlt.model_metrics(model, X_test, y_test)
# # mlt.model_show_tree(model, X_train, n_tree=3, output_path='results/trees/')

# # ------------------- CODE TO EVALUATE MULTIPLE IMAGES -------------------
# # # PHASE 1
# img_list = ['im_extra_001.JPG', 'im_Outro_001.JPG', 'im_Outro_002.JPG', 'im_Outro_003.JPG',
#             'im10_001.JPG', 'im10_002.JPG', 'im10_003.JPG', 'im10_004.JPG', 'im10_005.JPG', 'im10_006.JPG',
#             'im30_001.JPG', 'im30_002.JPG', 'im30_003.JPG', 'im30_004.JPG', 'im30_005.JPG',
#             'teste1.JPG', 'teste2.JPG', 'teste3.JPG', 'teste4.JPG', 'teste5.JPG']
#
# img_directory = r'D:\\Drive\\PROFISSIONAL\\Projeto Drone\\Imagens\\Imagens Drone\\Selecionadas\\'
# img_db_dir = r'C:\\Projetos\\Demeter\\codes\\core\\database\\Output\\Imagens\\'

# # # PHASE 2
# img_list = ['POS IV - 2.JPG', 'POS II - 2.JPG', 'pos-V-1.JPG', 'POS-V-3.JPG', 'PRE I - 1.JPG',
#             'pre-I-1.JPG', 'pre-I-4.JPG', 'PRE-III- 4.JPG']
#
# img_directory = r'D:\\Drive\\PROFISSIONAL\\Projeto Drone\\Imagens\\Imagens Drone\\Selecionadas2\\'
# img_db_dir = r'C:\\Projetos\\Demeter\\codes\\core\\database\\Output2\\2000\\'

# # selected_img = [img_list[0]]  # Modo manual
# selected_img = img_list  # Modo lista
#
# for img_name in selected_img:
#     print(img_name)
#     img_path = img_directory + img_name
#     db_path = img_db_dir + img_name + '.csv'
#     img_label, img_out, img_class = dbt.img_test(image_path=img_path,
#                                                  database_path=db_path,
#                                                  # model_name='model_29_07_2020_H10_52',
#                                                  model_name='RF_1596561713',
#                                                  debug=True,
#                                                  save=True)
#
# spt.img_show(img_label)
# spt.img_show(img_out)