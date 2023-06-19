"""
Criado por: Ivan Perissini
Data: 15/11/2020
Função: Weather detector
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
from sklearn.ensemble import RandomForestRegressor

# # # #---------------------WEATHER DETECTOR----------------------
img_directory = r"D:\\Demeter Banco de imagens\\.Selecionadas\\Clima RUN 1\\"
# img_list = ['A_PANICUM_2D_PRE_I_3_R2', 'A_PANICUM_2D_PRE_III_ 3_R3', 'A_PANICUM_3D_PRE_I_2_R3',
#             'A_PANICUM_3D_PRE_II_3_R1', 'A_PANICUM_3D_PRE_IV_2_R1', 'A_PANICUM_3D_PRE_V_3_R1', 'S_CAYANA_PRE_I_3',
#             'S_PANICUM_1B_PRE_III_2', 'S_PANICUM_1B_PRE_III_3', 'S_PANICUM_1B_PRE_IV_4', 'S_PANICUM_1B_PRE_V_2',
#             'S_PANICUM_1B_PRE_V_3', 'S_PANICUM_3D_PRE_V_3', 'S_PANICUM_6B_PRE_IV_1_R3', 'S_PANICUM_6B_PRE_V_2_R1',
#             'T_PIQUETE153_PRE_I_3_R3', 'T_PIQUETE153_PRE_II_1_R1', 'T_PIQUETE153_PRE_II_4_R3',
#             'T_PIQUETE153_PRE_III_3_R1', 'T_PIQUETE153_PRE_IV_3_R2', 'T_PIQUETE153_PRE_V_3_R2']
#
#
# full_dir = "D:/Demeter Banco de imagens/.Selecionadas/Todas/"
#
# for root, directories, files in os.walk(full_dir):
#     for file in files:
#         path = root + file
#         img = spt.img_read(path)
#         out = info.weather_detect(img, debug=False)
#         print(file, out)
#
# # ~~~~~~~ Descriptor generation ~~~~~~
# for img_name in img_list:
#     img_path = img_directory + img_name + '.JPG'
#     img = spt.img_read(img_path)
#
#     out = info.weather_detect(img, debug=True)
#     print(out)
#     # data_df = dbt.descriptor_db(img, img_identifier=None,
#     #                             conversion_list=parameters.get_conversion_list(),
#     #                             index_list=parameters.get_index_list(),
#     #                             img_results=None,
#     #                             show_time=True)
#     #
#     # print(data_df)
#     #
#     # # Append data to existing data base file
#     # output_path = img_directory + 'descritores_full.csv'
#     # data_df.to_csv(output_path, sep=';', mode='a', header=not os.path.exists(output_path), index=False)
#     # print('Database file updated due to ', output_path)

# ~~~~~~~ Descriptor evaluation ~~~~~~
output_path = img_directory + 'descritores_run1.csv'
print(output_path)
data_raw = pd.read_csv(output_path, delimiter=';', encoding='ISO-8859-1')

change_dic = {'Secas': 1, 'Trans': 2, 'Aguas': 3,
              'Dia': 1, 'Extremo': 0,
              'Sol aberto': 1, 'Nublado': 0,
              'Baixo': 1, 'Medio': 2, 'Alto': 3}

data_ml = data_raw.copy()
for original, new in change_dic.items():
    data_ml.replace(original, new, inplace=True)

n_features = 5

# y_analysis = 'Cg_noise'
y_analysis = 'Epoca'
# y_analysis = 'Luz'
class_names = ('Secas', 'Trans', 'Aguas')
# class_names = ('Nublado', 'Sol')
# y_analysis = 'Epoca'
# y_analysis = 'Horario'
non_descriptor_list = ['Image', 'Data', 'hist', 'Epoca', 'Horario', 'Luz', 'Luz_orig', 'Grupo', 'Clima', 'Cg_noise']
keep_list = ['Mean']
# keep_list = ['']
non_descriptor_list.remove(y_analysis)

X_train, X_test, y_train, y_test = mlt.prepare_data(data_ml,
                                                    remove_list=non_descriptor_list,
                                                    keep_only_list=keep_list,
                                                    y_name=y_analysis,
                                                    test_size=0.3)


# # Categorical evaluation
# mlt.descriptor_evaluation(data_raw,
#                           y_type='',
#                           remove_descriptor=non_descriptor_list,
#                           keep_descriptor=keep_list,
#                           y_name=y_analysis,
#                           n_features=n_features, random_state=20, test_size=0)

# # Numerical evaluation
# mlt.descriptor_evaluation(data_ml,
#                           remove_descriptor=non_descriptor_list,
#                           keep_descriptor=keep_list,
#                           y_name=y_analysis,
#                           n_features=n_features, random_state=20, test_size=0)

from sklearn.cluster import KMeans

# kmeans = KMeans(init="k-means++", n_clusters=5, n_init=4, random_state=0)
# bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)
# print(bench_k_means)

# model = RandomForestClassifier(n_estimators=5, random_state=20, max_depth=2)
# model = mlt.train_model(model, X_train, y_train, model_name='RF', save_model=False)
# mlt.model_metrics(model, X_test, y_test, class_names=class_names)

feat_list, model = mlt.feature_analysis_quantitative(X_train, y_train, k=2, method='mir', random_state=10)
feat_list, model = mlt.feature_analysis_quantitative(X_train, y_train, k=2, method='tree', random_state=10)

# model = RandomForestRegressor(max_depth=5, random_state=20)
# model = mlt.train_model(model, X_train, y_train, model_name='RF', save_model=False)
mlt.model_metrics(model, X_test, y_test, class_names=class_names)

# mlt.model_show_tree(model, X_train, class_names=class_names, n_tree=3)

print('Finalizado')
