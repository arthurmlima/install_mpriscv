"""
Criado por: Ivan Perissini
Data: 15/11/2020
Função: Model
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
from sklearn.model_selection import train_test_split

# # #---------------------MODELING----------------------

# Required r"\\" format
full_db_path = r"D:\\Demeter Banco de imagens\\.Selecionadas\\Dados\\DBfull.csv"
# full_db_path = r"D:\\Demeter Banco de imagens\\.Selecionadas\\Dados\\DBfull_Sabia.csv"
model_out_path = r"D:\\Demeter Banco de imagens\\.Selecionadas\\Modelos\\"

save_db = False
save_model = False
show_results = True

dict_list = parameters.get_models_list()[1:3]

print('\n~~~~~~~ SEARCHING DATABASE ~~~~~~~')
print("Searching Database in", full_db_path)
df_full = pd.read_csv(full_db_path, delimiter=';')
print("Database loaded with", df_full.shape[1], 'parameters and', df_full.shape[0], 'lines')

result_df = pd.DataFrame()
for model_dictionary in dict_list:
    random_state = 10

    model, X, y = mlt.generate_model(model_dictionary,
                                     data_base=df_full,
                                     output_path=model_out_path,
                                     save_db=save_db,
                                     save_model=save_model,
                                     random_state=random_state)

    print("Current database for model has", X.shape[1], 'parameters and', X.shape[0], 'lines')

    if model is not None:
        # # -------------------EVALUATE THE MODELS -------------------
        print('\n~~~~~~~ MODEL EVALUATION ~~~~~~~')
        class_dic, class_count = mlt.category_evaluation(y, show_results=show_results)

        if model_dictionary['balance'] == 'true':
            X_balance, y_balance = mlt.data_base_balance(X, y, method='auto', sampling_strategy='moderate')
            class_dic, class_count = mlt.category_evaluation(y_balance, show_results=show_results)
            X, y = X_balance, y_balance
            print("Rebalanced database for model has", X.shape[1], 'parameters and', X.shape[0], 'lines')

        score_dic = mlt.cross_validation_metrics(model, X, y, cv=5)

        # Ideally no simulated data should go to model testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
        report_dic = mlt.model_metrics(model, X_test, y_test, class_names=[], y_type='categorical',
                                       show_matrix=show_results)

        model_dictionary = {key: val.replace('+', 'keep_') for key, val in model_dictionary.items()}
        model_dictionary = {key: val.replace('-', 'remove_') for key, val in model_dictionary.items()}
        result_dic = model_dictionary
        result_dic.update(class_dic)
        result_dic.update(score_dic)
        result_dic.update(report_dic)

        # Flattens the dictionary into a single df entry
        new_result_df = pd.json_normalize(result_dic, sep='_')

        # new_result_df = pd.DataFrame(result_dic, index=[0])
        result_df = pd.concat([result_df, new_result_df])

print('\n~~~~~~~ RESULTS ~~~~~~~')
print(result_df)
result_path = model_out_path + str(int(time.time())) + '_results.csv'
result_df.to_csv(result_path, sep=';', mode='a', header=not os.path.exists(result_path), index=False)
print('New database file generated at:', result_path)

print('\n ~~~~~~~> E N D <~~~~~~~')
