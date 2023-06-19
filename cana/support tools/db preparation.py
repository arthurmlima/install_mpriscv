"""
Criado por: Ivan Perissini
Data: 15/11/2020
Função: Data base Preparation
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

# operation = "image"
# operation = "label"
operation = "merge"

# # Remove SettingWithCopyWarning Warnings
pd.set_option('mode.chained_assignment', None)

# # #---------------------Image Data Base Preparation----------------------
if operation == "image":
    print("Image database generation started...")
    # Required r"\\" format
    img_dir = r"D:\\Demeter Banco de imagens\\.Selecionadas\\Todas\\"
    img_db_dir = r"D:\\Demeter Banco de imagens\\.Selecionadas\\Dados\\Image DB\\"
    file_filter = ''

    for root, directories, files in os.walk(img_dir):
        for file in files:
            if file_filter in file:
                print("Image being processed:", file)
                img_path = root + file
                img = spt.img_read(img_path)
                additional_info = info.image_info(img_path)
                dbt.image_db_full(img_path, img_db_dir, image_info=additional_info, debug=False)

# # #---------------------Label per Image Data Base Preparation----------------------
if operation == "label":
    print("label database generation started...")
    # Required r"\\" format
    label_dir = r"D:\\Demeter Banco de imagens\\.Selecionadas\\Dados\\RUN\\Run3\\"
    label_db_dir = r"D:\\Demeter Banco de imagens\\.Selecionadas\\Dados\\Label DB\\"
    db_master_path = r"D:\\Demeter Banco de imagens\\.Selecionadas\\Master DB.csv"
    file_filter = ''
    debug = True

    # ---- a. Access image numbers and labelers ----
    img_number = []
    person_list = []
    name_position = -3
    for root, directories, files in os.walk(label_dir):
        for file in files:
            img_number.append(file.split(sep='_')[0])
            # person_list.append("label_" + file.split(sep='_')[2])
            person_list.append("label_" + file.split(sep='_')[name_position])


    # # Remove duplicates
    img_number = dict.fromkeys(img_number)
    person_list = dict.fromkeys(person_list)
    # print(img_number, person)

    # ---- b. Access data from Master DB ----
    run_name = "Rotulação Run 3"
    df_master = pd.read_csv(db_master_path, delimiter=';')
    # print(df_master)

    df_run_names = df_master.filter(["Filename", run_name])
    # # --- Removes NaN ----
    # df_run_names = df_run_names.dropna(thresh=2)
    # df_run_names = df_run_names.dropna(how='all', subset=[run_name])
    df_run_names = df_run_names.loc[pd.notna(df_run_names[run_name])]
    # print(df_run_names)

    # # --- Do the file names correspondence ----
    original_name = df_run_names["Filename"]
    run_name = df_run_names[run_name]
    for key, value in img_number.items():
        number = str(int(key)) + '_'
        name = original_name.loc[run_name.str.startswith(number)]
        img_number[key] = list(name)[0]
    # print(img_number, person_list)

    # ---- c. Creates a default header for DB ----
    label_db_header = {"test_filename": None,
                       "original_filename": None,
                       "SP": None,
                       "ID": None
                       }

    label_db_header.update(person_list)

    label_df = pd.DataFrame()  # Set to null the dataframe
    label_df['ID'] = None

    if debug:
        print('Data found verification:')
        print('-Run images found and original file names:\n', img_number)
        print('-Labelers found:\n', person_list)
        print(label_db_header)
        print()
        input("Press Enter to continue...")

    # ---- d. DB verification and generation Loop ----
    # For each image db file
    for test_name, orig_name in img_number.items():
        # For each labeler
        for person in list(person_list):
            # Find the respective file in dir
            for root, directories, files in os.walk(label_dir):
                for file in files:
                    if file_filter in file:
                        if file.split(sep='_')[0] == test_name and \
                                "label_" + file.split(sep='_')[name_position] == person:
                            df_loop = pd.read_csv(label_dir + file, delimiter=';')
                            # e. Verify ID and add new value
                            for index in df_loop.index:
                                # For each row, creates an ID
                                ID = orig_name + '_SP' + str(df_loop['BLOCO'][index])

                                # Check if ID exists
                                if (label_df['ID'] == ID).any():
                                    label_df[person].loc[label_df['ID'] == ID] = df_loop['CLASSIFICACAO'][index]
                                else:
                                    # If ID is not found, create new entry
                                    label_db_header.update(person_list)  # Clear all labelers values
                                    label_db_header["test_filename"] = test_name
                                    label_db_header["original_filename"] = orig_name
                                    label_db_header["SP"] = df_loop['BLOCO'][index]
                                    label_db_header["ID"] = ID
                                    label_db_header[person] = df_loop['CLASSIFICACAO'][index]

                                    # Create data frame from dictionary
                                    new_label_df = pd.DataFrame(label_db_header, index=[0])
                                    label_df = pd.concat([label_df, new_label_df])

        output_file = label_db_dir + orig_name + "_label.csv"
        label_df.to_csv(output_file, sep=';', mode='a', header=not os.path.exists(output_file), index=False)
        print('New database file generated:', output_file)
        label_df = pd.DataFrame()  # Set to null the dataframe
        label_df['ID'] = None

# # #---------------------Merge Image Data Base and label Data Base to single file----------------------
if operation == "merge":
    print("Consolidated database generation started...")
    # Required r"\\" format
    img_db_dir = r"D:\\Demeter Banco de imagens\\.Selecionadas\\Dados\\Image DB\\"
    label_db_dir = r"D:\\Demeter Banco de imagens\\.Selecionadas\\Dados\\Label DB\\"
    full_db_path = r"D:\\Demeter Banco de imagens\\.Selecionadas\\Dados\\"

    file_filter = ''
    labeler_rank = ("label_ROGESTER GOMES", "label_NÍCOLAS", "label_CAMILA BRIZOLARI")

    label_only = False

    # For each label DB file
    df_full = pd.DataFrame()
    for root_label, directories_label, files_label in os.walk(label_db_dir):
        for file_label in files_label:
            if file_filter in file_label:

                if label_only:
                    print("Processing ", label_db_dir + file_label)
                    df_label = pd.read_csv(label_db_dir + file_label, delimiter=';')
                    # Filters the columns with label prefix
                    df_only_label = df_label.loc[:, df_label.columns.str.contains("label")]

                    # Change labels to only numbers use external parameters to perform label conversion
                    label_change = {value[2]: value[0] for key, value in parameters.get_label_info().items()}
                    for original, new in label_change.items():
                        df_only_label.replace(original, new, inplace=True)
                        df_label.replace(original, new, inplace=True)
                    # print('Labels were exchanged to numbers only')

                    # Divergence measure for the labels (1.5 for inicial cut and 3 labelers)
                    df_label["label_std"] = df_only_label.std(axis=1, skipna=True, numeric_only=None)
                    df_label["label_std"].loc[
                        pd.isna(df_label["label_std"])] = 0  # Ensures 0 std to null values

                    # Mean value rounded to int
                    df_label["label_mean"] = df_only_label.mean(axis=1, skipna=True, numeric_only=None).round(0)

                    # Most frequent value
                    df_label["label_mode"] = df_only_label.mode(axis=1, dropna=True, numeric_only=None)[0]

                    # Label by best labeler
                    df_label["label_rank"] = None
                    for labeler in labeler_rank:
                        df_label["label_rank"].loc[pd.isna(df_label["label_rank"])] = df_label[labeler]

                    # Label mix
                    df_label["label_mix"] = None
                    df_label["label_mix"].loc[df_label["label_std"] > 1.5] = df_label["label_rank"]
                    df_label["label_mix"].loc[df_label["label_std"] <= 1.5] = df_label["label_mean"]

                    # Label Best Compromise
                    df_label["label_best"] = None
                    df_label["label_best"].loc[df_label["label_std"] <= 1.5] = df_label["label_mean"]
                    df_label["label_best"].loc[(df_label["label_rank"] > 5) | (df_label["label_std"] > 1.5)] \
                        = df_label["label_rank"]

                    df_full_img = df_label
                    df_full = pd.concat([df_full, df_full_img])

                else:
                    # Finds the correspondent image DB file
                    for root_img, directories_img, files_img in os.walk(img_db_dir):
                        for file_img in files_img:
                            if file_img.split(sep='_img')[0] == file_label.split(sep='_label')[0]:
                                orig_name = file_label.split(sep='_label')[0]

                                print("Image DB found at", img_db_dir + file_img)
                                df_label = pd.read_csv(label_db_dir + file_label, delimiter=';')
                                df_img = pd.read_csv(img_db_dir + file_img, delimiter=';')

                                # Generate the species group column
                                df_img["inf_species_group"] = df_img["inf_species"]
                                for original, new in parameters.get_species_group().items():
                                    df_img["inf_species_group"].replace(original, new, inplace=True)

                                # Filters the columns with label prefix
                                df_only_label = df_label.loc[:, df_label.columns.str.contains("label")]

                                # Change labels to only numbers use external parameters to perform label conversion
                                label_change = \
                                    {value[2]: value[0] for key, value in parameters.get_label_info().items()}
                                for original, new in label_change.items():
                                    df_only_label.replace(original, new, inplace=True)
                                    df_label.replace(original, new, inplace=True)
                                # print('Labels were exchanged to numbers only')

                                # Divergence measure for the labels (1.5 for inicial cut and 3 labelers)
                                df_label["label_std"] = df_only_label.std(axis=1, skipna=True, numeric_only=None)
                                df_label["label_std"].loc[
                                    pd.isna(df_label["label_std"])] = 0  # Ensures 0 std to null values

                                # Mean value rounded to int
                                df_label["label_mean"] = \
                                    df_only_label.mean(axis=1, skipna=True, numeric_only=None).round(0)

                                # Most frequent value
                                df_label["label_mode"] = \
                                    df_only_label.mode(axis=1, dropna=True, numeric_only=None)[0]

                                # Label by best labeler
                                df_label["label_rank"] = None
                                for labeler in labeler_rank:
                                    df_label["label_rank"].loc[pd.isna(df_label["label_rank"])] = df_label[labeler]

                                # Label mix
                                df_label["label_mix"] = None
                                df_label["label_mix"].loc[df_label["label_std"] > 1.5] = df_label["label_rank"]
                                df_label["label_mix"].loc[df_label["label_std"] < 1.5] = df_label["label_mean"]

                                # Label Best Compromise
                                df_label["label_best"] = None
                                df_label["label_best"].loc[df_label["label_std"] < 1.5] = df_label["label_mean"]
                                df_label["label_best"].loc[(df_label["label_rank"] > 5) |
                                                           (df_label["label_std"] > 1.5)] = df_label["label_rank"]

                                # # Visualize the label preparation
                                # output_file = full_db_path + orig_name + "_CleanLabel.csv"
                                # df_label.to_csv(output_file, sep=';', mode='w', header=True, index=False)

                                # # Remove all descriptors if needed
                                # df_img = df_img.filter(regex='inf|ID', axis=1)

                                # Merge DF based on label information at column ID
                                df_full_img = pd.merge(df_label, df_img, how="left", on=["ID"])
                                df_full = pd.concat([df_full, df_full_img])

                                # # For a Full DB per Image
                                # output_file = full_db_path + orig_name + "_full.csv"
                                # df_full_img.to_csv(output_file, sep=';', mode='a',
                                # header=not os.path.exists(output_file), index=False)
                                # print('New database file generated:', output_file)
                                # df_full_img = pd.DataFrame()  # Set to null the dataframe

    if label_only:
        output_file = full_db_path + "DB_label_full.csv"
    else:
        output_file = full_db_path + "DBfull.csv"

    # df_full.to_csv(output_file, sep=';', mode='a', header=not os.path.exists(output_file), index=False)
    df_full.to_csv(output_file, sep=';', mode='w', header=True, index=False)
    print('New database file generated:', output_file)
    df_full = pd.DataFrame()  # Set to null the dataframe

print('\n ~~~~~~~~END~~~~~~~~')
