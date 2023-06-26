"""
Criado por: Andre Luiz Santos e Ivan C Perissini
Data: 15/07/2020
Função: Módulo com conjunto usual de ferramentas para trabalhar com o banco de dados de imagens e pandas
Última alteração: 15/07/2020
"""

from core.demeter import sptools as spt
from core.demeter import descriptors as info
from core.demeter import colorindex
from core.demeter import metadata
from core.demeter import results
from core.demeter import parameters
import time
from datetime import datetime
import numpy as np
import pandas as pd
import os
from numba import jit
import cv2
import math
from joblib import dump, load


def __version__():
    return 'dbtools version: v0.2'


# ======================== SUPPORT TOOLS =========================

# ~~~~~~~~~~~~~~~~~~ Removes duplicates ~~~~~~~~~~~~~~~~~~~
def remove_duplicates(input_dir, database_name, column_name='item'):
    """
    Given a path and database, the function removes duplicates based on a specific column name
    :param input_dir:
    :param database_name:
    :param column_name:
    :return:
    """
    input_path = input_dir + database_name + '.csv'

    original_df = pd.read_csv(input_path, delimiter=';')
    new_df = original_df.drop_duplicates(subset=column_name, keep='last', inplace=False)

    duplicates_found = original_df.shape[0] - new_df.shape[0]
    original_df = pd.DataFrame()  # Set to null the original dataframe

    print(duplicates_found, 'Duplicates were found and removed')

    # Append data to existing data base
    new_path = input_path.split('.csv')[0] + '_unique.csv'
    new_df.to_csv(new_path, sep=';', mode='w', index=False)
    print('New database generated at:', new_path)

    return 0


# ~~~~~~~~~~~~~~~~~~ Cleans the database ~~~~~~~~~~~~~~~~~~~
# Removes duplicates, change labels to number only and round data to 8 decimals
def clean_db(input_dir, database_name, column_name='item'):
    input_path = input_dir + database_name + '.csv'

    original_df = pd.read_csv(input_path, delimiter=';')
    new_df = original_df.drop_duplicates(subset=column_name, keep='last', inplace=False)

    duplicates_found = original_df.shape[0] - new_df.shape[0]
    original_df = pd.DataFrame()  # Set to null the original dataframe

    print(duplicates_found, 'Duplicates were found and removed')

    # Change labels to only numbers use external parameters to perform label conversion
    label_change = {value[2]: value[0] for key, value in parameters.get_label_info().items()}
    for original, new in label_change.items():
        new_df['label'].replace(original, new, inplace=True)
    print('Labels were exchanged to numbers only')

    new_df = new_df.round(8)
    print('All data was rounded to float32')

    # Append data to existing data base
    new_path = input_path.split('.csv')[0] + '_clean.csv'
    new_df.to_csv(new_path, sep=';', mode='w', index=False)
    print('New database generated at:', new_path)

    return 0


# ======================== DATABASE TOOLS =========================

# ~~~~~~~~~~~~~~~~~~ Descriptor Generator ~~~~~~~~~~~~~~~~~~~
# The main function to generate the database, given an image the command integrates different data sources
# The function also calls for the conversion and indexes, and in sequence calculates the respective descriptors
# Can be used for database generation for training or simply for descriptor calculation for model usage
def descriptor_db(image, img_identifier, conversion_list, index_list, img_results, show_time=False):
    data = {}
    show_cvt_time = show_time
    show_desc_time = show_time

    if img_identifier is not None:
        data.update(img_identifier)

    # Original Image Descriptors
    descriptor_list = ['full_statistics', 'histogram', 'moments', 'glcm', 'lbp', 'multiple_hog']
    data.update(info.multiple_descriptors(image, descriptor_list, label='img', show_time=show_desc_time))

    # Color Space Descriptors
    if conversion_list is not None:
        descriptor_list = ['statistics', 'histogram']
        image_vet = colorindex.multiple_conversions(image, conversion_list, no_background=True, show_time=show_cvt_time)
        for index, img in enumerate(image_vet):
            data.update(info.multiple_descriptors(img, descriptor_list,
                                                  label=conversion_list[index],
                                                  show_time=show_desc_time))

    # Color Index Descriptors
    if index_list is not None:
        descriptor_list = ['statistics', 'histogram']
        image_vet = colorindex.multiple_indexes(image, index_list, no_background=True, show_time=show_cvt_time)
        for index, img in enumerate(image_vet):
            data.update(info.multiple_descriptors(img, descriptor_list,
                                                  label=index_list[index],
                                                  show_time=show_desc_time))

    # Image Results
    if img_results is not None:
        data.update(img_results)

    # Create data frame from dictionary
    entry_df = pd.DataFrame(data, index=[0])

    return entry_df


# ~~~~~~~~~~~~~~~~~~ Create database ~~~~~~~~~~~~~~~~~~~
# This function simply automates the database generation process
# Given a label file and a image folder
def create_db(input_dir, output_dir, image_dir, img_format='.JPG', database_name='',
              conversion_list='', index_list='', debug=False):
    # Case no database name is provided uses date/time as name
    if database_name == '':
        database_name = 'DB_D' + datetime.now().strftime('%d_%m_%Y') + datetime.now().strftime('_H%H_%M')

    # Case no list is provided uses all available
    if conversion_list == '':
        conversion_list = parameters.get_conversion_list()

    if index_list == '':
        index_list = parameters.get_index_list()

    # Generates output path
    output_path = output_dir + database_name + '.csv'

    file_filter = ''
    if debug:
        print('----DEBUG MODE----')
    # Access all files at input folder
    for root, directories, files in os.walk(input_dir):
        for file in files:
            # Consider if provided a file name filter
            if file_filter in file:
                try:
                    file_name, file_extension = os.path.splitext(file)

                    # Based on label file name, searches for the respective image
                    nome_img = (file_name.split('_raw_data')[0] + img_format)
                    image_path = image_dir + nome_img
                    if debug:
                        print('Image found at:', image_path)

                    # Generate all the image SuperPixels
                    h, lat, long = metadata.get_gps(image_path)
                    _, _, _, sp_vector, box = spt.img_superpixel(image_path, debug=debug)
                    center_x = np.uint16((box[1] + box[0]) / 2)
                    center_y = np.uint16((box[3] + box[2]) / 2)

                    # Access data from input file
                    label_path = input_dir + file_name + '.csv'
                    df = pd.read_csv(label_path, delimiter=';')
                    column_label = df['CLASSIFICACAO']  # Class
                    column_sp = df['BLOCO']             # Block
                    column_date = df['DATA']            # Date
                    column_time = df['HORA']            # Time

                    # If the database already exists, access the row number
                    if os.path.exists(output_path):
                        original_db = pd.read_csv(output_path, delimiter=';')
                        max_row = original_db.shape[0] + 1
                        original_db = pd.DataFrame()  # Set to null the dataframe
                    else:
                        max_row = 1

                    mode = 'original'
                    df_db = pd.DataFrame()  # Set to null the dataframe
                    # For each SuperPixel in label file
                    for index, n_sp in enumerate(column_sp):
                        # Capture the basic image information
                        img_identifier = {'ID': (max_row + index),
                                          'item': file_name.split('_raw_data')[0] + '_' + str(n_sp) + '_' + mode,
                                          'image_name': nome_img,
                                          '#_SP': n_sp,
                                          'mode': mode,
                                          'label_date': column_date[index],
                                          'label_time': column_time[index],
                                          'pos_x': center_x[n_sp],
                                          'pos_y': center_y[n_sp],
                                          'GPS_alt': h,
                                          'GPS_lat': lat,
                                          'GPS_long': long}

                        # Saves the label information
                        img_results = {'label': column_label[index],
                                       'date': datetime.now().strftime('%d/%m/%Y'),
                                       'time': datetime.now().strftime('%H:%M:%S')}

                        # Generates the SP and descriptors information
                        df_new_img = descriptor_db(sp_vector[n_sp],
                                                   img_identifier,
                                                   conversion_list,
                                                   index_list,
                                                   img_results)

                        if debug:
                            print('# SuperPixel:', n_sp, '\t', (index + 1), 'Processados')

                        # Append the entry to the database
                        df_db = pd.concat([df_db, df_new_img])

                    if debug:
                        print('First entries for new image:', file_name)
                        print(df_db.head())

                    # Append data to existing data base file
                    df_db.to_csv(output_path, sep=';', mode='a', header=not os.path.exists(output_path), index=False)
                    print('Database file updated due to ', file_name)

                except:
                    print('Access error:', file_name)

    return 0


# ~~~~~~~~~~~~~~~~~~ Image database ~~~~~~~~~~~~~~~~~~~
# This function generates the file containing the complete image descriptor
# Given an image path, the imagem is divided in SP and each generate a new line at the file output
def image_db(image_path, output_dir, debug=True):
    _, _, _, sp_vector, box = spt.img_superpixel(image_path, debug=debug)

    conversion_list = parameters.get_conversion_list()
    index_list = parameters.get_index_list()
    # index_list = parameters.get_old_index_list()

    # New dataframe
    df_database = pd.DataFrame()
    print('------------------------------------------------------')
    print('Image database generation started')
    ti = time.time()
    # For each SP calculate all the descriptors selected
    for index, SP_img in enumerate(sp_vector):
        n_sp = {'nSP': index}
        df_new_img = descriptor_db(SP_img,
                                   img_identifier=n_sp,
                                   conversion_list=conversion_list,
                                   index_list=index_list,
                                   img_results=None)
        # Append data to the database
        df_database = pd.concat([df_database, df_new_img])

    print('Process ended in [s]:' + str(time.time() - ti))
    print('Database preview:')
    print(df_database)

    # Save results
    img_name = image_path.split(sep=r'\\')[-1]
    output_path = output_dir + img_name + '.csv'
    df_database.to_csv(output_path, sep=';', mode='a', header=not os.path.exists(output_path), index=False)
    print('New database file generated at:', output_path)

    return 0


# ~~~~~~~~~~~~~~~~~~ Image database ~~~~~~~~~~~~~~~~~~~
# This function generates the file containing the complete image descriptor
# Given an image path, the imagem is divided in SP and each generate a new line at the file output
def image_db_full(image_path, output_dir, image_info={}, debug=True):
    _, _, _, sp_vector, box = spt.img_superpixel(image_path, debug=debug)
    img_name = image_path.split(sep=r'\\')[-1]
    conversion_list = parameters.get_conversion_list()
    index_list = parameters.get_index_list()

    # New dataframe
    df_database = pd.DataFrame()
    print('------------------------------------------------------')
    print('Image database generation started')
    ti = time.time()
    # For each SP calculate all the descriptors selected
    for index, SP_img in enumerate(sp_vector):
        n_sp = {'inf_nSP': index,
                'ID': img_name + '_SP' + str(index)}
        image_info.update(n_sp)

        # Crop only the relevant data to the calculations
        xi = box[0][index]
        xf = box[1][index]
        yi = box[2][index]
        yf = box[3][index]

        # The SP is croped to fit only the relevant data
        SP_crop = SP_img[0:xf - xi, 0:yf - yi, :]

        # # Debug lines
        # if index > 10:
        #     break
        # if index > 530:
        #     spt.img_show(img1=SP_img, img2=SP_crop)


        if debug:
            print(n_sp)

        df_new_img = descriptor_db(SP_crop,
                                   img_identifier=image_info,
                                   conversion_list=conversion_list,
                                   index_list=index_list,
                                   img_results=None,
                                   show_time=False)
        # Append data to the database
        df_database = pd.concat([df_database, df_new_img])

    print('Process ended in [s]:' + str(time.time() - ti))
    print('Database preview:')
    print(df_database)

    # Save results
    print('original file:', img_name)
    output_path = output_dir + img_name + '_img.csv'
    df_database.to_csv(output_path, sep=';', mode='a', header=not os.path.exists(output_path), index=False)
    print('New database file generated at:', output_path)

    return 0



# ~~~~~~~~~~~~~~~~~~ Image database fast ~~~~~~~~~~~~~~~~~~~
# This function generates the file containing the complete image descriptor
# Given an image path, the imagem is divided in SP and each generate a new line at the file output
# The difference from the previous function is that the SP is croped to fit only the relevant data
def image_db_fast(image_path, output_dir, debug=True):
    _, _, _, sp_vector, box = spt.img_superpixel(image_path, debug=debug)

    conversion_list = parameters.get_conversion_list()
    index_list = parameters.get_index_list()

    # New dataframe
    df_database = pd.DataFrame()
    print('------------------------------------------------------')
    print('Image database generation started')
    ti = time.time()
    # for each SP calculate all the descriptors selected
    for index, SP_img in enumerate(sp_vector):
        n_sp = {'nSP': index}

        xi = box[0][index]
        xf = box[1][index]
        yi = box[2][index]
        yf = box[3][index]

        # The SP is croped to fit only the relevant data
        SP_crop = SP_img[0:xf - xi, 0:yf - yi, :]
        df_new_img = descriptor_db(SP_crop,
                                   img_identifier=n_sp,
                                   conversion_list=conversion_list,
                                   index_list=index_list,
                                   img_results=None)
        # Append data to the database
        df_database = pd.concat([df_database, df_new_img])

    print('Process ended in [s]:' + str(time.time() - ti))
    print('Database preview:')
    print(df_database)

    # Save results
    img_name = image_path.split(sep=r'\\')[-1]
    output_path = output_dir + img_name + '.csv'
    print('Output path:', output_path)
    df_database.to_csv(output_path, sep=';', mode='a', header=not os.path.exists(output_path), index=False)
    print('New database file generated at:', output_path)

    return 0


# ~~~~~~~~~~~~~~~~~~ Image database generation ~~~~~~~~~~~~~~~~~~~
# This function calls the image database function for all the files at a given directory
def image_db_generation(image_dir, output_dir, file_filter='', debug=False):
    if file_filter == '':
        file_filter = '.JPG'

    # Access all files at input directory
    for root, directories, files in os.walk(image_dir):
        for file in files:
            # Filters based on string provided
            if file_filter in file:
                img_path = image_dir + file
                print('Image found at:', img_path)
                image_db(image_path=img_path, output_dir=output_dir, debug=debug)

    return 0


# ~~~~~~~~~~~~~~~~~~ Image Test ~~~~~~~~~~~~~~~~~~~
# Given a database file path and an imagem the function runs the model
# and generates all the graphical results, numerical data and report
def img_test(image_path, database_path, model_name, debug=False, save=False):
    df_img = pd.read_csv(database_path, delimiter=';')
    if debug:
        print(df_img)

    model_path = parameters.get_model_folder()
    model = load(model_path + model_name + '.joblib')
    print("Model Loaded:", model)

    column_sp = df_img['nSP']
    column_sp = pd.Series(column_sp)
    df_img.drop(columns=['nSP'], inplace=True)

    prediction_labels = model.predict(df_img)
    prediction_labels = pd.Series(prediction_labels)

    img, img_class, img_label, img_out = spt.class_map(image_path,
                                                       sp_values=column_sp,
                                                       sp_labels=prediction_labels,
                                                       alpha=0.5,
                                                       debug=False)

    if debug:
        spt.img_show(img1=img, label1='Original image',
                     img2=img_class, label2='Class image',
                     img3=img_label, label3='Label image',
                     img4=img_out, label4='Color Image')

    resultados = results.label_percent(img_class)
    # results.pie_chart(resultados, save_img=save)
    results.results_img(img, img_label, resultados, save_img=True)

    if save:
        results.pdf_report()

    return img_label, img_out, img_class

# #===============================================================
# #===========================BACKUP==============================
# #===============================================================

# # #--------------------- DATA BASE GENERATION CODE SAMPLE ----------------------
# db_name = 'small_db'
# img_directory = r'D:\\Drive\\PROFISSIONAL\\Projeto Drone\\Imagens\\Imagens Drone\\Selecionadas\\'
# in_directory = 'database/Input/'
# out_dir = 'database/Output/'
#
# dbt.create_db(input_dir=in_directory,
#               output_dir=out_dir,
#               database_name=db_name,
#               image_dir=img_directory,
#               debug=False)
#
# dbt.clean_db(input_dir=out_dir, database_name=db_name)

# # #---------------------- IMAGE PRE-PROCESSING SAMPLE ---------------------
# img_directory = r'D:\\Demeter Banco de imagens\\Testes\\Cor\\referencia.JPG'
# out_dir = img_directory
# img = spt.img_read(img_directory)
#
# dbt.descriptor_db(img, img_identifier=None,
#                   conversion_list=parameters.get_conversion_list(),
#                   index_list=parameters.get_index_list(),
#                   img_results=None)

# # #--------------------- MANUAL ENTRY SAMPLE ----------------------
# conversion_list = ['XYZ', 'YUV', 'LAB', 'LUV', 'HSV', 'YCrCb', 'CrCgCb', 'rgb', 'I1I2I3', 'l1l2l3', 'TSL']
#
# index_list = ['NDI', 'ExG', 'ExR', 'ExGR', 'CIVE', 'VEG', 'MExG', 'COM1', 'COM2',
#               'YmX', 'mA', 'BmA', 'mExU', 'VmU', 'mCrpCb', 'Cg', 'ExCg', 'l3', 'I3']
#
# img_identifier_p = {'image_name': 'P1.png', '#_SP': 123, 'Autor': 'Ivan', 'pos_x': 120, 'pos_y': 240,
#                     'altura': 10, 'GPS_x': 145.1, 'GPS_y': 333.3}
#
# img_results_p = {'label': 'planta3', 'Biomassa': 2, 'alturaPasto': 35, 'NDVI': 0.4, 'Status': 'bom'}
#
# img_identifier_t = {'image_name': 'T1.png', '#_SP': 234, 'Autor': 'Ivan', 'pos_x': 240, 'pos_y': 260,
#                     'altura': 11, 'GPS_x': 15, 'GPS_y': 30.2}
#
# img_results_t = {'label': 'terra'}
#
# df_img1 = dbt.create_db_entry(image_p_RGB, img_identifier_p, conversion_list, index_list, img_results_p)
# df_img2 = dbt.create_db_entry(image_t_RGB, img_identifier_t, conversion_list, index_list, img_results_t)
# df_database = pd.concat([df_img1, df_img2])
