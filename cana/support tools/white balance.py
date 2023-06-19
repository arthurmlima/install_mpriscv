"""
Criado por: Ivan Perissini
Data: 28/10/2020
Função: Codigo para pré-processar imagens e reduzir efeito do clima
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import winsound

##~~~~~~~~~~~~~~~~~~~ImageMax~~~~~~~~~~~~~~~~~~~~~
def image_max(mat):
    # mat=np.array(frame,np.uint8)

    max_r = np.max(mat[:, :, 0])
    max_g = np.max(mat[:, :, 1])
    max_b = np.max(mat[:, :, 2])
    return (max_r, max_g, max_b)
    # Metodo se mostrou mais demorado
    # return np.max(mat,(0,1))

##~~~~~~~~~~~~~~~~~~~ImageMax~~~~~~~~~~~~~~~~~~~~~
def image_mean(mat):
    # mat=np.array(frame,np.uint8)

    mean_r = np.mean(mat[:, :, 0])
    mean_g = np.mean(mat[:, :, 1])
    mean_b = np.mean(mat[:, :, 2])
    return (mean_r, mean_g, mean_b)
    # Metodo se mostrou mais demorado
    # return np.max(mat,(0,1))

##~~~~~~~~~~~~~~ImageMaxPercentile~~~~~~~~~~~~~~~~~~
def image_max_p(mat, p):
    max_r = np.percentile(mat[:, :, 0], p)
    max_g = np.percentile(mat[:, :, 1], p)
    max_b = np.percentile(mat[:, :, 2], p)
    return (max_r, max_g, max_b)


##~~~~~~~~~~~~~~Pré-Processamento~~~~~~~~~~~~~~~~~~
def white_adjust(mat, adjust):
    out = np.array((mat * adjust).clip(min=0, max=255), np.uint8)
    return out


# --------------------------------------
# ------------MAIN CODE-----------------
# --------------------------------------

# ---- PARAMETROS ----
input_dir = 'D:/Demeter Banco de imagens/Testes/Cor/'
file = '13_PIQUETE156.JPG'
file = 'original.JPG'
# file = 'referencia.JPG'

debug = True
save_output = True

white_crop = None
mode = 'ref'

# ---- CODIGO ----

image_path = input_dir + file
image = cv2.imread(image_path)
image = np.array(image)

# Converte do BRG original do openCV para o tradicional RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Manual
if mode == 'm':
    white_ref = np.array([240, 240, 240])

# Percentile
if mode == 'p':
    white_ref = image_max_p(image, 99.9)

# Gray World
if mode == 'gw':
    white_ref = image_mean(image)

# #Reference
if mode == 'ref':
    cx_r = 3640
    cy_r = 2970
    margin = 25

    cxi = cx_r - margin
    cyi = cy_r - margin
    cxf = cx_r + margin
    cyf = cy_r + margin

    white_crop = image[cyi:cyf, cxi:cxf, :]

    white_ref = image_mean(white_crop)

if white_crop is None:
    white_crop = np.array(np.ones((50, 50, 3)) * white_ref, np.uint8)

# Color adjustment
adjust = np.array([255, 255, 255])/white_ref

print('white reference: ', white_ref)
print('adjust: ', adjust)
pos_image = white_adjust(image, adjust)

if debug:
    fig = plt.figure()

    axi = fig.add_subplot(1, 2, 1)  # Gera os subplots
    axi.set_title("Imagem original: " + file)  # Nomeia cada subplot
    axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
    axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
    plt.imshow(image)

    axi = fig.add_subplot(7, 7, 25)  # Gera os subplots
    axi.set_title("White reference ")  # Nomeia cada subplot
    axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
    axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
    plt.imshow(white_crop)

    axi = fig.add_subplot(1, 2, 2)  # Gera os subplots
    axi.set_title("Color corrected")  # Nomeia cada subplot
    axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
    axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
    plt.imshow(pos_image)

    plt.show()

if save_output:
    img_out = cv2.cvtColor(pos_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(input_dir + 'out_' + file, img_out)

