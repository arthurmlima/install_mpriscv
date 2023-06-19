"""
Criado por: Ivan Perissini
Data: 30/04/2020
Função: Módulo com um conjunto usual de ferramentas para trabalhar com SuperPixel
Última alteração: 12/06/2020
"""

import cv2
import math
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from fast_slic import Slic
from numba import jit
import numpy as np
import time
# from demeter import metadata
import pandas as pd


def __version__():
    return 'sptools version: v0.2'


# #========================FUNÇÕES============================

# ~~~~Multiple image Show~~~~~~
def img_show(img1=[0], label1='Image1', img2=[0], label2='Image2', img3=[0], label3='Image3', img4=[0],
             label4='Image4'):
    fig = plt.figure()

    size = int(len(img1) > 1) + int(len(img2) > 1) + int(len(img3) > 1) + int(len(img4) > 1)
    if size == 4:
        row = 2
        col = 2
    else:
        row = 1
        col = size

    # Inicializa a posição do plot
    p = 0

    if len(img1) > 1:
        p = p + 1
        axi = fig.add_subplot(row, col, p)  # Gera os subplots
        axi.set_title(label1)  # Nomeia cada subplot
        axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
        axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
        plt.imshow(img1)

    if len(img2) > 2:
        p = p + 1
        axi = fig.add_subplot(row, col, p)  # Gera os subplots
        axi.set_title(label2)  # Nomeia cada subplot
        axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
        axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
        plt.imshow(img2)

    if len(img3) > 3:
        p = p + 1
        axi = fig.add_subplot(row, col, p)  # Gera os subplots
        axi.set_title(label3)  # Nomeia cada subplot
        axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
        axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
        plt.imshow(img3)

    if len(img4) > 4:
        p = p + 1
        axi = fig.add_subplot(row, col, p)  # Gera os subplots
        axi.set_title(label4)  # Nomeia cada subplot
        axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
        axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
        plt.imshow(img4)

    # plt.tight_layout(0.1)
    plt.show()

    return 1


# ~~~~fast_slic~~~~~~
# Versão rápida do calculo do Slic, a função encapsula as diferentes operações necessárias para assegurar
# similaridade com a implementação original
def fast_slic(image, n_segments=100, compactness=10.0, sigma=0, convert2lab=True, min_size_factor=0.5):
    # Por padrão a abordagem converte a imagem para LAB
    img_original = np.copy(image)

    if convert2lab:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Caso um sigma seja designado a função aplica um filtro gaussiano na imagem
    if sigma > 0:
        image = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma)

    slic = Slic(num_components=n_segments, compactness=compactness, min_size_factor=min_size_factor)
    segment = slic.iterate(image)  # Cluster Map

    # Marca as bordas dos SuperPixeis nas imagens, sua saída é uma imagem float
    img_slic = mark_boundaries(img_original, segment, color=(0, 1, 1), outline_color=(0, 1, 1))
    img_slic = cv2.convertScaleAbs(img_slic, alpha=255)  # Converte novamente de float para Uint8

    return img_slic, segment


# ~~~~sp_slicer~~~~~~
# Dada a imagem de entrada e os rotulos dos segmentos, a função retorna um vetor de
# imagens e uma tupla contendo as coordenadas da caixa delimitadora associada ao
# SuperPixel, sendo o indice de ambos igual ao rotulo do SuperPixel
@jit
def sp_slicer(image, segments):
    # Captura as dimensões da imagem
    (length, height, depth) = image.shape

    # Registra o numero de SuperPixeis na imagem
    n_segments = np.max(segments) + 1

    # Inicializa as variáveis que irão definir os limites da caixa
    min_x = np.full(n_segments, length, dtype=np.uint16)
    min_y = np.full(n_segments, height, dtype=np.uint16)
    max_x = np.zeros(n_segments, dtype=np.uint16)
    max_y = np.zeros(n_segments, dtype=np.uint16)
    box = (min_x, max_x, min_y, max_y)  # Tupla com as 4 Coordenadas da caixa

    # Primeira varredura da imagem, gera as coordenadas da caixa delimitadora
    for x in range(0, length):
        for y in range(0, height):

            # Calcula os extremos da caixa delimitadora para cada SuperPixel
            if x < min_x[segments[x, y]]:
                min_x[segments[x, y]] = x

            if x > max_x[segments[x, y]]:
                max_x[segments[x, y]] = x

            if y < min_y[segments[x, y]]:
                min_y[segments[x, y]] = y

            if y > max_y[segments[x, y]]:
                max_y[segments[x, y]] = y

    # Calculo das maiores dimensões entre os SuperPixel
    max_delta_x = np.max(max_x - min_x)
    max_delta_y = np.max(max_y - min_y)

    # Gera um vetor de imagens com as maiores resoluções encontradas e com indice associado ao SuperPixel equivalente
    sp_img_vector = np.zeros((n_segments, max_delta_x, max_delta_y, 3), dtype=image.dtype)
    # sp_img_vector = np.zeros((n_segments, max_delta_x, max_delta_y, 3), dtype=np.uint8)

    # Segunda varredura, Posiciona o SuperPixel na coordenada 0,0 dentro vetor de imagens
    for x in range(0, length):
        for y in range(0, height):
            # Gera um vetor de imagens com os SuperPixel
            sp_img_vector[segments[x, y], x - min_x[segments[x, y]], y - min_y[segments[x, y]]] = image[x, y]

    return sp_img_vector, box


# ~~~~Image Superpixels~~~~~~
# Principal função que compreende todas as etapas e saídas necessárias para os códigos
# return img, img_slic, segments, sp_vector, box
def img_superpixels(image_path, mode='RGB', debug=False, segment=0):
    img = imread(image_path)

    # Parâmetros do SuperPixel
    try:
        pass
        #h, _, _ = metadata.get_gps(image_path)
    except:
        h = 100
        print('Altura da imagem não encontrada nos metadados, valor utilizado de:', h, 'm')

    if segment == 0:
        segment = metadata.segment_estimation(h)
    compactness = 30
    sigma = 2
    min_size = 0.5

    if mode == 'BGR':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Fast Slic
    img_slic, segments = fast_slic(img, n_segments=segment, compactness=compactness,
                                   sigma=sigma,
                                   min_size_factor=min_size, convert2lab=True)

    if mode == 'BGR':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_slic = cv2.cvtColor(img_slic, cv2.COLOR_BGR2RGB)

    # Função de separação dos SuperPixeis
    sp_vector, box = sp_slicer(img, segments)

    if debug:
        img_show(img1=img, label1='Original image',
                 img2=img_slic, label2='Slic Image',
                 img3=segments, label3='Segments Image',
                 img4=sp_vector[1], label4='Sample Image')

    return img, img_slic, segments, sp_vector, box