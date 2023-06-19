"""
Criado por: Ivan Perissini
Data: 18/11/2020
Função: Texture test codes
Última alteração:
"""
import cv2
from codes.core.demeter import descriptors as info
from codes.core.demeter import sptools as spt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import time
from scipy.special import expit
from codes.core.demeter import colorindex
import math

# # #---------------------TEXTURE TEST CODE----------------------
img_directory = "D:/Drive/PROFISSIONAL/Projeto Drone/Banco de dados/Imagens/Texture/"
img_list = ['Tteste', 'Tnoise', 'Tzig', 'To', 'Ti', 'T45',
            'baixo1s', 'baixo2', 'baixo3', 'medio1s', 'medio2',
            'alto1s', 'alto2s', 'alto3', 'alto4']

# img_list = ['Tnoise', 'baixo1s', 'baixo2', 'baixo3', 'medio1s', 'medio2',
#             'alto1s', 'alto2s', 'alto3', 'alto4']


for img_name in img_list:
    img_path = img_directory + img_name + '.PNG'
    img = spt.img_read(img_path)
    debug = True

    glcm_descriptor = info.img_glcm(img, label='img', distance=5, levels=256, debug=debug)
    # print(glcm_descriptor)

    lbp_descriptor = info.img_lbp(img, label=img_name, radius=1, sampling_pixels=8, debug=debug)
    # print(lbp_descriptor)

    hog_descriptor = info.img_hog(img, debug=debug)
    # print(hog_descriptor)

    info.img2dft(img, debug=debug)
    # dft_img = colorindex.dft(img)
    # print(info.img_statistics(dft_img))

print('END')
