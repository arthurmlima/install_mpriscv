"""
Criado por: André Carmona Hernandes
Data: 09/05/2020
Função: Busca todas as imagens no diretório path, gera super pixel das imagens baseado nos segments e compactness
 fixos, divide os superpixels em imagens separadas e faz roda descritores estatísticos, salvando os mesmos em um csv
Última alteração:
"""

import cv2
import os
import codes.functions.createSuperpixelCut as getSPimg
from codes.functions.resize import resize
from codes.functions.computeEntropy import computeEntropy
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
import numpy as np
import csv


def nan_if(arr, value):
    return np.where(arr == value, np.nan, arr)

n = 1
path_root = 'C:/Users/Andre/PycharmProjects/demeter/demeters-vision/' #TODO: Pensar em uma maneira de não precisar ficar resetando o path, sem precisar mandar as imagens para o bit.
path_test = path_root + 'codes/images/'
path_save = path_root + 'results/'
path_sp = path_save + 'sp/'
segment = 1000
compactness = 28
selectedList = os.listdir(path_test)
with open('imageData.csv', 'w', newline='') as csvfile:
    fieldnames = ['ImageName', 'segmentID', 'Rmean', 'Gmean', 'Bmean', 'Rstd', 'Gstd', 'Bstd', 'Rentropy', 'Gentropy', 'Bentropy']

    dataWriter = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    dataWriter.writeheader()

    # para cada imagem na variável path
    for image_file_name in selectedList:
        # leia a imagem contida no diretório com o nome passado, sem alterar nada na mesma e converta para float
        image = cv2.imread(path_test+image_file_name, cv2.IMREAD_UNCHANGED)
        # Demonstrativo da imagem original
        cv2.imshow('Imagem Original', image)
        #imResized = resize(image)
        imResized = image
        img = img_as_float(imResized)
        # Executa o slic na imagem
        segmentsSlic = slic(img, n_segments=segment, compactness=compactness, sigma=1.5)
        # Demonstrativo da imagem original
        #cv2.imshow('Imagem Original', img)

        # Demonstrativo da imagem SLIC
        imgSlic = mark_boundaries(imResized, segmentsSlic, color=(255, 255, 0), outline_color=(255, 255, 0))
        #cv2.imshow('Imagem SLIC', imgSlic)
       # fig, ax = plt.subplots()
        # mantem as dimensões da imagem
       # fig.set_size_inches(39.68, 28.76)  # TODO - pegar o metadado da imagem passando as dimensões
        # plota a imagem com as cores amarelo nas linhas de cada pixel
        #ax.imshow(imgSlic)
        # desativa a visualização do x e y na plotagem
        #plt.axis("off")
        # salva a imagem na pasta criando a sequencia no inicio do nome mantendo uma boa qualidade, deixando a box espremida e salvando com um dpi de 100
        #plt.savefig(path_save + str(n) + '_' + image_file_name, quality=95, bbox_inches='tight', dpi=100)
        test = 255*imgSlic
        test = test.astype(np.uint8)
        cv2.imwrite(path_save + str(n) + '_' + image_file_name, test)
        # Converte em numpy
        segmentsSlic = np.uint16(segmentsSlic)
        # Captura as dimensões da imagem
        (length, height, depth) = img.shape
        idMax = np.max(segmentsSlic)
        #teste
        #sp_teste = getSPimg.create_superpixel_cut(image, segmentsSlic, 4)
        classes = range(idMax+1)
        for _id in classes:
            print([n, _id])
            itSP = getSPimg.create_superpixel_cut(imResized, segmentsSlic, _id)
            cv2.imwrite(path_sp + 'im' + str(n) + '_SP' + str(_id) + '.jpg', itSP)
            bent, gent, rent = computeEntropy(itSP)
            convImg = nan_if(itSP, -1)
            #cv2.imwrite(path_sp + 'imNan' + str(n) + '_SP' + str(_id) + '.jpg', convImg)
            redMean = np.nanmean(np.nanmean(convImg, axis=0), axis=0)[2]
            greenMean = np.nanmean(np.nanmean(convImg, axis=0), axis=0)[1]
            blueMean = np.nanmean(np.nanmean(convImg, axis=0), axis=0)[0]
            redSTD = np.nanstd(np.nanstd(convImg, axis=0), axis=0)[0]
            greenSTD = np.nanstd(np.nanstd(convImg, axis=0), axis=0)[1]
            blueSTD = np.nanstd(np.nanstd(convImg, axis=0), axis=0)[2]
            dataWriter.writerow(dict(ImageName=n, segmentID=_id, Rmean=redMean, Gmean=greenMean, Bmean=blueMean,
                                     Rstd=redSTD, Gstd=greenSTD, Bstd=blueSTD, Rentropy=rent, Gentropy=gent,
                                     Bentropy=bent))
        n = n + 1
        # Aguarda qualquer tecla para fechar as janelas
cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image







