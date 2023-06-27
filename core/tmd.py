# Criado por: Raphael P Ferreira
# Data: 22/07/2020
# Função: Módulo com conjunto usual de ferramentas para o pré-processamento de imagens contendo trilhas
# Última alteração: 25/02/2021

import cv2
import numpy as np
from demeter import sptools
#import matplotlib.pyplot as plt


# *****************************************************************************************
# *****************************************************************************************


def split(img_input, save_path, vertical_slices, horizontal_slices, name, image_original):

    img = img_input.copy()
    height, width = img.shape
    img2 = img
    img_blk = np.zeros((height, width), np.uint8)  # create an empty image
    img_blk.fill(255)  # fills white pixels the image created
    img_dots = img_blk

    CROP_W_SIZE = horizontal_slices  # Number of pieces Horizontally
    CROP_H_SIZE = vertical_slices  # n° of pieces Vertically to each Horizontal

    for ih in range(CROP_H_SIZE):
        for iw in range(CROP_W_SIZE):

            x = int(width/CROP_W_SIZE * iw)
            y = int(height/CROP_H_SIZE * ih)
            h = int(height / CROP_H_SIZE)
            w = int(width / CROP_W_SIZE)
            img = img[y:y+h, x:x+w]
            array_draws = vertical_projection(img)

            for ii in range(0, array_draws.size):
                #img_colorida = cv2.circle(img_colorida, (int(array_draws[ii]), int(y+(h/2))), 3, 0, 2)
                img_dots = cv2.circle(img_blk, (int(array_draws[ii]), int(y+(h/2))), 1, 0, -1, cv2.LINE_AA)

            img = img2
    print("img dots", img_dots.shape)
    return img_dots

# *****************************************************************************************
# *****************************************************************************************


def vertical_projection(img):

    img_copy = img
    (height, width) = img_copy.shape
    img_copy[img_copy == 0] = 0     # Convert black spots to ones
    img_copy[img_copy == 255] = 1   # Convert white spots to zeros

    # --------- Calculates the vertical projection and apply the threshold

    vertical_proj = np.sum(img_copy, axis=0)  # detect the intesit of the white
    zeros_thr = np.zeros(width)
    for k in range(0, width):
        if vertical_proj[k] > (0.9*height):
            zeros_thr[k] = vertical_proj[k]

    # -------- Calculates the differencial of the threshold vertical projection

    diff_zeros_thr = np.diff(zeros_thr, n=3)

    # -------- Calculates the threshold for the differencial

    thr_diff = np.zeros(width)

    diff_positive = []
    for ll in range(0, (width-3)):
        if diff_zeros_thr[ll] > (1.5*height):
            thr_diff[ll] = diff_zeros_thr[ll]
            diff_positive.append(ll)

    diff_negative = []
    for m in range(0, (width-3)):
        if diff_zeros_thr[m] < (-1.5*height):
            thr_diff[m] = diff_zeros_thr[m]
            diff_negative.append(m)

    # --------- Gets the position of the borders

    index_pos = np.array(diff_positive)
    index_neg = np.array(diff_negative)

    if ((index_neg.size == 0) or (index_pos.size == 0)):
        index_pos = np.array([])
        return index_pos
    else:
        if(index_neg.size >= 1):
            if (index_neg[0] > index_pos[0]):  # Make sure the first border is left crop
                # print(index_neg)
                # print(index_pos)
                index_pos = np.delete(index_pos, 0)
            if(index_neg.size != index_pos.size):  # Make sure that entire crop limit has been obtained
                index_neg = np.delete(index_neg, -1)   # delete the last element

    # --------- Gets and draws the center of the crops  #

    pixel_draw_pos = np.zeros(index_neg.size)

    for p in range(0, index_neg.size):
        pixel_draw_pos[p] = int(index_neg[p] + ((index_pos[p] - index_neg[p])/2))

    # -------- Don't delete this section: Its evaluates feature extraction  #
    # plt.imshow(img_copy,cmap='gray')                                      #
    # plt.plot(vertical_proj,label='vertical proj')                         #
    # plt.plot(zeros_thr,label='threshold')                             #
    # plt.plot(thr_diff,label='diff')                                       #
    # plt.legend()                                                          #
    # plt.show()                                                            #
    # --------------------------------------------------------------------- #

    img_copy[img_copy == 0] = 0  # Convert black spots to ones
    img_copy[img_copy == 1] = 255  # Convert back white to original scale
    return pixel_draw_pos

# *****************************************************************************************
# *****************************************************************************************





