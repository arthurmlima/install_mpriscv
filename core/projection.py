# Criado por: Raphael P Ferreira
# Data: 21/06/2021

import cv2
import numpy as np
from demeter import sptools
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import common
from demeter import colorindex


# *****************************************************************************************
# *****************************************************************************************

print_on_terminal = False

def main_projection(img, img_bw, angle_resolution, filter, coordinates, debug="false"):
    how_many_zeros = []
    how_many_savitzky = []
    angles = []

    #  -  -  -  -  -  -  -  -  -  -  -  -  -
    for angle in range(0, angle_resolution*180, 1):
        angle = angle/angle_resolution
        img_rot = rotate_image(img_bw, angle)  # To do: why rotate returns gray scale?
        vet_thr = vertical_proj(img_rot, filter, debug)

        half_len = int(len(vet_thr)/2)
        quarter_len = int(len(vet_thr)/4)

        if (filter == "zeros"):
            zeros_wrap = np.take(vet_thr, range(half_len-quarter_len, half_len+quarter_len))
            how_many_zeros.append(np.count_nonzero(zeros_wrap == 0))  # append zeros qtts
        if (filter == "savitzky"):
            savitzky_wrap = np.take(vet_thr, range(half_len-quarter_len, half_len+quarter_len))
            how_many_savitzky.append(np.count_nonzero(savitzky_wrap)) # append nonzero qtts


        #--------------------------------------
        if coordinates == "0to180Right":
            angles.append(angle)

        #--------------------------------------
        elif coordinates == "-90to90Front":
            if angle <= 90:
                angles.append(angle)
            elif angle > 90:
                angles.append(angle-180)
        #--------------------------------------


        if debug == "true1":
            print("Angle: %d°" % (angle))
            print("width da img rotacionada", img_rot.shape[1])
            print("------------------------------------")
    #  -  -  -  -  -  -  -  -  -  -  -  -  -

    if debug == "true":
        plt.xlabel('Angle [°]')
        if (filter == "zeros"):
            plt.ylabel('Soil Predominance [pixel]')
            plt.plot(angles, how_many_zeros, label='zeros')
        if (filter == "savitzky"):
            #plt.title('Variation of the sum of intensity of the vertically projected pixels as a function of the image rotation angle')
            plt.ylabel('Scaled Savitzky-Golay Filter Output')
            plt.plot(angles, how_many_savitzky)
        # plt.legend()
        plt.show()

    if (filter == "zeros"):
        index_max_zeros = np.argmax(how_many_zeros)
        detected_angle = angles[index_max_zeros]
        if(print_on_terminal):
            print("detected_angle by zeros: ", detected_angle)
    if (filter == "savitzky"):
        index_max_gol = np.argmax(how_many_savitzky)
        detected_angle = angles[index_max_gol]
        if(print_on_terminal):
            print("detected_angle by savitzsky: ", detected_angle)

    return detected_angle


# *****************************************************************************************
# *****************************************************************************************


def vertical_proj(img_bw, filter, debug="false"):
    window_sav = 15
    img_copy = img_bw
    (height, width) = img_copy.shape
    img_copy = np.where(img_copy > 0, 1, img_copy)  # convert any pixel >0 to 1

    vertical_proj = np.sum(img_copy, axis=0)  # detect the intesit of the white

    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    if (filter == "zeros"):
        zeros_thr = np.zeros(width)
        for k in range(0, width):
            if vertical_proj[k] > (0.005*height):  # Few vegetation is soil
                zeros_thr[k] = vertical_proj[k]
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    if (filter == "savitzky"):
        gol = savgol_filter(vertical_proj, window_sav, 2, 1)  # window, polyorder, deri
        gol = gol*gol  # filter based on power
        savizky_thr = np.zeros(len(gol))
        for j in range(len(gol)):
            if gol[j] > 50:
                savizky_thr[j] = gol[j]
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

    if debug == "true1":
        plt.figure(1)
        plt.subplot(121)
        plt.title("Image rotated")
        plt.imshow(img_copy, "gray")
        plt.subplot(122)
        plt.xlabel('Width [pixels]')
        plt.ylabel('Sum of vertical pixels')
        plt.plot(vertical_proj, label='vertical projection')
        if (filter == "zeros"):
            plt.plot(zeros_thr, 'r', label='thresholded')
        if (filter == "savitzky"):
            plt.plot(savizky_thr, label='Filtered')
        plt.legend()
        plt.show()

    if (filter == "zeros"):
        return zeros_thr
    if (filter == "savitzky"):
        return savizky_thr

# *****************************************************************************************
# *****************************************************************************************


def rotate_image(img_bw, angle_o):
    angle = float(angle_o*(np.pi/180))
    height = img_bw.shape[0]  # original image height
    width = img_bw.shape[1]  # original image width
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle_o, 1)  # Rotate the image by angle
    heightNew = int(width * np.fabs(np.sin(angle)) + height * np.fabs(np.cos(angle)))
    widthNew = int(height * np.fabs(np.sin(angle)) + width * np.fabs(np.cos(angle)))

    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img_bw, rotateMat, (widthNew, heightNew), borderValue=(0, 0, 0))
    return imgRotation

# *****************************************************************************************
# *****************************************************************************************

def segmentation(img, veg_index='EXCESS_GREEN', type_det='CROP', kernel = 2, debug="false"):

    if veg_index == 'EXCESS_GREEN':
        imgVegIndex = colorindex.ExG(img, adjust='clip')
    elif veg_index == 'EXCESS_GREEN_RED':
        imgVegIndex = colorindex.ExGR(img, adjust='clip')

    imgVegIndex = np.array(255 * imgVegIndex, np.uint8)
    thr1, binary = cv2.threshold(imgVegIndex, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1*kernel, 1*kernel))
    se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel*3, kernel*3))
    se5 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel*4, kernel*4))

    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se1)

    if type_det == 'CROP':
        if debug == "true":
            sptools.img_show(img, "original", binary, "Otsu", imgVegIndex, "ExG only", mask, "Closing")
        return mask  # if crop
    elif type_det == 'LAND':
        if debug == "true":
            sptools.img_show(img, "original",  cv2.bitwise_not(imgVegIndex), "ExG only",  cv2.bitwise_not(mask), "Inversion from ExG + Closing")
        return cv2.bitwise_not(mask)  # if land




