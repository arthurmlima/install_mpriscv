# Criado por: Raphael P Ferreira
# Data: 12/07/2022
# Função: Radon transform

import cv2
import numpy as np
from skimage.transform import radon
#import matplotlib.pyplot as plt
import common
from demeter import colorindex
from demeter import sptools



# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

def RadonPreProcessing(img):
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    I = I - np.mean(I)  # Demean; make the brightness extend above and below zero

    return I


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

def radonProcess(I, scan_resolution, path, counter_img, debug='false'):
    theta = np.linspace(0., 180., scan_resolution * 180, endpoint=False)

    # Do the radon transform
    sinogram = radon(I, theta=theta)

    if (debug=='true'):
        print('Image Shape 2nd step: ', I.shape)
        plt.figure(1)
        plt.title("Sinogram")
        plt.xlabel('Angle [degrees]')
        plt.ylabel('Radial coordinate')
        plt.imshow(sinogram, aspect='auto')
        plt.gray()
        plt.show()

    rotation = sinogramLinesAbsoluteMean(sinogram, scan_resolution,debug)

    rotation = rotation/(scan_resolution)
    if rotation < 90:
        rotation = (-rotation)
    elif rotation >= 90:
        rotation = (180-rotation)

    return rotation


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

def sinogramLinesAbsoluteMean(sinogram, scan_resolution,debug='false'):
    #this should be only a absolute average. @todo mean of abs(lines)
    r = np.array([np.mean(np.abs(line)) for line in sinogram.transpose()]) # MRS
    #r = np.array([np.mean(np.sqrt(np.abs(line) ** 2)) for line in sinogram.transpose()]) # MRS

    #r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()]) # RMS

    if (debug=='true'):
        plt.xlabel('Angle [°]')
        plt.ylabel(' Pixels absolute average')
        plt.plot(r, label='Mean')
        plt.show()

    rotation = np.argmax(r)
    return rotation

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
def scale2NEDdomain(scan_resolution):
    array_size = (180 * scan_resolution)
    NEDdomain_array = np.linspace(0, 180, num=array_size, endpoint=False)

    a_array = []
    b_array = []

    for i in range(len(NEDdomain_array)):
        if NEDdomain_array[i] > 0 and NEDdomain_array[i] <= 90:
            a_array.append(-NEDdomain_array[i])
        elif NEDdomain_array[i] > 90:
            b_array.append(180-NEDdomain_array[i])
    c = b_array + [0] + a_array

    return c

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

def sinogramDrawLine(sinogram, path, counter_img, rotation):
    sin_t = sinogram.transpose().astype(np.float).copy()
    #plt.imsave(path + 'sinogram_%02i.png' % counter_img, sin_t, cmap='hsv')
    sin_line = common.draw_horizontal_line_matplot(sin_t,rotation)
    plt.imsave(path + 'sinogram_%02i.png' % counter_img, sin_line, cmap='gray')
    read_img = plt.imread(path + 'sinogram_%02i.png' % counter_img)
    #plt.imshow(read_img)
    #plt.show()
    print("shape", read_img.shape)
    print("class", type(read_img))
    print("type",read_img.dtype)

    return

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

def radonRotate(img, rotation):
    width = img.shape[0]
    height = img.shape[1]

    # Rotate and save with the original resolution
    M = cv2.getRotationMatrix2D((width/2, height/2), 90 - rotation, 1)
    final_image = cv2.warpAffine(img, M, (width, height))

    return final_image
