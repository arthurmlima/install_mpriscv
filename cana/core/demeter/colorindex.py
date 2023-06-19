"""
Criado por: Ivan C Perissini
Data: 06/2020
Função: Color and vegetation index functions
Última alteração: 13/11/2020
"""



import cv2
import numpy as np
import time
import math
#import matplotlib.pyplot as plt

def __version__():
    return 'colorindex version: v0.2'

# ======================== SUPPORT TOOLS =========================

# ~~~~~~~~~~~~~~~~~~ Image Basic Information ~~~~~~~~~~~~~~~~~~~
def img_info(img):
    print('---Image Info--- ')
    print('Minimum: ', np.min(img))
    print('Mean: ', np.mean(img, (0, 1)))
    print('Maximum: ', np.max(img))

    return 1


# ~~~~~~~~~~~~~~~~~~ Contrast Adjustment~~~~~~~~~~~~~~~~~~~
def image_adjustment(img, adjust=''):
    data_type = img.dtype

    # Define a escala da imagem
    if data_type == 'uint8':
        max_scale = 255
    else:
        max_scale = 1

    if adjust == 'clip':
        mat = img.clip(min=0, max=max_scale)

    elif adjust == 'raw':
        mat = np.array(img, np.float32)

    else:
        mat = np.array(img, np.float32)
        mat = mat - np.min(mat)  # Ajuste linear
        mat = mat * (max_scale / np.max(mat)) / max_scale  # Ajuste de escala

    return mat


# ~~~~~~~~~~~~~~~~~~ Remove SuperPixel background~~~~~~~~~~~~~~~~~~~
def remove_background(img, original_img):
    # Transform absolute zero values to virtual zeros
    if img.dtype == 'uint8':
        min_value = 1
    else:
        min_value = 1 / 255

    img = np.where(img == 0, min_value, img)

    # In case of multiple dimensions image input do
    if len(img.shape) > 2:
        ret, mask = cv2.threshold(original_img, 1 / 256, 1, cv2.THRESH_BINARY)
    else:
        ret, mask = cv2.threshold(original_img[:, :, 0], 1 / 256, 1, cv2.THRESH_BINARY)

    # Applies mask to the input image
    img = img * mask

    return img


# ======================== COLOR CONVERSION =========================

# ~~~~~~~~~~~~~~~~~ RGB to rgb ~~~~~~~~~~~~~~~~~~
def RGB2rgb(img):
    mat = np.array(img, np.float32)
    img_rgb = np.copy(mat)
    R = mat[:, :, 0]
    G = mat[:, :, 1]
    B = mat[:, :, 2]

    denominator = R + G + B + 0.01
    img_rgb[:, :, 0] = R / denominator
    img_rgb[:, :, 1] = G / denominator
    img_rgb[:, :, 2] = B / denominator

    # Image output conditioning
    img_rgb = img_rgb.clip(min=0, max=1)
    if img.dtype == 'uint8':
        img_rgb = np.array(img_rgb * 255, np.uint8)

    return img_rgb


# ~~~~~~~~~~~~~~~~~ RGB to LAB ~~~~~~~~~~~~~~~~~~
def RGB2LAB(img):
    # Image input conditioning
    input_float = True
    if img.dtype == 'uint8':
        input_float = False
        img = np.array(img, np.float32) / 255

    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)  # Outputs a 255 image depth

    # Image output conditioning
    img_lab = img_lab.clip(min=0, max=255)
    if input_float:
        img_lab = np.array(img_lab, np.float32) / 255
    else:
        img_lab = np.array(img_lab, np.uint8)

    return img_lab


# ~~~~~~~~~~~~~~~~~ RGB to LUV ~~~~~~~~~~~~~~~~~~
def RGB2LUV(img):
    # Image input conditioning
    input_float = True
    if img.dtype == 'uint8':
        input_float = False
        img = np.array(img, np.float32) / 255

    img_luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)  # Outputs a 255 image depth

    # Image output conditioning
    img_luv = img_luv.clip(min=0, max=255)
    if input_float:
        img_luv = np.array(img_luv, np.float32) / 255
    else:
        img_luv = np.array(img_luv, np.uint8)

    return img_luv


# ~~~~~~~~~~~~~~~~~ RGB to HSV ~~~~~~~~~~~~~~~~~~
def RGB2HSV(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if img.dtype == 'float32':
        img_hsv[:, :, 0] = img_hsv[:, :, 0] / 360  # Hue channel is scaled to 360 degrees

    return img_hsv


# ~~~~~~~~~~~~~~~~~ RGB to YCrCb ~~~~~~~~~~~~~~~~~~
def RGB2YCrCb(img):
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    return img_ycrcb


# ~~~~~~~~~~~~~~~~~ RGB to CrCgCb ~~~~~~~~~~~~~~~~~~
def RGB2CrCgCb(img):
    # Image input conditioning
    input_float = True
    if img.dtype == 'uint8':
        input_float = False
        img = np.array(img, np.float32) / 255

    # Converte para YCrCb
    img_crcgcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    # Index calculation
    # Cg = 1,211.(G - Y) + 0.5
    cg = 1.211 * (img[:, :, 1] - img_crcgcb[:, :, 0]) + 0.5

    img_crcgcb[:, :, 0] = img_crcgcb[:, :, 1]
    img_crcgcb[:, :, 1] = cg

    # Image output conditioning
    if not input_float:
        img_crcgcb = np.array(img_crcgcb * 255, np.uint8)

    return img_crcgcb


# ~~~~~~~~~~~~~~~~~ RGB to XYZ D65~~~~~~~~~~~~~~~~~~
def RGB2XYZ(img):
    img_xyz = cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)
    return img_xyz


# ~~~~~~~~~~~~~~~~~ RGB to YUV ~~~~~~~~~~~~~~~~~~
def RGB2YUV(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return img_yuv


# ~~~~~~~~~~~~~~~~~ RGB to I1I2I3 ~~~~~~~~~~~~~~~~~~
def RGB2I1I2I3(img):
    # Input preparation and channel split
    mat = np.array(img, np.float32)
    img_i1i2i3 = np.copy(mat)

    R = mat[:, :, 0]
    G = mat[:, :, 1]
    B = mat[:, :, 2]

    # Index calculation
    img_i1i2i3[:, :, 0] = (R + G + B) / 3
    img_i1i2i3[:, :, 1] = (R - B) / 2
    img_i1i2i3[:, :, 2] = (2 * G - R - B) / 4

    # Image output conditioning
    if img.dtype == 'float32':
        img_i1i2i3 = img_i1i2i3.clip(min=0, max=1)
    else:
        img_i1i2i3 = img_i1i2i3.clip(min=0, max=255)
        img_i1i2i3 = np.array(img_i1i2i3, np.uint8)

    return img_i1i2i3


# ~~~~~~~~~~~~~~~~~ RGB to l1l2l3 ~~~~~~~~~~~~~~~~~~
def RGB2l1l2l3(img):
    # Input preparation and channel split
    mat = np.array(img, np.float32)
    img_l1l2l3 = np.copy(mat)
    R = mat[:, :, 0]
    G = mat[:, :, 1]
    B = mat[:, :, 2]
    denominator = (R - G) * (R - G) + (R - B) * (R - B) + (G - B) * (G - B) + 0.01

    # Index calculation
    img_l1l2l3[:, :, 0] = ((R - G) * (R - G)) / denominator
    img_l1l2l3[:, :, 1] = ((R - B) * (R - B)) / denominator
    img_l1l2l3[:, :, 2] = ((G - B) * (G - B)) / denominator

    # Image output conditioning
    if img.dtype == 'uint8':
        img_l1l2l3 = np.array(img_l1l2l3 * 255, np.uint8)

    return img_l1l2l3


# ~~~~~~~~~~~~~~~~~ RGB to TSL ~~~~~~~~~~~~~~~~~~
def RGB2TSL(img):
    # Input preparation and channel split
    mat = np.array(img, np.float32)
    img_tsl = np.copy(mat)

    scale = 1
    if img.dtype == 'uint8':
        scale = 255

    R = mat[:, :, 0] / scale
    G = mat[:, :, 1] / scale
    B = mat[:, :, 2] / scale

    denominator = R + G + B + 0.001
    r = R / denominator
    g = G / denominator
    # b = B / denominator

    # Index calculation
    rl = r - 1 / 3
    gl = g - 1 / 3
    img_tsl[:, :, 0] = np.arctan2(rl, gl)
    img_tsl[:, :, 1] = np.sqrt((rl * rl + gl * gl) * 9 / 5)
    img_tsl[:, :, 2] = 0.299 * R + 0.587 * G + 0.114 * B

    # Image output conditioning
    img_tsl = img_tsl.clip(min=0, max=1)
    if img.dtype == 'uint8':
        img_tsl = np.array(img_tsl * 255, np.uint8)

    return img_tsl


# ~~~~~~~~~~~~~~~~~ Multiple space color conversions ~~~~~~~~~~~~~~~~~~
def multiple_conversions(image, conversion_list, no_background=True, show_results=False, show_time=False):
    # Saves the number of methods provided
    n_method = len(conversion_list)

    # Generates a image vector to hold the conversion output, using the input image depth as default
    index_images = np.zeros((n_method, image.shape[0], image.shape[1], image.shape[2]), dtype=image.dtype)

    # Executes and save into the vector each of the methods on the list
    for index, method in enumerate(conversion_list):
        ti = time.time()
        index_images[index] = eval('RGB2' + method)(image)

        # Removes background for future calculations
        if no_background:
            index_images[index] = remove_background(index_images[index], image)

        if show_time:
            print('Tempo de conversão para ' + method + ' [s]: ' + str(time.time() - ti))

    if show_results:
        fig = plt.figure()
        row = math.ceil(np.sqrt(n_method + 1))
        col = row

        for index, method in enumerate(conversion_list):
            if index == 0:
                axi = fig.add_subplot(row, col, index + 1)  # Gera os subplots
                axi.set_title('Original')  # Nomeia cada subplot
                axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
                axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
                plt.imshow(image)

            axi = fig.add_subplot(row, col, index + 2)  # Gera os subplots
            axi.set_title(method)  # Nomeia cada subplot
            axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
            axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
            plt.imshow(index_images[index])

        plt.tight_layout(0.1)
        plt.show()

    return index_images


# ======================== VEGETATIVE COLOR INDEXES =========================

# ~~~~~~~~~~~~~~~~~ Normalised Difference Index - NDI ~~~~~~~~~~~~~~~~~~
def NDI(img, adjust='clip'):
    # Input preparation and channel split
    mat = np.array(img, np.float32) + 0.01
    R = mat[:, :, 0]
    G = mat[:, :, 1]
    # B = mat[:, :, 2]

    # Index calculation
    img_NDI = 0.5 * (((G - R) / (G + R)) + 1)

    # Image output conditioning
    img_NDI = image_adjustment(img_NDI, adjust)

    return img_NDI


# ~~~~~~~~~~~~~~~~~~ Excess Green - ExG ~~~~~~~~~~~~~~~~~~~
def ExG(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = RGB2rgb(img)
    r = mat[:, :, 0]
    g = mat[:, :, 1]
    b = mat[:, :, 2]

    # Index calculation
    img_ExG = (2 * g - r - b)

    # Image output conditioning
    img_ExG = image_adjustment(img_ExG, adjust)

    return img_ExG


# ~~~~~~~~~~~~~~~~~~Excess Red - ExR~~~~~~~~~~~~~~~~~~~
def ExR(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = RGB2rgb(img)
    r = mat[:, :, 0]
    g = mat[:, :, 1]
    # b = mat[:, :, 2]

    # Index calculation
    img_ExR = 1.3 * r - g

    # Image output conditioning
    img_ExR = image_adjustment(img_ExR, adjust)

    return img_ExR


# ~~~~~~~~~~~~~~~~~~Colour Index of Vegetation Extraction - CIVE~~~~~~~~~~~~~~~~~~~
def CIVE(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = np.array(img, np.float32)
    R = mat[:, :, 0]
    G = mat[:, :, 1]
    B = mat[:, :, 2]

    # Index Calculation
    img_CIVE = 0.441 * R - 0.811 * G + 0.385 * B + (18.78745 / 255)

    # Image output conditioning
    img_CIVE = image_adjustment(img_CIVE, adjust)

    return img_CIVE


# ~~~~~~~~~~~~~~~~~~Excess Green minus Excess Red Index - ExGR~~~~~~~~~~~~~~~~~~~
def ExGR(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = RGB2rgb(img)
    r = mat[:, :, 0]
    g = mat[:, :, 1]
    b = mat[:, :, 2]

    # Index calculation
    img_ExGR = (3 * g - 2.3 * r - b)

    # Image output conditioning
    img_ExGR = image_adjustment(img_ExGR, adjust)

    return img_ExGR*255


# ~~~~~~~~~~~~~~~~~~Colour Index of Vegetation Extraction - CIVE~~~~~~~~~~~~~~~~~~~
def VEG(img, adjust='clip', a=0.667):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = np.array(img, np.float32) + 0.001
    R = mat[:, :, 0]
    G = mat[:, :, 1]
    B = mat[:, :, 2]

    # Index Calculation
    img_VEG = G / (np.power(R, a) * np.power(B, (1 - a)))

    # Image output conditioning
    img_VEG = image_adjustment(img_VEG, adjust)

    return img_VEG


# ~~~~~~~~~~~~~~~~~~Modified Excess Green Index - MExG~~~~~~~~~~~~~~~~~~~
def MExG(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = np.array(img, np.float32)
    R = mat[:, :, 0]
    G = mat[:, :, 1]
    B = mat[:, :, 2]

    # Index Calculation
    img_MExG = (1.262 * G - 0.884 * R - 0.311 * B)

    # Image output conditioning
    img_MExG = image_adjustment(img_MExG, adjust)

    return img_MExG


# ~~~~~~~~~~~~~~~~~~Combined ExG, ExGR, CIVE e VEG Indexes - COM1~~~~~~~~~~~~~~~~~~~
def COM1(img, adjust='clip'):
    # Input preparation and channel split
    method = 'raw'
    img_ExG = ExG(img, adjust=method)
    img_ExR = ExR(img, adjust=method)
    img_CIVE = CIVE(img, adjust=method)
    img_VEG = VEG(img, adjust=method)

    # Index Calculation
    # Todo: Configurar a proporção correta do método
    img_COM1 = 0.25 * img_ExG + 0.25 * img_ExR + 0.25 * img_CIVE + 0.25 * img_VEG

    # Image output conditioning
    img_COM1 = image_adjustment(img_COM1, adjust)

    return img_COM1


# ~~~~~~~~~~~~~~~~~~Combined ExG, CIVE e VEG Indexes - COM2~~~~~~~~~~~~~~~~~~~
def COM2(img, adjust='clip'):
    # Input preparation and channel split
    method = 'raw'
    img_ExG = ExG(img, adjust=method)
    img_CIVE = CIVE(img, adjust=method)
    img_VEG = VEG(img, adjust=method)

    # Index Calculation
    img_COM2 = 0.36 * img_ExG + 0.47 * img_CIVE + 0.17 * img_VEG

    # Image output conditioning
    img_COM2 = image_adjustment(img_COM2, adjust)

    return img_COM2


# ~~~~~~~~~~~~~~~~~~Visible Atmospheric Resistant Index - VARI~~~~~~~~~~~~~~~~~~~
# Used for highlighting vegetation in images
def VARI(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = np.array(img, np.float32) + 0.01
    R = mat[:, :, 0]
    G = mat[:, :, 1]
    B = mat[:, :, 2]

    # Index calculation
    img_VARI = (G - R) / (G + R - B)

    # Image output conditioning
    img_VARI = image_adjustment(img_VARI, adjust)

    return img_VARI


# ~~~~~~~~~~~~~~~~~~Triangular Greenness Index - TGI~~~~~~~~~~~~~~~~~~~
# Estimate leaf chlorophyll and, indirectly plant nitrogen content, using visible-spectrum
def TGI(img, adjust='clip'):

    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    # Input preparation and channel split
    mat = np.array(img, np.float32) + 0.01
    R = mat[:, :, 0]
    G = mat[:, :, 1]
    B = mat[:, :, 2]

    # Index calculation
    img_TGI = G - 0.39 * R - 0.61 * B

    # Image output conditioning
    img_TGI = image_adjustment(img_TGI, adjust)

    return img_TGI


# ~~~~~~~~~~~~~~~~~~NGRDI~~~~~~~~~~~~~~~~~~~
# Highlights vegetation cover against the rest of land cover types
def NGRDI(img, adjust='clip'):

    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    # Input preparation and channel split
    mat = np.array(img, np.float32) + 0.01
    R = mat[:, :, 0]
    G = mat[:, :, 1]
    #B = mat[:, :, 2]

    # Index calculation
    img_NGRDI = (G - R)/(G + R)

    # Image output conditioning
    img_NGRDI = image_adjustment(img_NGRDI, adjust)

    return img_NGRDI


# ~~~~~~~~~~~~~~~~~~RGBVI~~~~~~~~~~~~~~~~~~~
# Is newly introduced normalized red green blue vegetation index
def RGBVI(img, adjust='clip'):

    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    # Input preparation and channel split
    mat = np.array(img, np.float32) + 0.01
    R = mat[:, :, 0]
    G = mat[:, :, 1]
    B = mat[:, :, 2]

    # Index calculation
    img_RGBVI = (G * G - R * B)/(G * G + R * B)

    # Image output conditioning
    img_RGBVI = image_adjustment(img_RGBVI, adjust)

    return img_RGBVI


# ~~~~~~~~~~~~~~~~~~GLI~~~~~~~~~~~~~~~~~~~
# Is focused on using reflectance of green vegetation cover to highlight it in the images
def GLI(img, adjust='clip'):

    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    # Input preparation and channel split
    mat = np.array(img, np.float32) + 0.01
    R = mat[:, :, 0]
    G = mat[:, :, 1]
    B = mat[:, :, 2]

    # Index calculation
    img_GLI = (2*G - R - B)/(2*G + R + B)

    # Image output conditioning
    img_GLI = image_adjustment(img_GLI, adjust)

    return img_GLI


# ~~~~~~~~~~~~~~~~~~Normalized Difference Vegetation Index - NDVI~~~~~~~~~~~~~~~~~~~
def NDVI(img, adjust='clip'):
    # Input preparation and channel split
    mat = np.array(img, np.float32) + 0.01
    O = mat[:, :, 0]
    # C = mat[:, :, 1]
    NIR = mat[:, :, 2]

    # Index calculation
    img_NDVI = 0.5 * (((NIR - O) / (NIR + O)) + 1)

    # Image output conditioning
    img_NDVI = image_adjustment(img_NDVI, adjust)

    return img_NDVI


# ======================== ALTERNATIVE VEGETATIVE COLOR INDEXES =========================

# ~~~~~~~~~~~~~~~~~~Y - X (XYZ)~~~~~~~~~~~~~~~~~~~
def YmX(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = RGB2XYZ(img)
    X = mat[:, :, 0]
    Y = mat[:, :, 1]
    # Z = mat[:, :, 2]

    # Index Calculation
    img_YmX = Y - X

    # Image output conditioning
    img_YmX = image_adjustment(img_YmX, adjust)

    return img_YmX


# ~~~~~~~~~~~~~~~~~~Minus A (LAB)~~~~~~~~~~~~~~~~~~~
def mA(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = RGB2LAB(img)
    # L = mat[:, :, 0]
    A = mat[:, :, 1]
    # B = mat[:, :, 2]

    # Index Calculation
    img_A = 1 - A

    # Image output conditioning
    img_A = image_adjustment(img_A, adjust)

    return img_A


# ~~~~~~~~~~~~~~~~~~ B - A (LAB)~~~~~~~~~~~~~~~~~~~
def BmA(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = RGB2LAB(img)
    # L = mat[:, :, 0]
    A = mat[:, :, 1]
    B = mat[:, :, 2]

    # Index Calculation
    img_BmA = B - A

    # Image output conditioning
    img_BmA = image_adjustment(img_BmA, adjust)

    return img_BmA


# ~~~~~~~~~~~~~~~~~~ Minus ExU (LUV)~~~~~~~~~~~~~~~~~~~
def mExU(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = RGB2LUV(img)
    # L = mat[:, :, 0]
    U = mat[:, :, 1]
    V = mat[:, :, 2]

    # Index Calculation
    img_mExU = V - 1.3 * U

    # Image output conditioning
    img_mExU = image_adjustment(img_mExU, adjust)

    return img_mExU


# ~~~~~~~~~~~~~~~~~~ V - U (LUV)~~~~~~~~~~~~~~~~~~~
def VmU(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = RGB2LUV(img)
    # L = mat[:, :, 0]
    U = mat[:, :, 1]
    V = mat[:, :, 2]

    # Index Calculation
    img_VmU = V - U

    # Image output conditioning
    img_VmU = image_adjustment(img_VmU, adjust)

    return img_VmU


# ~~~~~~~~~~~~~~~~~~ minus Cr + Cb (YCrCb)~~~~~~~~~~~~~~~~~~~
def mCrpCb(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = RGB2YCrCb(img)
    # Y = mat[:, :, 0]
    Cr = mat[:, :, 1]
    Cb = mat[:, :, 2]

    # Index Calculation
    img_mCrpCb = 1 - (Cr + Cb)

    # Image output conditioning
    img_mCrpCb = image_adjustment(img_mCrpCb, adjust)

    return img_mCrpCb


# ~~~~~~~~~~~~~~~~~~ Cg (CrCgCb)~~~~~~~~~~~~~~~~~~~
def Cg(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = RGB2CrCgCb(img)
    # Cr = mat[:, :, 0]
    Cg = mat[:, :, 1]
    # Cb = mat[:, :, 2]

    # Index Calculation
    img_Cg = Cg

    # Image output conditioning
    img_Cg = image_adjustment(img_Cg, adjust)

    return img_Cg


# ~~~~~~~~~~~~~~~~~~ ExCg (CrCgCb)~~~~~~~~~~~~~~~~~~~
def ExCg(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = RGB2CrCgCb(img)
    Cr = mat[:, :, 0]
    Cg = mat[:, :, 1]
    Cb = mat[:, :, 2]

    # Index Calculation
    img_ExCg = 2 * Cg - Cr - Cb

    # Image output conditioning
    img_ExCg = image_adjustment(img_ExCg, adjust)

    return img_ExCg


# ~~~~~~~~~~~~~~~~~~ l3 (l1l2l3)~~~~~~~~~~~~~~~~~~~
def l3(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = RGB2l1l2l3(img)
    # l1 = mat[:, :, 0]
    # l2 = mat[:, :, 1]
    l3 = mat[:, :, 2]

    # Index Calculation
    img_l3 = l3

    # Image output conditioning
    img_l3 = image_adjustment(img_l3, adjust)

    return img_l3


# ~~~~~~~~~~~~~~~~~~ I3 (I1I2I3)~~~~~~~~~~~~~~~~~~~
def I3(img, adjust='clip'):
    # Input preparation and channel split
    if img.dtype == 'uint8':
        img = np.array(img, np.float32) / 255

    mat = RGB2I1I2I3(img)
    # I1 = mat[:, :, 0]
    # I2 = mat[:, :, 1]
    I3 = mat[:, :, 2]

    # Index Calculation
    img_I3 = I3

    # Image output conditioning
    img_I3 = image_adjustment(img_I3, adjust)

    return img_I3


# ~~~~~~~~~~~~~~~~~~ canny ~~~~~~~~~~~~~~~~~~~
def canny(img, adjust='', sigma=0.33):
    # Input preparation
    if img.dtype == 'float32':
        img = np.array(img*255, np.uint8)

    # compute the median of the single channel pixel intensities
    v = np.median(img)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    img_canny = cv2.Canny(img, lower, upper)

    # Image output conditioning
    img_canny = np.where(img_canny == 0, 1, img_canny)
    img_canny = np.array(img_canny, np.float32) / 255
    # return the edged image
    return img_canny


# ~~~~~~~~~~~~~~~~~~ Discrete Fourier Transform (DFT) Image ~~~~~~~~~~~~~~~~~~~
def dft(image, adjust=''):

    # Convert image to grayscale
    img_gray = np.copy(image)
    if len(image.shape) > 2:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    dft_shift = np.where(dft_shift == 0, 0.01, dft_shift) # Ensure no zero values to log
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # Image output conditioning
    img_dft = image_adjustment(magnitude_spectrum, adjust='')

    return img_dft


# ~~~~~~~~~~~~~~~~~ Multiple color indexes~~~~~~~~~~~~~~~~~~
def multiple_indexes(image, method_list, adjust='clip', no_background=True, color_map='gray', show_results=False,
                     show_time=False):
    # Saves the number of methods provided
    n_method = len(method_list)

    # Generates a image vector to hold the conversion output
    index_images = np.zeros((n_method, image.shape[0], image.shape[1]))

    # Executes and save into the vector each of the methods on the list
    for index, method in enumerate(method_list):
        ti = time.time()
        index_images[index] = eval(method)(image, adjust='clip')

        # Removes background for future calculations
        if no_background:
            index_images[index] = remove_background(index_images[index], image)

        if show_time:
            print('Tempo para ' + method + ' [s]: ' + str(time.time() - ti))

    if show_results:
        fig = plt.figure()
        row = math.ceil(np.sqrt(n_method + 1))
        col = row

        for index, method in enumerate(method_list):
            if index == 0:
                axi = fig.add_subplot(row, col, index + 1)  # Gera os subplots
                axi.set_title('Original')  # Nomeia cada subplot
                axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
                axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
                plt.imshow(image)  # Usa mapa de cor escala de cinza

            axi = fig.add_subplot(row, col, index + 2)  # Gera os subplots
            axi.set_title(method)  # Nomeia cada subplot
            axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
            axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
            plt.imshow(index_images[index], cmap=color_map)  # Usa mapa de cor escala de cinza

        plt.tight_layout(0.1)
        plt.show()

    return index_images

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~BACKUP~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # #---------USAGE SAMPLE---------

# # Direct usage
# ExG_p = colorindex.ExG(image_t_RGB, adjust='clip')
# ExG_p = colorindex.remove_background(ExG_t, image_t_RGB)

# # Multiple calculations
# mapcolor='RdYlGn'
# # mapcolor = 'gray'
#
# # # Color index
# # approach_list = ['NDI', 'ExG', 'ExR', 'ExGR', 'CIVE', 'VEG', 'MExG', 'COM1', 'COM2']
# # colorindex.multiple_indexes(imageRGB, approach_list, adjust='clip', color_map=mapcolor, show_time=True)
#
# # Alternative color index
# approach_list = ['YmX', 'mA', 'BmA', 'mExU', 'VmU', 'mCrpCb', 'Cg', 'ExCg', 'l3', 'I3']
# colorindex.multiple_indexes(imageRGB, approach_list, adjust='clip', color_map=mapcolor, show_time=True)
#
# # NDVI
# approach_list = ['NDVI']
# colorindex.multiple_indexes(imageNIR, approach_list, adjust='clip', color_map=mapcolor, show_time=True)
#
# # Alternative color conversion
# conversion_list = ['XYZ', 'LAB', 'LUV', 'HSV', 'YCrCb', 'CrCgCb', 'rgb', 'I1I2I3', 'l1l2l3', 'TSL']
# colorindex.multiple_conversions(imageRGB, conversion_list, show_time=True)


