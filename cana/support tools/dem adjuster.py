"""
Criado por: Ivan Perissini
Data: 09/06/2020
Função: Codigo para integração dos módulos e desenvolvimento base do programa
Última alteração:
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np

def bounding_box(mask):
    image_w = np.sum(mask, axis=0)
    image_h = np.sum(mask, axis=1)

    min_x = min_y = max_x = max_y = None
    val_ant = 0
    for index, val in enumerate(image_w):
        if index == 0:
            val_ant = val
        else:
            if val > val_ant == 0:
                min_x = index
            if val_ant > val == 0:
                max_x = index
            val_ant = val

    val_ant = 0
    for index, val in enumerate(image_h):
        if index == 0:
            val_ant = val
        else:
            if val > val_ant == 0:
                min_y = index
            if val_ant > val == 0:
                max_y = index
            val_ant = val

    return min_x, min_y, max_x, max_y


# --------------------------------------
# ------------MAIN CODE-----------------
# --------------------------------------

# ---- PARAMETERS ----
input_dir = "C:/Users/ivanp/OneDrive/Desktop/Agisoft/Retry/"
dem_path = input_dir + 'DEM.PNG'
orto_path = input_dir + 'ORTO.PNG'
output_dir = input_dir

debug = True
save_output = False
adjust_mode = ''  # 'norm' - normaliza , 'scale' - scale by 100, default - no changes

# ---- CODE ----
# Open image without modifying the data type
img_dem = cv2.imread(dem_path, -1)

# Converts to np array
image_pos = np.array(img_dem, np.float32)

# Convert background to nan
image_pos = np.where(image_pos == np.min(image_pos), np.nan, image_pos)

# Offset to only positives values
image_pos = image_pos - np.nanmin(image_pos)  # Ajuste linear

if adjust_mode == 'norm':
    # Scales to 0 and 1
    h_max = np.nanmax(image_pos)
    image_pos = image_pos * (1 / h_max)

if adjust_mode == 'scale':
    # Scales by 100
    image_pos = image_pos * (1 / 100)

image_mask = np.where(img_dem == np.min(img_dem), 0, 1)

min_x, min_y, max_x, max_y = bounding_box(image_mask)
dem_crop = image_pos[min_y:max_y, min_x:max_x]

img_orto = cv2.imread(orto_path)
img_orto = cv2.cvtColor(img_orto, cv2.COLOR_BGR2RGB)

# resize image to match orto
dem_resize = cv2.resize(dem_crop, (img_orto.shape[1], img_orto.shape[0]), interpolation=cv2.INTER_AREA)

# img_out = np.dstack((img_orto[:, :, 0]/255, img_orto[:, :, 1]/255, img_orto[:, :, 2]/255, dem_resize))
# img_out = img_orto * np.dstack((dem_resize, dem_resize, dem_resize))/255

if debug:
    fig = plt.figure()

    axi = fig.add_subplot(3, 1, 1)  # Gera os subplots
    axi.set_title("Original DEM")  # Nomeia cada subplot
    axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
    axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
    plt.imshow(img_dem)

    axi = fig.add_subplot(3, 1, 2)  # Gera os subplots
    axi.set_title("Original ORTO")  # Nomeia cada subplot
    axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
    axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
    plt.imshow(img_orto)

    axi = fig.add_subplot(3, 1, 3)  # Gera os subplots
    axi.set_title("DEM adjusted")  # Nomeia cada subplot
    axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
    axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
    plt.imshow(dem_resize)

    plt.show()

if save_output:
    # Tiff is mandatory to allow 32 bit output
    cv2.imwrite(input_dir + "DEM_adjust.tiff", dem_resize)
