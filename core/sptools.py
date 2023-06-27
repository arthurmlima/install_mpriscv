#"""
#Criado por: Ivan Perissini
#Data: 30/04/2020
#Função: Módulo com um conjunto usual de ferramentas para trabalhar com SuperPixel
#Última alteração: 18/11/2020
#"""

import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from fast_slic import Slic
#from numba import jit
import numpy as np
from core.demeter import metadata


def __version__():
    return 'sptools version: v0.3'


# ======================== SUPPORT =========================

# ~~~~~~~~~~~~~~~~~~ Image Read ~~~~~~~~~~~~~~~~~~~
def img_read(image_path, tofloat=False):
    """
    Own version of the image read function in order to assure the input conditions for the other image processing
    functions
    :param image_path: path of the image
    :param tofloat: if true scales the output to a float condition within 0 and 1
    :return: RGB image without zeros
    """
    image = cv2.imread(image_path)

    # Converts the BGR original from openCV to the traditional RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Removes all zeros from the image, to allow background differentiation in future operations
    image = np.where(image == 0, 1, image)

    if tofloat:
        image = np.array(image, np.float32) / 255

    return image


# # ~~~~~~~~~~~~~~~~~~ Multiple image show ~~~~~~~~~~~~~~~~~~~
def img_show(img1=[0], label1='Image1', img2=[0], label2='Image2',
             img3=[0], label3='Image3', img4=[0], label4='Image4',
             share=False, title='Imagens', color_map='gray'):
    """
    Matplotlib imagem plot prepared for up to 4 images
    :return: 0 if sucessfull
    """
    # color_map = 'gray'

    # Compute the adequate plot grid size
    size = int(len(img1) > 1) + int(len(img2) > 1) + int(len(img3) > 1) + int(len(img4) > 1)
    if size == 4:
        row = 2
        col = 2
    else:
        row = 1
        col = size

    # Generate the subplot frame
    fig, axs = plt.subplots(nrows=row, ncols=col, sharex=share, sharey=share)

    axs = np.array(axs)
    ax = axs.flatten()

    p = -1
    if len(img1) > 1:
        p = p + 1
        ax[p].axis('off')
        ax[p].set_title(label1)  # Name it
        ax[p].imshow(img1, cmap=color_map)

    if len(img2) > 2:
        p = p + 1
        ax[p].axis('off')
        ax[p].set_title(label2)
        ax[p].imshow(img2, cmap=color_map)

    if len(img3) > 3:
        p = p + 1
        ax[p].axis('off')
        ax[p].set_title(label3)
        ax[p].imshow(img3, cmap=color_map)

    if len(img4) > 4:
        p = p + 1
        ax[p].axis('off')
        ax[p].set_title(label4)
        ax[p].imshow(img4, cmap=color_map)

    # plt.tight_layout(0.1)
    plt.suptitle(title, wrap=True, size=16)
    # plt.text(20.1, 20.1, text, ha='center', va='top', rotation=0, wrap=True, size=16)
    plt.show()

    return 0


# ======================== SLIC =========================

# ~~~~~~~~~~~~~~~~~~ Fast slic ~~~~~~~~~~~~~~~~~~~
# Fast version for slic calculation, the function copiles other operations to assure similarity with the original slic
def fast_slic(image, n_segments=100, compactness=10.0, sigma=0, convert2lab=True, min_size_factor=0.5):
    # Create a copy of the original image
    img_original = np.copy(image)

    # Convert to LAB as standard to this method
    if convert2lab:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # If sigma is provided, performs a gaussian blur to the image
    if sigma > 0:
        image = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma)

    # Fast slic
    slic = Slic(num_components=n_segments, compactness=compactness, min_size_factor=min_size_factor)
    # Cluster Map, an image where each pixel contains the super pixel identifier
    segment = slic.iterate(image)

    # Mark the SuperPixel boundaries for demonstration
    img_slic = mark_boundaries(img_original, segment, color=(0, 1, 1), outline_color=(0, 1, 1))
    # The previous operation generates a float output, so a conversion to uint8 is required
    img_slic = cv2.convertScaleAbs(img_slic, alpha=255)

    return img_slic, segment


# ~~~~~~~~~~~~~~~~~~ Super pixel slicer ~~~~~~~~~~~~~~~~~~~
# Given a reference image and the super pixel map image, the function returns relevant information of each super pixel
# >> box - that contain each SP bounding box coordinate
# >> sp_img_vector - an image vector of each SuperPixel
#@jit
def sp_slicer(image, segments, box_only=False):
    (length, height, depth) = image.shape

    # Saves the numbers of SP at the image
    n_segments = np.max(segments) + 1

    # Initialize the vectors that saves the bounding box coordinates
    min_x = np.full(n_segments, length, dtype=np.uint16)
    min_y = np.full(n_segments, height, dtype=np.uint16)
    max_x = np.zeros(n_segments, dtype=np.uint16)
    max_y = np.zeros(n_segments, dtype=np.uint16)
    # Copiles the box coordinates
    box = (min_x, max_x, min_y, max_y)

    # First image sweep, to compute the SP coordinates
    for x in range(0, length):
        for y in range(0, height):

            # For each pixel, identify the SP and updates if needed the coordinates
            if x < min_x[segments[x, y]]:
                min_x[segments[x, y]] = x

            if x > max_x[segments[x, y]]:
                max_x[segments[x, y]] = x

            if y < min_y[segments[x, y]]:
                min_y[segments[x, y]] = y

            if y > max_y[segments[x, y]]:
                max_y[segments[x, y]] = y

    # Saves the biggest SP dimensions
    max_delta_x = np.max(max_x - min_x)
    max_delta_y = np.max(max_y - min_y)

    # Create a black image vector that matches the size of the biggest SP found
    sp_img_vector = np.zeros((n_segments, max_delta_x, max_delta_y, 3), dtype=image.dtype)

    if not box_only:
        # Second image sweep populates the SP image vector with the reference image
        for x in range(0, length):
            for y in range(0, height):
                # For each pixel the reference pixel is matched at the correspondent SP vector index
                sp_img_vector[segments[x, y], x - min_x[segments[x, y]], y - min_y[segments[x, y]]] = image[x, y]

    return sp_img_vector, box


# ~~~~~~~~~~~~~~~~~~ Super pixel mask ~~~~~~~~~~~~~~~~~~~
# Given a reference image and the super pixel map image, the function returns relevant information of each super pixel
# >> box - that contain each SP bounding box coordinate
# >> sp_mask_vector - an image mask vector of each SuperPixel
#@jit
def sp_mask(segments, box_only=False):
    (length, height) = segments.shape

    # Saves the numbers of SP at the image
    n_segments = np.max(segments) + 1

    # Initialize the vectors that saves the bounding box coordinates
    min_x = np.full(n_segments, length, dtype=np.uint16)
    min_y = np.full(n_segments, height, dtype=np.uint16)
    max_x = np.zeros(n_segments, dtype=np.uint16)
    max_y = np.zeros(n_segments, dtype=np.uint16)
    # Copiles the box coordinates
    box = (min_x, max_x, min_y, max_y)

    # First image sweep, to compute the SP coordinates
    for x in range(0, length):
        for y in range(0, height):

            # For each pixel, identify the SP and updates if needed the coordinates
            if x < min_x[segments[x, y]]:
                min_x[segments[x, y]] = x

            if x > max_x[segments[x, y]]:
                max_x[segments[x, y]] = x

            if y < min_y[segments[x, y]]:
                min_y[segments[x, y]] = y

            if y > max_y[segments[x, y]]:
                max_y[segments[x, y]] = y

    # Saves the biggest SP dimensions
    max_delta_x = np.max(max_x - min_x)
    max_delta_y = np.max(max_y - min_y)

    # Create a black image vector that matches the size of the biggest SP found
    sp_mask_vector = np.zeros((n_segments, max_delta_x, max_delta_y, 3), dtype=np.uint8)

    if not box_only:
        # Second image sweep populates the SP image vector with the reference image
        for x in range(0, length):
            for y in range(0, height):
                # For each pixel the reference pixel is matched at the correspondent SP vector index
                sp_mask_vector[segments[x, y], x - min_x[segments[x, y]], y - min_y[segments[x, y]]] \
                    = np.array([1, 1, 1])

    return sp_mask_vector, box


# ~~~~~~~~~~~~~~~~~~ Image Superpixel ~~~~~~~~~~~~~~~~~~~
# Main Super Pixel function copiles the slic and support operation in a single function
# Output: img, img_slic, segments, sp_img_vector, box
def img_superpixel(image_path, debug=False, segment=0):
    img = img_read(image_path)

    # If segment value is not provided it estimates based on image height
    if segment == 0:
        try:
            h, _, _ = metadata.get_gps(image_path)
        except:
            h = 50
            print('Height not found at image metadata , default value of', h, 'meters used')
        segment = metadata.segment_estimation(h)

    # Other slic parameters
    compactness = 30
    sigma = 2
    min_size = 0.5

    # Fast Slic
    img_slic, segments = fast_slic(img, n_segments=segment, compactness=compactness,
                                   sigma=sigma,
                                   min_size_factor=min_size, convert2lab=True)

    # retrieve super pixel information
    sp_img_vector, box = sp_slicer(img, segments)

    # if debug mode, show image results
    if debug:
        img_show(img1=img, label1='Original image',
                 img2=img_slic, label2='Slic Image',
                 img3=segments, label3='Segments Image',
                 img4=sp_img_vector[1], label4='Sample Image',
                 share=False)

    return img, img_slic, segments, sp_img_vector, box


# ~~~~~~~~~~~~~~~~~~ Class map ~~~~~~~~~~~~~~~~~~~
# Generates the class map based on the pair sp value and label
# The results are returned as a class image, a color image based on label color and a blend version with the original
def class_map(image_path, sp_values, sp_labels, alpha=0.25, debug=False):
    if len(sp_values) != len(sp_labels):
        print('Error - Incompatible input sizes')
        print('sp_values size:', len(sp_values), 'sp_labels size:', len(sp_labels))
        return -1

    # Generates a colors dictionary
    color_dictionary = {'0': (1, alpha, alpha),  # Solo
                        '1': (1, 1, alpha),  # Planta 1
                        '2': (0.75 * (1 - alpha) + alpha, 1, alpha),  # Planta 2
                        '3': (0.50 * (1 - alpha) + alpha, 1, alpha),  # Planta 3
                        '4': (0.25 * (1 - alpha) + alpha, 1, alpha),  # Planta 4
                        '5': (alpha, 1, alpha),  # Planta 5
                        '6': (alpha, 1, 1),  # Animal
                        '7': (alpha, alpha, alpha),  # Resto
                        '8': (alpha, alpha, 1)}  # Daninha

    # Generate Super Pixels and support information
    img, _, segments, _, box = img_superpixel(image_path, debug=debug)

    # Create label canvas
    label_img = np.float32(np.ones_like(img))
    class_img = np.float32(np.zeros_like(img[:, :, 0]))

    # For each SP do:
    for index, nSP in enumerate(sp_values):
        label = str(sp_labels.iloc[index])
        if debug:
            print(index, nSP, sp_labels.iloc[index], color_dictionary[label])

        # Saves the coordinates of the bounding box of the SP for future use
        (x1, x2, y1, y2) = (box[0][nSP], box[1][nSP], box[2][nSP], box[3][nSP])

        # Create a mask for the SP at the relevant interval
        mask = np.where(segments[x1:x2, y1:y2] == nSP, 1, 0)

        # Increase mask dimension and multiply by label color
        mask3d = np.dstack((mask, mask, mask)) * color_dictionary[label]

        # Change background to 1 to maintain image values
        mask3d = np.where(mask3d == 0, 1, mask3d)

        # Generates the label image by multiplying the mask
        label_img[x1:x2, y1:y2, :] = mask3d * label_img[x1:x2, y1:y2, :]

        # Generates a class image for future calculations
        class_img[x1:x2, y1:y2] = np.where(segments[x1:x2, y1:y2] == nSP, int(label), class_img[x1:x2, y1:y2])

    # Generates the output image
    out_img = label_img * img

    if img.dtype == 'uint8':
        out_img = np.uint8(out_img)

    if debug:
        img_show(img1=class_img, label1='Class image',
                 img2=label_img, label2='Label image',
                 img3=out_img, label3='Output Image')

    return img, class_img, label_img, out_img
