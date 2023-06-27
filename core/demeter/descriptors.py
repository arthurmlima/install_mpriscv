"""
Criado por: Ivan Perissini
Data: 30/04/2020
Função: Gera imagens individuais para cada um dos SuperPixeis gerados pelo algoritimo Slic
Última alteração:
"""

import cv2
import numpy as np
import time
import pandas as pd
import scipy.stats
from scipy.special import expit
from skimage.feature import greycomatrix, greycoprops
from core.demeter import sptools as spt
import matplotlib.pyplot as plt
from skimage import feature
from skimage.feature import hog
from core.demeter import parameters
from core.demeter import colorindex
from core.demeter import metadata


def __version__():
    return 'descriptors version: v0.2'


# ======================== SUPPORT =========================

# ~~~~~~~~~~~~~~~~~~ Fast visualization of the descriptors~~~~~~~~~~~~~~~~~~~
def print_descriptor(descriptor):
    # Saves the first dict key
    first_key = list(descriptor.keys())[0]
    # Split the key to find the label reference
    label = first_key.split('_')[0]

    print('\n====Descriptor information ' + label + '====')
    for key, value in descriptor.items():
        if isinstance(value, str):
            print(key, '\t=\t', value)
        elif value > 0.001 or value == 0:
            print(key, '\t=\t', round(value, 3))
        else:
            print(key, '\t=\t', "{:.3e}".format(value))


# ~~~~~~~~~~~~~~~~~~ Fastest entropy calculation tested ~~~~~~~~~~~~~~~~~~~
def entropy(data, base=2):
    data_series = pd.Series(data)
    p_data = data_series.value_counts()  # counts occurrence of each value
    value = scipy.stats.entropy(p_data, base=base)  # get entropy from counts
    return value


# ~~~~~~~~~~~~~~~~~~ Merge descriptors ~~~~~~~~~~~~~~~~~~~
def merge_descriptor(descriptor1, descriptor2):
    out_descriptor = descriptor1.copy()
    out_descriptor.update(descriptor2)
    return out_descriptor


# ~~~~~~~~~~~~~~~~~ Multiple descriptors ~~~~~~~~~~~~~~~~~~
def multiple_descriptors(image, descriptor_list, label='img', show_time=False):
    # Initialize the dictionary
    descriptor = {}

    # Executes and merge the dictionaries based on the provided descriptor list
    for index, descriptor_type in enumerate(descriptor_list):
        ti = time.time()
        descriptor_aux = eval('img_' + descriptor_type)(image, label=label)
        descriptor.update(descriptor_aux)
        if show_time:
            print('Computation time for descritor ' + descriptor_type + ' [s]: ' + str(time.time() - ti))

    return descriptor


# ======================== DESCRIPTORS =========================

# ======================== STATISTICS =========================
# ~~~~~~~~~~~~~~~~~~ Simple descriptive statistics~~~~~~~~~~~~~~~~~~~
# At image level remove the zeros and compute all the basic descriptive statistics
# If a 3 channel image is provided the function also compute the metrics for each channel
def img_statistics_old(image, label='img', ignore_zero=True):
    if image.dtype == 'float32':
        image = np.array(image * 255, np.uint8)

    # Denominator of 255 to cancel final normalization operation
    norm_entropy = 32 / 255

    # Initialize the dict
    descriptor = {}

    # if ignore flag is true zeros are removed from calculation
    if ignore_zero:
        aux_image = image[image > 0]
    else:
        aux_image = np.copy(image)

    descriptor[label + '_Mean'] = np.mean(aux_image)
    descriptor[label + '_Max'] = np.max(aux_image)
    descriptor[label + '_Min'] = np.min(aux_image)
    descriptor[label + '_Med'] = np.median(aux_image)
    descriptor[label + '_Std'] = np.std(aux_image)
    descriptor[label + '_Entropy'] = entropy(aux_image) / norm_entropy

    # Case the imagem has multiple channels
    if len(image.shape) > 2:
        # For each channel do
        for n in range(image.shape[2]):
            channel = image[:, :, n]
            if ignore_zero:
                channel = channel[channel > 0]

            descriptor[label + '_MeanCh' + str(n + 1)] = np.mean(channel)
            descriptor[label + '_MaxCh' + str(n + 1)] = np.max(channel)
            descriptor[label + '_MinCh' + str(n + 1)] = np.min(channel)
            descriptor[label + '_MedCh' + str(n + 1)] = np.median(channel)
            descriptor[label + '_StdCh' + str(n + 1)] = np.std(channel)
            descriptor[label + '_EntropyCh' + str(n + 1)] = entropy(channel) / norm_entropy

    # If input is integer adjust values to 0 and 1
    for key, value in descriptor.items():
        descriptor[key] = np.float32(value / 255)

    return descriptor


# ~~~~~~~~~~~~~~~~~~ Simple descriptive statistics~~~~~~~~~~~~~~~~~~~
# At image level remove the zeros and compute all the basic descriptive statistics
# If a 3 channel image is provided the function also compute the metrics for each channel
def img_full_statistics(image, label='img', ignore_zero=True, norm='sigmoid'):
    if image.dtype == 'float32':
        image = np.array(image * 255, np.uint8)

    # Denominator of 255 to cancel final normalization operation
    norm_entropy = 32 / 255
    norm_skew = 10 / 255
    norm_kurtosis = 10 / 255

    # Initialize the dict
    descriptor = {}

    aux_image = np.copy(image)
    # If a color imagem is provided the image is converted to grayscale
    if len(image.shape) > 2:
        aux_image = cv2.cvtColor(aux_image, cv2.COLOR_RGB2GRAY)

        # if ignore flag is true zeros are removed from calculation
    if ignore_zero:
        aux_image = aux_image[aux_image > 0]

    descriptor[label + '_Mean'] = np.mean(aux_image)
    descriptor[label + '_Max'] = np.max(aux_image)
    descriptor[label + '_Min'] = np.min(aux_image)
    descriptor[label + '_Med'] = np.median(aux_image)
    descriptor[label + '_Std'] = np.std(aux_image)
    descriptor[label + '_Entropy'] = entropy(aux_image) / norm_entropy
    if norm == 'sigmoid':
        descriptor[label + '_Skew'] = 255 * expit(scipy.stats.skew(aux_image, bias=True, axis=None))
        descriptor[label + '_Kurtosis'] = 255 * expit(scipy.stats.kurtosis(aux_image, bias=True, axis=None))
    elif norm == 'custom':
        descriptor[label + '_Skew'] = 128 + scipy.stats.skew(aux_image, bias=True, axis=None) / norm_skew
        descriptor[label + '_Kurtosis'] = 128 + scipy.stats.kurtosis(aux_image, bias=True, axis=None) / norm_kurtosis

    # Case the imagem has multiple channels
    if len(image.shape) > 2:
        # For each channel do
        for n in range(image.shape[2]):
            channel = image[:, :, n]
            if ignore_zero:
                channel = channel[channel > 0]

            descriptor[label + '_MeanCh' + str(n + 1)] = np.mean(channel)
            descriptor[label + '_MaxCh' + str(n + 1)] = np.max(channel)
            descriptor[label + '_MinCh' + str(n + 1)] = np.min(channel)
            descriptor[label + '_MedCh' + str(n + 1)] = np.median(channel)
            descriptor[label + '_StdCh' + str(n + 1)] = np.std(channel)
            descriptor[label + '_EntropyCh' + str(n + 1)] = entropy(channel) / norm_entropy
            if norm == 'sigmoid':
                descriptor[label + '_SkewCh' + str(n + 1)] = \
                    255 * expit(scipy.stats.skew(channel, bias=True, axis=None))
                descriptor[label + '_KurtosisCh' + str(n + 1)] = \
                    255 * expit(scipy.stats.kurtosis(channel, bias=True, axis=None))
            elif norm == 'custom':
                descriptor[label + '_Skew' + str(n + 1)] = \
                    128 + scipy.stats.skew(channel, bias=True, axis=None) / norm_skew
                descriptor[label + '_Kurtosis' + str(n + 1)] = \
                    128 + scipy.stats.kurtosis(channel, bias=True, axis=None) / norm_kurtosis

    # If input is integer adjust values to 0 and 1
    for key, value in descriptor.items():
        descriptor[key] = np.float32(value / 255)

    return descriptor


# ~~~~~~~~~~~~~~~~~~ Simple descriptive statistics~~~~~~~~~~~~~~~~~~~
# At image level remove the zeros and compute all the basic descriptive statistics
# If a 3 channel image is provided the function also compute the metrics for each channel
def img_statistics(image, label='img', ignore_zero=True, norm='sigmoid'):
    if image.dtype == 'float32':
        image = np.array(image * 255, np.uint8)

    # Denominator of 255 to cancel final normalization operation
    norm_entropy = 16 / 255  # 16 levels in theory
    norm_skew = 10 / 255
    norm_kurtosis = 10 / 255

    # Initialize the dict
    descriptor = {}

    # Case the imagem has multiple channels
    if len(image.shape) > 2:
        # For each channel do
        for n in range(image.shape[2]):
            channel = image[:, :, n]
            if ignore_zero:
                channel = channel[channel > 0]

            descriptor[label + '_MeanCh' + str(n + 1)] = np.mean(channel)
            descriptor[label + '_MaxCh' + str(n + 1)] = np.max(channel)
            descriptor[label + '_MinCh' + str(n + 1)] = np.min(channel)
            descriptor[label + '_MedCh' + str(n + 1)] = np.median(channel)
            descriptor[label + '_StdCh' + str(n + 1)] = np.std(channel)
            descriptor[label + '_EntropyCh' + str(n + 1)] = entropy(channel) / norm_entropy
            if norm == 'sigmoid':
                descriptor[label + '_SkewCh' + str(n + 1)] = \
                    255 * expit(scipy.stats.skew(channel, bias=True, axis=None))
                descriptor[label + '_KurtosisCh' + str(n + 1)] = \
                    255 * expit(scipy.stats.kurtosis(channel, bias=True, axis=None))
            elif norm == 'custom':
                descriptor[label + '_Skew' + str(n + 1)] = \
                    128 + scipy.stats.skew(channel, bias=True, axis=None) / norm_skew
                descriptor[label + '_Kurtosis' + str(n + 1)] = \
                    128 + scipy.stats.kurtosis(channel, bias=True, axis=None) / norm_kurtosis
    # If image has only one channel do
    else:
        # if ignore flag is true zeros are removed from calculation
        if ignore_zero:
            aux_image = image[image > 0]

        descriptor[label + '_Mean'] = np.mean(aux_image)
        descriptor[label + '_Max'] = np.max(aux_image)
        descriptor[label + '_Min'] = np.min(aux_image)
        descriptor[label + '_Med'] = np.median(aux_image)
        descriptor[label + '_Std'] = np.std(aux_image)
        descriptor[label + '_Entropy'] = entropy(aux_image) / norm_entropy
        if norm == 'sigmoid':
            descriptor[label + '_Skew'] = 255 * expit(scipy.stats.skew(aux_image, bias=True, axis=None))
            descriptor[label + '_Kurtosis'] = 255 * expit(scipy.stats.kurtosis(aux_image, bias=True, axis=None))
        elif norm == 'custom':
            descriptor[label + '_Skew'] = 128 + scipy.stats.skew(aux_image, bias=True, axis=None) / norm_skew
            descriptor[label + '_Kurtosis'] = 128 + scipy.stats.kurtosis(aux_image, bias=True,
                                                                         axis=None) / norm_kurtosis

    # Adjust values to 0 and 1
    for key, value in descriptor.items():
        descriptor[key] = np.float32(value / 255)

    return descriptor


# ======================== HISTOGRAM =========================
# ~~~~~~~~~~~~~~~~~~ Image normalized histogram~~~~~~~~~~~~~~~~~~~
# Generates a normalized histogram (sum of bins equals 1) for a specified number of bins
def img_histogram(image, label='img', bins=8, ignore_zero=True):
    # Initialize the dict
    descriptor = {}

    data_type = image.dtype
    # Define the scale based on image type
    if data_type == 'uint8':
        max_scale = 256
    else:
        max_scale = 1

    # Case the imagem has multiple channels
    if len(image.shape) > 2:
        # For each channel do
        for n in range(image.shape[2]):
            channel = image[:, :, n]
            if ignore_zero:
                # Remove zero Pixels related to background
                channel = channel[channel > 0]

            # Compute the density histogram
            histogram, edges = np.histogram(channel, bins, (0, max_scale), normed=None, weights=None, density=True)

            # Normalizes the output to a unity sum
            histogram_norm = histogram * (max_scale / bins)

            for key, value in enumerate(histogram_norm):
                descriptor[label + '_histogramCh' + str(n + 1) + '_b' + str(key + 1)] = np.float32(value)

    else:
        if ignore_zero:
            # Remove zero Pixels related to background
            image = image[image > 0]

        # Compute the density histogram
        histogram, edges = np.histogram(image, bins, (0, max_scale), normed=None, weights=None, density=True)

        # Normalizes the output to a unity sum
        histogram_norm = histogram * (max_scale / bins)

        for key, value in enumerate(histogram_norm):
            descriptor[label + '_histogram_b' + str(key + 1)] = np.float32(value)

    return descriptor


# ======================== SHAPE =========================
# ~~~~~~~~~~~~~~~~~~ Image moments~~~~~~~~~~~~~~~~~~~
# Computes the images moments and hu moments
def img_moments(image, label='img', hu_only=False):
    # Initialize the dict
    descriptor = {}

    # Case the imagem has multiple channels converts to grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Convert nonzero values to 1 and compute the moments
    img_moments = cv2.moments(image, binaryImage=True)
    # Computes Hu moments
    huMoments = cv2.HuMoments(img_moments)

    if not hu_only:
        # Saves the image moments
        for key, value in img_moments.items():
            # Salva apenas os momentos normalizados
            if key[0] == 'n':
                # Adjust to the 0 to 1 interval
                descriptor[label + '_' + key] = np.float32((value / 2) + 0.5)

    # Save the Hu moments
    for key, value in enumerate(huMoments):
        descriptor[label + '_huMoments_' + str(key + 1)] = np.float32((value[0] / 2) + 0.5)

    return descriptor


# ======================== TEXTURE =========================
# ~~~~~~~~~~~~~~~~~~ Crop image margins to reduce background noise~~~~~~~~~~~~~~~~~~~
def img_remove_margin(image, margin=0):
    if margin == 0:
        return image

    img_size = image.shape
    margin_limit = 30

    # If margin is negative find the minimum margin to image
    if margin < 0:
        mask = image[:, :, 0] == 0
        mask_loop = mask
        margin_loop = 0.1
        while mask_loop.any() and margin_loop <= margin_limit / 100.0:
            margin_loop = margin_loop + 0.01
            mask_loop = mask[int(img_size[0] * margin_loop):int(img_size[0] * (1 - margin_loop)),
                        int(img_size[1] * margin_loop):int(img_size[1] * (1 - margin_loop))]
        margin = margin_loop * 100

    # Crop the input image to meet the margin provided
    if margin <= margin_limit:
        margin = margin / 100.0
        if len(img_size) > 2:
            image = image[int(img_size[0] * margin):int(img_size[0] * (1 - margin)),
                    int(img_size[1] * margin):int(img_size[1] * (1 - margin)), :]

        elif len(img_size) <= 2:
            image = image[int(img_size[0] * margin):int(img_size[0] * (1 - margin)),
                    int(img_size[1] * margin):int(img_size[1] * (1 - margin))]

    return image


# ~~~~~~~~~~~~~~~~~~ Gray-Level Co-Occurrence Matrix~~~~~~~~~~~~~~~~~~~
def img_glcm(image, label='img', distance=5, levels=256, norm='custom',
             debug=False, margin=parameters.get_default_margin()):
    # Initialize the dict
    descriptor = {}

    # Convert image to grayscale
    img_gray = np.copy(image)
    if len(image.shape) > 2:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Crop image if requested
    img_gray = img_remove_margin(img_gray, margin=margin)

    # Matches the image to the intended level for analysis
    if img_gray.dtype == 'uint8':
        img_gray = np.array(img_gray * (levels / 256), np.uint8)
    else:
        img_gray = np.array(img_gray * (levels - 1), np.uint8)

    glcm = greycomatrix(img_gray, distances=[distance], angles=[0], levels=levels, symmetric=True, normed=True)
    glcm_img = glcm[:, :, 0, 0]

    if debug:
        spt.img_show(img1=image, label1='Original',
                     img2=img_gray, label2='Gray',
                     img3=glcm_img, label3='GLCM',
                     share=False)

    descriptor[label + '_maxp'] = np.max(glcm_img)
    descriptor[label + '_contrast'] = greycoprops(glcm, 'contrast')[0, 0]
    descriptor[label + '_dissimilarity'] = greycoprops(glcm, 'dissimilarity')[0, 0]
    descriptor[label + '_homogeneity'] = greycoprops(glcm, 'homogeneity')[0, 0]
    descriptor[label + '_ASM'] = greycoprops(glcm, 'ASM')[0, 0]
    descriptor[label + '_energy'] = greycoprops(glcm, 'energy')[0, 0]
    descriptor[label + '_correlation'] = greycoprops(glcm, 'correlation')[0, 0]

    # Adjust values to 0 and 1
    if norm == 'sigmoid':
        for key, value in descriptor.items():
            descriptor[key] = expit(value)

    if norm == 'custom':
        descriptor[label + '_contrast'] = descriptor[label + '_contrast'] / (
                (levels - 1) * (levels - 1))  # interval [0,(l-1)²]
        descriptor[label + '_dissimilarity'] = descriptor[label + '_dissimilarity'] / (levels - 1)  # interval [0, l-1]
        descriptor[label + '_correlation'] = 0.5 + descriptor[label + '_correlation'] / 2  # interval [-1,1]

    return descriptor


# ~~~~~~~~~~~~~~~~~~ Local Binary Patterns ~~~~~~~~~~~~~~~~~~~
def img_lbp(image, label='img', radius=1, sampling_pixels=8, debug=False, margin=parameters.get_default_margin()):
    # Initialize the dict
    descriptor = {}

    # Convert image to grayscale
    img_gray = np.copy(image)
    if len(image.shape) > 2:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Crop image if requested
    img_gray = img_remove_margin(img_gray, margin=margin)

    # Ensures uint8 type
    if img_gray.dtype == 'float32':
        img_gray = np.array(img_gray * 256, np.uint8)

    # Compute LBP
    lbp = feature.local_binary_pattern(img_gray, sampling_pixels, radius, method="uniform")

    # LBP returns a matrix with the codes, so we compute the histogram
    (histogram, _) = np.histogram(lbp.ravel(), bins=np.arange(0, sampling_pixels + 3), range=(0, sampling_pixels + 2))

    # Hist normalization
    histogram = histogram.astype("float")
    histogram /= (histogram.sum() + 1e-6)

    if debug:
        fig = plt.figure()
        vals = range(len(histogram))

        axi = fig.add_subplot(1, 3, 1)  # Gera os subplots
        axi.set_title('Original')  # Nomeia cada subplot
        axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
        axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
        plt.imshow(image, cmap='gray')

        axi = fig.add_subplot(1, 3, 2)  # Gera os subplots
        axi.set_title('Gray')  # Nomeia cada subplot
        axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
        axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
        plt.imshow(img_gray, cmap='gray')

        axi = fig.add_subplot(1, 3, 3)
        plt.ylim(0, 1)
        axi.set_title('LBP code histogram')
        plt.bar(vals, histogram)

        # plt.savefig('LBP_' + label +'.png', dpi=800)
        plt.show()

    for key, value in enumerate(histogram):
        descriptor[label + '_LBP' + '_b' + str(key + 1)] = np.float32(value)

    return descriptor


# ~~~~~~~~~~~~~~~~~~ Histogram of oriented gradients ~~~~~~~~~~~~~~~~~~~
# Block normalization types: ‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’
def img_hog(image, label='img', grid=5, orientations=8, norm='L2-Hys', debug=False, margin=0, mode='default'):
    # Initialize the dict
    descriptor = {}

    # Convert image to grayscale
    # img_gray = np.copy(image)
    # if len(image.shape) > 2:
    #     img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = img_remove_margin(image, margin=margin)

    w, h, _ = image.shape
    size_cell_w = int(w / grid)
    size_cell_h = int(h / grid)

    if debug:
        feature_vector, hog_image = hog(image, orientations=orientations, pixels_per_cell=(size_cell_w, size_cell_h),
                                        cells_per_block=(1, 1), visualize=True, multichannel=True, feature_vector=True,
                                        block_norm=norm)

        print('Cell size:', size_cell_w, 'x', size_cell_h, '/ Vector Size:', len(feature_vector),
              '/ Orientations:', str(orientations), '/ grid number:', str(grid * grid))

        fig = plt.figure()
        vals = range(len(feature_vector))

        axi = fig.add_subplot(1, 3, 1)  # Gera os subplots
        axi.set_title('Original')  # Nomeia cada subplot
        axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
        axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
        plt.imshow(image, cmap='gray')

        axi = fig.add_subplot(1, 3, 2)  # Gera os subplots
        axi.set_title('HOG')  # Nomeia cada subplot
        axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
        axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
        plt.imshow(hog_image, cmap='gray')

        axi = fig.add_subplot(1, 3, 3)
        plt.ylim(0, 1)
        axi.set_title('HOG histogram')
        plt.bar(vals, feature_vector)
        plt.show()
    else:
        feature_vector = hog(image, orientations=orientations, pixels_per_cell=(size_cell_w, size_cell_h),
                             cells_per_block=(1, 1), visualize=False, multichannel=True, feature_vector=True,
                             block_norm=norm)

    if mode == 'statistics':
        feature_vector_2d = feature_vector.reshape(int(orientations), int(len(feature_vector) / orientations))

        for index, orientation_vet in enumerate(feature_vector_2d):
            base_str = label + '_HOG_g' + str(grid * grid) + '_o' + str(orientations) \
                       + '_m' + str(int(margin)) + '_f' + str(index + 1)

            descriptor[base_str + '_Mean'] = np.mean(orientation_vet)
            descriptor[base_str + '_Max'] = np.max(orientation_vet)
            descriptor[base_str + '_Min'] = np.min(orientation_vet)
            descriptor[base_str + '_Std'] = np.median(orientation_vet)
            descriptor[base_str + '_Med'] = np.std(orientation_vet)
            descriptor[base_str + '_Entropy'] = entropy(orientation_vet)
            descriptor[base_str + '_Skew'] = expit(scipy.stats.skew(orientation_vet, bias=True, axis=None))
            descriptor[base_str + '_Kurtosis'] = expit(scipy.stats.kurtosis(orientation_vet, bias=True, axis=None))
    else:
        for key, value in enumerate(feature_vector):
            # Turn around due to round errors
            if key < (grid * grid * orientations):
                descriptor[label + '_HOG_g' + str(grid * grid) + '_o' + str(orientations) +
                           '_m' + str(int(margin)) + '_f' + str(key + 1)] = np.float32(value)
            else:
                # print('Grid overflow, extra data excluded: ', mode, key+1, value, w / grid, h / grid)
                break

    # print(descriptor)
    return descriptor


# ~~~~~~~~~~~~~~~~~~ Multiple configurations for HOG ~~~~~~~~~~~~~~~~~~~
def img_multiple_hog(image, label='img', grid_orientation_set=parameters.get_hog_parameters(), norm='L2-Hys'):
    multi_hog = {}

    for i in grid_orientation_set:
        grid = i[0]
        orientations = i[1]
        margin = i[2]
        mode = i[3]
        hog_descriptor = img_hog(image, label=label, grid=grid, orientations=orientations,
                                 norm=norm, margin=margin, mode=mode)
        multi_hog.update(hog_descriptor)

    return multi_hog


# ~~~~~~~~~~~~~~~~~~ Discrete Fourier Transform (DFT) ~~~~~~~~~~~~~~~~~~~
def img2dft(image, debug=False):
    # Convert image to grayscale
    img_gray = np.copy(image)
    if len(image.shape) > 2:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    raw_dft = dft_shift
    img_dft = magnitude_spectrum

    if debug:
        spt.img_show(img1=image, label1='Original',
                     img2=img_gray, label2='Gray',
                     img3=img_dft, label3='DFT',
                     share=False)

    return img_dft, raw_dft


# ~~~~~~~~~~~~~~~~~~ Inverse Discrete Fourier Transform (DFT)  ~~~~~~~~~~~~~~~~~~~
def dft2img(raw_dft):
    f_ishift = np.fft.ifftshift(raw_dft)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


# ======================== HIGH LEVEL DESCRIPTOR =========================

def basic_info(image_path, mode='default'):
    # Name
    image_name = image_path.split(sep=r'\\')[-1]

    # GPS
    try:
        h, lat, long = metadata.get_gps(image_path)
    except:
        h, lat, long = 'NA', 'NA', 'NA'

    # Species
    try:
        species_name = image_name.split(sep='_')[1]
    except:
        species_name = 'NA'

    basic_info_dictionary = {'image_name': image_name,
                             'mode': mode,
                             'species': species_name,
                             'GPS_alt': h,
                             'GPS_lat': lat,
                             'GPS_long': long}

    return basic_info_dictionary


# ~~~~~~~~~~~~~~~~~~ Blur Detection  ~~~~~~~~~~~~~~~~~~~
def blur_detect(image):
    blur_dictionary = {}
    low_limit = 1250
    high_limit = 2000

    focus_index = cv2.Laplacian(image, cv2.CV_64F).var()
    blur_dictionary['focus index'] = focus_index

    if focus_index < low_limit:
        blur_dictionary['quality'] = 'low'
    elif focus_index < high_limit:
        blur_dictionary['quality'] = 'medium'
    else:
        blur_dictionary['quality'] = 'high'

    return blur_dictionary


# ~~~~~~~~~~~~~~~~~~ Period Detection  ~~~~~~~~~~~~~~~~~~~
def day_cycle_detect(image_path):
    period_dictionary = {}

    try:
        date_time = metadata.get_datetime(image_path)
    except:
        date_time = '2000:01:01 13:00:00'

    str_date = date_time[0:10]
    str_time = date_time[11:19]

    hour = int(str_time[0:2])
    hour_reference = 13
    light_index = abs(hour - hour_reference)

    period_dictionary['date'] = str_date
    period_dictionary['time'] = str_time
    period_dictionary['chroma_index'] = light_index

    if 6 <= hour < 10:
        period_dictionary['period'] = 'Dawn'
    elif 10 <= hour < 16:
        period_dictionary['period'] = 'Day'
    elif 16 <= hour < 18:
        period_dictionary['period'] = 'Dusk'
    else:
        period_dictionary['period'] = 'Night'

    return period_dictionary


# ~~~~~~~~~~~~~~~~~~ weather detection  ~~~~~~~~~~~~~~~~~~~
def weather_detect(image, debug=False):
    weather_dictionary = {}

    # Limits for each condition
    season_ll = 0.0680
    season_ul = 0.0753
    light_l = 0.070

    if image.dtype == 'uint8':
        image = np.array(image, np.float32) / 255

    # Most promising color channels for weather detection
    s_img = colorindex.RGB2TSL(image)[:, :, 1]
    b_img = colorindex.RGB2rgb(image)[:, :, 2]

    s_mean = np.mean(s_img)
    b_mean = np.mean(b_img)
    dry_index = s_mean * s_mean + b_mean * b_mean
    weather_dictionary['sunny_index'] = s_mean
    weather_dictionary['dry_index'] = dry_index

    if s_mean < light_l:
        weather_dictionary['light'] = 'cloudy'
    else:
        weather_dictionary['light'] = 'sunny'

    if dry_index < season_ll:
        weather_dictionary['season'] = 'water'
    elif dry_index > season_ul:
        weather_dictionary['season'] = 'dry'
    else:
        weather_dictionary['season'] = 'transition'

    if debug:
        spt.img_show(img1=image, label1='Original',
                     img2=s_img, label2='S in TSI',
                     img3=b_img, label3='b in rgb', title=str(weather_dictionary))

    return weather_dictionary


# ~~~~~~~~~~~~~~~~~~ image info  ~~~~~~~~~~~~~~~~~~~
def image_info(image_path):
    image_info_dictionary = {}
    image = spt.img_read(image_path)

    image_info_dictionary.update(basic_info(image_path))
    image_info_dictionary.update(day_cycle_detect(image_path))
    image_info_dictionary.update(weather_detect(image))
    image_info_dictionary.update(blur_detect(image))

    prefix = 'inf_'
    image_info_dictionary = {prefix + str(key): val for key, val in image_info_dictionary.items()}

    return image_info_dictionary
