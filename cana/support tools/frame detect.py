"""
Criado por: Ivan Perissini
Data: 28/10/2020
Função: Codigo para pré-processar imagens com frame e expandir sua cobertura
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import winsound
import sptools as spt


def relevant_contorns(img, bin, thickness, max_obj, area_min, max_perimeter, mean_aspect_ratio):
    # Find the binary image contourns
    contours, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out_img = np.copy(img)

    # Create a structured array for the object propreties
    dtype = [('ID', int), ('area', int), ('perimeter', int), ('Cx', int), ('Cy', int), ('aspect_ratio', float)]
    values = np.ones((max_obj), dtype=int) * -1
    objects = np.array(values, dtype=dtype)

    ratio_tolerance = 0.4  # 0.2
    thick_size = thickness
    area_max = 10000  # 8000
    min_perimeter = 100  # 200
    max_aspect_ratio = 1.0  # 1.2
    contours_color = (0, 0, 0)
    # contours_color = (0, 0, 255)
    # cv2.drawContours(out_img, contours, -1, color=(0,255,0), thickness=thick_size)

    n = 0
    count = cx_s = cy_s = cx_r = cy_r = 0
    for id, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        # Verify the criteria for the selection of the objects
        if cv2.contourArea(contour) > area_min:
            # print('contourArea', cv2.contourArea(contour))

            # Temporary records the object moments in M
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Capture the frame
            if (cv2.arcLength(contour, 1) < max_perimeter
                    and n < max_obj
                    and mean_aspect_ratio - ratio_tolerance < aspect_ratio < mean_aspect_ratio + ratio_tolerance):


                objects[n] = (
                    id,
                    cv2.contourArea(contour),
                    cv2.arcLength(contour, 1),
                    cx,
                    cy,
                    float(aspect_ratio))

                # Draw the relevant contourns
                cv2.drawContours(out_img, [contour], -1, color=contours_color, thickness=thick_size)
                n = n + 1

            # # Capture the numbers
            # elif (cv2.contourArea(contour) < area_max
            #       and max_perimeter + 200 > cv2.arcLength(contour, 1) > min_perimeter
            #       and aspect_ratio < max_aspect_ratio):
            #     count = count + 1
            #     cx_s = cx_s + cx
            #     cy_s = cy_s + cy
            #
            #     # Draw the relevant contourns
            #     # cv2.drawContours(out_img, [contour], -1, color=(255, 0, 255), thickness=thick_size)
            #
            #     print(id,
            #           cv2.contourArea(contour),
            #           cv2.arcLength(contour, 1),
            #           cx,
            #           cy,
            #           float(aspect_ratio))

    if count > 0:
        cx_r = int(cx_s / count)
        cy_r = int(cy_s / count)

    # Sort the relevant results in respect to the contourn area
    objects = np.sort(objects, order='area')[::-1]
    objects = objects[0:n]

    return out_img, objects, cx_r, cy_r


# --------------------------------------
# ------------MAIN CODE-----------------
# --------------------------------------

# ---- PARAMETROS ----
input_dir = r'D:\\Demeter Banco de imagens\\.Selecionadas\\Massas\\Massa RUN 3\\Fix\\'
output_dir = r'D:\\Demeter Banco de imagens\\.Selecionadas\\Massas\\Massa RUN 3\\Pos\\'

debug = True
save_output = True
# threshold_azul = 200  # 200
threshold = 190 # 185
kernel_size = 3  # 3
min_area = 2000  # 2000
max_perimeter = 400
mean_aspect_ratio = 1
thickness = 6 #7
margin = 70
name_filter = ''
# Beep
frequency = 900
duration = 300

# ---- CODIGO ----
for root, directories, files in os.walk(input_dir):
    for file in files:
        # try:
        if name_filter in file:
            print(file)

            file_name, file_extension = os.path.splitext(file)

            image_path = input_dir + file
            image = cv2.imread(image_path)
            image = np.array(image)

            # Converte do BRG original do openCV para o tradicional RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            mask = cv2.inRange(image, (threshold - 80, threshold, threshold), (255, 255, 255))
            # mask = cv2.inRange(image, (0, 0, threshold), (30, 30, 255))
            # mask = cv2.inRange(img_hsv, (50, 0, 100), (150, 255, 255))

            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

            img_out, objects, cx_r, cy_r = relevant_contorns(image, mask, thickness, 10,
                                                             min_area, max_perimeter, mean_aspect_ratio)
            # Manual
            # cx_r = 1870
            # cy_r = 1470
            # spt.img_show(img_slic)

            img_crop = np.copy(img_out)
            if cx_r + cy_r > 0:
                cxi = cx_r - margin
                cyi = cy_r - margin
                cxf = cx_r + margin
                cyf = cy_r + margin

                img_crop[cyi:cyf, cxi:cxf, :] = np.ones((margin * 2, margin * 2, 3)) * 255

            img_slic, segments = spt.fast_slic(img_crop, n_segments=2800, compactness=40,
                                               sigma=2,
                                               min_size_factor=0.1, convert2lab=True)

            # Função de separação dos SuperPixeis
            sp_vector, box = spt.sp_slicer(img_crop, segments)

            num_frames = len(objects)
            # If no frame is found ignores the output
            if num_frames > 0:
                print('Numero de frames encontrados', num_frames)
                cy_img, cx_img, _ = np.array(image.shape)/2

                center_frame = 0
                min_dist = 10000000
                for n in range(num_frames):
                    sp_cx = objects['Cx'][n]
                    sp_cy = objects['Cy'][n]
                    dist = (cx_img-sp_cx)*(cx_img-sp_cx) + (cy_img-sp_cy)*(cy_img-sp_cy)
                    print(dist)
                    if dist < min_dist:
                        center_frame = n
                        min_dist = dist
                print('frame central', center_frame)

                sp_cx = objects['Cx'][center_frame]
                sp_cy = objects['Cy'][center_frame]
                print('centro frame', sp_cx, sp_cy)
                n_sp = segments[sp_cy, sp_cx]
                print('N_sp correspondente', n_sp)
            else:
                img_out = image

            xi = box[0][n_sp]
            xf = box[1][n_sp]
            yi = box[2][n_sp]
            yf = box[3][n_sp]

            sp_crop = sp_vector[n_sp][0:xf - xi, 0:yf - yi, :]
            # sp_image = img[xi:xf, yi:yf, :] * sp_mask_vector[n_sp][0:xf - xi, 0:yf - yi, :]

            if debug:
                print('Relevant Objects:')
                print(objects)

                fig = plt.figure()

                axi = fig.add_subplot(2, 2, 1)  # Gera os subplots
                axi.set_title("Imagem original: " + file)  # Nomeia cada subplot
                axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
                axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
                plt.imshow(image)

                axi = fig.add_subplot(2, 2, 3)  # Gera os subplots
                axi.set_title("Mascara")  # Nomeia cada subplot
                axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
                axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
                plt.imshow(mask)

                axi = fig.add_subplot(2, 2, 2)  # Gera os subplots
                axi.set_title("Saída")  # Nomeia cada subplot
                axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
                axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
                plt.imshow(img_out)

                axi = fig.add_subplot(2, 2, 4)  # Gera os subplots
                axi.set_title("Recorte")  # Nomeia cada subplot
                axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
                axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
                plt.imshow(img_crop)

                axi = fig.add_subplot(5, 5, 13)  # Gera os subplots
                axi.set_title("SP")  # Nomeia cada subplot
                axi.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
                axi.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
                plt.imshow(sp_crop)

                plt.show()

            if save_output:
                print("Imagem", file_name, "salva em", output_dir + file_name + "out" + file_extension)
                sp_crop = cv2.cvtColor(sp_crop, cv2.COLOR_RGB2BGR)
                img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_dir + file_name + '_sp' + file_extension, sp_crop)
                cv2.imwrite(output_dir + file_name + file_extension, img_crop)

        # except:
        # print('Erro para acessar ', file_name)

    # End code beep
    winsound.Beep(frequency, duration)

# --------------------------------------
# ------------PARKING LOT-----------------
# --------------------------------------

# # DICTIONARY APPROACH
# objects_dic = {'ID': -1, 'area': -1, 'Cx': -1, 'Cy': -1, 'aspect_ratio': -1}
# dic_list = []


# objects_dic['ID'] = id
# objects_dic['area'] = cv2.contourArea(contour)
# objects_dic['Cx'] = int(M['m10'] / M['m00'])
# objects_dic['Cy'] = int(M['m01'] / M['m00'])
# objects_dic['aspect_ratio'] = aspect_ratio
# dic_list.append(objects_dic)
