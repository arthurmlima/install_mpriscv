"""
Criado por: Ivan C Perissini
Data: 25/07/2020
Função: Módulo com conjunto usual de ferramentas para trabalhar com aprendizagem de máquina
Última alteração: 25/07/2020
"""
from core.demeter import parameters
from core.demeter import sptools as spt
from core.demeter import mltools as mlt
from core.demeter import metadata
from core.demeter import dbtools as dbt
from core.demeter import descriptors as info
import matplotlib.pyplot as plt
from core.demeter import colorindex
import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import os
import math
from joblib import dump, load
import matplotlib.colors as mcolors
from reportlab.pdfgen import canvas


def __version__():
    return 'metrics version: v0.2'


# #======================== RESULTS ============================

# ~~~~~~~~~~~~~~~~~~ Get image coverage ~~~~~~~~~~~~~~~~~~~
# Based on the camera parameters and image height the pixel and dimension correlations are returned
def get_coverage(height, camera_info=parameters.get_camera_info(), show_info=False):
    if height > 0:
        scale = 1000 * (height / camera_info['f'])
        m_pix_widht = (camera_info['widht_sensor'] / camera_info['widht_pixels']) * scale / 1000
        m_pix_heigth = (camera_info['heigth_sensor'] / camera_info['heigth_pixels']) * scale / 1000

        # Use the bigger mm to pixel value as reference
        m_pix = np.max([m_pix_widht, m_pix_heigth])  # meter per imagem pixel

        widht_m = camera_info['widht_pixels'] * m_pix_widht
        heigth_m = camera_info['heigth_pixels'] * m_pix_heigth
        area = widht_m * heigth_m

        m2_pix = m_pix_heigth * m_pix_widht  # square meter per imagem pixel

        if show_info:
            print('---------Image Projection calculations---------')
            print('Considering a flight height of: {0:1.2f}'.format(height))
            print('meters per image pixel: {0:1.5f}'.format(m_pix))
            print('Square meters per image pixel {0:1.5f}'.format(m2_pix))
            print('Estimated coverage area: {0:1.1f} x {1:1.1f} = {2:1.1f} m²'.format(widht_m, heigth_m, area))
    else:
        print('Incompatible height value')

    return area, m_pix, m2_pix


# ~~~~~~~~~~~~~~~~~~ Label percent ~~~~~~~~~~~~~~~~~~~
# Based on the class imagem, where each pixel contains the label number a percent summary is returned
def label_percent(img_class, label_dictionary=parameters.get_label_info()):
    percent_results = {}
    size = img_class.shape[0] * img_class.shape[1]
    for name, value in label_dictionary.items():
        count = np.sum(img_class == value[0])
        percent_results[name] = count / size

    return percent_results


# ~~~~~~~~~~~~~~~~~~ Pie chart ~~~~~~~~~~~~~~~~~~~
# Based on percent summary a pie chart graph is show
def pie_chart(percent_results, label_dictionary=parameters.get_label_info(), save_img=False):
    # Remove zero values from results
    results_filter = {key: value for key, value in percent_results.items() if value > 0.01}
    # results_filter = {key: value for key, value in percent_results.items() if value != 0}
    # results_filter = percent_results

    # Select only the relevant colors
    label_color = [label_dictionary[key][1] for key, value in results_filter.items()]

    fig = plt.figure()

    # Generate pie chart
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Vegetation coverage")
    ax.axis('equal')

    names = results_filter.keys()
    values = results_filter.values()

    wedges, texts, texts = ax.pie(values, labels=names, colors=label_color, autopct='%1.1f%%')

    # # Adding legend
    # ax.legend(wedges, names,
    #           title="Classes",
    #           loc="center left")

    plt.setp(texts, size=8, weight="bold")

    if save_img:
        plt.savefig('results/results.png', dpi=800)
        plt.savefig('results/results.pdf', dpi=500)

    # show plot
    plt.show()

    return 0


# ~~~~~~~~~~~~~~~~~~ Results image ~~~~~~~~~~~~~~~~~~~
# Based on percent summary, and some additional image the function merge all the results into a single image
def results_img(image, label_image, percent_results, label_dictionary=parameters.get_label_info(), save_img=False):
    # Remove zero values from results
    # results_filter = {key: value for key, value in percent_results.items() if value != 0}
    results_filter = {key: value for key, value in percent_results.items() if value > 0.01}

    # Select only the relevant colors
    label_color = [label_dictionary[key][1] for key, value in results_filter.items()]

    fig = plt.figure()

    # Original Image
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title('Imagem Original')  # Nomeia cada subplot
    ax.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
    ax.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
    plt.imshow(image)

    # Label Image
    ax = fig.add_subplot(2, 2, 2)
    ax.set_title('Classes estimadas')  # Nomeia cada subplot
    ax.get_xaxis().set_visible(False)  # Esconde a legenda do eixo x
    ax.get_yaxis().set_visible(False)  # Esconde a legenda do eixo y
    plt.imshow(label_image)

    # Table data
    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("Resultados")
    percent = [round(100 * val, 1) for key, val in results_filter.items()]
    data_text = np.transpose(np.array((list(results_filter.keys()), percent)))
    col_label = ('Classes', "Percentual [%]")
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=data_text, colLabels=col_label, loc='center')
    table.auto_set_font_size(True)
    # table.set_fontsize(20)

    # Pie Chart
    ax = fig.add_subplot(2, 2, 4)
    ax.set_title("Cobertura vegetal")
    ax.axis('equal')

    names = results_filter.keys()
    values = results_filter.values()

    wedges, texts, texts = ax.pie(values, labels=names, colors=label_color, autopct='%1.1f%%')

    plt.setp(texts, size=8, weight="bold")

    if save_img:
        result_path = parameters.get_result_folder()
        plt.savefig(result_path + 'results.png', dpi=800)
        plt.savefig(result_path + 'results.pdf', dpi=500)

    # show plot
    plt.show()

    return 0

from reportlab.pdfgen import canvas

# ~~~~~~~~~~~~~~~~~~ pdf report ~~~~~~~~~~~~~~~~~~~
# Sample of a simple pdf report with some hardcoded parameters
def pdf_report(results_img_path=parameters.get_result_folder() + 'results.png'):
    print('Geração de pdf iniciada')
    nome_pdf = 'Relatório de análise'
    data_text = datetime.now().strftime('%d/%m/%Y')
    time_text = datetime.now().strftime('%H:%M:%S')
    gps_text = 'alt :119    long: 18.0;52.0;57.0    lat: 48.0;20.0;11.0'
    name_text = 'José da Silva'

    h = 119
    area, _, _ = get_coverage(h)
    area_text = '{0:1.1f} m²'.format(area)

    pdf = canvas.Canvas('{}.pdf'.format(nome_pdf))
    pdf.setTitle(nome_pdf)

    # Default sizes
    new_line = 20
    lat = 50
    par = 20
    line = 780

    logo = parameters.get_result_folder() + 'demeter_logo.jpg'
    size = 70
    pdf.drawImage(logo, par, line - 30, width=int(size * 1.5), height=size)

    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(150, line, 'Massa de forragem e cobertura do solo')
    line -= new_line
    line -= new_line
    line -= new_line
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(lat, line, 'Informações Gerais:')
    pdf.setFont("Helvetica", 12)
    line -= new_line
    pdf.drawString(lat + par, line, 'Data: ' + data_text)
    line -= new_line
    pdf.drawString(lat + par, line, 'Hora: ' + time_text)
    line -= new_line
    pdf.drawString(lat + par, line, 'GPS: ' + gps_text)
    line -= new_line
    pdf.drawString(lat + par, line, 'Área estimada: ' + area_text)
    line -= new_line
    pdf.drawString(lat + par, line, 'Operador: ' + name_text)
    line -= new_line
    line -= new_line

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(lat, line, 'Descrição:')
    line -= new_line
    pdf.setFont("Helvetica", 12)
    pdf.drawString(lat + par, line, 'Estimativa de cobertura vegetal através do sistema de captura Demeter.')
    line -= new_line
    pdf.drawString(lat + par, line, 'Autuando na região de campo grande, no Sitio do Picapau Amarelo. ')
    line -= new_line
    line -= new_line

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(lat, line, 'Resultados:')
    line -= new_line
    line -= new_line
    line -= new_line

    size = 420
    pdf.drawImage(results_img_path, 10, 50, width=int(size * 1.33), height=size)

    pdf.setFont("Helvetica", 8)
    pdf.drawString(10, 10, 'Relatório automático gerado pelo sistema Deméter versão 0.1')

    pdf.save()
    print('{}.pdf criado com sucesso!'.format(nome_pdf))

    return 0
