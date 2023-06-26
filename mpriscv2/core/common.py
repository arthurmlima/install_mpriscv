# Criado por: Raphael P Ferreira
# Data: 21/06/2021

import cv2
import numpy as np
from demeter import sptools

# *****************************************************************************************
# *****************************************************************************************

def edge_detection(img):
    height, width = img.shape
    img_blk = np.zeros((height, width, 1), np.uint8)  # create an empty image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    for idx in hierarchy[idx][0]:
        frame = cv2.drawContours(img_blk, contours, idx, (128, 128, 128), 2, 8, hierarchy)
    thr1, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    sptools.img_show(frame, "edges")
    return frame


# *****************************************************************************************
# ********* DOESN'T MAKE SENSE USING THIS METRIC FOR THIS APPLICATION   *******************
# ********* WHY? EX.: ERROR OF 0.7° @ 1.4° DIRECTION GIVE 50% OF PRECISION   **************

def accuracy_calc(data_list):
    accur_vet = []
    for i in range(len(data_list)):
        observed = data_list[i][1]
        truth = data_list[i][2]
        diff = (observed - truth)
        percent_error = abs((diff/truth)*100)
        percent_accuracy = 100 - percent_error
        if abs(percent_accuracy) > 100:
            percent_accuracy = 0
        accur_vet.append(percent_accuracy)
        print("accuracy", data_list[i][0], ": ", percent_accuracy)

    return sum(accur_vet) / len(data_list)

# *****************************************************************************************
# *****************************************************************************************


def error_absolute(data_list):
    abs_vet = []
    for i in range(len(data_list)):
        observed = data_list[i][1]
        truth = data_list[i][2]
        diff = round(abs(observed - truth),3)
        abs_vet.append(diff)
        print(data_list[i][0], '@', data_list[i][1], "| ", diff)


    std_dev = round(np.std(abs_vet),3)
    print("---------------------------------")
    print("---> STD DEV", std_dev)

    #green_diamond = dict(markerfacecolor='g', marker='D')
    #fig1, ax1 = plt.subplots()
    #ax1.set_title('Absolute angle error')
    #ax1.boxplot(abs_vet, flierprops=green_diamond)
    #plt.show()

    return round(sum(abs_vet) / len(data_list),3)


# *****************************************************************************************
# *****************************************************************************************

def save_results_to_csv(data_list):
    arrayFig = []
    arrayAngleAlg = []
    #arrayAngleSpe = []
    #arraySBU = []
    for i in range(len(data_list)):
        arrayFig.append(data_list[i][0])
        arrayFig.append(data_list[i][0])
        #arrayAngleAlg.append(data_list[i][0][1])
        #arrayAngleAlg.append(data_list[i][0][1])
        #arrayAngleSpe.append(data_list[i][1])
        #arraySBU.append(data_list[i][2])


    #newList = list(zip(arrayFig,arrayAngleAlg,arrayAngleSpe,arraySBU))
    newList = list(zip(arrayFig,arrayAngleAlg))
    np.savetxt('final_results.csv',newList,fmt='%s',delimiter=',')


# *****************************************************************************************
# *****************************************************************************************


def draw_orientation(img, angle):
    angle_radian = angle * (np.pi/180)
    (height, width, layers) = img.shape
    center = (int(width/2), int(height/2))
    cv2.circle(img, center, 10, (255, 0, 0), 10)  # Draw a circle at the center of image
    hyp = (height/2) / np.cos(angle_radian)
    p = hyp * np.sin(angle_radian)
    if p > 10000000000:
        p = 0
    cv2.line(img, (int(width/2), int(height/2)), (int((width/2)+p), int(0)), (255, 0, 0), 10, cv2.LINE_AA)
    cv2.line(img, (int(width/2), int(height/2)), (int((width/2)-p), int(height)), (255, 0, 0), 10, cv2.LINE_AA)

    cv2.putText(img, str(angle), center, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 10)

    return img


# *****************************************************************************************
# *****************************************************************************************
'''
This draws a horizontal line at the maximum accumulation white pixel on the image
Which in turn represents the angle orientation of the crop row
OPEN CV
'''

def draw_horizontal_line(img, y_pos):

    color = (255, 255, 255)
    (height, width, z) = img.shape
    center = (int(width/2), int(height/2))
    print("shape",img.shape)
    print("center",center)
    cv2.line(img, (int(0), int(y_pos)), (int(width), int(y_pos)), color, 1)

    cv2.putText(img, str(y_pos), center, cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)

    return img

# *****************************************************************************************
# *****************************************************************************************
'''

This draws a horizontal line at the maximum accumulation white pixel on the image
Which in turn represents the angle orientation of the crop row
MATPLOTLIB

'''

def draw_horizontal_line_matplot(img, y_pos):

    color = (0, 0, 0, 1)

    height = img.shape[0]
    width = img.shape[1]
    center = (int(width/2), int(height/2))
    x = [int(0), int(width)]
    y = [int(y_pos), int(y_pos)]


    #plt.plot(x, y, color=color, linewidth=3)

    cv2.line(img, (int(0), int(y_pos)), (int(width), int(y_pos)), color, 2)
    #cv2.putText(img, str(y_pos), center, cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)

    return img

# *****************************************************************************************
# *****************************************************************************************
