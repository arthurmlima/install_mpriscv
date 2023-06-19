# Criado por: Raphael P Ferreira
# Data: 21/06/2021

from sklearn.decomposition import PCA
import numpy as np
import cv2
#import matplotlib.pyplot as plt
from demeter import sptools


# *****************************************************************************************
# *****************************************************************************************


def main_pca(src, bw, preprocess="true"):
    img_blk = np.zeros((src.shape[0], src.shape[1], src.shape[2]), np.uint8)  # create an empty image
    img_blk.fill(255)  # fills white pixels the image created

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~ If source image needs preprocessing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if preprocess == "false":
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) # Convert image to grayscale
        th, bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        plt.figure(1)
        plt.imshow(bw, cmap='gray')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~ Finding, filtering, get orientation for all contours ~~~~~~~~~~~~~
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    areas = []
    areas_angle = []
    hypotenuse_vector = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)  # Calculate the area of each contour
        # Ignore contours that are too small or too large
        if area < 1e3 or 1e4 < area:
            continue
        areas.append(area)
        cv2.drawContours(img_blk, contours, i, (0, 0, 255), 2)   # Draw each contour only for visualisation purposes
        angle, hypotenuse = getOrientation(c, img_blk)  # Find the orientation of each shape
        angle = (angle - (np.pi/2))*(180/np.pi)
        areas_angle.append(angle)
        hypotenuse_vector.append(hypotenuse)

    plt.figure(2)
    plt.subplot(131)
    plt.imshow(img_blk, cmap='gray')
    plt.subplot(132)
    plt.title("Angle orientation histogram")
    plt.hist(areas_angle, 40)  # arguments are passed to np.histogram
    plt.subplot(133)
    plt.xlabel("Angle orientation")
    plt.ylabel("Module vector")
    plt.stem(areas_angle, hypotenuse_vector, 'ro')
    plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Calculate the direction of the largest connected domain
    # ind = np.argmax(areas)

    return src

# *****************************************************************************************
# *****************************************************************************************


def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # mean, eigenvectors, eigenvalues = cv.PCACompute(data_pts, mean, 2) #image, mean=None, maxComponents=10
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv2.circle(img, cntr, 3, (255, 0, 255), 2)  # Draw a circle at the center of PCA
    p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    error_minimum = drawAxis(img, cntr, p1, (0, 0, 0), 1)      # black
    drawAxis(img, cntr, p2, (255, 255, 0), 1)  # yellow
    angle = np.arctan((eigenvectors[0, 1])/(eigenvectors[0, 0]))  # orientation in radians #PCA first dimension angle
    return [angle, error_minimum]

# *****************************************************************************************
# *****************************************************************************************


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    angle = np.arctan((p[1] - q[1])/(p[0] - q[0]))  # angle in radians
    hypotenuse = np.sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * np.cos(angle)
    q[1] = p[1] - scale * hypotenuse * np.sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * np.cos(angle + np.pi / 4)
    p[1] = q[1] + 9 * np.sin(angle + np.pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)
    p[0] = q[0] + 9 * np.cos(angle - np.pi / 4)
    p[1] = q[1] + 9 * np.sin(angle - np.pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)
    return hypotenuse

# *****************************************************************************************
# *****************************************************************************************


def rotate_img(img, angle):
    """ center rotates the image, the input angle is radians """
    angle_o = (angle - np.pi/2)*180/np.pi
    height = img.shape[0]  # original image height
    width = img.shape[1]  # original image width
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle_o, 1)  # Rotate the image by angle
    heightNew = int(width * np.fabs(np.sin(angle)) + height * np.fabs(np.cos(angle)))
    widthNew = int(height * np.fabs(np.sin(angle)) + width * np.fabs(np.cos(angle)))

    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation
