import sys
import os
import math
import cv2 as cv
import numpy as np
#import matplotlib.pyplot as plt
from demeter import sptools
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny
from skimage.draw import line
from skimage import data
from skimage import io
from skimage import util
from skimage.color import rgb2gray
from matplotlib import cm

# *****************************************************************************************
# *****************************************************************************************


def find_lines(img, vows):

    # # [edge_detection]
    dst = cv.Canny(img, 50, 200, None, 3)
    #sptools.img_show(dst, "hough cdst")

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2RGB)

    #  Standard Hough Line Transform
    #lines = cv.HoughLines(dst, 2, np.pi / 180, vows, None, 0, 0, -0.4*np.pi, 0.4*np.pi)
    lines = cv.HoughLinesP(dst, 1, np.pi / 180, 20, None, 5, 0)
    print(len(lines))

    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

            cv.line(cdst, pt1, pt2, (0, 255, 255), 3, cv.LINE_AA)

    # Show results
    return cdst

# *****************************************************************************************
# *****************************************************************************************

'''
This is the standart hough transform. The only parameter is the angle resolution.
'''

def hough_std(img):

    #img_gray_w = rgb2gray(img)
    img_gray = util.invert(img)
    print(img_gray.shape)

    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
    h, theta, d = hough_line(img_gray, theta=tested_angles)

    # Generating figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    ax = axes.ravel()

    ax[0].imshow(img_gray, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    # Generating figure 2
    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
              np.rad2deg(theta[-1] + angle_step),
              d[-1] + d_step, d[0] - d_step]

    ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    # Generating figure 3
    ax[2].imshow(img_gray, cmap=cm.gray)
    ax[2].set_ylim((img_gray.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d,10)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))

    plt.tight_layout()
    plt.show()

    return 0

# *****************************************************************************************
# *****************************************************************************************
'''
This is the probabilist hough transform. You can select line spacing and line quantities
threshold
'''

def hough_probab(img):

    image = util.invert(img)
    print(image.shape)

    edges = canny(image, 3)
    lines = probabilistic_hough_line(edges, threshold=4, line_length=80,
                                     line_gap=15)

    # Generating figure 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edges')

    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.show()

    return 0
