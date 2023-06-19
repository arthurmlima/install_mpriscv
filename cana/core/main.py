import cv2
from glob import glob
import os
import numpy as np
import time
import projection
import common
import pca
import hough
import tmd
import radon
from natsort import natsorted
#from demeter import sptools



# ****************    SET PATHS      *******************************************************

load_path = '/home/ubuntu/app-sav-gol/core/Input_Images/'
save_path = '/home/ubuntu/app-sav-gol/core/Output_Images/'
sinogram_path = '/home/ubuntu/app-sav-gol/core/Sinogram/'

# ****************    SET IMAGE NAMES      *************************************************
#img_names = glob(os.path.join(load_path, 'conf_086*.png'))  # article for discipline
img_names = glob(os.path.join(load_path, 'conf_*.png'))  # article for discipline
#img_names = glob(os.path.join(load_path, 'test.png'))  # article for discipline


# *************   PUBLIC VARIABLES  ***************************************************
i = 0
list_images = []
list_angles = []
time_process_imgs = []
print_on_terminal = False

# *************   HEURISTIC BASIC SETUP  **********************************************
algorithm = "RADON"    # Pick up one of heuristic available [PROJECTION, MIDDLE, HOUGH, PCA, RADON]
scan_resolution = 1    # Factor for angle step image rotation in degrees [1/scan_resolution]°
image_rezise    = 4    # image downsize factor [original/image_rezise]

# *************   START PROCESSING ***********************************************
print("================  START PROCESSING  ================")
for fn in img_names:
    base = os.path.basename(fn)
    file_name = (os.path.splitext(base)[0])
    list_images.append(file_name)
    if(print_on_terminal):
        print("************ NEW IMAGE", file_name, "****************************")
    start_time = time.time()
    img = cv2.imread(fn)  # load images
    [height, width, layer] = img.shape
    #print("height:", height, " width:", width)

    if(algorithm == "PROJECTION"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converts BGR to RGB
        stage = 1
    elif(algorithm == "MIDDLE"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converts BGR to RGB
        stage = 1
    elif(algorithm == "HOUGH"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converts BGR to RGB
        stage = 2
    elif(algorithm == "PCA"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converts BGR to RGB
        stage = 1
    elif(algorithm == "RADON"):
        stage = 1

    img = cv2.resize(img, (int(width/image_rezise), int(height/image_rezise)))
    [height, width, layer] = img.shape
    #print("height:", height, " width:", width)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~ 1° STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    if stage == 1:
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
        if algorithm == "PROJECTION":
            first_stage = projection.segmentation(img, 'EXCESS_GREEN', 'CROP', 1, 'false')
            stage = 2
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
        elif algorithm == "RADON":
            first_stage = radon.RadonPreProcessing(img)
            stage = 2

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  #
    # ~~~~~~~~~~~~ 2° STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  #
    if stage == 2:
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
        if algorithm == "PROJECTION":            

            # ----> input the desired filter
            #filter = "zeros"
            filter = "savitzky"

            # ----> input the desired coordinate system
            #coordinates = "0to180Right"
            coordinates = "-90to90Front"

            rotation = projection.main_projection(img, first_stage, scan_resolution, filter, coordinates, debug="false")
            list_angles.append(rotation)
            stage = "final"
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
        elif algorithm == "MIDDLE":
            final_stage = tmd.split(first_stage, save_path, 80, 1, 'Teste_', img)
            hough.hough_std(final_stage)
            stage = "final"

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
        elif algorithm == "PCA":
            final_stage = pca.main_pca(img, first_stage)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
        elif algorithm == "HOUGH":
            vows = 200
            final_stage = hough.find_lines(img, vows)
            stage = "final"

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
        elif algorithm == "RADON":
            rotation = radon.radonProcess(first_stage,scan_resolution ,sinogram_path,i,'false')
            if(print_on_terminal):
                print("Angle detected: ",rotation)
            list_angles.append(rotation)            
            stage = "final"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # ~~~~~~~~~~~~ FINAL STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if stage == "final":
        time_img = time.time() - start_time
        time_process_imgs.append(time_img)

        final_stage = common.draw_orientation(img,rotation)

        cv2.imwrite(save_path + file_name + '_%02i.png' % i, cv2.cvtColor(final_stage, cv2.COLOR_RGB2BGR))
        #sptools.img_show(final_stage, file_name)
    i += 1

#mean_time = sum(time_process_imgs) / len(time_process_imgs)
print("================  FINISH PROCESSING  ================")

# *************   END OF MAIN APPLICATION  ***********************************************************
# ****************************************************************************************************


# ----------------------------------------------#
#     (1)   PRINT ALL RESULTS ON TERMINAL       #
# ----------------------------------------------#
print("-------- Summary Results -----------------")
print(" N° images:", i)
print(" Algorithm:", algorithm)
print(" Scan Angle Resolution:", scan_resolution)
#print(" Processing time: %.3f seconds" % mean_time)
print("------------------------------------------")

# ----------------------------------------------#
#     (2)   SAVE MAIN RESULTS IN A CSV FILE     #
# ----------------------------------------------#
if (False):
    local_results = list(zip(list_images,list_angles))
    data_ordered = natsorted(local_results)
    np.savetxt('final_results.csv',data_ordered,fmt='%s',delimiter=',')

# ----------------------------------------------#
#     (3)   CALL RESULTS ANALYSES               #
# ----------------------------------------------#
if (False):
    os.chdir("../result_analyses")
    os.system('python main.py ' + algorithm)
