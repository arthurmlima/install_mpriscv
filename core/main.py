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
import mpcall
from natsort import natsorted
#from demeter import sptools

# ****************    SET PATHS      *******************************************************

load_path = '../images/output_images/'
save_path = '../images/output_images/'
sinogram_path = '../images/output_images/'

# ****************    SET IMAGE NAMES      *************************************************
img_names = glob(os.path.join('../images/conf_*.hpp'))  # article for discipline


# *************   PUBLIC VARIABLES  ***************************************************
i = 0
list_images = []
list_angles = []
time_process_imgs = []
print_on_terminal = False

iname = []
t0 = []
t1 = []
t2 = []
t3 = []
t4 = []
t5 = []
tc = []

# *************   HEURISTIC BASIC SETUP  **********************************************
algorithm = "RADON"    # Pick up one of heuristic available [PROJECTION, MIDDLE, HOUGH, PCA, RADON]
scan_resolution = 1    # Factor for angle step image rotation in degrees [1/scan_resolution]°
image_rezise    = 1    # image downsize factor [original/image_rezise]
ig_index = 0            # image index
# *************   START PROCESSING ***********************************************
print("================  START PROCESSING  ================")
for fn in img_names:
    print(fn)
    base = os.path.basename(fn)
    file_name = (os.path.splitext(base)[0])
    st0 = 0
    st1 = 0
    st2 = 0
    st3 = 0
    st4 = 0
    st5 = 0
    stc = 0
    start_time = time.time()
    img ,st0, st1, st2, st3, st4, st5  = mpcall.mpriscv(ig_index)
    iname.append(file_name)
    t0.append(str(st0.value))    
    t1.append(str(st1.value))
    t2.append(str(st2.value))
    t3.append(str(st3.value))
    t4.append(str(st4.value))
    t5.append(str(st5.value))

    ig_index += 1
    [height, width, layer] = img.shape

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

        cv2.imwrite(save_path + '%02i.png' % i, cv2.cvtColor(final_stage, cv2.COLOR_RGB2BGR))
        #sptools.img_show(final_stage, file_name)
    i += 1

#mean_time = sum(time_process_imgs) / len(time_process_imgs)
print("================  FINISH PROCESSING  ================")

# *************   END OF MAIN APPLICATION  ***********************************************************
# ****************************************************************************************************
local_results = list(zip(iname, t0, t1, t2, t3,t4, time_process_imgs))
np.savetxt('final_results.csv',local_results,fmt='%s',delimiter=',')

import csv

# Specify the filename for the CSV file
filename = "data.csv"

# Open the CSV file in write mode
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write the header row
    header = ["Image Name", "T0", "T1", "T2", "T3", "T4","Tappl"]
    writer.writerow(header)

    # Write the data rows
    for i in range(len(iname)):
        row = [iname[i], t0[i], t1[i], t2[i], t3[i],t4[i], time_process_imgs[i]]
        writer.writerow(row)

print("Data written to", filename)

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
