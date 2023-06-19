# This Python file uses the following encoding: utf-8
import sys
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy.stats as stats

image_original_lenght = 2716
image_original_height = 1524

image_quarter_lenght = image_original_lenght/4
image_quarter_height = image_original_height/4

forward_speed_Km_h = 6
forward_speed_m_s = forward_speed_Km_h/3.6

FOV_degree = 70
FOV_rad = (70 * np.pi)/180
overlap = 0.8
uav_height =  np.arange(0, 61, 5)
#print("----> uav_height:", uav_height)

# **************** IMAGE SIZE DIMENSIONS  ******************************************

halfXSideImage = np.round(uav_height * np.tan(FOV_rad/2),3)
Xside = np.round(2*halfXSideImage,3)
Yside = np.round((image_original_height/image_original_lenght)*Xside,3)
#print("----> Yside:", Yside)

plt.figure(1)
plt.title('Flight height vs. Side dimensions')
plt.grid()
plt.xlabel('Flight height [m]')
plt.ylabel('Distance [m]')
plt.plot(uav_height, Xside)
plt.plot(uav_height, Yside)
plt.scatter(uav_height, Xside,label='X-side')
plt.scatter(uav_height, Yside, label='Y-side')
plt.annotate("(%i, %0.2f)" % (uav_height[5],Yside[5]),
            xy=(uav_height[5], Yside[5]),
            xycoords='data',
            xytext=(20, 5),
            textcoords='data',
            arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
plt.annotate("(%i, %0.2f)" % (uav_height[3],Xside[3]),
            xy=(uav_height[3], Xside[3]),
            xycoords='data',
            xytext=(10, 35),
            textcoords='data',
            arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
plt.legend()
plt.savefig('side_dim.png')

# ********************* PIXEL RESOLUTION  **************************************************

resolution_horizontal_cm = np.round(100*(uav_height * np.tan(FOV_rad/2))/(image_quarter_lenght/2), 3)
resolution_vertical_cm = np.round(100*(uav_height * np.tan(FOV_rad/2))/(image_quarter_height/2), 3)

#print("----> resolution_horizontal_cm:", resolution_horizontal_cm)
#print("----> resolution_vertical_cm:", resolution_vertical_cm)

plt.figure(2)
plt.title('Flight height vs. Pixel resolution')
plt.grid()
plt.xlabel('Flight height [m]')
plt.ylabel('Resolution size [cm]')
plt.plot(uav_height, resolution_horizontal_cm, label = "X resolution")
plt.plot(uav_height, resolution_vertical_cm, label = "Y resolution")
plt.scatter(uav_height, resolution_horizontal_cm)
plt.scatter(uav_height, resolution_vertical_cm)
plt.annotate("(%i, %0.2f)" % (uav_height[11],resolution_horizontal_cm[11]),
            xy=(uav_height[11], resolution_horizontal_cm[11]),
            xycoords='data',
            xytext=(50, 15),
            textcoords='data',
            arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
plt.legend()
plt.savefig('pixel_resol.png')


# **************** TIME CONSTRAINTS *******************************************************

plt.figure(3)
plt.title('UAV height vs. Frame processing vs Overlap')
plt.grid()
plt.xlabel('UAV height [m]')
plt.ylabel('Maximum time processing [s]')
for over in range(80,100,5):
    over_a = over/100
    frame_processing = np.round(((1-over_a) * Yside)/forward_speed_m_s, 3)
    #print("----> frame_processing:", frame_processing)
    plt.plot(uav_height, frame_processing, label = "Overlap = %0.2f " %over_a )
    plt.scatter(uav_height, frame_processing)
    if over == 80:
        plt.annotate("(%i, %0.2f)" % (uav_height[10],frame_processing[10]),
                    xy=(uav_height[10], frame_processing[10]),
                    xycoords='data',
                    xytext=(30, 5),
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

plt.legend()
plt.savefig('fps_req.png')




# ********************* PRECISION EFFECTS  **************************************************

plt.figure(4)
plt.title('Time processing  vs. Lateral displacement vs. Mean Absolute Error (MAE)')
plt.grid()
plt.xlabel('Real time processing [s]')
plt.ylabel('Lateral displacement [cm]')

time_process = np.arange(1,5.4,0.2)
new_cover_vertical = time_process * forward_speed_m_s
#print(new_cover_vertical)
elem_5 = 12
elem_6 = 6

for mae in range(1,10,1):
    mae_a = mae/2
    lateral = 100 * np.round((np.tan((mae_a * np.pi)/180) * new_cover_vertical),4)
    plt.plot(time_process, lateral, label = "MAE = %0.2f°" % mae_a)
    plt.scatter(time_process, lateral)
    if mae_a == 2:
        plt.annotate("(%0.2f, %0.2f)" % (time_process[elem_5],lateral[elem_5]),
                    xy=(time_process[elem_5], lateral[elem_5]),
                    xycoords='data',
                    xytext=(time_process[elem_5], 0),
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
        plt.annotate("(%0.2f, %0.2f)" % (time_process[elem_6],lateral[elem_6]),
                    xy=(time_process[elem_6], lateral[elem_6]),
                    xycoords='data',
                    xytext=(time_process[elem_6], 0),
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
plt.legend()
plt.savefig('mae_effect.png')

# ********************* COMBINATION GAUSSIAN **************************************************
plt.figure(5)
plt.title('Conf_002 - Specialists Angles Fusion')

mu_spec_0 = 36.55
mu_spec_1 = 39.1
mu_spec_2 = 38.15
mu_final = 38.531
std_spec_0 = 1.92
std_spec_1 = 0.96
std_spec_2 = 2.19
std_final = 0.799

xfinal = np.linspace(mu_final - 3*std_final, mu_final + 3*std_final, 100)
x1 = np.linspace(mu_spec_0 - 3*std_spec_0, mu_spec_0 + 3*std_spec_0, 100)
x2 = np.linspace(mu_spec_1 - 3*std_spec_1, mu_spec_1 + 3*std_spec_1, 100)
x3 = np.linspace(mu_spec_2 - 3*std_spec_2, mu_spec_2 + 3*std_spec_2, 100)

plt.plot(x1, stats.norm.pdf(x1, mu_spec_0, std_spec_0),label = "spec_0")
plt.plot(x2, stats.norm.pdf(x2, mu_spec_1, std_spec_1),label = "spec_1")
plt.plot(x3, stats.norm.pdf(x3, mu_spec_2, std_spec_2),label = "spec_2")
plt.plot(xfinal, stats.norm.pdf(xfinal, mu_final, std_final),label = "Final",linewidth = "3")

plt.xlabel('Orientation line angle [°]')
plt.ylabel('Probability')
plt.grid()
plt.legend()
plt.savefig('gaussian.png')

# ********************* PLOTS **************************************************

plt.show()
