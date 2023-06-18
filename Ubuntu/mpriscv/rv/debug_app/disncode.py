import re
import sys
import numpy as np
import matplotlib.pyplot as plt

words=[]
pixels = []
   
with open(sys.argv[1],'r') as f:
    for line in f:        
            words = line.strip().split(",")
            pixels.append(words[0])
pixels = np.array(pixels)
pixels = np.reshape(pixels,(240,240))
pixels = pixels.astype(np.uint8)
plt.imshow(pixels, cmap='gray')
plt.show()

