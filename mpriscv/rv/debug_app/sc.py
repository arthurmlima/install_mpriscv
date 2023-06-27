import serial
import io
import numpy as np
import matplotlib.pyplot as plt


image = [] 
ser = serial.Serial()
ser.baudrate = 115200
ser.port = '/dev/ttyUSB2'
ser.close()
ser.open()
sio = io.TextIOWrapper(io.BufferedRWPair(ser, ser))
for y in range(240):
    for x in range(240):
        arrd = ser.readline().decode("utf-8").strip()
        image.append(arrd)
    print(arrd)
image = np.array(image)
image = np.reshape(image,(240,240))
image = image.astype(np.uint8)
plt.imshow(image, cmap='gray')
plt.show()



