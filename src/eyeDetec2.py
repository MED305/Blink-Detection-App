import numpy as np
import cv2

import pandas as pd
from skimage import io
from PIL import Image
import matplotlib.pylab as plt
import copy
import pathlib


cap = cv2.VideoCapture('data/vid/heccevidCropped.mov')

while(cap.isOpened()):
    red=0
    ret , frame = cap.read()
    color = frame
    subtract = copy.copy(frame)
    for x in range(color.shape[0]):
        for y in range(color.shape[1]):
            if (color[x,y,1] < 150 and color[x,y,2] > 80 and color[x,y,2] < 255):
                color[x,y,1] = 255
                color[x,y,2] = 255
                color[x,y,0] = 255
            elif (color[x,y,0] < 50):
                color[x,y,1] = 255
                color[x,y,2] = 255
                color[x,y,0] = 255
            elif (color[x,y,0] > 135 and color[x,y,1] > 120 and color[x,y,2] > 100):
                color[x,y,1] = 0
                color[x,y,2] = 0
                color[x,y,0] = 0
            elif (color[x,y,0] > 20 and color[x,y,0] < 80 and color[x,y,1] > 20 and color[x,y,2] > 20 ):
                color[x,y,2] = 255
                red+=1
    if (red > 2000): 
        print("eyes are open")
    elif (red < 2000):
        print("eyes are closed")

    subtract_median = copy.copy(color)
    print(red)

    cv2.imshow("Hector",color)
    cv2.imshow("jojo",subtract)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()