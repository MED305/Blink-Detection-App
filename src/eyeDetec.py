import numpy as np
import cv2
import pandas as pd
from skimage import io
from PIL import Image
import matplotlib.pylab as plt
import copy
import pathlib


cap = cv2.VideoCapture('video1cropped.mov')

while(cap.isOpened()):
    ret, frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    low_blue = np.array([70, 60, 20])
    high_blue = np.array([150, 255, 255])
    blue_mask = cv2.inRange(color, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
    print()

    cv2.imshow("HSV", blue)
    cv2.imshow("Cutout", color)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
