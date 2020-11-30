import numpy as np
import cv2
from PIL import Image
import matplotlib.pylab as plt
import copy
import pathlib
import time

starttime = time.time()

cap = cv2.VideoCapture('data/vid/video4.mp4')

file = open("out/results_video4.txt", "w+")

while(cap.isOpened()):
    red = 0
    ret, frame = cap.read()
    color = frame
    subtract = copy.copy(frame)

    for x in range(color.shape[0]):
        for y in range(color.shape[1]):
            if (color[x, y, 1] < 150 and color[x, y, 2] > 80 and color[x, y, 2] < 255):
                color[x, y, 1] = 255
                color[x, y, 2] = 255
                color[x, y, 0] = 255
            elif (color[x, y, 0] < 50):
                color[x, y, 1] = 255
                color[x, y, 2] = 255
                color[x, y, 0] = 255
            elif (color[x, y, 0] > 135 and color[x, y, 1] > 120 and color[x, y, 2] > 100):
                color[x, y, 1] = 0
                color[x, y, 2] = 0
                color[x, y, 0] = 0
            elif (color[x, y, 0] > 20 and color[x, y, 0] < 80 and color[x, y, 1] > 20 and color[x, y, 2] > 20):
                color[x, y, 2] = 255
                red += 1

    if (red > 2000):
        file.write(str(time.time()) + ": " + "eyes are open" + "\n")
    elif (red < 2000):
        file.write(str(time.time()) + ": " + "eyes are closed" + "\n")

    subtract_median = copy.copy(color)
    file.write("Number of detected pixels: " + str(red) + "\n")
    cv2.imshow("Hector", color)
    cv2.imshow("jojo", subtract)
    file.write("Time: " + str((time.time() - starttime)) + "\n")
    file.write(" " + "\n")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
