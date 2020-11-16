import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pylab as plt
import copy
import pathlib

image = cv.imread("data/img/test_image.jpg")
imageIN = cv.cvtColor(image, cv.COLOR_BGR2RGB)

eye = 0

for y in range(imageIN.shape[0]):
    for x in range(imageIN.shape[1]):
        for c in range(3):

            white = (imageIN[y, x, 0] + imageIN[y, x, 1] +
                     imageIN[y, x, 2]) / 3

            if(imageIN[y, x, 0] + 5 > white and image[y, x, 0] - 5 < white):  # finder øjne
                if(imageIN[y, x, 1] + 5 > white and image[y, x, 1] - 5 < white):
                    if(imageIN[y, x, 2] + 5 > white and image[y, x, 2] - 5 < white):

                        imageIN[y, x, 0] = 255
                        eye += 1

            if(imageIN[y, x, 2] > white + 150):
                imageIN[y, x, 1] = 0
                imageIN[y, x, 0] = 0

if(eye < 1000):
    print("The eyes are closed!")
if(eye > 1000 and eye < 10000):
    print("eyes is open!")
if(eye > 10000):
    print("Something might have went wrong")


cv.imshow(imageIN)
