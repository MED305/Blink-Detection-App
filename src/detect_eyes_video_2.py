import numpy as np
import cv2 as cv

capture = cv.VideoCapture('data/vid/video2.mp4')

# Get user supplied values
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv.CascadeClassifier("src/haarcascade_frontalface_default.xml")

while(capture.isOpened()):
    label = 0
    ret, cap = capture.read()

    scale_percent = 60
    width = int(cap.shape[1] * scale_percent / 100)
    height = int(cap.shape[0] * scale_percent / 100)
    dim = (width, height)

    rezise = cv.resize(cap, dim, interpolation=cv.INTER_AREA)

    imageRezise = cv.rotate(rezise, cv.ROTATE_90_CLOCKWISE)

    gray = cv.cvtColor(imageRezise, cv.COLOR_BGR2GRAY)
    image = cv.cvtColor(imageRezise, cv.COLOR_BGR2HSV)

    height, width, channels = cap.shape

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        for i in range(h):
            for j in range(w):
                if (image[i + y, j + x, 0] < 300 and image[i + y, j + x, 0] > 260 and image[i + y, j + x, 1] < 20 and image[i + y, j + x, 2] > 60):
                    image[i + y, j + x, :] = 255

        newImage = cv.cvtColor(image, cv.COLOR_HSV2BGR)
        cv.rectangle(newImage, (x, y), (x+w, y+h), (0, 255, 0), 2)

    print("Found {0} faces!".format(len(faces)))

    cv.imshow("Hector", newImage)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
