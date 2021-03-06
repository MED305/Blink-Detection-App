# Based on Eye blink detection with OpenCV, Python, and dlib by Adrian Rosebrock at
# https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/

from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import cv2
import time
import dlib
import os
import keyboard


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear


def detectBlink():
    starttime = time.time()
    frame = 0
    faceFrames = 0
    count = 0
    blinks = 0

    cap = cv2.VideoCapture(0)
    take_picture = False
    while(cap.isOpened()):
        ret, img = cap.read()

        if not ret:
            break

        scale_percent = 60
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        newImg = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # Uncomment to rotate image
        #newImg = cv2.rotate(newImg, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)

        faces = detector(gray, 0)

        for face in faces:
            faceFrames = faceFrames + 1
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(newImg, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(newImg, [rightEyeHull], -1, (0, 255, 0), 1)

            if keyboard.is_pressed('c') or take_picture == True:
                take_picture = True
                if ear < EYE_AR_THRESH:
                    cv2.waitKey(400)
                    break

                else:
                    cv2.imwrite(filepath + str(frame) + ".bmp", newImg)
                    take_picture = False

                cv2.putText(newImg, "Blinks: {}".format(blinks),
                            (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (50, 50, 255), 2)
                cv2.putText(newImg, "EAR: {:.2f}".format(
                    ear), (300, 70), cv2.FONT_HERSHEY_PLAIN, 3, (100, 200, 100), 2)

        cv2.imshow("Frame", newImg)
        key = cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


EYE_AR_THRESH = 0.3


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "data/face_predictor/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

filepath = "out/blinkshots/"
detectBlink()
