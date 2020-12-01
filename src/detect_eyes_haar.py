import cv2
import os
import time

starttime = time.time()


def detectBlink(file, output):
    first_read = True

    # Starting the video capture
    cap = cv2.VideoCapture(file)

    while(cap.isOpened()):
        ret, img = cap.read()

        if not ret:
            break

        # Scale and rotation
        scale_percent = 60
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        rezise = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        newImg = cv2.rotate(rezise, cv2.ROTATE_90_CLOCKWISE)

        # Coverting the recorded image to grayscale
        gray = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)
        # Applying filter to remove impurities
        gray = cv2.bilateralFilter(gray, 5, 1, 1)

        # Detecting the face for region of image to be fed to eye classifier
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
        if(len(faces) > 0):
            for (x, y, w, h) in faces:
                img = cv2.rectangle(newImg, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # roi_face is face which is input to eye classifier
                roi_face = gray[y:y+h, x:x+w]
                roi_face_clr = img[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(
                    roi_face, 2.6, 5, minSize=(50, 50))

                if(len(eyes) >= 2):
                    cv2.putText(newImg,
                                "Eyes open!", (70, 70),
                                cv2.FONT_HERSHEY_PLAIN, 2,
                                (255, 255, 255), 2)
                else:
                    output.write(
                        "Time: " + str((time.time() - starttime)) + " - Blink Detected" + "\n")
                    cv2.putText(newImg,
                                "Blink!", (70, 70),
                                cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 255), 2)

        else:
            cv2.putText(newImg,
                        "No face detected", (100, 100),
                        cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 255, 0), 2)

        # Controlling the algorithm with keys
        cv2.imshow('img', newImg)
        a = cv2.waitKey(1)
        if(a == ord('q')):
            break
        if(ret == False):
            break

    cap.release()
    cv2.destroyAllWindows()


face_cascade = cv2.CascadeClassifier(
    'data/cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    'data/cascades/haarcascade_eye.xml')

for file in os.listdir("data/test"):

    fileName = os.path.splitext(file)[0]
    outputPath = os.path.join("out/results2", fileName + "results.txt")
    output = open(outputPath, "w+")

    if file.endswith(".mp4"):
        path = os.path.join("data/test", file)
        detectBlink(path, output)
