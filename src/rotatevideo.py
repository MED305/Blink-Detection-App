import cv2
import os


def rotate(file, outputPath):
    cap = cv2.VideoCapture(file)

    out = cv2.VideoWriter(outputPath, 0x7634706d, 20.0, (640, 480))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            imageWrite = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            out.write(imageWrite)
        else:
            break

    # Define the codec and create VideoWriter object

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


for file in os.listdir("data/test"):

    fileName = os.path.splitext(file)[0]
    outputPath = os.path.join("data/rotatedvideos", fileName + ".mp4")

    if file.endswith(".mp4"):
        path = os.path.join("data/test", file)
        rotate(path, outputPath)
