import numpy as np
import cv2 as cv
import dlib
import argparse
import imutils
from imutils import face_utils


faceDetector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeDetector = cv.CascadeClassifier("haarcascade_eye.xml")

input = cv.imread("input.jpg")
gray = cv.cvtColor(input, cv.COLOR_BGR2GRAY)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

faces = faceDetector.detectMultiScale(gray, 1.3, 5)

rects = dlib.rectangles()
for (x, y, w, h) in faces:
    rects.append(dlib.rectangle(x, y, x+w, y+h))

for(i, rect) in enumerate(rects):
    shape = predictor(input, rect)
    shape = face_utils.shape_to_np(shape)

    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv.rectangle(input, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.putText(input, "Face #{}".format(i + 1), (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for(x, y) in shape:
        cv.circle(input, (x, y), 1, (0, 0, 255), -1)

    roi_gray = input[y:y + h, x:x + w]
    roi_color = input[y:y + h, x:x + w]

    eyes = eyeDetector.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

cv.imshow("Output", input)
cv.waitKey(0)

cv.destroyAllWindows()