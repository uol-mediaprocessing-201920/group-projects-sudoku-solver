import cv2 as cv
from definitions import detection, transformation, extraction, recognition, ar

cv.namedWindow("Webcam Test")
vc = cv.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    exit(1)

while True:
    cv.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv.waitKey(20)
    if key == 27: # exit on ESC
        break
    if not rval:
        break

cv.destroyWindow("preview")