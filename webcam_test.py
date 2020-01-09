import cv2 as cv

WINDOW_NAME = "Webcam Test"

cv.namedWindow(WINDOW_NAME)
vc = cv.VideoCapture(0)

if not vc.isOpened(): # try to get the first frame
    exit(1)

while True:
    key = cv.waitKey(20)
    if key == 27:  # exit on ESC
        print("ESC pressed!")
        break

    rval, frame = vc.read()
    if not rval:  # exit if unable to capture
        print("Unable to capture!")
        break

    cv.imshow(WINDOW_NAME, frame)

cv.destroyWindow(WINDOW_NAME)
