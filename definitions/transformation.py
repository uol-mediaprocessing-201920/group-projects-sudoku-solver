import cv2 as cv
import numpy as np

def sort_contour(contour):
    corners = np.reshape(contour, (4, 2))
    indices = np.zeros((4,), dtype="int")
    sums = np.sum(corners, axis=1)
    # [0]: top left corner (smallest sum x + y)
    indices[0] = np.argmin(sums)
    # [2]: bottom right corner (largest sum x + y)
    indices[2] = np.argmax(sums)
    diffs = np.diff(corners, axis=1)
    # [1]: bottom left corner (largest diff. y - x)
    indices[1] = np.argmax(diffs)
    # [3]: top right corner (smallest diff. y - x)
    indices[3] = np.argmin(diffs)
    return corners[indices]


def get_side_length(contour):
    return int(cv.arcLength(contour, closed=True) / 4)


def get_target_contour(side_length):
    a = [0, 0]
    b = [0, side_length]
    c = [side_length, side_length]
    d = [side_length, 0]
    return np.array([a, b, c, d])


def get_transformation(source, destination):
    source = source.astype("float32")
    destination = destination.astype("float32")
    return cv.getPerspectiveTransform(source, destination)


def apply_transformation(image, transformation, side_length):
    return cv.warpPerspective(image, transformation, (side_length, side_length))
