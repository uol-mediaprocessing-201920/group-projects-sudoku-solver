import cv2 as cv
import numpy as np
from common import *


def convert_to_grayscale(image, input_format):
    return apply_opencv_color_conversion(image, input_format, "GRAY")


def threshold_image(image, input_format=None,
                    method=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                    block_size=0.1,
                    bias=32):
    # make sure image is grayscale
    image = apply_opencv_color_conversion(image, input_format, "GRAY")
    # convert relative block size to absolute block size
    width, height = image.shape
    block_size = int(block_size * min(width, height))
    # if block_size is even, we must make it uneven (see def. of adaptiveThreshold)
    if block_size % 2 == 0:
        block_size += 1
    return cv.adaptiveThreshold(image, 255, method, cv.THRESH_BINARY, block_size, bias)


def find_contours(image):
    # image must be inverted, because findContours treats white as foreground and black as background
    # if we don't invert the image, the image border will be detected as a contour
    image = cv.bitwise_not(image)
    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours(image, contours, input_format=None, output_format=None, thickness=0.001, color=(0, 0, 255)):
    # make sure image has three color channels
    image = apply_opencv_color_conversion(image, input_format, output_format, always_copy=True)
    # calculate absolute thickness of contours
    height, width = image.shape[:2]
    thickness = (height + width) / 2 * thickness
    # draw contours using OpenCV
    if not any(x is None for x in contours):
        cv.drawContours(image, contours, -1, color, int(thickness))
    return image


def approximate_contour(contour, precision=0.1):
    # convert relative precision to absolute precision (epsilon)
    _, _, width, height = cv.boundingRect(contour)
    epsilon = (width + height) / 2 * precision
    return cv.approxPolyDP(contour, epsilon, closed=True)


def get_foursided_contours(contours):
    candidates = []
    for contour in contours:
        if len(contour) != 4:
            continue  # contour has more than four sides, so it cannot be a rectangle
        candidates.append(contour)
    return candidates


def get_largest_contour(contours):
    lengths = [cv.contourArea(x) for x in contours]
    if len(lengths) == 0:
        return None
    else:
        longest = np.argmax(lengths)
        return contours[longest]


def check_sudoku_contour(input_image, contour, threshold=0.1):
    if contour is None:
        return False
    width, height = input_image.shape[:2]
    total_image_area = width * height
    contour_area = cv.contourArea(contour)
    return contour_area / total_image_area > threshold
