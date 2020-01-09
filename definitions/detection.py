import cv2 as cv
import pandas as pd
import numpy as np


def convert_to_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)


def threshold_image(image,
                    method=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                    block_size=0.1,
                    bias=32):
    # convert relative block size to absolute block size
    width, height = image.shape
    block_size = int(block_size * min(width, height))
    # if block_size is even, we must make it uneven (see def. of adaptiveThreshold)
    if block_size % 2 == 0:
        block_size += 1
    return cv.adaptiveThreshold(image, 255, method, cv.THRESH_BINARY, block_size, bias)

