import tensorflow
import numpy as np
from common import *

INPUT_SIZE = (28, 28)
INPUT_SHAPE = (*INPUT_SIZE, 1)


def load_model(path="./sudoku_recognition_model.h5"):
    return tensorflow.keras.models.load_model(path)


def preprocess(cell, input_format):
    # convert to grayscale
    cell = apply_opencv_color_conversion(cell, input_format, "GRAY")
    # invert:
    cell = 255 - cell
    # resize:
    cell = cv.resize(cell, INPUT_SIZE, cv.INTER_AREA)
    # normalize:
    cell = cell.astype("float32") / 255.0
    # reshape:
    cell = cell.reshape(INPUT_SHAPE)
    return cell


def predict_batch(batch, model):
    prediction = model.predict(batch)
    classes = np.argmax(prediction, axis=1)
    probabilities = np.max(prediction, axis=1)
    return classes, probabilities

