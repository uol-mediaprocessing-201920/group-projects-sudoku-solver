import sys
sys.path.append("./definitions")  # hack? allows files like detection.py to find common.py

from definitions import detection, transformation, extraction, recognition, solving, solving_dlx, ar
from definitions.pipeline import Stage, Pipeline

import numpy as np
import cv2 as cv
import multiprocessing as mp
import threading

gui_queue = []
gui_lock = threading.Semaphore(1)


def show_images(window_name, images, padding_width=2, padding_color=127, scaling=None):
    frame = []
    for row in images:
        frame_row = []
        for col in row:
            image = col
            if len(col.shape) == 2:
                image = cv.cvtColor(col, cv.COLOR_GRAY2BGR)
            elif not (len(col.shape) == 3 and col.shape[-1] == 3):
                raise Exception("Unknown image format (must be either BGR or GRAY)!")
            p = padding_width
            if scaling is not None:
                h, w = image.shape[:2]
                h = int(h * scaling)
                w = int(w * scaling)
                image = cv.resize(image, (w, h))
            image = np.pad(image, ((p, p), (p, p), (0, 0)), constant_values=padding_color)
            frame_row.append(image)
        frame.append(frame_row)
    frame = [np.hstack(x) for x in frame]
    frame = np.vstack(frame)
    gui_lock.acquire()
    gui_queue.append(lambda: cv.imshow(window_name, frame))
    gui_lock.release()

def draw_centered_text(image, text, font_face=cv.FONT_HERSHEY_SIMPLEX, font_scale=1 / 32, thickness=2,
                       color=(0, 0, 255)):
    image_width, image_height = image.shape[:2]
    font_scale = min(image_width, image_height) * font_scale
    text = str(text)
    (text_width, text_height), _ = cv.getTextSize(text, font_face, font_scale, thickness)
    text_origin = int((image_width - text_width) / 2), \
                  int(image_height - (image_height - text_height) / 2)
    cv.putText(image, text, org=text_origin,
               fontFace=font_face,
               fontScale=font_scale,
               thickness=thickness,
               color=color)


DETECTION_WINDOW_NAME = "1. Detection"
TRANSFORM_WINDOW_NAME = "2. Transformation"
EXTRACTION_WINDOW_NAME = "3. Extraction"
RECOGNITION_WINDOW_NAME = "4. Recognition"
SOLVING_WINDOW_NAME = "5. Solving"
AR_WINDOW_NAME = "6. Artificial Reality"


class SudokuData():
    def __init__(self):
        self.input_image = None
        self.detected_contour = None
        self.transform_source = None
        self.transform_target = None
        self.transform_matrix = None
        self.transform_image = None
        self.extracted_cells = None
        self.recognized_digits = None
        self.solved_puzzle = None


class DetectionStage(Stage):
    def compute(self, data: SudokuData):
        grayscale_image = detection.convert_to_grayscale(data.input_image, input_format="BGR")
        threshold_image = detection.threshold_image(grayscale_image)
        contours = detection.find_contours(threshold_image)
        contours = [detection.approximate_contour(x) for x in contours]
        contours = detection.get_foursided_contours(contours)
        contours_image = detection.draw_contours(threshold_image, contours,
                                                 input_format="GRAY", output_format="BGR",
                                                 thickness=0.001, color=(0, 0, 255))
        data.detected_contour = detection.get_largest_contour(contours)
        contour_valid = detection.check_sudoku_contour(data.input_image, data.detected_contour)
        contour_color = (0, 255, 0) if contour_valid else (0, 0, 255)
        contour_image = detection.draw_contours(contours_image, [data.detected_contour], thickness=0.01, color=contour_color)
        show_images(DETECTION_WINDOW_NAME, [
            [data.input_image, grayscale_image],
            [threshold_image, contour_image]
        ], scaling=0.5)
        if contour_valid:
            return data


class TransformationStage(Stage):
    def compute(self, data: SudokuData):
        data.transform_source = transformation.sort_contour(data.detected_contour)
        side_length = transformation.get_side_length(data.transform_source)
        data.transform_target = transformation.get_target_contour(side_length)
        data.transform_matrix = transformation.get_transformation(data.transform_source, data.transform_target)
        data.transform_image = transformation.apply_transformation(data.input_image, data.transform_matrix, side_length)
        show_images(TRANSFORM_WINDOW_NAME, [[data.transform_image]])
        return data


class ExtractionStage(Stage):
    def compute(self, data: SudokuData):
        data.extracted_cells = extraction.extract_cells(data.transform_image)
        show_images(EXTRACTION_WINDOW_NAME, data.extracted_cells)
        return data


class RecognitionStage(Stage):
    def __init__(self):
        super().__init__()
        self.model = None

    def setup(self):
        print("Loading model...")
        self.model = recognition.load_model()
        print("Model loaded!")

    def compute(self, data: SudokuData):
        # preprocess grid of extracted cells for neural network
        preprocessed_cells = np.array([[recognition.preprocess(cell, "BGR") for cell in row] for row in data.extracted_cells])
        # reshape grid of preprocessed cells into linear array
        preprocessed_cells = preprocessed_cells.reshape(-1, *recognition.INPUT_SHAPE)
        # predict classes and corresponding probabilities for all preprocessed cells
        classes, probabilities = recognition.predict_batch(preprocessed_cells, self.model)
        # reshape classes output as grid
        data.recognized_digits = classes.reshape(9, 9)  # TODO: (9, 9) should not be hardcoded here

        recognized_cell_images = []
        for i_row in range(len(data.extracted_cells)):
            row_images = []
            for i_col in range(len(data.extracted_cells[i_row])):
                image = data.extracted_cells[i_row][i_col].copy()
                clazz = data.recognized_digits[i_row][i_col]
                draw_centered_text(image, clazz)
                row_images.append(image)
            recognized_cell_images.append(row_images)
        show_images(RECOGNITION_WINDOW_NAME, recognized_cell_images)

        return data


class SolvingStage(Stage):
    def compute(self, data: SudokuData):
        # data.solved_puzzle = data.recognized_digits.copy()
        # try:
        #     solved = solving.solve(data.solved_puzzle, timeout=5.0)
        #     if not solved:
        #         print("Could not find solution for Sudoku!")
        #         return None
        # except TimeoutError:
        #     print("Took too long to solve Sudoku!")
        #     return None

        try:
            data.solved_puzzle = solving_dlx.solve(data.recognized_digits)
            if data.solved_puzzle is False:
                print("Could not find solution for Sudoku!")
                return None
        except:
            print("Some unexpected error occurred while solving the Sudoku :(")
            return None

        solved_cell_images = []
        for i_row in range(len(data.extracted_cells)):
            row_images = []
            for i_col in range(len(data.extracted_cells[i_row])):
                image = data.extracted_cells[i_row][i_col].copy()
                digit = data.solved_puzzle[i_row][i_col]
                draw_centered_text(image, digit)
                row_images.append(image)
            solved_cell_images.append(row_images)
        show_images(SOLVING_WINDOW_NAME, solved_cell_images)
        return data


class ARStage(Stage):
    def compute(self, data: SudokuData):
        ar.draw_solution(data.transform_image, data.recognized_digits, data.solved_puzzle)
        # inverting the transformation matrix does not always work
        # a more robust solution is to use the target and destination points for calculation
        M = transformation.get_transformation(data.transform_target, data.transform_source)
        ar_image = ar.overlay_transformed_image(background_image=data.input_image,
                                                foreground_image=data.transform_image,
                                                transformation_matrix=M)
        show_images(AR_WINDOW_NAME, [[ar_image]])


cv.namedWindow(DETECTION_WINDOW_NAME)
cv.namedWindow(TRANSFORM_WINDOW_NAME)
cv.namedWindow(EXTRACTION_WINDOW_NAME)
cv.namedWindow(RECOGNITION_WINDOW_NAME)
cv.namedWindow(SOLVING_WINDOW_NAME)
cv.namedWindow(AR_WINDOW_NAME)
vc = cv.VideoCapture(0)

if not vc.isOpened():
    raise Exception("Cannot open video capturing device!")

pipeline = Pipeline([DetectionStage(),
                     TransformationStage(),
                     ExtractionStage(),
                     RecognitionStage(),
                     SolvingStage(),
                     ARStage()])

while True:
    key = cv.waitKey(20)
    if key == 27:  # exit on ESC
        print("ESC pressed!")
        break

    rval, input_image = vc.read()
    if not rval:  # exit if unable to capture
        print("Unable to capture!")
        break

    data = SudokuData()
    #data.input_image = cv.resize(input_image, (512, 512), interpolation=cv.INTER_AREA).copy()
    data.input_image = input_image
    pipeline.feed(data)

    gui_lock.acquire()
    while len(gui_queue) > 0:
        action = gui_queue.pop(0)
        action()
    gui_lock.release()

exit(0)
