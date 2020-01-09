from definitions import detection, transformation, extraction, recognition, solving, ar

import numpy as np
import cv2 as cv


def show_images(window_name, images, padding_width=2, padding_color=127):
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
            image = np.pad(image, ((p, p), (p, p), (0, 0)), constant_values=padding_color)
            frame_row.append(image)
        frame.append(frame_row)
    frame = [np.hstack(x) for x in frame]
    frame = np.vstack(frame)
    cv.imshow(window_name, frame)


def draw_centered_text(image, text, font_face=cv.FONT_HERSHEY_SIMPLEX, font_scale=1/32, thickness=2, color=(0, 0, 255)):
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

cv.namedWindow(DETECTION_WINDOW_NAME)
cv.namedWindow(TRANSFORM_WINDOW_NAME)
cv.namedWindow(EXTRACTION_WINDOW_NAME)
cv.namedWindow(RECOGNITION_WINDOW_NAME)
cv.namedWindow(SOLVING_WINDOW_NAME)
cv.namedWindow(AR_WINDOW_NAME)
vc = cv.VideoCapture(0)

if not vc.isOpened():
    raise Exception("Cannot open video capturing device!")

model = recognition.load_model()

while True:
    key = cv.waitKey(20)
    if key == 27:  # exit on ESC
        print("ESC pressed!")
        break

    rval, input_image = vc.read()
    if not rval:  # exit if unable to capture
        print("Unable to capture!")
        break

    noise_image = np.random.rand(*input_image.shape)
    noise_image *= 255
    noise_image = noise_image.astype("uint8")

    grayscale_image = detection.convert_to_grayscale(input_image, input_format="BGR")
    threshold_image = detection.threshold_image(grayscale_image)
    contours = detection.find_contours(threshold_image)
    contours = [detection.approximate_contour(x) for x in contours]
    contours = detection.get_foursided_contours(contours)
    contours_image = detection.draw_contours(threshold_image, contours,
                                             input_format="GRAY", output_format="BGR",
                                             thickness=0.001, color=(0, 0, 255))
    contour = detection.get_largest_contour(contours)
    contour_valid = detection.check_sudoku_contour(input_image, contour)

    if contour_valid:
        contour_image = detection.draw_contours(contours_image, [contour],
                                                thickness=0.01, color=(0, 255, 0))

        show_images(DETECTION_WINDOW_NAME, [
            [input_image, grayscale_image],
            [threshold_image, contour_image]
        ])

        transform_source = transformation.sort_contour(contour)
        side_length = transformation.get_side_length(transform_source)
        transform_target = transformation.get_target_contour(side_length)
        transform_matrix = transformation.get_transformation(transform_source, transform_target)
        transform_image = transformation.apply_transformation(input_image, transform_matrix, side_length)
        show_images(TRANSFORM_WINDOW_NAME, [[transform_image]])

        cells = extraction.extract_cells(transform_image)
        show_images(EXTRACTION_WINDOW_NAME, cells)

        # preprocess grid of extracted cells for neural network
        preprocessed_cells = np.array([[recognition.preprocess(cell, "BGR") for cell in row] for row in cells])
        # reshape grid of preprocessed cells into linear array
        preprocessed_cells = preprocessed_cells.reshape(-1, *recognition.INPUT_SHAPE)
        # predict classes and corresponding probabilities for all preprocessed cells
        classes, probabilities = recognition.predict_batch(preprocessed_cells, model)
        # reshape classes output as grid
        classes = classes.reshape(9, 9)  # TODO: (9, 9) should not be hardcoded here
        # reshape probabilities output as grid
        probabilities = probabilities.reshape(9, 9)  # TODO: (9, 9) should not be hardcoded here

        recognized_cell_images = []
        for i_row in range(len(cells)):
            row_images = []
            for i_col in range(len(cells[i_row])):
                image = cells[i_row][i_col].copy()
                clazz = classes[i_row][i_col]
                prob = probabilities[i_row][i_col]
                draw_centered_text(image, clazz)
                row_images.append(image)
            recognized_cell_images.append(row_images)
        show_images(RECOGNITION_WINDOW_NAME, recognized_cell_images)

        unsolved_grid = classes.copy()
        solved_grid = unsolved_grid.copy()
        solving.solve(solved_grid)

        solved_cell_images = []
        for i_row in range(len(cells)):
            row_images = []
            for i_col in range(len(cells[i_row])):
                image = cells[i_row][i_col].copy()
                digit = solved_grid[i_row][i_col]
                draw_centered_text(image, digit)
                row_images.append(image)
            solved_cell_images.append(row_images)
        show_images(SOLVING_WINDOW_NAME, solved_cell_images)

        ar.draw_solution(transform_image, unsolved_grid, solved_grid)
        ar_image = ar.overlay_transformed_image(input_image, transform_image, np.linalg.inv(transform_matrix))
        show_images(AR_WINDOW_NAME, [[ar_image]])
    else:
        contour_image = detection.draw_contours(contours_image, [contour], thickness=0.01, color=(0, 0, 255))
        # always update detection window even if we did not find the Sudoku
        show_images(DETECTION_WINDOW_NAME, [
            [input_image, grayscale_image],
            [threshold_image, contour_image]
        ])

exit(0)
