import cv2 as cv


def draw_digit(image, digit,
               i_row, i_col,
               n_rows=9, n_cols=9,
               font_face=cv.FONT_HERSHEY_SIMPLEX,
               font_scale=1 / 32,
               thickness=0.01,
               color=(0, 0, 0)):
    image_height, image_width = image.shape[:2]
    cell_width = int(image_width / n_rows)
    cell_height = int(image_height / n_cols)
    cell_left = int((i_col / n_cols) * image_width)
    cell_top = int((i_row / n_rows) * image_height)
    cell_right = cell_left + cell_width
    cell_bottom = cell_top + cell_height
    font_scale = min(cell_width, cell_height) * font_scale
    thickness = int(image_height * thickness)
    digit_text = str(digit)
    (digit_width, digit_height), _ = cv.getTextSize(digit_text, font_face, font_scale, thickness)
    digit_origin = cell_left + int((cell_width - digit_width) / 2), \
                   cell_bottom - int((cell_height - digit_height) / 2)
    cv.putText(image, digit_text, digit_origin, font_face, font_scale, color, thickness)


def draw_solution(image, solve_input, solve_output, n_rows=9, n_cols=9):
    for i_row in range(n_rows):
        for i_col in range(n_cols):
            if solve_input[i_row][i_col] > 0:
                continue
            draw_digit(image, solve_output[i_row][i_col], i_row, i_col)
    return image


def overlay_transformed_image(background_image, foreground_image, transformation_matrix):
    return cv.warpPerspective(src=foreground_image,
                              dst=background_image,
                              M=transformation_matrix,
                              dsize=(background_image.shape[1], background_image.shape[0]),
                              borderMode=cv.BORDER_TRANSPARENT)
