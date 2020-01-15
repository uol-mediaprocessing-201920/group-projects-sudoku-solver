import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import math
import random
import cv2 as cv


def show_images(images, titles=None,
                rows=None, columns=None,
                cmap=None, scaling=(5, 5),
                n_samples=None, i_samples=None,
                normalize=False,
                figure=True):
    assert not (n_samples is not None and i_samples is not None)

    if titles is None:
        titles = ["[%d]" % i for i in range(len(images))]

    if n_samples is not None:
        indices = random.sample(range(len(images)), n_samples)
        images = [images[i] for i in indices]
        titles = [titles[i] for i in indices]
    elif i_samples is not None:
        images = [images[i] for i in i_samples]
        titles = [titles[i] for i in i_samples]

    if rows is None and columns is None:
        rows = math.ceil(math.sqrt(len(images)))
    if rows is None:
        rows = math.ceil(len(images) / columns)
    if columns is None:
        columns = math.ceil(len(images) / rows)

    if figure:
        plt.figure(figsize=(scaling[1] * columns, scaling[0] * rows))

    for i, image in enumerate(images):
        if len(image.shape) == 3 and image.shape[-1] == 1:
            # assume grayscale image with single channel has been passed
            # matplotlib does not like that
            image = image.reshape(image.shape[:-1])
        if not normalize and image.dtype.name.startswith("uint"):
            normalizer = matplotlib.colors.Normalize(vmin=0, vmax=255)
        elif not normalize and image.dtype.name.startswith("float"):
            normalizer = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
        else:
            normalizer = matplotlib.colors.Normalize()
        plt.subplot(rows, columns, i + 1)
        plt.title(titles[i])
        plt.imshow(image, cmap=cmap, norm=normalizer)

    if figure:
        plt.tight_layout()
        plt.show()


def show_cells(cells, n_rows=9, n_cols=9):
    assert len(cells) == n_rows
    plt.figure(figsize=(10, 10))
    counter = 0
    for i_row, row in enumerate(cells):
        assert len(row) == n_cols
        for i_col, col in enumerate(row):
            counter += 1
            plt.subplot(n_rows, n_cols, counter)
            plt.title(str((i_row, i_col)))
            plt.axis("off")
            plt.imshow(col)
    plt.tight_layout()
    plt.show()


def grid_to_string(grid, n_rows=9, n_cols=9):
    output = []
    cell_width = 3
    total_spacing = n_cols + 1
    total_cell_width = n_cols * cell_width
    total_width = total_spacing + total_cell_width
    for i_row in range(n_rows):
        output.append("-" * total_width + "\n")
        for i_col in range(n_cols):
            digit = grid[i_row][i_col]
            if digit >= 1 and digit <= 9:
                output.append("| %d " % grid[i_row][i_col])
            else:
                output.append("|   ")
        output.append("|\n")
    output.append("-" * total_width + "\n")
    return "".join(output)


def grids_to_string(grids, spacing=1, **kwargs):
    strings = [grid_to_string(g, **kwargs).split("\n") for g in grids]
    n_lines = min(len(s) for s in strings)
    padding = " " * spacing
    output = [padding.join(s[i_row] for s in strings) + "\n" for i_row in range(n_lines)]
    return "".join(output)


def print_grid(grid, **kwargs):
    string = grid_to_string(grid, **kwargs)
    print(string)


def get_opencv_color_conversion_code(source, target):
    return getattr(cv, "COLOR_" + source.upper() + "2" + target.upper())


def apply_opencv_color_conversion(image, source, target, always_copy=False):
    if source is None or target is None or source == target:
        if always_copy:
            return image.copy()
        else:
            return image
    name = "COLOR_" + source.upper() + "2" + target.upper()
    if not hasattr(cv, name):
        raise Exception("Unknown OpenCV color conversion code: " + name)
    code = getattr(cv, name)
    return cv.cvtColor(image, code)
