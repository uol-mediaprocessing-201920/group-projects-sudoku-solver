import numpy as np


def extract_cells(image, n_rows=9, n_cols=9):
    split = np.array_split(image, n_rows)
    for i, x in enumerate(split):
        split[i] = np.array_split(x, n_cols, axis=1)
    # show_images(split[0])
    return split
