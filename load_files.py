import numpy as np
import os
import matplotlib.pyplot as plt


def txt2matrix(txt_file, no_zeros = False):
    data = np.loadtxt(txt_file, skiprows=19)

    # Step 2: Determine the dimensions of the image
    max_row = int(np.max(data[:, 0])) + 1  # Adding 1 because index starts at 0
    max_col = int(np.max(data[:, 1])) + 1  # Adding 1 because index starts at 0

    # print(max_row)
    # print(max_col)

    # Step 3: Create an empty array
    image_matrix = np.zeros((max_row, max_col), dtype=np.uint16)

    # Step 4: Fill the array with pixel values
    for row in data:
        r, c, val = int(row[0]), int(row[1]), row[2]
        image_matrix[r, c] = val + 1

    return image_matrix 

