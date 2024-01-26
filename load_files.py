import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from loguru import logger

def txt2matrix(txt_file, no_zeros = False):
    data = np.loadtxt(txt_file, skiprows=19)

    # Step 1: Determine the dimensions of the image
    max_row = int(np.max(data[:, 0])) + 1  # Adding 1 because index starts at 0
    max_col = int(np.max(data[:, 1])) + 1  # Adding 1 because index starts at 0

    # Step 2: Create an empty array
    image_matrix = np.zeros((max_row, max_col), dtype=np.uint16)

    # Step 3: Fill the array with pixel values
    for row in data:
        r, c, val = int(row[0]), int(row[1]), row[2]
        image_matrix[r, c] = val + 1

    return image_matrix 

def open_bmp(bmp_file_path):
    # Read the image
    bmp_image = cv2.imread(bmp_file_path)

    logger.debug(f"{type(bmp_image) = }")
    logger.debug(f"{bmp_image.shape = }")
    # Check if image is loaded properly
    if bmp_image is not None:
        # Display the image
        cv2.imshow('Image', bmp_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("Error: Image not loaded. Check the file path.")
    return bmp_image


def main():
    # txt_file = (r"test_data\\D325150\\500V Crossed 25mA.txt")
    # bmp = open_bmp(r"test_data\D325150\500V Crossed 25mA.bmp")
    txt_file = (r"test_data\\D325150\\0V Parallel.txt")
    bmp = open_bmp(r"test_data\D325150\0V Parallel.bmp")
    image = txt2matrix(txt_file)
    cv2.imshow('image', image)
    plt.imshow(image)
    plt.show()
    

if __name__ == '__main__':
    main()