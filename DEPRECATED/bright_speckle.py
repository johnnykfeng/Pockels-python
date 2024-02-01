from loguru_pack import loguru_config
from loguru import logger
import cv2
import matplotlib.pyplot as plt
import numpy as np
from DEPRECATED.load_files import txt2matrix

def remove_bright_speckles(image, threshold_scale=2.0):
    """
    Remove bright speckles from an image by comparing each pixel to its neighbors.

    :param image: Input image (grayscale or single channel)
    :return: Image with bright speckles removed
    """
    # Ensure the image is in grayscale
    if len(image.shape) > 2:
        logger.warning()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a copy of the image to modify
    cleaned_image = np.copy(image)

    # Iterate through the image (excluding borders)
    for y in range(1, image.shape[0] - 2):
        for x in range(1, image.shape[1] - 2):
            # Calculate the average of the neighbors
            neighbors = [
                image[y-1, x], image[y+1, x],  # top and bottom
                image[y, x-1], image[y, x+1],   # left and right
                image[y-1, x-1], image[y+1, x-1],  # corners
                image[y-1, x+1], image[y+1, x+1]   # corners
            ]
            avg_neighbors = np.mean(neighbors)

            # Check if the current pixel is a bright speckle
            if image[y, x] > threshold_scale * avg_neighbors:
                # Replace with the minimum of left and right neighbors
                # cleaned_image[y, x] = np.min([image[y, x-1], image[y, x+1], 
                #                                image[y, x-2], image[y, x+2]])
                cleaned_image[y, x] = np.min([image[y, x-1], image[y, x+1]])

    return cleaned_image

def compare_images(image1, image2, verbose = True):
    """
    Compare two images and return the count of pixels that are exactly the same
    and the count of pixels that are different.

    :param image1: First input image
    :param image2: Second input image
    :return: A tuple (count_same, count_different)
    """
    if image1.shape != image2.shape:
        raise ValueError("Images do not have the same dimensions")

    # Calculate the difference
    difference = image1 != image2
    total = difference.size

    # Count the number of different pixels
    count_different = np.count_nonzero(difference)

    # Count the number of same pixels
    count_same = total - count_different

    # Calculate the fraction
    same_fraction = round(count_same/total, 3)
    diff_fraction = round(count_different/total, 3)
    if verbose:
        print(f"Same pixels: {count_same}, Different pixels: {count_different}")
        print(f"Same pixels: {same_fraction}, Different pixels: {diff_fraction}")

    return count_same, count_different, same_fraction, diff_fraction

image_path = r'test_data\D325150\700V Crossed 25mA.txt'
# image_path = r"test_data\D325150\500V Crossed 0mA.txt"
# image_path = r"test_data\D325150\0V Parallel.txt"
image_array = txt2matrix(image_path)

# Remove bright speckles
cleaned_image = remove_bright_speckles(image_array, threshold_scale=2.0)

# Compare the images
compare_images(image_array, cleaned_image)

vmin, vmax = 0, 600

plt.figure()
plt.imshow(image_array, cmap='jet', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.figure()
plt.imshow(cleaned_image, cmap='jet', vmin=vmin, vmax=vmax)
plt.colorbar()

plt.show()