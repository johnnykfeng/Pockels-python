# import logging.handlers
# import logging.config
# import json
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from load_files import txt2matrix
from skimage import filters, feature, io
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from loguru_pack import loguru_config
from loguru import logger
from utils import plot_array, plot_image, plot_two_arrays

# loguru_config.try_all_levels()

# NOT USED, OTHER METHOD BETTER
# def skimage_extract_edges(image, method='sobel', threshold=0.1):
#     """
#     Extract edges from an image using either the Sobel operator or Canny edge detector.

#     Parameters:
#     image (numpy.ndarray): The input image (2D array).
#     method (str): The edge detection method ('sobel' or 'canny').
#     threshold (float): The threshold value for edge detection.

#     Returns:
#     numpy.ndarray: The image with edges highlighted.
#     """
#     if method.lower() == 'sobel':
#         # Apply Sobel operator
#         edges = filters.sobel(image)
#         # Apply threshold
#         edges = edges > threshold
#     elif method.lower() == 'canny':
#         # Apply Canny edge detector
#         edges = feature.canny(image, sigma=threshold)
#     else:
#         raise ValueError("Method should be either 'sobel' or 'canny'")

#     return edges

        
def linear_fit_edge(edge_array):
    """
    Applies a linear fit to an edge array.
    Args:
        edge_array: A 1D numpy array containing edge positions.
    Returns:
        linear_edge_array: A 1D numpy array of the same length as `edge_array`,
                       containing the linearly fitted edge values.
    """
    x1 = np.arange(len(edge_array))
    p = np.polyfit(x1, edge_array, 1)
    linear_edge_array = np.round(np.polyval(p, x1)).astype(int)

    return linear_edge_array

def find_sensor_edges(image, threshold=0.8, show_dim = True, show_plots = False):
    """
    Detects the top and bottom edges of signals in an image.

    Parameters:
    image: A 2D numpy array representing the input image. The array elements are
        expected to be intensity values of the image.

    Returns:
    edge_1, edge_2: 1-D array of size image_width, maps the column position of the edge
    edge_1_fit, edge_2_fit: linear fit of edge_1 and edge_2
    """

    image_length, image_width = image.shape
    if show_dim:
        logger.debug(f"{image_length = }")
        logger.debug(f"{image_width = }")

    # Threshold for judging signal
    image_center = image[:, int(image_width/2)]  # rounds down image_length/2
    image_center_smooth = uniform_filter1d(image_center, size=20)
    threshold_line = threshold * np.max(image_center_smooth)

    # Initialize arrays
    edge_1 = np.zeros(image_width, dtype=int)
    edge_2 = np.zeros(image_width, dtype=int)

    for i in range(image_width):
        Iline = image[:, i]  # image per line
        I_signal_rough = Iline[Iline > threshold_line]
        if I_signal_rough.size > 0:
            Signal_median = np.median(I_signal_rough)
            Th = Signal_median / 2  # Threshold for judging an edge
            signal_range = np.where(Iline > Th)[0]

            if signal_range.size > 0:
                edge_1[i] = signal_range[0] 
                edge_2[i] = signal_range[-1]
            else:
                edge_1[i] = 0
                edge_2[i] = image_width - 1
        else:
            edge_1[i] = 0
            edge_2[i] = image_width - 1

    edge_1_fit = linear_fit_edge(edge_1)
    edge_2_fit = linear_fit_edge(edge_2)
    if show_plots:  
        plot_two_arrays(edge_1, edge_1_fit, "upper edge")
        plot_two_arrays(edge_2, edge_2_fit, "lower edge")

    return edge_1, edge_2, edge_1_fit, edge_2_fit


def correct_distortion(image_raw, upper_edge_line, lower_edge_line, padding = 0):

    # upper_bound = min(upper_edge_line) - padding
    # lower_bound = max(lower_edge_line) + padding
    # image_cropped = image_raw[upper_bound: lower_bound, :]
    image_cropped = image_raw

    # _, _, upper_edge_line, lower_edge_line = find_sensor_edges(image_cropped, threshold = threshold)  # Assuming you have this function implemented

    length, width = image_cropped.shape
    # print(f"{length = }")
    # print(f"{width = }")

    sensor_thickness_fit = np.mean(lower_edge_line - upper_edge_line)
    image_corrected = np.full((length, width), np.nan)

    if sensor_thickness_fit + 2 * padding >= length or \
        min(upper_edge_line) < padding or \
        max(lower_edge_line) > length - padding:
        padding = 0

    logger.debug(f"{padding = }")

    Y_standard = np.arange(round(np.mean(upper_edge_line) - padding), round(np.mean(lower_edge_line)) + padding)

    logger.debug(f"{len(Y_standard) = }")

    for col_index in range(width):
        # Calculate the vertical bounds for the current column, adjusted by padding
        bottom_edge = int(upper_edge_line[col_index] - padding)
        top_edge = int(lower_edge_line[col_index] + padding)
        # Extract a vertical slice (column) of the image within the calculated bounds
        column_slice = image_cropped[bottom_edge:top_edge, col_index]
        # Create an array representing the original vertical positions in the slice
        original_positions = np.arange(len(column_slice))
        # Create a linearly spaced array for the new vertical positions
        new_positions = np.arange(1, len(Y_standard))
        # Perform linear interpolation from the original positions to the new positions
        # 'interp1d' creates a linear interpolation function
        # 'fill_value='extrapolate'' allows the function to handle values outside the domain
        interpolated_values = interp1d(original_positions, column_slice, 
                                    kind='linear', fill_value='extrapolate')(new_positions)

        # Assign the interpolated values to the corresponding column in the corrected image
        # 'Y_standard' defines the vertical range in the corrected image where the interpolated values are placed
        image_corrected[Y_standard[0]: Y_standard[-1], col_index] = interpolated_values

    return image_corrected



# Example usage:
def test_correct_distortion():

    data_folder = r"test_data\D325150"
    bias_txt = os.path.join(data_folder, '700V Crossed 25mA.txt')
    bias_image = txt2matrix(bias_txt)
    parallel_0V_txt = os.path.join(data_folder, '0V Parallel.txt')
    parallel_image = txt2matrix(parallel_0V_txt)

    upper_edge, lower_edge, upper_edge_fit, lower_edge_fit = find_sensor_edges(parallel_image, threshold=0.8, show_plots=True)
    logger.debug(f"{upper_edge[0:10] = }")
    logger.debug(f"{lower_edge[0:10] = }")

    upper_bound = np.round(np.mean(upper_edge)-50).astype(int)
    lower_bound = np.round(np.mean(lower_edge)+50).astype(int)

    padding = 10
    image_corrected = correct_distortion(bias_image, upper_edge_fit, lower_edge_fit, padding = padding)

    image_cropped = bias_image[upper_bound: lower_bound, :]

    logger.success("Making plots")

    plt.figure("Parallel image")
    plt.imshow(parallel_image, cmap='viridis')  # 'gray' colormap for grayscale image
    rows = np.arange(parallel_image.shape[1])
    plt.scatter(rows, upper_edge_fit, color = 'red', label = 'Top Edge fit', s=2, alpha=0.2)
    plt.scatter(rows, lower_edge_fit, color = 'blue', label = 'Bottom Edge fit', s=2, alpha=0.2)
    plt.scatter(rows, upper_edge, color = 'black', label = 'Top Edge fit', s=2, alpha=0.2)
    plt.scatter(rows, lower_edge, color = 'black', label = 'Bottom Edge fit', s=2, alpha=0.2)
    plt.legend()

    plt.figure("bias image with edge lines")
    plt.imshow(bias_image, cmap='viridis')  # 'gray' colormap for grayscale image
    rows = np.arange(bias_image.shape[1])
    plt.scatter(rows, upper_edge_fit, color = 'red', label = 'Top Edge', s=2, alpha=0.2)
    plt.scatter(rows, lower_edge_fit, color = 'blue', label = 'Bottom Edge', s=2, alpha=0.2)
    plt.scatter(rows, upper_edge_fit - padding, color = 'magenta', label = 'edge with padding', s=2, alpha=0.2)
    plt.scatter(rows, lower_edge_fit + padding, color = 'magenta', label = 'edge with padding', s=2, alpha=0.2)
    plt.legend()

    plt.figure("corrected and cropped bias image")
    plt.imshow(image_corrected, cmap='viridis')  # 'gray' colormap for grayscale image
    plt.show()


if __name__ == '__main__':
    test_correct_distortion()