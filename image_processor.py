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
        
class ImageProcessor:
    def __init__(self, image_raw):
        self.image_raw = image_raw
        self.cropped_image = None
        self.corrected_image = None
        self.self.edge_1 = None
        self.self.edge_2 = None
        self.self.self.edge_2_fit = None
        self.self.self.edge_2_fit = None
        
    def linear_fit_edge(self, edge_array):
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

    def find_sensor_edges(self, image, threshold=0.8, show_dim = True, show_plots = False):
        """
        Detects the top and bottom edges of signals in an image.

        Parameters:
        image: A 2D numpy array representing the input image. The array elements are
            expected to be intensity values of the image.

        Returns:
        self.edge_1, self.edge_2: 1-D array of size image_width, maps the column position of the edge
        self.edge_1_fit, self.self.edge_2_fit: linear fit of self.edge_1 and edge_2
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
        self.self.edge_1 = np.zeros(image_width, dtype=int)
        self.self.edge_2 = np.zeros(image_width, dtype=int)

        for i in range(image_width):
            Iline = image[:, i]  # image per line
            I_signal_rough = Iline[Iline > threshold_line]
            if I_signal_rough.size > 0:
                Signal_median = np.median(I_signal_rough)
                Th = Signal_median / 2  # Threshold for judging an edge
                signal_range = np.where(Iline > Th)[0]

                if signal_range.size > 0:
                    self.self.edge_1[i] = signal_range[0] 
                    self.self.edge_2[i] = signal_range[-1]
                else:
                    self.self.edge_1[i] = 0
                    self.self.edge_2[i] = image_width - 1
            else:
                self.self.edge_1[i] = 0
                self.self.edge_2[i] = image_width - 1

        self.edge_1_fit = self.linear_fit_edge(self.edge_1)
        self.self.edge_2_fit = self.linear_fit_edge(edge_2)
        if show_plots:  
            plot_two_arrays(self.edge_1, self.edge_1_fit, "upper edge")
            plot_two_arrays(self.edge_2, self.edge_2_fit, "lower edge")

        return self.edge_1, self.edge_2, self.edge_1_fit, self.edge_2_fit

    def correct_distortion(self, padding = 10):

        length, width = self.image_raw.shape

        sensor_thickness_fit = np.mean(self.edge_2_fit - self.edge_1_fit)
        self.image_corrected = np.full((length, width), np.nan)

        if sensor_thickness_fit + 2 * padding >= length or \
            min(self.edge_1_fit) < padding or \
            max(self.edge_2_fit) > length - padding:
            padding = 0

        logger.debug(f"{padding = }")

        Y_standard = np.arange(round(np.mean(self.edge_1_fit) - padding), round(np.mean(self.edge_2_fit)) + padding)

        logger.debug(f"{len(Y_standard) = }")

        for col_index in range(width):
            # Calculate the vertical bounds for the current column, adjusted by padding
            bottom_edge = int(self.edge_1_fit[col_index] - padding)
            top_edge = int(self.edge_2_fit[col_index] + padding)
            # Extract a vertical slice (column) of the image within the calculated bounds
            column_slice = self.image_raw[bottom_edge:top_edge, col_index]
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
            self.image_corrected[Y_standard[0]: Y_standard[-1], col_index] = interpolated_values

data_folder = r"test_data\D325150"
bias_txt = os.path.join(data_folder, '700V Crossed 25mA.txt')
bias_image = txt2matrix(bias_txt)
parallel_0V_txt = os.path.join(data_folder, '0V Parallel.txt')
parallel_image = txt2matrix(parallel_0V_txt)

image_processor = ImageProcessor(bias_image)

# def main():



# if __name__ == '__main__':
    # main()