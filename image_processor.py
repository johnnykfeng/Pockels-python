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
    
    measurement = 'Pockels' # not useful, only to demo class variables
    
    def __init__(self, image_calib):
        self.image_calib = image_calib
        self.cropped_image = None
        self.image_corrected = None
        self.edge_1 = None
        self.edge_2 = None
        self.edge_1_fit = None
        self.edge_2_fit = None
        self.padding = 10
        
    @staticmethod
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

    def find_sensor_edges(self, threshold=0.8, show_dim = False, show_plots = False):
        """
        Detects the top and bottom edges of signals in an image.

        Parameters:
        image: A 2D numpy array representing the input image. The array elements are
            expected to be intensity values of the image.

        Returns:
        self.edge_1, self.edge_2: 1-D array of size image_width, maps the column position of the edge
        self.edge_1_fit, self.edge_2_fit: linear fit of self.edge_1 and edge_2
        """

        image_length, image_width = self.image_calib.shape
        if show_dim:
            logger.debug(f"{image_length = }")
            logger.debug(f"{image_width = }")

        # Threshold for judging signal
        image_center = self.image_calib[:, int(image_width/2)]  # rounds down image_length/2
        image_center_smooth = uniform_filter1d(image_center, size=20)
        threshold_line = threshold * np.max(image_center_smooth)

        # Initialize arrays
        self.edge_1 = np.zeros(image_width, dtype=int)
        self.edge_2 = np.zeros(image_width, dtype=int)

        for i in range(image_width):
            Iline = self.image_calib[:, i]  # image per line
            I_signal_rough = Iline[Iline > threshold_line]
            if I_signal_rough.size > 0:
                Signal_median = np.median(I_signal_rough)
                Th = Signal_median / 2  # Threshold for judging an edge
                signal_range = np.where(Iline > Th)[0]

                if signal_range.size > 0:
                    self.edge_1[i] = signal_range[0] 
                    self.edge_2[i] = signal_range[-1]
                else:
                    self.edge_1[i] = 0
                    self.edge_2[i] = image_width - 1
            else:
                self.edge_1[i] = 0
                self.edge_2[i] = image_width - 1

        self.edge_1_fit = self.linear_fit_edge(self.edge_1)
        self.edge_2_fit = self.linear_fit_edge(self.edge_2)
        
        #shows the edge fitting close ip
        if show_plots:  
            plot_two_arrays(self.edge_1, self.edge_1_fit, "upper edge")
            plot_two_arrays(self.edge_2, self.edge_2_fit, "lower edge")

        return self.edge_1, self.edge_2, self.edge_1_fit, self.edge_2_fit

    def correct_distortion(self, raw_image):

        length, width = raw_image.shape

        sensor_thickness_fit = np.mean(self.edge_2_fit - self.edge_1_fit)
        self.image_corrected = np.full((length, width), np.nan)

        # set self.padding=0 if beyond image boundaries
        if sensor_thickness_fit + 2 * self.padding >= length or \
            min(self.edge_1_fit) < self.padding or \
            max(self.edge_2_fit) > length - self.padding:
            self.padding = 0

        logger.debug(f"{self.padding = }")

        # array of new pixel positions in the y-axis
        y_corrected = np.arange(round(np.mean(self.edge_1_fit) - self.padding), round(np.mean(self.edge_2_fit)) + self.padding)

        logger.debug(f"{len(y_corrected) = }")

        for col_index in range(width):
            # Calculate the vertical bounds for the current column, adjusted by self.padding
            edge_1_crop = int(self.edge_1_fit[col_index] - self.padding)
            edge_2_crop = int(self.edge_2_fit[col_index] + self.padding)
            # Extract a vertical slice (column) of the image within the calculated bounds
            column_slice = raw_image[edge_1_crop:edge_2_crop, col_index]
            # Create an array representing the original vertical positions in the slice
            original_positions = np.arange(len(column_slice))
            # Create a linearly spaced array for the new vertical positions
            new_positions = np.arange(1, len(y_corrected))
            # Perform linear interpolation from the original positions to the new positions
            # 'interp1d' creates a linear interpolation function
            # 'fill_value='extrapolate'' allows the function to handle values outside the domain
            interpolated_values = interp1d(original_positions, column_slice, 
                                        kind='linear', fill_value='extrapolate')(new_positions)

            # Assign the interpolated values to the corresponding column in the corrected image
            # 'y_corrected' defines the vertical range in the corrected image where the interpolated values are placed
            self.image_corrected[y_corrected[0]: y_corrected[-1], col_index] = interpolated_values

        return self.image_corrected

    def plot_edge_fit(self):
        # Ensure edge_1_fit and edge_2_fit are defined and accessible
        plt.figure("Image with Edges")

        # Plot image_calib
        plt.imshow(self.image_calib, cmap='jet')

        # Generate x-coordinates for the edges
        rows = np.arange(self.image_calib.shape[1])

        # Plot edge_1_fit and edge_2_fit
        plt.plot(rows, self.edge_1_fit, color='black', label='Edge 1 fit', alpha=1)
        plt.plot(rows, self.edge_2_fit, color='green', label='Edge 2 fit', alpha=1)

        # Plot edges with padding
        plt.plot(rows, self.edge_1_fit - self.padding, color='magenta', label='padding')
        plt.plot(rows, self.edge_2_fit + self.padding, color='magenta')

        plt.colorbar()
        plt.legend()
        plt.show()

def main():
    data_folder = r"test_data\D325150"
    bias_txt = os.path.join(data_folder, '700V Crossed 25mA.txt')
    bias_image = txt2matrix(bias_txt)
    parallel_0V_txt = os.path.join(data_folder, '0V Parallel.txt')
    parallel_image = txt2matrix(parallel_0V_txt)

    processor = ImageProcessor(parallel_image)
    processor.find_sensor_edges()
    img_corrected = processor.correct_distortion(raw_image=bias_image)
    print(processor.__dict__.keys())
    processor.plot_edge_fit()

    # # second edge detection does not work!
    # processor2 = ImageProcessor(image_calib=img_corrected)
    # processor2.find_sensor_edges(show_dim=True, show_plots=True)
    # img_corrected_again = processor2.correct_distortion(raw_image=img_corrected)
    # processor2.plot_edge_fit()

    plt.figure("corrected and cropped bias image")
    # set vmin and vmax to the same values to compare images
    plt.imshow(img_corrected, cmap='jet', vmin=0, vmax=700)  # 'gray' colormap for grayscale image
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()