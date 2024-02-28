import numpy as np
import os
import matplotlib.pyplot as plt
from data_parser import DataParser
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from loguru_pack import logger, loguru_config
from utils import plot_array, plot_image, plot_two_arrays


class EdgeFinder:
    """EdgeFinder only applies to parallel image because edges are most
    distinct. The edges are used to correct the distortion in the bias image.
    """

    measurement = "Pockels"  # not useful, only to demo class variables

    def __init__(self, image_calib):
        self.image_calib = image_calib
        self.edge_1 = None
        self.edge_2 = None
        self.edge_1_fit = None
        self.edge_2_fit = None
        self.padding = 10
        self.image_with_edges = None
        self.image_corrected = None  # output of correct_distortion
        self.image_corrected_cropped = None
        self.cleaned_image = None  # output of remove_bright_speckes

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

    @staticmethod
    def cubic_fit_edge(edge_array):
        """
        Applies a cubic fit to an edge array.
        Args:
            edge_array: A 1D numpy array containing edge positions.
        Returns:
            fitted_edge_array: A 1D numpy array of the same length as `edge_array`,
                        containing the cubic fitted edge values.
        """
        x1 = np.arange(len(edge_array))
        p = np.polyfit(x1, edge_array, 3)
        fitted_edge_array = np.round(np.polyval(p, x1)).astype(int)

        return fitted_edge_array

    def find_sensor_edges(
        self, threshold=0.8, fit_func=linear_fit_edge, show_dim=False, show_plots=False
    ):
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
        image_center = self.image_calib[
            :, int(image_width / 2)
        ]  # rounds down image_length/2
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

        self.edge_1_fit = fit_func(self.edge_1)
        self.edge_2_fit = fit_func(self.edge_2)

        return self.edge_1, self.edge_2, self.edge_1_fit, self.edge_2_fit

    def show_edge_fit(self):
        scale = -1
        plt.figure("plot array utils")
        plt.title("Edge fit")
        plt.plot(scale*self.edge_1, marker=".", lw = 0.2, label="edge 1")
        plt.plot(scale*self.edge_2, marker=".", lw = 0.2, label="edge 2")
        plt.plot(scale*self.edge_1_fit, linestyle="-", label="edge 1 fit")
        plt.plot(scale*self.edge_2_fit, linestyle="-", label="edge 2 fit")
        plt.legend()
        plt.grid(True)
        plt.show()


class ImageProcessor:
    def __init__(self, raw_image, edge_finder: EdgeFinder):

        self.image_calib = edge_finder.image_calib
        self.edge_1 = edge_finder.edge_1
        self.edge_2 = edge_finder.edge_2
        self.edge_1_fit = edge_finder.edge_1_fit
        self.edge_2_fit = edge_finder.edge_2_fit

        self.padding = 10
        self.raw_image = raw_image
        self.cropped_raw_image = None
        self.y_crop = None
        self.image_corrected = None
        self.image_corrected_cropped = None
        self.image_with_edges = None
        self.cleaned_image = None

    @property
    def raw_length(self):
        return self.raw_image.shape[0]

    @property
    def raw_width(self):
        return self.raw_image.shape[1]

    def crop_image(self):

        sensor_thickness_fit = np.mean(self.edge_2_fit - self.edge_1_fit)

        # set self.padding=0 if beyond image boundaries
        if (
            sensor_thickness_fit + 2 * self.padding >= self.raw_length
            or min(self.edge_1_fit) < self.padding
            or max(self.edge_2_fit) > self.raw_length - self.padding
        ):
            self.padding = 0

        logger.debug(f"{self.padding = }")

        # array of new pixel positions in the y-axis
        self.y_crop = np.arange(
            round(np.mean(self.edge_1_fit) - self.padding),
            round(np.mean(self.edge_2_fit)) + self.padding,
        )
        # crop self.image_corrected to the region of interest
        self.cropped_raw_image = self.raw_image[self.y_crop[0] : self.y_crop[-1], :]

        logger.debug(f"{self.cropped_raw_image.shape = }")

    def correct_distortion(self):
        if self.cropped_raw_image is None:
            self.crop_image()
        
        self.image_corrected = np.zeros_like(self.raw_image)
        self.image_corrected_cropped = np.zeros_like(self.cropped_raw_image)

        for col_index in range(self.raw_width):
            # Calculate the vertical bounds for the current column, adjusted by self.padding
            edge_1_crop = int(self.edge_1_fit[col_index] - self.padding)
            edge_2_crop = int(self.edge_2_fit[col_index] + self.padding)
            # Extract a vertical slice (column) of the image within the calculated bounds
            column_slice_raw = self.raw_image[edge_1_crop:edge_2_crop, col_index]

            original_positions = np.arange(len(column_slice_raw))
            logger.debug(f"{len(original_positions) = }")

            new_positions = np.arange(1, len(self.y_crop))
            logger.debug(f"{len(new_positions) = }")

            interpolated_values = interp1d(
                original_positions,
                column_slice_raw,
                kind="linear",
                fill_value="extrapolate",
            )(new_positions)

            logger.debug(f"{len(interpolated_values) = }")
            self.image_corrected[self.y_crop[0] : self.y_crop[-1], col_index] = (
                interpolated_values
            )
            self.image_corrected_cropped[:, col_index] = interpolated_values

        return self.image_corrected, self.image_corrected_cropped

    def show_image_with_edges(self, plot_image=False):
        # Ensure edge_1_fit and edge_2_fit are defined and accessible
        plt.figure("Image with Edges")

        # Plot image_calib
        plt.imshow(self.image_calib, cmap="jet")

        # Generate x-coordinates for the edges
        rows = np.arange(self.image_calib.shape[1])

        # create a new image with the edges and padding marked, by setting the values of the edges to 0 and padding to max
        self.image_with_edges = np.copy(self.image_calib)
        self.image_with_edges[self.edge_1_fit, rows] = 0
        self.image_with_edges[self.edge_2_fit, rows] = 0
        self.image_with_edges[self.edge_1_fit - self.padding, rows] = np.max(
            self.image_with_edges
        )
        self.image_with_edges[self.edge_2_fit + self.padding, rows] = np.max(
            self.image_with_edges
        )

        if plot_image:
            plt.figure("Image with Edges")
            plt.imshow(self.image_with_edges, cmap="jet")
            plt.colorbar()
            plt.show()

    def remove_bright_speckles(self, threshold_scale=1.5):
        """
        args:
            threshold_scale: float, default=1.2
                Lower values will remove more speckles, but may also remove valid features.
                Best values are between 1.2 and 1.6, determined by trial and error.
        """
        self.cleaned_image = np.copy(self.image_corrected_cropped)
        for y in range(2, self.cleaned_image.shape[0] - 2):
            for x in range(2, self.cleaned_image.shape[1] - 2):
                neighbors = self.cleaned_image[y - 2 : y + 3, x - 2 : x + 3].ravel()
                avg_neighbors = np.mean(neighbors)

                # Check if the current pixel is a bright speckle
                if self.cleaned_image[y, x] > threshold_scale * avg_neighbors:
                    # Replace with the minimum of left and right neighbors
                    self.cleaned_image[y, x] = np.min(
                        [self.cleaned_image[y, x - 1], self.cleaned_image[y, x + 1]]
                    )

        return self.cleaned_image


def main():
    data_folder = r"test_data\D325150"
    bias_txt = os.path.join(data_folder, "700V Crossed 25mA.txt")
    DP = DataParser()
    bias_image = DP.txt2array(bias_txt)
    parallel_0V_txt = os.path.join(data_folder, "0V Parallel.txt")
    parallel_image = DP.txt2array(parallel_0V_txt)
    
    edge_finder = EdgeFinder(parallel_image)
    edge_finder.find_sensor_edges(fit_func=edge_finder.cubic_fit_edge)
    edge_finder.show_edge_fit()

    processor = ImageProcessor(raw_image=bias_image, edge_finder=edge_finder)
    img_corrected, img_corrected_cropped = processor.correct_distortion()
    print(processor.__dict__.keys())
    processor.show_image_with_edges(plot_image=True)

    # crop the image to the region of interest
    xmin, xmax = 450, 690
    ymin, ymax = 115, 225
    img_corrected, img_corrected_cropped = processor.correct_distortion()
    plt.figure("corrected and cropped bias image")
    # set vmin and vmax to the same values to compare images
    # plt.imshow(img_corrected[ymin:ymax, xmin:xmax], cmap='jet', vmin=0, vmax=700)
    plt.imshow(img_corrected_cropped, cmap="jet", vmin=0, vmax=700)
    plt.colorbar()

    for threshold_scale in [1.2, 1.4, 1.6]:
        cleaned = processor.remove_bright_speckles(threshold_scale=threshold_scale)
        plt.figure(f"cleaned image threshold_scale={threshold_scale}")
        # plt.imshow(cleaned[ymin:ymax, xmin:xmax], cmap='jet', vmin=0, vmax=700)
        plt.imshow(cleaned, cmap="jet", vmin=0, vmax=700)
        plt.colorbar()

    plt.show()


if __name__ == "__main__":
    main()
