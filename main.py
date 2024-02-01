from data_parser import DataParser
from image_processor import ImageProcessor
import h5py
import numpy as np
from pprint import pprint as pp
from loguru_pack import logger, loguru_config

# Open the HDF5 file in append mode
hdf5_filepath = r'D325150.hdf5'
# open the file in append mode
f = h5py.File(hdf5_filepath, 'a') # make sure to close the file at the end of the script

DP = DataParser()
# IP = ImageProcessor()

def initial_edge_processing():
    # load the parallel image
    parallel_image_key = '0V Parallel.txt'
    parallel_image = f[parallel_image_key]["raw_image"][:]
    IP = ImageProcessor(parallel_image) # create an instance with parallel_image as calib_image
    IP.find_sensor_edges()
    img_corrected = IP.correct_distortion(raw_image=parallel_image)
    IP.show_image_with_edges(plot_image=False)

    # load the image with edges into the hdf5 file
    if 'edge_detection' in f[parallel_image_key]:
        del f[parallel_image_key]['edge_detection']
    f[parallel_image_key].create_dataset('edge_detection', 
                                        data=IP.image_with_edges)
    logger.success("Edge detection data saved to hdf5 file")

    return IP

IP = initial_edge_processing()
logger.debug(IP.edge_1_fit[0:10])
logger.debug(IP.edge_1_fit[0:10])

image_keys = list(f.keys())
for i, image_key in enumerate(image_keys):
    # print the key
    pp("Key: %s" % image_key)
    if image_key == '0V Parallel.txt': # skip the parallel image
        continue

    raw_image = (f[image_key]["raw_image"][:])
    print(raw_image.shape)

    # correct the distortion
    _, image_corrected_cropped = IP.correct_distortion(raw_image=raw_image)
    # print(image_corrected_cropped.shape)

    cleaned_image = IP.remove_bright_speckles(threshold_scale=1.5)
    logger.success("Image cleaned")

    # create a new dataset in the file
    if 'corrected_cropped' in f[image_key]:
        del f[image_key]['corrected_cropped']
    f[image_key].create_dataset('corrected_cropped', data=image_corrected_cropped)

    if 'corrected_cropped_cleaned' in f[image_key]:
        del f[image_key]['corrected_cropped_cleaned']
    f[image_key].create_dataset('corrected_cropped_cleaned', data=cleaned_image)


# close hdf5 file
f.close()
