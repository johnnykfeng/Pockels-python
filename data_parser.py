from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from pprintpp import pprint as pp
import h5py
from loguru_pack import logger, loguru_config
from abc import ABC, abstractmethod

class DataParser(ABC):
    @abstractmethod
    def txt2array(self, txt_file:str, no_zeros=False) -> np.ndarray:
        """ Used in the store_in_hdf5 method.
        Convert a .txt file to a 2D numpy array."""
        pass
    def txt2metadata(self, txt_file, start_line:int = 5, end_line:int = 17) -> dict:
        """ Used in the store_in_hdf5 method.
        Extract metadata from a .txt file. The metadata is stored in the file between lines 5 and 17.
        """
        pass


class DataParser:
    """
    A class to parse and store data from .txt files into an HDF5 file.
    This class has no instance variables and methods are static.
    """
    def __init__(self):
        pass

    def txt2array(self, txt_file:str, no_zeros=False) -> np.ndarray:
        """ Used in the store_in_hdf5 method.
        Convert a .txt file to a 2D numpy array."""
        data = np.loadtxt(txt_file, skiprows=19)

        # Step 1: Determine the dimensions of the image
        max_row = int(np.max(data[:, 0])) + 1  # Adding 1 because index starts at 0
        max_col = int(np.max(data[:, 1])) + 1  # Adding 1 because index starts at 0

        # Step 2: Create an empty 2D array
        image_array = np.zeros((max_row, max_col), dtype=np.uint16)

        # Step 3: Fill the array with pixel values
        for row in data:
            r, c, val = int(row[0]), int(row[1]), row[2]
            image_array[r, c] = val + 1

        return image_array

    def txt2metadata(self, txt_file, start_line:int = 5, end_line:int = 17) -> dict:
        """ Used in the store_in_hdf5 method.
        Extract metadata from a .txt file. The metadata is stored in the file between lines 5 and 17.
        """
        camera_data = {}
        with open(txt_file, 'r') as f:
            lines = list(islice(f, start_line, end_line)) # read only the metadata lines
            for i, line in enumerate(lines):
                line = line.strip() # remove leading and trailing whitespaces
                if not line: # skip empty lines
                    continue    
                key, value = line.split(':', 1)
                camera_data[key.strip()] = value.strip()
                
        return camera_data  

    def extract_metadata_filename(self, filename):
        """ Used in the store_in_hdf5 method.
        Extract metadata from the filename of a .txt file."""
        # Split the filename at spaces
        components = filename.split()

        # Remove the '.txt' from the last component
        components[-1] = components[-1].replace('.txt', '')

        # Extract the bias, pol_posn, and flux
        bias = int(re.sub('\D', '', components[0])) # bias voltage
        pol_posn = re.sub('\d', '', components[1]) # polarizer position
        try: # somtimes the flux is not present
            flux = int(re.sub('\D', '', components[2])) # X-ray flux current
        except:
            flux = 0

        return {'pol_posn': pol_posn, 'bias': bias, 'flux': flux}


class HDF5Parser():

    def map_subdirectories(self, root_directory):  # Define a function to map the subdirectory tree of a given directory
        dir_tree = {}  # Initialize an empty dictionary to store the directory tree
        for name in os.listdir(root_directory):  # Iterate over each item in the directory
            if os.path.isdir(os.path.join(root_directory, name)):  # If the item is a directory
                dir_tree[name] = self.map_subdirectories(os.path.join(root_directory, name))

        return dir_tree  # Return the directory tree


    def store_in_hdf5(self, folderpath, hdf5_file):
        hdf5_file = h5py.File(hdf5_file, 'w')

        dir_tree = self.map_subdirectories(folderpath)  # Map the subdirectory tree of the root directory
        
        for sensor_id in dir_tree:  # Iterate over the sensor_id and files in the root directory
            logger.debug(f"{sensor_id = }")
            sensor_grp = hdf5_file.create_group(sensor_id)  # Create a group in the HDF5 file for the current directory
            # Iterate over the test_id in the current directory
            for test_id in dir_tree[sensor_id]:  
                # check if the subdirectory is a "test", otherwise skip
                if not test_id.lower().startswith("test"):
                    continue

                test_grp = sensor_grp.create_group(test_id) # Create sub-group for each test
                logger.debug(f"{test_id = }")

                for txt_filename in os.listdir(os.path.join(folderpath, sensor_id, test_id)):
                    if not txt_filename.endswith('.txt') or not txt_filename != 'P.txt': 
                        # check if the file is a .txt file, otherwise skip
                        continue

                    logger.debug(f"{txt_filename = }")
                    
                    txt_filepath = os.path.join(folderpath, sensor_id, test_id, txt_filename)
                    image_array = self.txt2array(txt_filepath)
                    camera_data = self.txt2metadata(txt_filepath)
                    l, w = image_array.shape
                    apparatus_data = self.extract_metadata_filename(txt_filename)
                    # logger.info(f"{apparatus_data = }")
                    # logger.info(f"{camera_data = }")
                    metadata = {**apparatus_data, **camera_data}
                    # logger.info(f"{metadata = }")
                    
                    image_grp = test_grp.create_group(txt_filename)
                    # load the image array into the HDF5 file
                    image_grp.create_dataset("raw_image", shape=(l,w), 
                                             dtype= np.int16, data=image_array)
                    # load the metadata into the HDF5 file
                    for k, v in metadata.items():
                        image_grp.attrs[k] = v

        hdf5_file.close()  # Close the HDF5 file
        logger.success(f"Data stored in {hdf5_file}")

    def show_hdf(self, hdf5_filepath):
        """ Display the contents of an HDF5 file.
        """
        with h5py.File(hdf5_filepath, 'r') as hdf_file:
            # List all the groups in the file
            for sensor_id in hdf_file:
                logger.info(sensor_id)
                for test_id in hdf_file[sensor_id]:
                    logger.info(test_id)
                    for txt_filename in hdf_file[sensor_id][test_id]:
                        logger.info(txt_filename)
                        # Load the image array
                        image_array = hdf_file[sensor_id][test_id][txt_filename]['raw_image'][:]
                        # Load the metadata
                        metadata = dict(hdf_file[sensor_id][test_id][txt_filename].attrs)
                        logger.debug(metadata)
                        logger.debug(image_array.shape)
                        # logger.debug(image_array.dtype)
                        # logger.debug(image_array)
                        print("-----")


# example usage
def main():
    DP = DataParser() # Create an instance of the DataParser class
    ## analyze the data folder
    # data_folder = r"big_data"
    # dir_tree = DP.map_subdirectories(data_folder)
    # pp(dir_tree)

    ## Create an HDF5 file and store the data
    # hdf5_filepath = r'big_data.hdf5'
    # DP.store_in_hdf5(data_folder, hdf5_filepath)
    # DP.load_hdf5(hdf5_filepath)
    
    HP = HDF5Parser()

    hdf5_filepath = r'big_data.hdf5'
    HP.show_hdf(hdf5_filepath)
    hdf_file = h5py.File(hdf5_filepath, 'r')
    # List all groups
    pp("Keys: %s" % hdf_file.keys())
    group_key = list(hdf_file.keys())[0]
    pp(f"{group_key = }")

    # Get the data
    data = hdf_file[group_key]["Test 1"]
    # pp("Keys: %s" % data.keys())
    print(data["0V Parallel.txt"].attrs.keys())
    print(data["0V Parallel.txt"].attrs.values())
    print(data["0V Parallel.txt"].attrs.items())
    # get raw image
    raw_image = data["0V Parallel.txt"]["raw_image"][:]
    plt.imshow(raw_image)
    plt.show()

    # Close the file after reading
    hdf_file.close()

if __name__ == "__main__":
    main()  # Run the main function