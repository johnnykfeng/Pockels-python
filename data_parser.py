from itertools import islice
import numpy as np
import re
import os
from pprintpp import pprint as pp
import h5py
from loguru_pack import logger, loguru_config

class DataParser:
    def __init__(self):
        pass

    def txt2array(self, txt_file:str, no_zeros=False) -> np.ndarray:

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

# DEPRECATED
    # def parse_datafolder(self, folder_path):

    #     # List all files in the directory
    #     files = os.listdir(folder_path)

    #     # Filter the list to include only '.txt' files
    #     txt_filenames = []
    #     image_arrays = []
    #     metadatas = []
    #     for f in files:
    #         if f.endswith('.txt') and f != 'P.txt':
    #             txt_filename = f
    #             txt_filepath = os.path.join(folder_path, txt_filename)
    #             image_array = self.txt2array(txt_filepath)
    #             camera_data = self.txt2metadata(txt_filepath)
    #             pol_posn, bias, flux = self.extract_metadata_filename(txt_filename)
    #             # metadata = {'bias': bias, 'pol_posn': pol_posn, 'flux': flux}

    #             txt_filenames.append(txt_filename)
    #             image_arrays.append(image_array)
    #             metadatas.append(metadata)

    #     results = {'txt_filenames': txt_filenames, 'image_arrays': image_arrays, 'metadatas': metadatas}
    #     return results


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
            grp = hdf5_file.create_group(sensor_id)  # Create a group in the HDF5 file for the current directory

            for test_id in dir_tree[sensor_id]:  # Iterate over the test_id in the current directory

                if not test_id.lower().startswith("test"): # check if the subdirectory is a "test", otherwise skip
                    continue

                test_grp = grp.create_group(test_id) # Create sub-group for each test
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
                    image_grp.create_dataset("raw_image", shape=(l,w), dtype= np.int16, data=image_array)
                    for k, v in metadata.items():
                        image_grp.attrs[k] = v

        hdf5_file.close()  # Close the HDF5 file
        logger.success(f"Data stored in {hdf5_file}")

    def load_hdf5(self, hdf5_filepath):
        """
        NEED TO REWRITE THIS FUNCTION
        """
        with h5py.File(hdf5_filepath, 'r') as f:
            txt_filenames = list(f.keys())
            image_arrays = []
            metadatas = []
            for txt_filename in txt_filenames:
                grp = f[txt_filename]
                image_array = grp['raw_image'][()]
                metadata = {k: grp.attrs[k] for k in grp.attrs}
                image_arrays.append(image_array)
                metadatas.append(metadata)

        logger.success(f"Data loaded from {hdf5_filepath}")
        results = {'txt_filenames': txt_filenames, 'image_arrays': image_arrays, 'metadatas': metadatas}
        return results

def main():
    DP = DataParser()
    data_folder = r"big_data"
    # results = DP.parse_datafolder(folder_path=data_folder)
    dir_tree = DP.map_subdirectories(data_folder)
    pp(dir_tree)

    hdf5_filepath = r'big_data.hdf5'
    DP.store_in_hdf5(data_folder, hdf5_filepath)



if __name__ == "__main__":
    main()  # Run the main function