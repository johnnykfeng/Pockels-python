import numpy as np
import re
import os
from pprintpp import pprint as pp
import h5py

class DataParser:
    def __init__(self):
        pass

    def txt2array(self, txt_file:str, no_zeros=False) -> np.ndarray:
        """
        Convert a text file containing pixel data into a 2D array.

        Parameters:
        - txt_file (str): The path to the text file.
        - no_zeros (bool): If True, exclude zero-valued pixels from the array.

        Returns:
        - image_array (ndarray): The 2D array representing the image.

        """
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

        return pol_posn, bias, flux


    def parse_datafolder(self, folder_path):
        # List all files in the directory
        files = os.listdir(folder_path)

        # Filter the list to include only '.txt' files
        txt_filenames = []
        image_arrays = []
        metadatas = []
        for f in files:
            if f.endswith('.txt') and f != 'P.txt':
                txt_filename = f
                txt_filepath = os.path.join(folder_path, txt_filename)
                image_array = self.txt2array(txt_filepath)
                pol_posn, bias, flux = self.extract_metadata_filename(txt_filename)
                metadata = {'bias': bias, 'pol_posn': pol_posn, 'flux': flux}

                txt_filenames.append(txt_filename)
                image_arrays.append(image_array)
                metadatas.append(metadata)
        
        results = {'txt_filenames': txt_filenames, 'image_arrays': image_arrays, 'metadatas': metadatas}
        return results

    def store_in_hdf5(self, results, hdf5_file):

        with h5py.File(hdf5_file, 'w') as f:
            for i, txt_filename in enumerate(results['txt_filenames']):
                image_array = results['image_arrays'][i]
                metadata = results['metadatas'][i]
                grp = f.create_group(txt_filename)
                grp.create_dataset('raw_image', data=image_array)
                for k, v in metadata.items():
                    grp.attrs[k] = v

    def load_hdf5(self, hdf5_filepath):
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
        results = {'txt_filenames': txt_filenames, 'image_arrays': image_arrays, 'metadatas': metadatas}
        return results

def main():
    DP = DataParser()
    data_folder = r"test_data\D325150"
    results = DP.parse_datafolder(folder_path=data_folder)
    hdf5_filepath = r'test_data\D325150.hdf5'
    DP.store_in_hdf5(results, hdf5_filepath)
    results = DP.load_hdf5(hdf5_filepath)

    # pp(results['txt_filenames'])
    # pp(f"{results['txt_filenames'] = }")
    # pp(f"{results['metadatas'] = }")
    # pp(results['txt_filenames'])
    # pp(results['metadatas'])

if __name__ == "__main__":
    main()  # Run the main function