import numpy as np
import os
import matplotlib.pyplot as plt
from DEPRECATED.load_files import txt2matrix

# data_folder = r"C:\Users\10552\OneDrive - Redlen Technologies\Code\Pockels_python\test_data\D325150"
data_folder = r"test_data\D325150"
filename = '0V Parallel.txt'
# filename = 'P.txt'
image_path = os.path.join(data_folder, filename)
# print(image_path)

parallel_0V_txt = os.path.join(data_folder, '0V Parallel.txt')
cross_0V_txt = os.path.join(data_folder, '0V Crossed.txt')
bias_txt = os.path.join(data_folder, "500V Crossed 0mA.txt")

parallel_0V_img = txt2matrix(parallel_0V_txt)
cross_0V_img = txt2matrix(cross_0V_txt)
bias_img = txt2matrix(bias_txt)

def convert_zeros_to_ones(arr):
    arr[arr == 0] = 1
    return arr

# parallel_0V_img = convert_zeros_to_ones(parallel_0V_img)

def count_zeros(arr):
    total_elements = arr.size
    non_zero_count = np.count_nonzero(arr)
    zero_count = total_elements - non_zero_count
    return zero_count

print("Number of zeros:", count_zeros(parallel_0V_img))
print("Number of zeros:", count_zeros(cross_0V_img))
print("Number of zeros:", count_zeros(bias_img))

def normalize(bias_img, cross_bcg, parallel_bcg):
    transmittance = (bias_img - cross_bcg)/parallel_bcg

    # transmittance = (bias_img - cross_bcg)/(parallel_bcg - cross_bcg)
    return transmittance

T = normalize(bias_img, cross_0V_img, parallel_0V_img)
# # Step 5: Plot the image
img = plt.imshow(T, cmap='viridis')  # 'gray' colormap for grayscale image
plt.colorbar(img)
plt.axis('off')  # Turn off axis numbers
plt.show()
