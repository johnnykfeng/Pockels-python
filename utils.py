import matplotlib.pyplot as plt

def plot_image(image_file, title=""):
    plt.figure("plot image utils")
    plt.title(title)
    plt.image(image_file)
    plt.show()

def plot_array(array, title=""):
    plt.figure("plot array utils")
    plt.title(title)
    plt.plot(array)
    plt.show()

def plot_two_arrays(array1, array2, title=""):
    plt.figure("plot array utils")
    plt.title(title)
    plt.plot(array1)
    plt.plot(array2)
    plt.show()