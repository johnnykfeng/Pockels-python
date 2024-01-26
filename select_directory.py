import tkinter as tk
from tkinter import filedialog

def select_directory():
    # Create a root window, but don't display it
    root = tk.Tk()
    root.withdraw()

    # Open the dialog to choose a directory
    directory = filedialog.askdirectory()

    # Close the root window
    root.destroy()

    return directory

# Ask the user to select a directory
selected_directory = select_directory()

# Print the selected directory
print(f"You selected: {selected_directory}")
