from tkinter import filedialog
import czifile
import tkinter as tk

def read_czi(file_path):
    # Open the CZI file
    with czifile.CziFile(file_path) as czi:
        # Get image data
        image_data = czi.asarray()
        # Print out the shape of the data
        print("Image shape (dimensions):", image_data.shape)

        # Get metadata
        metadata = czi.metadata()
        print("Metadata:", metadata)

def convert_czi_tif(file_path):
    # Open the CZI file
    with czifile.CziFile(file_path) as czi:
        # Get image data
        image_data = czi.asarray()
        # Print out the shape of the data
        print("Image shape (dimensions):", image_data.shape)

        # Get metadata
        metadata = czi.metadata()
        print("Metadata:", metadata)

        # Save the image data as a TIFF file
        tif_file_path = file_path.replace(".czi", ".tif")
        czifile.imsave(tif_file_path, image_data)

# Specify the path to your CZI file
file_dialog = tk.Tk()
file_dialog.withdraw()
file_path = filedialog.askopenfilename(filetypes=[("CZI files", "*.czi")])
read_czi(file_path)
