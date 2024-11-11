from tkinter import filedialog
import czifile
import tkinter as tk
import xml.etree.ElementTree as ET
import tifffile

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

def convert_czi_to_tiff_with_updated_metadata(czi_filepath, tiff_filepath):
    # Load CZI file
    with czifile.CziFile(czi_filepath) as czi:
        image_data = czi.asarray()
        metadata = czi.metadata()

    # Parse the original XML metadata (if needed, depending on your specific updates)
    root = ET.fromstring(metadata)

    # Example modification: updating the description or any other metadata
    for elem in root.iter('Description'):
        elem.text = 'Updated Description Here'

    # Convert the updated XML back to a string
    updated_metadata = ET.tostring(root, encoding='unicode')

    # Save image data and updated metadata to a new TIFF file
    tifffile.imsave(tiff_filepath, image_data, description=updated_metadata)

# Specify the path to your CZI file
file_dialog = tk.Tk()
file_dialog.withdraw()
file_path = filedialog.askopenfilename(filetypes=[("CZI files", "*.czi")])
convert_czi_to_tiff_with_updated_metadata(file_path, 'output.tiff')
