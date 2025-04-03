import os
import tkinter as tk
from tkinter import filedialog
from Data_access.metadata_manager import read_tiff_metadata
from Data_access.storm_db import STORMDatabaseManager

def remove_tif_extension(filename):
    """
    Removes only the '.tif' or '.tiff' extension from the filename while preserving other dots.
    """
    if filename.lower().endswith(".tif"):
        return filename[:-4]  # Remove last 4 characters (.tif)
    elif filename.lower().endswith(".tiff"):
        return filename[:-5]  # Remove last 5 characters (.tiff)
    return filename  # Return unchanged if no match

def extract_and_save_metadata(folder_path, db_manager, tif_file_path):
    """Extracts metadata and saves it to the database."""
    filename = os.path.basename(tif_file_path)

    #Extract relative path file
    rel_file_path = os.path.relpath(tif_file_path, folder_path)

    # Remove only the ".tif" or ".tiff" extension, keeping other dots in the filename
    filename = remove_tif_extension(filename)

    # Extract metadata for both tags
    pulsestorm_metadata = read_tiff_metadata(tif_file_path, 'pulsestorm')
    czi_pulsestorm_metadata = read_tiff_metadata(tif_file_path, 'czi-pulsestorm')

    # Add tag field to each entry before saving
    for entry in pulsestorm_metadata:
        entry['tag'] = 'pulsestorm'
    
    for entry in czi_pulsestorm_metadata:
        entry['tag'] = 'czi-pulsestorm'

    # Combine metadata from both tags
    all_metadata = pulsestorm_metadata + czi_pulsestorm_metadata

    # Save to database
    db_manager.save_metadata(all_metadata, filename, rel_file_path)

    print(f"Metadata for {filename} saved successfully.")


def process_all_tif_files(db_manager, folder_path):
    """Recursively searches for all .tif files in the given folder and processes them."""
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".tif", ".tiff")):
                tif_file_path = os.path.join(root, file)
                print(f"Processing: {tif_file_path}")
                extract_and_save_metadata(folder_path, db_manager, tif_file_path)

def clear_collections(storm_db):
        """Clears all documents from experiments, molecules, and localizations collections."""
        storm_db.experiments.delete_many({})
        storm_db.molecules.delete_many({})
        storm_db.localizations.delete_many({})
        print("All collections have been cleared.")


def main():
    """Main function to handle folder selection and metadata extraction for all .tif files."""
    # Initialize Tkinter to select folder
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    folder_path = filedialog.askdirectory(title="Select the main folder containing TIFF files")

    if not folder_path:
        print("No folder selected. Exiting.")
        return

    print(f"Selected folder: {folder_path}")

    # Initialize STORM database
    storm_db = STORMDatabaseManager()

    clear_collections(storm_db=storm_db)

    # Process all .tif files in the folder (recursively)
    process_all_tif_files(storm_db, folder_path)
    
    print("Process completed.")


if __name__ == "__main__":
    main()
